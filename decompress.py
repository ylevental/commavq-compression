#!/usr/bin/env python3
"""
CommaVQ GPT + Arithmetic Coding Decompression

This is the decompression script to be included in the submission.
It reverses the compression done by compress_final.py.

Usage:
    OUTPUT_DIR=/path/to/output python decompress.py
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path

# Import torch directly (needed for decorator)
import torch
import torch.nn.functional as F

# Constants
TOKENS_PER_FRAME = 128
FRAMES_PER_SEGMENT = 1200
VOCAB_SIZE = 1024
BOS_TOKEN = 1024
GPT_CONTEXT_FRAMES = 20

# Compression config (must match compressor!)
TEMPORAL_BOOST = 3.0
TEMPORAL_SIGMA = 30.0
PRECISION_BITS = 16


# ===========================================================================
# Arithmetic Decoder
# ===========================================================================

class ArithmeticCoder:
    PRECISION = 32
    WHOLE = (1 << 32) - 1
    HALF = 1 << 31
    QUARTER = 1 << 30
    
    def __init__(self, precision_bits=16):
        self.precision_bits = precision_bits
        self.prob_scale = 1 << precision_bits
        
    def probs_to_cdf(self, probs):
        scaled = (probs * self.prob_scale).astype(np.int64)
        scaled = np.maximum(scaled, 1)
        total = scaled.sum()
        if total != self.prob_scale:
            diff = self.prob_scale - total
            max_idx = np.argmax(scaled)
            scaled[max_idx] += diff
        cdf = np.zeros(len(probs) + 1, dtype=np.int64)
        cdf[1:] = np.cumsum(scaled)
        return cdf


class ArithmeticDecoder(ArithmeticCoder):
    def __init__(self, data, precision_bits=16):
        super().__init__(precision_bits)
        self.bits = []
        for byte in data:
            for i in range(7, -1, -1):
                self.bits.append((byte >> i) & 1)
        self.bit_pos = 0
        self.low = 0
        self.high = self.WHOLE
        self.value = 0
        for _ in range(self.PRECISION):
            self.value = (self.value << 1) | self._read_bit()
            
    def _read_bit(self):
        if self.bit_pos < len(self.bits):
            bit = self.bits[self.bit_pos]
            self.bit_pos += 1
            return bit
        return 0
        
    def decode_symbol(self, cdf):
        range_size = self.high - self.low + 1
        offset = self.value - self.low
        scaled_value = ((offset + 1) * cdf[-1] - 1) // range_size
        symbol = int(np.searchsorted(cdf, scaled_value, side='right')) - 1
        symbol = max(0, min(symbol, len(cdf) - 2))
        self.high = self.low + (range_size * cdf[symbol + 1]) // cdf[-1] - 1
        self.low = self.low + (range_size * cdf[symbol]) // cdf[-1]
        self._renormalize()
        return symbol
        
    def _renormalize(self):
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.value -= self.HALF
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                self.value -= self.QUARTER
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
            self.value = 2 * self.value + self._read_bit()


# ===========================================================================
# GPT Model
# ===========================================================================

def load_gpt_model(device='cuda'):
    """Load the commaVQ GPT model."""
    # Try to import from utils
    try:
        from utils.gpt import GPT, GPTConfig
    except ImportError:
        # Add parent to path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from utils.gpt import GPT, GPTConfig
    
    print(f"Loading GPT model on {device}...")
    config = GPTConfig()
    
    with torch.device('meta'):
        model = GPT(config)
    
    model.load_state_dict_from_url(
        'https://huggingface.co/commaai/commavq-gpt2m/resolve/main/pytorch_model.bin',
        assign=True
    )
    model = model.eval().to(device=device, dtype=torch.float32)
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model


# ===========================================================================
# Decompressor
# ===========================================================================

class GPTDecompressor:
    """Decompresses segments using GPT + arithmetic decoding."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = load_gpt_model(device)
    
    @torch.no_grad()
    def get_probs(self, context):
        """Get probability distribution for the next token."""
        if context.dim() == 1:
            context = context.unsqueeze(0)
        logits = self.model(context)
        logits = logits[0, -1, :VOCAB_SIZE]
        probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()
    
    def apply_temporal_bias(self, probs, prev_token):
        """Must match compressor exactly!"""
        distances = np.abs(np.arange(VOCAB_SIZE) - prev_token)
        boost = np.exp(-0.5 * (distances / TEMPORAL_SIGMA) ** 2)
        boost = 1 + (TEMPORAL_BOOST - 1) * boost
        boost[prev_token] *= 2.0
        biased = probs * boost
        biased /= biased.sum()
        return biased
    
    def decompress_segment(self, data, original_shape):
        """
        Decompress a segment - must match compressor exactly!
        
        Note: This is slower than compression because we must decode token-by-token
        (can't use teacher forcing since we don't know the tokens yet).
        
        Args:
            data: Compressed bytes
            original_shape: Original token array shape
            
        Returns:
            Decompressed token array
        """
        from tqdm import tqdm
        
        decoder = ArithmeticDecoder(data, PRECISION_BITS)
        coder = ArithmeticCoder(PRECISION_BITS)
        
        # Context tracks: [BOS, frame1_tokens..., BOS, frame2_tokens..., ...]
        context_tokens = []
        decoded_frames = []
        prev_frame = None
        
        for frame_idx in tqdm(range(FRAMES_PER_SEGMENT), desc="Decompressing"):
            # Trim context BEFORE the frame (must match compressor!)
            max_context = 2580 - 129  # Leave room for BOS + 128 tokens
            if len(context_tokens) > max_context:
                context_tokens = context_tokens[-(max_context):]
                # Ensure starts with BOS
                if context_tokens[0] != BOS_TOKEN:
                    for i, tok in enumerate(context_tokens):
                        if tok == BOS_TOKEN:
                            context_tokens = context_tokens[i:]
                            break
            
            # Start frame with BOS (fixed context for all 128 tokens)
            frame_context = context_tokens + [BOS_TOKEN]
            frame_tokens = []
            
            for pos in range(TOKENS_PER_FRAME):
                # Build context: fixed frame_context + decoded tokens so far
                working_context = frame_context + frame_tokens
                
                # Get probabilities
                context_tensor = torch.tensor(working_context, device=self.device, dtype=torch.long)
                probs = self.get_probs(context_tensor)
                
                # Apply temporal bias (must match compressor!)
                if prev_frame is not None:
                    prev_token = int(prev_frame[pos])
                    probs = self.apply_temporal_bias(probs, prev_token)
                
                # Decode
                cdf = coder.probs_to_cdf(probs)
                token = decoder.decode_symbol(cdf)
                frame_tokens.append(token)
            
            decoded_frames.extend(frame_tokens)
            
            # Update context: add BOS + frame tokens
            context_tokens.append(BOS_TOKEN)
            context_tokens.extend(frame_tokens)
            
            prev_frame = np.array(frame_tokens, dtype=np.int64)
        
        tokens = np.array(decoded_frames, dtype=np.int16).reshape(original_shape)
        return tokens


def decompress_all():
    """Decompress all files in the current directory."""
    HERE = Path(__file__).resolve().parent
    output_dir = Path(os.environ.get('OUTPUT_DIR', HERE / 'decompressed'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Find compressed files
    compressed_files = [f for f in HERE.iterdir() 
                       if f.is_file() 
                       and not f.suffix 
                       and f.name not in ('decompress', 'decompress.py')]
    
    if not compressed_files:
        print("No compressed files found!")
        return
    
    print(f"Found {len(compressed_files)} files to decompress")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize decompressor
    decompressor = GPTDecompressor(device)
    
    # Decompress
    for filepath in compressed_files:
        print(f"\nDecompressing {filepath.name}...")
        
        with open(filepath, 'rb') as f:
            shape = struct.unpack('III', f.read(12))
            data = f.read()
        
        tokens = decompressor.decompress_segment(data, shape)
        np.save(output_dir / filepath.name, tokens)
    
    print(f"\nDone! Output in {output_dir}")


if __name__ == "__main__":
    decompress_all()
