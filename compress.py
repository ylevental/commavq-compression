#!/usr/bin/env python3
"""
CommaVQ GPT + Arithmetic Coding Compression
============================================

This compresses the commaVQ dataset using:
1. The pretrained commaVQ GPT world model for probability prediction
2. Temporal biasing (exploiting 35% copy rate between frames)
3. Arithmetic coding for near-optimal entropy coding

Target: >2.6x compression ratio (beating ZPAQ's 2.2x)

Hardware Requirements:
- GPU with 4GB+ VRAM (Tesla P40 with 24GB is plenty)
- 16GB+ system RAM
- ~6-12 hours for full dataset on single GPU

Usage:
    # Test on one segment first (5-10 minutes)
    python compress_final.py --test --device cuda:0
    
    # Full compression
    python compress_final.py --device cuda:0
    
    # Multi-GPU (run in separate terminals)
    python compress_final.py --device cuda:0 --shard 0 --num-shards 3
    python compress_final.py --device cuda:1 --shard 1 --num-shards 3
    python compress_final.py --device cuda:2 --shard 2 --num-shards 3
"""

import os
import sys
import struct
import argparse
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time

# Import torch (required for this script)
import torch
import torch.nn.functional as F


# ===========================================================================
# Constants
# ===========================================================================

TOKENS_PER_FRAME = 128      # 8x16 spatial grid
FRAMES_PER_SEGMENT = 1200   # 1 minute at 20 FPS  
VOCAB_SIZE = 1024           # Token values 0-1023
BOS_TOKEN = 1024            # Beginning of sequence token
GPT_CONTEXT_FRAMES = 20     # GPT context window


# ===========================================================================
# Arithmetic Coding (copied here for self-contained submission)
# ===========================================================================

class ArithmeticCoder:
    """Base class with shared constants and CDF conversion."""
    PRECISION = 32
    WHOLE = (1 << 32) - 1
    HALF = 1 << 31
    QUARTER = 1 << 30
    
    def __init__(self, precision_bits: int = 16):
        self.precision_bits = precision_bits
        self.prob_scale = 1 << precision_bits
        
    def probs_to_cdf(self, probs: np.ndarray) -> np.ndarray:
        """Convert probability distribution to cumulative distribution function."""
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


class ArithmeticEncoder(ArithmeticCoder):
    """Arithmetic encoder using integer arithmetic."""
    
    def __init__(self, precision_bits: int = 16):
        super().__init__(precision_bits)
        self.reset()
        
    def reset(self):
        self.low = 0
        self.high = self.WHOLE
        self.pending_bits = 0
        self.output_bits: List[int] = []
        
    def encode_symbol(self, symbol: int, cdf: np.ndarray):
        """Encode a single symbol given its CDF."""
        range_size = self.high - self.low + 1
        self.high = self.low + (range_size * cdf[symbol + 1]) // cdf[-1] - 1
        self.low = self.low + (range_size * cdf[symbol]) // cdf[-1]
        self._renormalize()
        
    def _renormalize(self):
        while True:
            if self.high < self.HALF:
                self._output_bit_plus_pending(0)
            elif self.low >= self.HALF:
                self._output_bit_plus_pending(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                self.pending_bits += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
            
    def _output_bit_plus_pending(self, bit: int):
        self.output_bits.append(bit)
        for _ in range(self.pending_bits):
            self.output_bits.append(1 - bit)
        self.pending_bits = 0
        
    def finish(self) -> bytes:
        """Finish encoding and return compressed bytes."""
        self.pending_bits += 1
        if self.low < self.QUARTER:
            self._output_bit_plus_pending(0)
        else:
            self._output_bit_plus_pending(1)
        while len(self.output_bits) % 8 != 0:
            self.output_bits.append(0)
        result = bytearray()
        for i in range(0, len(self.output_bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.output_bits[i + j]
            result.append(byte)
        return bytes(result)


class ArithmeticDecoder(ArithmeticCoder):
    """Arithmetic decoder for decompression."""
    
    def __init__(self, data: bytes, precision_bits: int = 16):
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
            
    def _read_bit(self) -> int:
        if self.bit_pos < len(self.bits):
            bit = self.bits[self.bit_pos]
            self.bit_pos += 1
            return bit
        return 0
        
    def decode_symbol(self, cdf: np.ndarray) -> int:
        """Decode a single symbol given its CDF."""
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
# GPT Model Loading
# ===========================================================================

def load_gpt_model(device: str = 'cuda'):
    """Load the commaVQ GPT model."""
    # Import from commavq utils
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.gpt import GPT, GPTConfig
    
    print(f"Loading commaVQ GPT model on {device}...")
    config = GPTConfig()
    
    with torch.device('meta'):
        model = GPT(config)
    
    model.load_state_dict_from_url(
        'https://huggingface.co/commaai/commavq-gpt2m/resolve/main/pytorch_model.bin',
        assign=True
    )
    model = model.eval().to(device=device, dtype=torch.float32)
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {num_params/1e6:.1f}M parameters")
    
    return model, config


# ===========================================================================
# Compressor
# ===========================================================================

@dataclass
class CompressionConfig:
    """Configuration for compression."""
    # Temporal biasing
    temporal_boost: float = 3.0        # Boost factor for previous token value
    temporal_sigma: float = 30.0       # Spread of Gaussian boost
    
    # Arithmetic coding precision
    precision_bits: int = 16
    
    # Processing
    device: str = 'cuda'
    

class GPTCompressor:
    """
    Compresses token sequences using GPT + temporal biasing + arithmetic coding.
    """
    
    def __init__(self, config: CompressionConfig = None):
        if config is None:
            config = CompressionConfig()
        self.config = config
        
        self.model, self.gpt_config = load_gpt_model(config.device)
        self.device = config.device
        
    @torch.no_grad()
    def get_probs(self, context: 'torch.Tensor') -> np.ndarray:
        """Get probability distribution for next token."""
        if context.dim() == 1:
            context = context.unsqueeze(0)
        logits = self.model(context)
        logits = logits[0, -1, :VOCAB_SIZE]
        probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()
    
    def apply_temporal_bias(self, probs: np.ndarray, prev_token: int) -> np.ndarray:
        """
        Boost probabilities for tokens near the previous frame's value.
        
        This exploits the high temporal correlation in driving video:
        - 35% of tokens are identical to previous frame
        - Many more are within a small distance
        """
        # Gaussian-like boost centered on previous token
        distances = np.abs(np.arange(VOCAB_SIZE) - prev_token)
        boost = np.exp(-0.5 * (distances / self.config.temporal_sigma) ** 2)
        boost = 1 + (self.config.temporal_boost - 1) * boost
        
        # Extra boost for exact match
        boost[prev_token] *= 2.0
        
        # Apply and renormalize
        biased = probs * boost
        biased /= biased.sum()
        
        return biased
    
    @torch.no_grad()
    def get_frame_probs_batched(self, context_tokens: list, frame_tokens: np.ndarray) -> np.ndarray:
        """
        Get probability distributions for all 128 tokens in a frame with ONE forward pass.
        
        IMPORTANT: The GPT was trained with 129 tokens per frame:
        [BOS, tok1, tok2, ..., tok128] for each frame.
        So we must include BOS at the start of each frame!
        
        Args:
            context_tokens: List of previous tokens (formatted with BOS per frame)
            frame_tokens: The 128 data tokens of the current frame
            
        Returns:
            probs: Array of shape (128, 1024) - probability distribution for each token
        """
        # GPT max context is 2580 tokens (block_size = 20 * 129)
        # Each frame is 129 tokens (BOS + 128 data tokens)
        # We need room for context + current frame (129 tokens)
        max_context = 2580 - 129  # 2451
        
        # Trim context if needed (trim whole frames to maintain structure)
        if len(context_tokens) > max_context:
            # Keep only the last max_context tokens, but start from a BOS
            context_tokens = context_tokens[-(max_context):]
            # Make sure it starts with BOS
            if context_tokens[0] != BOS_TOKEN:
                # Find first BOS and trim to there
                for i, tok in enumerate(context_tokens):
                    if tok == BOS_TOKEN:
                        context_tokens = context_tokens[i:]
                        break
        
        # Build sequence for this frame: context + [BOS, frame_tokens...]
        frame_with_bos = [BOS_TOKEN] + frame_tokens.tolist()
        full_seq = context_tokens + frame_with_bos
        
        # Verify we don't exceed block_size
        if len(full_seq) > 2580:
            # Trim context more aggressively
            excess = len(full_seq) - 2580
            context_tokens = context_tokens[excess:]
            full_seq = context_tokens + frame_with_bos
        
        assert len(full_seq) <= 2580, f"Sequence too long: {len(full_seq)}"
        
        # Convert to tensor and run forward pass
        seq_tensor = torch.tensor(full_seq, device=self.device, dtype=torch.long).unsqueeze(0)
        logits = self.model(seq_tensor)  # (1, seq_len, vocab_size)
        
        # Extract logits for the 128 data token positions
        # Frame structure: [BOS, tok1, tok2, ..., tok128]
        # - logits[-129] predicts tok1 (position after BOS)
        # - logits[-128] predicts tok2
        # - ...
        # - logits[-2] predicts tok128
        # So we need logits[-129:-1] for the 128 data tokens
        frame_logits = logits[0, -129:-1, :VOCAB_SIZE]  # (128, 1024)
        
        # Convert to probabilities
        probs = F.softmax(frame_logits, dim=-1)
        
        return probs.cpu().numpy()
    
    def compress_segment(self, tokens: np.ndarray, verbose: bool = True) -> bytes:
        """
        Compress a single segment using BATCHED inference (much faster!).
        
        Args:
            tokens: Array of shape (1200, 128) or (1200, 8, 16)
            verbose: Print progress
            
        Returns:
            Compressed bytes
        """
        tokens = tokens.reshape(FRAMES_PER_SEGMENT, TOKENS_PER_FRAME)
        
        encoder = ArithmeticEncoder(self.config.precision_bits)
        coder = ArithmeticCoder(self.config.precision_bits)
        
        # Context starts empty (first frame will just have BOS)
        context_tokens = []
        prev_frame = None
        
        if verbose:
            from tqdm import tqdm
            frame_iter = tqdm(range(FRAMES_PER_SEGMENT), desc="Compressing")
        else:
            frame_iter = range(FRAMES_PER_SEGMENT)
        
        for frame_idx in frame_iter:
            frame_tokens = tokens[frame_idx]
            
            # Get probabilities for ALL 128 tokens in ONE forward pass
            frame_probs = self.get_frame_probs_batched(context_tokens, frame_tokens)
            
            # Now encode each token using the pre-computed probabilities
            for pos in range(TOKENS_PER_FRAME):
                token = int(frame_tokens[pos])
                probs = frame_probs[pos]
                
                # Apply temporal bias if we have a previous frame
                if prev_frame is not None:
                    prev_token = int(prev_frame[pos])
                    probs = self.apply_temporal_bias(probs, prev_token)
                
                # Encode
                cdf = coder.probs_to_cdf(probs)
                encoder.encode_symbol(token, cdf)
            
            # Update context: add BOS + frame tokens (129 tokens per frame)
            context_tokens.append(BOS_TOKEN)
            context_tokens.extend(frame_tokens.tolist())
            
            # Keep context bounded (will be trimmed in get_frame_probs_batched)
            max_context = 2580 - 129
            if len(context_tokens) > max_context + 129:  # Allow some slack
                context_tokens = context_tokens[-(max_context):]
            
            prev_frame = frame_tokens.copy()
        
        return encoder.finish()


# ===========================================================================
# Main Compression Pipeline
# ===========================================================================

def compress_dataset(args):
    """Compress the full dataset or a shard of it."""
    from datasets import load_dataset
    import shutil
    
    HERE = Path(__file__).resolve().parent
    output_dir = HERE / f'submission_shard{args.shard}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', data_files=data_files)
    
    # Determine shard range
    total = len(ds['train'])
    shard_size = total // args.num_shards
    start_idx = args.shard * shard_size
    end_idx = start_idx + shard_size if args.shard < args.num_shards - 1 else total
    
    print(f"Processing shard {args.shard}/{args.num_shards}: segments {start_idx} to {end_idx}")
    
    # Initialize compressor
    config = CompressionConfig(
        device=args.device,
        temporal_boost=args.temporal_boost,
        temporal_sigma=args.temporal_sigma,
    )
    compressor = GPTCompressor(config)
    
    # Compress segments
    total_original = 0
    total_compressed = 0
    
    for i in range(start_idx, end_idx):
        example = ds['train'][i]
        tokens = np.array(example['token.npy'])
        name = example['json']['file_name']
        
        print(f"\n[{i-start_idx+1}/{end_idx-start_idx}] {name}")
        
        start_time = time.time()
        compressed = compressor.compress_segment(tokens, verbose=True)
        elapsed = time.time() - start_time
        
        # Save
        with open(output_dir / name, 'wb') as f:
            f.write(struct.pack('III', *tokens.shape))
            f.write(compressed)
        
        # Stats
        original_bits = tokens.size * 10
        compressed_bits = len(compressed) * 8
        ratio = original_bits / compressed_bits
        total_original += original_bits
        total_compressed += compressed_bits
        
        print(f"  Ratio: {ratio:.2f}x | Time: {elapsed:.1f}s | Running: {total_original/total_compressed:.2f}x")
    
    print(f"\n{'='*50}")
    print(f"Shard {args.shard} complete!")
    print(f"  Segments: {end_idx - start_idx}")
    print(f"  Total ratio: {total_original/total_compressed:.2f}x")
    print(f"  Output: {output_dir}")


def test_single_segment(args):
    """Test on a single segment."""
    HERE = Path(__file__).resolve().parent.parent
    tokens = np.load(HERE / 'examples' / 'tokens.npy')
    
    print(f"Test segment shape: {tokens.shape}")
    print(f"Token range: [{tokens.min()}, {tokens.max()}]")
    
    config = CompressionConfig(
        device=args.device,
        temporal_boost=args.temporal_boost,
        temporal_sigma=args.temporal_sigma,
    )
    
    print(f"\nConfig: boost={config.temporal_boost}, sigma={config.temporal_sigma}")
    
    compressor = GPTCompressor(config)
    
    print("\nCompressing...")
    start_time = time.time()
    compressed = compressor.compress_segment(tokens, verbose=True)
    elapsed = time.time() - start_time
    
    original_bits = tokens.size * 10
    compressed_bits = len(compressed) * 8
    ratio = original_bits / compressed_bits
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Original:   {original_bits:,} bits ({original_bits//8:,} bytes)")
    print(f"Compressed: {compressed_bits:,} bits ({len(compressed):,} bytes)")
    print(f"Ratio:      {ratio:.2f}x")
    print(f"Time:       {elapsed:.1f}s")
    print()
    print(f"Leaderboard comparison:")
    print(f"  LZMA baseline: 1.6x")
    print(f"  ZPAQ:          2.2x")
    print(f"  This result:   {ratio:.2f}x {'✓' if ratio > 2.2 else ''}")
    print(f"  Target:        2.6-2.9x")


def main():
    parser = argparse.ArgumentParser(description="CommaVQ GPT Compression")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--test', action='store_true', help='Test on single segment')
    
    # Sharding for multi-GPU
    parser.add_argument('--shard', type=int, default=0, help='Shard index (0-based)')
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of shards')
    
    # Tunable parameters
    parser.add_argument('--temporal-boost', type=float, default=3.0,
                       help='Boost factor for temporal prediction')
    parser.add_argument('--temporal-sigma', type=float, default=30.0,
                       help='Spread of temporal boost')
    
    args = parser.parse_args()
    
    if args.test:
        test_single_segment(args)
    else:
        compress_dataset(args)


if __name__ == "__main__":
    main()
