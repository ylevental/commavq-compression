#!/usr/bin/env python3
"""
CommaVQ Decompression Script for Challenge Submission
Decompresses data.bin to OUTPUT_DIR with original filenames
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
import sys
import zlib

# Check for constriction - install if needed
try:
    import constriction
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'constriction'])
    import constriction


class FramePredictor(nn.Module):
    def __init__(self, n_context=8, d_model=256, n_heads=4, n_layers=6):
        super().__init__()
        self.n_context = n_context
        self.tok_embed = nn.Embedding(1024, d_model)
        self.pos_embed = nn.Embedding(128, d_model)
        self.frame_embed = nn.Embedding(n_context, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, 1024)
    
    def forward(self, context_frames):
        B, N, L = context_frames.shape
        x = self.tok_embed(context_frames)
        pos = torch.arange(L, device=x.device)
        x = x + self.pos_embed(pos)
        frame_idx = torch.arange(N, device=x.device)
        x = x + self.frame_embed(frame_idx).unsqueeze(0).unsqueeze(2)
        x = x.reshape(B, N * L, -1)
        x = self.transformer(x)
        x = x[:, -L:, :]
        return self.out_proj(x)


def decompress_model(compressed):
    """Decompress quantized model weights."""
    data = zlib.decompress(compressed)
    offset = 0
    
    n_tensors = struct.unpack_from('I', data, offset)[0]
    offset += 4
    
    state_dict = {}
    
    for _ in range(n_tensors):
        name_len = struct.unpack_from('I', data, offset)[0]
        offset += 4
        name = data[offset:offset + name_len].decode('utf-8')
        offset += name_len
        
        n_dims = struct.unpack_from('I', data, offset)[0]
        offset += 4
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack_from('I', data, offset)[0])
            offset += 4
        shape = tuple(shape)
        
        min_val, scale = struct.unpack_from('ff', data, offset)
        offset += 8
        
        arr_len = struct.unpack_from('I', data, offset)[0]
        offset += 4
        q_array = np.frombuffer(data[offset:offset + arr_len], dtype=np.uint8)
        offset += arr_len
        
        tensor = torch.from_numpy(q_array.astype(np.float32)) * scale + min_val
        state_dict[name] = tensor.reshape(shape)
    
    return state_dict


def decompress_segment(raw_tokens, coded_data, n_frames, model, device, n_context=8):
    """Decompress a segment using predictions + constriction."""
    tokens = torch.zeros(n_frames, 128, dtype=torch.long)
    
    # Restore raw frames
    tokens[:n_context] = torch.tensor(raw_tokens.reshape(n_context, 128))
    
    # Setup decoder
    decoder = constriction.stream.stack.AnsCoder(coded_data)
    model_family = constriction.stream.model.Categorical(perfect=False)
    
    # Decode frame by frame
    with torch.no_grad():
        for t in range(n_context, n_frames):
            ctx = tokens[t-n_context:t].unsqueeze(0).to(device)
            logits = model(ctx)
            probs = F.softmax(logits[0], dim=-1).cpu().numpy().astype(np.float64)
            
            # Ensure valid probabilities
            probs = np.clip(probs, 1e-9, 1.0)
            probs = probs / probs.sum(axis=1, keepdims=True)
            
            # Decode all 128 positions for this frame
            decoded = decoder.decode(model_family, probs)
            tokens[t] = torch.tensor(decoded)
    
    return tokens


def main():
    # Get output directory from command line or environment
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = os.environ.get('OUTPUT_DIR', './decompressed/')
    
    # Find data.bin in same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'data.bin')
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Reading {input_path}...")
    
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'CVQ4':
            raise ValueError(f"Invalid magic: {magic}, expected CVQ4")
        
        n_files = struct.unpack('I', f.read(4))[0]
        model_size = struct.unpack('I', f.read(4))[0]
        model_data = f.read(model_size)
        
        files_data = []
        for _ in range(n_files):
            fname_len = struct.unpack('I', f.read(4))[0]
            fname = f.read(fname_len).decode('utf-8')
            n_dims = struct.unpack('I', f.read(4))[0]
            shape = tuple(struct.unpack('I', f.read(4))[0] for _ in range(n_dims))
            raw_size = struct.unpack('I', f.read(4))[0]
            coded_size = struct.unpack('I', f.read(4))[0]
            raw_bytes = f.read(raw_size * 2)
            coded_bytes = f.read(coded_size * 4)
            
            raw_tokens = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.int32)
            coded_data = np.frombuffer(coded_bytes, dtype=np.uint32)
            files_data.append((fname, shape, raw_tokens, coded_data))
    
    print(f"Found {n_files} files to decompress")
    
    # Load model
    print("Loading model...")
    state_dict = decompress_model(model_data)
    model = FramePredictor(n_context=8, d_model=256, n_heads=4, n_layers=6)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (fname, shape, raw_tokens, coded_data) in enumerate(files_data):
        if i % 100 == 0:
            print(f"Decompressing {i}/{n_files}...")
        
        n_frames = shape[0] if len(shape) <= 2 else shape[0]
        tokens = decompress_segment(raw_tokens, coded_data, n_frames, model, device)
        
        tokens = tokens.reshape(shape).numpy().astype(np.int16)
        output_file = os.path.join(output_dir, fname)
        np.save(output_file, tokens)
        # np.save adds .npy extension, but evaluate.py expects no extension
        os.rename(output_file + '.npy', output_file)
    
    print(f"Done! Decompressed {n_files} files to {output_dir}")


if __name__ == '__main__':
    main()
