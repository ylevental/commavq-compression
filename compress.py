"""
CommaVQ Fast Compression - Using constriction for entropy coding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
import sys
import time
import zlib
from pathlib import Path
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


def compress_segment(tokens, model, device, n_context=8):
    """Compress a segment using sequential predictions + constriction."""
    n_frames = tokens.shape[0]
    
    print("  Getting predictions (sequential)...")
    pred_start = time.time()
    
    # Use SINGLE predictions to match decode exactly
    all_probs = []
    with torch.no_grad():
        for t in range(n_context, n_frames):
            if t % 200 == 0:
                print(f"    Frame {t}/{n_frames}")
            ctx = tokens[t-n_context:t].unsqueeze(0).to(device)
            logits = model(ctx)
            probs = F.softmax(logits[0], dim=-1).cpu()  # (128, 1024)
            all_probs.append(probs)
    
    all_probs = torch.stack(all_probs)  # (n_frames - n_context, 128, 1024)
    print(f"  Predictions: {time.time() - pred_start:.1f}s")
    
    # Prepare data for entropy coding
    print("  Entropy coding...")
    code_start = time.time()
    
    # First n_context frames: store raw
    raw_tokens = tokens[:n_context].flatten().numpy().astype(np.int32)
    
    # Remaining frames: entropy code with predictions
    n_coded = (n_frames - n_context) * 128
    symbols = tokens[n_context:].flatten().numpy().astype(np.int32)
    
    # Flatten probs: (n_frames - n_context, 128, 1024) -> (n_coded, 1024)
    probs_flat = all_probs.reshape(-1, 1024).numpy().astype(np.float64)
    
    # Ensure valid probabilities
    probs_flat = np.clip(probs_flat, 1e-9, 1.0)
    probs_flat = probs_flat / probs_flat.sum(axis=1, keepdims=True)
    
    # Use constriction's ANS coder with categorical distribution
    encoder = constriction.stream.stack.AnsCoder()
    
    # Create categorical model family
    model_family = constriction.stream.model.Categorical(perfect=False)
    
    # Encode all symbols at once (encode_reverse handles LIFO ordering)
    encoder.encode_reverse(symbols, model_family, probs_flat)
    
    coded_data = encoder.get_compressed()
    print(f"  Entropy coding: {time.time() - code_start:.1f}s")
    
    return raw_tokens, coded_data


def compress_file(input_path, output_path, model_path, device='cuda'):
    """Compress a single file."""
    print(f"Loading model...")
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    state_dict = decompress_model(model_data)
    model = FramePredictor(n_context=8, d_model=256, n_heads=4, n_layers=6)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    print(f"Loading {input_path}...")
    original_shape = np.load(input_path).shape
    tokens = torch.tensor(np.load(input_path)).reshape(-1, 128)
    n_frames = tokens.shape[0]
    
    print(f"Compressing {n_frames} frames...")
    start_time = time.time()
    
    raw_tokens, coded_data = compress_segment(tokens, model, device)
    
    elapsed = time.time() - start_time
    print(f"Total compression: {elapsed:.1f}s ({n_frames/elapsed:.1f} frames/sec)")
    
    # Write output
    with open(output_path, 'wb') as f:
        f.write(b'CVQ3')  # New magic for constriction format
        f.write(struct.pack('I', len(original_shape)))
        for dim in original_shape:
            f.write(struct.pack('I', dim))
        f.write(struct.pack('I', len(model_data)))
        f.write(struct.pack('I', len(raw_tokens)))
        f.write(struct.pack('I', len(coded_data)))
        f.write(model_data)
        f.write(raw_tokens.astype(np.int16).tobytes())
        f.write(coded_data.tobytes())
    
    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    data_size = output_size - len(model_data)
    print(f"Compressed: {input_size} -> {output_size} bytes ({input_size/output_size:.2f}x)")
    print(f"  Model: {len(model_data)} bytes, Data: {data_size} bytes")


def compress_directory(input_dir, output_path, model_path, device='cuda', filenames=None):
    """Compress all segments in a directory."""
    input_dir = Path(input_dir)
    npy_files = sorted(input_dir.glob('*.npy'))
    
    print(f"Found {len(npy_files)} files")
    print(f"Loading model...")
    
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    state_dict = decompress_model(model_data)
    model = FramePredictor(n_context=8, d_model=256, n_heads=4, n_layers=6)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    all_raw = []
    all_coded = []
    all_shapes = []
    all_filenames = []
    
    total_start = time.time()
    
    for i, npy_file in enumerate(npy_files):
        print(f"\n[{i+1}/{len(npy_files)}] {npy_file.name}")
        
        original_shape = np.load(npy_file).shape
        tokens = torch.tensor(np.load(npy_file)).reshape(-1, 128)
        all_shapes.append(original_shape)
        
        # Store original filename if provided, else use index
        if filenames and i < len(filenames):
            all_filenames.append(filenames[i])
        else:
            all_filenames.append(npy_file.stem)
        
        raw_tokens, coded_data = compress_segment(tokens, model, device)
        
        all_raw.append(raw_tokens)
        all_coded.append(coded_data)
        
        print(f"  -> {len(coded_data) * 4} bytes")
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/len(npy_files):.1f}s/file)")
    
    # Write combined output
    with open(output_path, 'wb') as f:
        f.write(b'CVQ4')  # Multi-file with constriction
        f.write(struct.pack('I', len(npy_files)))
        f.write(struct.pack('I', len(model_data)))
        f.write(model_data)
        
        for shape, raw, coded, fname in zip(all_shapes, all_raw, all_coded, all_filenames):
            # Write filename
            fname_bytes = fname.encode('utf-8')
            f.write(struct.pack('I', len(fname_bytes)))
            f.write(fname_bytes)
            # Write shape
            f.write(struct.pack('I', len(shape)))
            for dim in shape:
                f.write(struct.pack('I', dim))
            f.write(struct.pack('I', len(raw)))
            f.write(struct.pack('I', len(coded)))
            f.write(raw.astype(np.int16).tobytes())
            f.write(coded.tobytes())
    
    output_size = os.path.getsize(output_path)
    print(f"\nTotal compressed: {output_size / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python compress.py <input.npy or dir> <output.bin> <model.bin>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_path = sys.argv[3]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if os.path.isdir(input_path):
        compress_directory(input_path, output_path, model_path, device)
    else:
        compress_file(input_path, output_path, model_path, device)
