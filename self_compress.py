"""
Self-compress the trained predictor model.
Applies quantization + entropy coding to shrink model weights.
"""
import torch
import torch.nn as nn
import numpy as np
import zlib
import struct

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


def quantize_weights(state_dict, bits=8):
    """Quantize weights to N bits per value."""
    quantized = {}
    scales = {}
    
    for name, tensor in state_dict.items():
        flat = tensor.float().flatten()
        min_val = flat.min().item()
        max_val = flat.max().item()
        
        # Scale to [0, 2^bits - 1]
        n_levels = 2**bits
        scale = (max_val - min_val) / (n_levels - 1) if max_val != min_val else 1.0
        
        quantized_flat = ((flat - min_val) / scale).round().clamp(0, n_levels - 1).to(torch.uint8 if bits <= 8 else torch.int16)
        
        quantized[name] = quantized_flat.numpy()
        scales[name] = (min_val, scale, tensor.shape)
    
    return quantized, scales


def dequantize_weights(quantized, scales):
    """Restore weights from quantized form."""
    state_dict = {}
    
    for name, q_array in quantized.items():
        min_val, scale, shape = scales[name]
        tensor = torch.from_numpy(q_array.astype(np.float32)) * scale + min_val
        state_dict[name] = tensor.reshape(shape)
    
    return state_dict


def compress_quantized(quantized, scales):
    """Entropy code the quantized weights using zlib."""
    # Pack everything into bytes
    data = b''
    
    # Header: number of tensors
    data += struct.pack('I', len(quantized))
    
    for name in quantized:
        q_array = quantized[name]
        min_val, scale, shape = scales[name]
        
        # Encode name
        name_bytes = name.encode('utf-8')
        data += struct.pack('I', len(name_bytes))
        data += name_bytes
        
        # Encode shape
        data += struct.pack('I', len(shape))
        for dim in shape:
            data += struct.pack('I', dim)
        
        # Encode scale info
        data += struct.pack('ff', min_val, scale)
        
        # Encode quantized data
        data += struct.pack('I', len(q_array))
        data += q_array.tobytes()
    
    # Compress with zlib (high compression)
    compressed = zlib.compress(data, level=9)
    return compressed


def decompress_quantized(compressed):
    """Decompress and restore quantized weights."""
    data = zlib.decompress(compressed)
    offset = 0
    
    n_tensors = struct.unpack_from('I', data, offset)[0]
    offset += 4
    
    quantized = {}
    scales = {}
    
    for _ in range(n_tensors):
        # Decode name
        name_len = struct.unpack_from('I', data, offset)[0]
        offset += 4
        name = data[offset:offset + name_len].decode('utf-8')
        offset += name_len
        
        # Decode shape
        n_dims = struct.unpack_from('I', data, offset)[0]
        offset += 4
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack_from('I', data, offset)[0])
            offset += 4
        shape = tuple(shape)
        
        # Decode scale info
        min_val, scale = struct.unpack_from('ff', data, offset)
        offset += 8
        
        # Decode quantized data
        arr_len = struct.unpack_from('I', data, offset)[0]
        offset += 4
        q_array = np.frombuffer(data[offset:offset + arr_len], dtype=np.uint8)
        offset += arr_len
        
        quantized[name] = q_array
        scales[name] = (min_val, scale, shape)
    
    return quantized, scales


def test_prediction_quality(model_orig, model_quant, device='cuda'):
    """Compare predictions between original and quantized model."""
    model_orig.eval()
    model_quant.eval()
    
    # Random test input
    test_input = torch.randint(0, 1024, (32, 8, 128)).to(device)
    
    with torch.no_grad():
        logits_orig = model_orig(test_input)
        logits_quant = model_quant(test_input)
    
    # Compare predictions
    pred_orig = logits_orig.argmax(-1)
    pred_quant = logits_quant.argmax(-1)
    
    match_rate = (pred_orig == pred_quant).float().mean().item()
    mse = ((logits_orig - logits_quant) ** 2).mean().item()
    
    return match_rate, mse


def main():
    print("Loading trained model...")
    model = FramePredictor(n_context=8, d_model=256, n_heads=4, n_layers=6)
    state_dict = torch.load('predictor.pt', map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Original size
    orig_size = sum(p.numel() * 4 for p in model.parameters())  # float32
    print(f"Original model: {orig_size / 1024 / 1024:.2f} MB")
    
    # Try different quantization levels
    for bits in [8, 6, 4]:
        print(f"\n{'='*50}")
        print(f"Quantizing to {bits} bits...")
        
        # Quantize
        quantized, scales = quantize_weights(state_dict, bits=bits)
        
        # Compress
        compressed = compress_quantized(quantized, scales)
        compressed_size = len(compressed)
        print(f"Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"Compression ratio: {orig_size / compressed_size:.2f}x")
        
        # Decompress and rebuild model
        q_restored, scales_restored = decompress_quantized(compressed)
        state_dict_restored = dequantize_weights(q_restored, scales_restored)
        
        model_quant = FramePredictor(n_context=8, d_model=256, n_heads=4, n_layers=6)
        model_quant.load_state_dict(state_dict_restored)
        
        # Test quality
        if torch.cuda.is_available():
            model.cuda()
            model_quant.cuda()
            match_rate, mse = test_prediction_quality(model, model_quant)
            model.cpu()
            model_quant.cpu()
        else:
            match_rate, mse = test_prediction_quality(model, model_quant, device='cpu')
        
        print(f"Prediction match rate: {match_rate*100:.1f}%")
        print(f"Logits MSE: {mse:.6f}")
        
        # Save best compressed model
        if bits == 8:  # Start with 8-bit as safe default
            with open(f'predictor_compressed_{bits}bit.bin', 'wb') as f:
                f.write(compressed)
            print(f"Saved predictor_compressed_{bits}bit.bin")

    print("\n" + "="*50)
    print("Summary:")
    print(f"Original: {orig_size / 1024 / 1024:.2f} MB")
    print("Run on server to test with real data!")


if __name__ == '__main__':
    main()
