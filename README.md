# CommaVQ Neural Compression

Lossless compression for the [comma.ai commaVQ challenge](https://github.com/commaai/commavq) using GPT world model + arithmetic coding.

## Results

| Method | Compression Ratio |
|--------|-------------------|
| LZMA baseline | 1.6x |
| ZPAQ | 2.2x |
| Leaderboard leader | 3.4x |
| **This implementation** | **3.55x** |

Tested on 2,250 segments (432 MB → 122 MB).

## Approach

1. **GPT World Model**: Use comma.ai's pretrained 307M parameter GPT to predict token probabilities conditioned on previous frames
2. **Temporal Biasing**: Boost probabilities near previous frame's token values (exploits ~35% frame-to-frame copy rate)
3. **Arithmetic Coding**: Encode tokens using predicted probability distributions for near-optimal compression

### Key Insight

The commaVQ dataset represents driving video as 1200 frames × 128 VQ tokens per segment. Adjacent frames are highly correlated, and the GPT model learns strong temporal priors. By using the model's probability predictions as the basis for arithmetic coding, we achieve compression close to the theoretical entropy limit.

## Usage
```bash
# Compress a segment
python compress.py --test --device cuda:0

# Compress full dataset with sharding (for multi-GPU)
python compress.py --shard 0 --num-shards 3 --device cuda:0
```

## Requirements

- PyTorch
- numpy
- tqdm
- datasets (HuggingFace)
- The commaVQ repository (for `utils/gpt.py`)

## Limitations

Decompression is ~128x slower than compression due to autoregressive token-by-token decoding (compression uses teacher forcing). On a Tesla P40, decompression takes ~10 hours per segment vs ~8 minutes for compression.

## Author

Yuval Levental
