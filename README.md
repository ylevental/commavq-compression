# CommaVQ Compression Challenge - 2.7x Solution

Lossless compression of 5000 minutes of driving video tokens using a self-compressed neural predictor + arithmetic coding.

## Results

| Metric | Value |
|--------|-------|
| Compression ratio | **2.7x** (official) |
| Raw data | 960 MB |
| Compressed | 353 MB |
| Model size | 4.5 MB (8-bit quantized) |

### Leaderboard Comparison
- 1st: szabolcs-cs - 3.4x
- 2nd: BradyWynn - 2.9x  
- **3rd: This solution - 2.7x**

## Approach

1. **Train a frame predictor** - Small transformer (5.3M params) that predicts all 128 tokens of a frame given 8 previous frames
2. **Quantize the model** - 8-bit quantization + zlib compression (20MB → 4.5MB)
3. **Arithmetic coding** - Use predictor probabilities with ANS entropy coding (constriction library)

The key insight: predict the whole frame at once (not autoregressive within frame), making inference 128x faster than token-by-token approaches.

## Hardware Requirements

- **Training**: 3 GPUs × ~19GB VRAM each (~57GB total), ~37 hours
- **Compression**: 1 GPU, ~220MB VRAM, ~12 hours  
- **Decompression**: 1 GPU, ~220MB VRAM, ~10 hours

Tested on 3x Tesla P40 (24GB each).

## Files

```
frame_predictor.py     # Train the predictor model
self_compress.py       # Quantize model to 8-bit
compress.py            # Compress data with trained model
decompress.py          # Decompress (standalone, for submission)
```

## Step-by-Step Replication

### Prerequisites

```bash
pip install torch numpy datasets constriction --break-system-packages
```

### Step 1: Train the Predictor (~37 hours)

```bash
nohup python -u frame_predictor.py > training.log 2>&1 &
tail -f training.log
```

This trains on 500 segments from the commaVQ dataset and produces:
- `predictor.pt` - 20MB model achieving 3.62 bits/token

Expected output:
```
Epoch 30: loss=2.511, acc=41.5%, bits/tok=3.62, compress=2.76x
Saved predictor.pt
```

### Step 2: Quantize the Model (~1 minute)

```bash
python self_compress.py
```

Produces:
- `predictor_compressed_8bit.bin` - 4.5MB quantized model

Expected output:
```
Original model: 20.21 MB
Quantizing to 8 bits...
Compressed size: 4.26 MB
Compression ratio: 4.75x
Prediction match rate: 91.0%
```

### Step 3: Extract Challenge Data (~10 minutes)

```bash
mkdir -p segments

python -c "
from datasets import load_dataset
import numpy as np
import json

data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
ds = load_dataset('commaai/commavq', data_files=data_files, split='train')
print(f'Total segments: {len(ds)}')

filenames = []
for i in range(len(ds)):
    if i % 100 == 0:
        print(f'Extracting {i}/{len(ds)}...')
    filenames.append(ds[i]['json']['file_name'])
    np.save(f'segments/{i:04d}.npy', np.array(ds[i]['token.npy']))

json.dump(filenames, open('filenames.json', 'w'))
print('Done')
"
```

### Step 4: Compress (~12 hours)

```bash
nohup python -u -c "
import json
from compress import compress_directory

filenames = json.load(open('filenames.json'))
compress_directory('segments', 'data.bin', 'predictor_compressed_8bit.bin', filenames=filenames)
" > compress.log 2>&1 &

tail -f compress.log
```

Produces:
- `data.bin` - ~339MB compressed data

### Step 5: Create Submission ZIP

```bash
zip -j compression_challenge_submission.zip data.bin decompress.py
ls -l compression_challenge_submission.zip
```

Expected: ~353MB ZIP file (shows as ~337 MiB)

## Verification

Use the official evaluation script from the [commaVQ repository](https://github.com/commaai/commavq):

```bash
git clone https://github.com/commaai/commavq
cd commavq
./compression/evaluate.sh /path/to/compression_challenge_submission.zip
```

Expected output:
```
compare (num_proc=40): 100%|██████████| 5000/5000 [00:31<00:00, 160.05 examples/s]
Compression rate: 2.7
```

## Potential Improvements

| Change | Potential Gain |
|--------|----------------|
| Train on all 5000 segments (not 500) | +0.1-0.2x |
| Use 16 context frames (not 8) | +0.1x |
| Larger model (10-20M params) | +0.1-0.2x |
| 6-bit quantization | +0.05x (if quality holds) |
| Parallel GPU decompression | 3x faster decompress |

## How It Works

### Frame Predictor Architecture

```
Input: 8 previous frames (8 × 128 tokens)
       ↓
Token Embedding (1024 → 256)
       ↓
Position Embedding (128 positions)
       ↓
Frame Embedding (8 frames)
       ↓
Transformer Encoder (6 layers, 4 heads)
       ↓
Output Projection (256 → 1024)
       ↓
Output: Probability distribution for next frame (128 × 1024)
```

- **Parameters**: 5.3M
- **Prediction**: All 128 tokens at once (not autoregressive)
- **Bits/token**: 3.62 (theoretical max compression ~2.76x)

### Compression Pipeline

```
Raw tokens (10 bits each)
       ↓
Frame predictor → probability distributions
       ↓
ANS entropy coding (constriction library)
       ↓
Compressed bitstream + quantized model
```

### Why This Works

1. **Temporal correlation**: Consecutive driving frames are very similar
2. **Learned predictions**: Model captures driving video patterns
3. **Efficient entropy coding**: ANS achieves near-optimal compression
4. **Small model overhead**: 4.5MB model amortized over 960MB data

## License

MIT
