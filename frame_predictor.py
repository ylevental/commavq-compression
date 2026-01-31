"""
Frame-level predictor: trains a transformer to predict next frame tokens
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from datasets import load_dataset

print("Loading data...")
ds = load_dataset('commaai/commavq', data_files={'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}, split='train')
n_seg = 500
all_tokens = torch.stack([torch.tensor(np.array(ds[i]['token.npy'])) for i in range(n_seg)])
all_tokens = all_tokens.reshape(n_seg, 1200, 128)
print(f"Shape: {all_tokens.shape}, Raw: {all_tokens.numel() * 10 / 8 / 1024 / 1024:.2f} MB")

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

print(f"Using {torch.cuda.device_count()} GPUs!")
n_context = 8
model = FramePredictor(n_context=n_context, d_model=256, n_heads=4, n_layers=6)
model = nn.DataParallel(model)
model = model.cuda()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,} = {n_params*4/1024/1024:.2f} MB")

# Create training data - KEEP ON CPU
print("Creating training pairs (on CPU)...")
contexts = []
targets = []
for seg in range(n_seg):
    for t in range(n_context, 1200):
        contexts.append(all_tokens[seg, t-n_context:t])
        targets.append(all_tokens[seg, t])

contexts = torch.stack(contexts)  # CPU
targets = torch.stack(targets)    # CPU
print(f"Training samples: {len(contexts):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

batch_size = 512
n_epochs = 30

for epoch in range(n_epochs):
    start = time.time()
    perm = torch.randperm(len(contexts))
    total_loss = total_correct = total_tokens = 0
    
    for i in range(0, len(contexts), batch_size):
        idx = perm[i:i+batch_size]
        ctx = contexts[idx].cuda()  # Move batch to GPU
        tgt = targets[idx].cuda()
        
        optimizer.zero_grad()
        logits = model(ctx)
        loss = F.cross_entropy(logits.reshape(-1, 1024), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * tgt.numel()
        total_correct += (logits.argmax(-1) == tgt).sum().item()
        total_tokens += tgt.numel()
    
    scheduler.step()
    elapsed = time.time() - start
    acc = total_correct / total_tokens * 100
    avg_loss = total_loss / total_tokens
    bits = avg_loss / np.log(2)
    print(f"Epoch {epoch+1}: loss={avg_loss:.3f}, acc={acc:.1f}%, bits/tok={bits:.2f}, compress={10/bits:.2f}x, time={elapsed:.0f}s")

print(f"\nFinal: {n_params*4/1024/1024:.2f} MB model, {bits:.2f} bits/tok, {10/bits:.2f}x compression")

# Save model
torch.save(model.module.state_dict(), 'predictor.pt')
print("Saved predictor.pt")
