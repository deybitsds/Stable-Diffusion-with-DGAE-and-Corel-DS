"""
Features:
- Cosine Annealing with Warm Restarts
- Early Stopping with oscillation detection
- Mixed precision training (FP16)
- Gradient accumulation
- Memory-efficient attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from collections import deque
import warnings


# ============================================================================
# 1. DIFFUSION SCHEDULE AND UTILITIES
# ============================================================================

class DiffusionSchedule:
    """Manages the noise schedule - DEFAULT: Linear (proven to work)"""
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type='linear'):
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == 'cosine':
            self.alphas_cumprod = self._cosine_beta_schedule(timesteps)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
            self.betas = 1 - (self.alphas_cumprod / self.alphas_cumprod_prev)
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)
            self.alphas = 1.0 - self.betas
        else:  # linear (DEFAULT)
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return alphas_cumprod[1:]


def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps."""
    batch_size = t.shape[0]
    out = a.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class EMA:
    """
    Exponential Moving Average 
    
    Updates shadow weights using: 
         shadow = decay * shadow + (1-decay) * current_weights
    where current_weights are the model parameters AFTER optimizer.step()
    """
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA shadow weights with current model weights (after backprop)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # param.data contains weights AFTER optimizer.step()
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Temporarily replace model weights with EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# EARLY STOPPING AND LEARNING RATE SCHEDULER
# ============================================================================

class EarlyStopping:
    """Early stopping with loss plateau and oscillation detection."""
    def __init__(self, patience=20, min_delta=1e-5, window_size=10, oscillation_threshold=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.oscillation_threshold = oscillation_threshold
        
        self.best_loss = float('inf')
        self.counter = 0
        self.loss_history = deque(maxlen=window_size)
        self.should_stop = False
        
    def __call__(self, loss):
        self.loss_history.append(loss)
        is_best = False
        reason = None
        
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            is_best = True
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            reason = f"Loss plateau detected ({self.patience} epochs without improvement)"
            return True, reason, is_best
        
        if len(self.loss_history) == self.window_size:
            mean_loss = np.mean(self.loss_history)
            std_loss = np.std(self.loss_history)
            
            if std_loss < self.oscillation_threshold and abs(loss - mean_loss) < self.oscillation_threshold:
                self.should_stop = True
                reason = f"Loss oscillating around {mean_loss:.4f} (std={std_loss:.6f})"
                return True, reason, is_best
        
        return False, None, is_best
    
    def get_stats(self):
        if len(self.loss_history) == 0:
            return None
        return {
            'mean': np.mean(self.loss_history),
            'std': np.std(self.loss_history),
            'best': self.best_loss,
            'patience_counter': self.counter
        }


# ============================================================================
# 2. U-NET ARCHITECTURE
# ============================================================================

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block - Standard diffusion model order.
    Order: conv → norm → activation (DDPM standard)
    """
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()
        num_groups_in = min(8, in_channels)
        while in_channels % num_groups_in != 0:
            num_groups_in -= 1
        num_groups_out = min(8, out_channels)
        while out_channels % num_groups_out != 0:
            num_groups_out -= 1
        
        # First conv path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups_out, out_channels)
        
        # Time embedding projection (no pre-activation)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        # Second conv path
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups_out, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        """CORRECTED forward: conv → norm → activation"""
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        time_emb_proj = self.time_mlp(time_emb)
        h = h + time_emb_proj[:, :, None, None]
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        
        # Residual connection
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Memory-efficient self-attention block.
    - Follows standard scaled dot-product attention
    - Chunked processing for memory efficiency
    """
    def __init__(self, channels):
        super().__init__()
        num_groups = min(8, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        
        # Chunked attention for memory efficiency
        if H * W > 1024:
            chunk_size = 512
            h_out = []
            for i in range(0, H * W, chunk_size):
                q_chunk = q[:, i:i+chunk_size]
                attn = torch.softmax(torch.bmm(q_chunk, k) / np.sqrt(C), dim=-1)
                h_chunk = torch.bmm(attn, v)
                h_out.append(h_chunk)
            h = torch.cat(h_out, dim=1)
        else:
            attn = torch.softmax(torch.bmm(q, k) / np.sqrt(C), dim=-1)
            h = torch.bmm(attn, v)
        
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        return x + self.proj(h)


class SmallUNet(nn.Module):
    """U-Net with proper residual block ordering."""
    def __init__(self, img_channels=3, base_channels=128, time_dim=256,
                 attention_resolutions=None, use_bottleneck_attention=True):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Downsampling
        self.down1 = nn.ModuleList([
            ResidualBlock(img_channels, base_channels, time_dim),
            ResidualBlock(base_channels, base_channels, time_dim),
        ])
        if attention_resolutions and 128 in attention_resolutions:
            self.down1.append(AttentionBlock(base_channels))
        
        self.down2 = nn.ModuleList([
            ResidualBlock(base_channels, base_channels * 2, time_dim),
            ResidualBlock(base_channels * 2, base_channels * 2, time_dim),
        ])
        if attention_resolutions and 64 in attention_resolutions:
            self.down2.append(AttentionBlock(base_channels * 2))
        
        self.down3 = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels * 4, time_dim),
            ResidualBlock(base_channels * 4, base_channels * 4, time_dim),
        ])
        if attention_resolutions and 32 in attention_resolutions:
            self.down3.append(AttentionBlock(base_channels * 4))
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 4, time_dim),
            ResidualBlock(base_channels * 4, base_channels * 4, time_dim),
        ])
        if use_bottleneck_attention:
            self.bottleneck.append(AttentionBlock(base_channels * 4))
        
        # Upsampling
        self.up3 = nn.ModuleList([
            ResidualBlock(base_channels * 8, base_channels * 4, time_dim),
            ResidualBlock(base_channels * 4, base_channels * 4, time_dim),
        ])
        if attention_resolutions and 32 in attention_resolutions:
            self.up3.append(AttentionBlock(base_channels * 4))
        
        self.up2 = nn.ModuleList([
            ResidualBlock(base_channels * 6, base_channels * 2, time_dim),
            ResidualBlock(base_channels * 2, base_channels * 2, time_dim),
        ])
        if attention_resolutions and 64 in attention_resolutions:
            self.up2.append(AttentionBlock(base_channels * 2))
        
        self.up1 = nn.ModuleList([
            ResidualBlock(base_channels * 3, base_channels, time_dim),
            ResidualBlock(base_channels, base_channels, time_dim),
        ])
        if attention_resolutions and 128 in attention_resolutions:
            self.up1.append(AttentionBlock(base_channels))
        
        # Output
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)
        
        # Pooling/Upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x, timesteps):
        time_emb = self.time_mlp(timesteps)
        
        # Encoder
        h1 = x
        for layer in self.down1:
            h1 = layer(h1, time_emb) if isinstance(layer, ResidualBlock) else layer(h1)
        h1_pool = self.pool(h1)
        
        h2 = h1_pool
        for layer in self.down2:
            h2 = layer(h2, time_emb) if isinstance(layer, ResidualBlock) else layer(h2)
        h2_pool = self.pool(h2)
        
        h3 = h2_pool
        for layer in self.down3:
            h3 = layer(h3, time_emb) if isinstance(layer, ResidualBlock) else layer(h3)
        h3_pool = self.pool(h3)
        
        # Bottleneck
        h = h3_pool
        for layer in self.bottleneck:
            h = layer(h, time_emb) if isinstance(layer, ResidualBlock) else layer(h)
        
        # Decoder
        h = self.upsample(h)
        h = torch.cat([h, h3], dim=1)
        for layer in self.up3:
            h = layer(h, time_emb) if isinstance(layer, ResidualBlock) else layer(h)
        
        h = self.upsample(h)
        h = torch.cat([h, h2], dim=1)
        for layer in self.up2:
            h = layer(h, time_emb) if isinstance(layer, ResidualBlock) else layer(h)
        
        h = self.upsample(h)
        h = torch.cat([h, h1], dim=1)
        for layer in self.up1:
            h = layer(h, time_emb) if isinstance(layer, ResidualBlock) else layer(h)
        
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# ============================================================================
# 3. IMAGE DATASET WITH AGGRESSIVE AUGMENTATION
# ============================================================================

class ImageDataset(Dataset):
    """
    Image Dataset with AGGRESSIVE augmentation for small dataset (less than thousands of images).
    
    For diffusion models on small datasets, we need heavy augmentation to prevent overfitting.
    """
    def __init__(self, image_dir, image_size=256, augmentation_level='aggressive'):
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            import glob
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} images in dataset")
        
        if augmentation_level == 'aggressive':
            # AGGRESSIVE augmentation for small datasets
            self.transform = transforms.Compose([
                # Random resize and crop (creates variation in scale)
                transforms.RandomResizedCrop(
                    image_size, 
                    scale=(0.8, 1.0),  # Random crop 80-100% of image
                    ratio=(0.95, 1.05)  # Slight aspect ratio variation
                ),
                
                # Geometric augmentations
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),  # Less common for natural images
                transforms.RandomRotation(degrees=15),  # ±15 degrees
                
                # Random affine transformation
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),  # 10% translation
                    scale=(0.9, 1.1),      # ±10% scale
                    shear=5                 # ±5 degrees shear
                ),
                # Random perspective (10% chance)
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                
                # Gaussian blur (10% chance)
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
                ], p=0.1),
                
                # Convert to tensor and normalize
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif augmentation_level == 'moderate':
            # Moderate augmentation
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:  # minimal
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


# ============================================================================
# 4. TRAINING
# ============================================================================

def train_diffusion_model(
    image_dir,
    output_dir='image_diffusion',
    num_epochs=500,
    batch_size=4,
    grad_accumulation_steps=4,
    learning_rate=2e-4,  # UPDATED: Increased from 1e-4 for better learning
    image_size=256,
    gpu_id=None,
    use_ema=True,  # Use --no-ema flag to disable
    ema_decay=0.995,
    schedule_type='linear',
    attention_resolutions=None,
    use_bottleneck_attention=False,  # Disable for small dataset
    use_mixed_precision=True,
    resume_checkpoint=None,
    use_lr_scheduler=True,
    lr_scheduler='cosine_warm_restarts',
    lr_warmup_epochs=10,
    lr_min=5e-5,  # UPDATED: Increased from 1e-6 (was too low!)
    early_stopping_patience=30,  # Higher for small dataset
    early_stopping_min_delta=1e-5,
    oscillation_window=10,
    oscillation_threshold=0.0005,
    augmentation_level='moderate'  # UPDATED: Changed from 'aggressive' for better stability
):
    """

    Train diffusion model on a given image dataset.  Optimized for
    small datasets with aggressive augmentation.

    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # Device setup
    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load dataset
    dataset = ImageDataset(image_dir, image_size, augmentation_level=augmentation_level)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stability
    )
    
    # Model and schedule
    model = SmallUNet(
        img_channels=3, 
        base_channels=128,
        attention_resolutions=attention_resolutions,
        use_bottleneck_attention=use_bottleneck_attention
    ).to(device)
    
    schedule = DiffusionSchedule(timesteps=1000, schedule_type=schedule_type)
    
    # Move schedule to device
    for attr in ['sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 
                 'betas', 'alphas', 'alphas_cumprod', 'posterior_variance']:
        setattr(schedule, attr, getattr(schedule, attr).to(device))
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning Rate Scheduler
    scheduler = None
    if use_lr_scheduler:
        if lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=lr_min
            )
            print(f"✓ Cosine Annealing LR scheduler")
        elif lr_scheduler == 'cosine_warm_restarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=2, eta_min=lr_min
            )
            print(f"✓ Cosine Annealing with Warm Restarts (RECOMMENDED)")
        elif lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, min_lr=lr_min, verbose=True
            )
            print(f"✓ ReduceLROnPlateau")
    
    # Early Stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        window_size=oscillation_window,
        oscillation_threshold=oscillation_threshold
    )
    print(f"✓ Early stopping (patience={early_stopping_patience})")
    
    # EMA
    ema = EMA(model, decay=ema_decay) if use_ema else None
    if ema:
        print(f"✓ EMA enabled (decay={ema_decay}, RECOMMENDED for quality)")
    
    # Mixed precision
    scaler = GradScaler() if use_mixed_precision else None
    if use_mixed_precision:
        print("✓ Mixed precision (FP16)")
    
    print()
    
    # Resume from checkpoint
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if ema and 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']
        print(f"Resumed from epoch {start_epoch}\n")
    
    # Training loop
    best_loss = float('inf')
    loss_history = []
    
    print("Starting training...\n")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device, non_blocking=True)
            batch_size_actual = batch.shape[0]
            
            # Sample timesteps
            t = torch.randint(0, schedule.timesteps, (batch_size_actual,), device=device).long()
            
            # Forward diffusion
            noise = torch.randn_like(batch)
            sqrt_alpha_t = extract(schedule.sqrt_alphas_cumprod, t, batch.shape)
            sqrt_one_minus_alpha_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, batch.shape)
            noisy_batch = sqrt_alpha_t * batch + sqrt_one_minus_alpha_t * noise
            
            # Mixed precision forward
            if use_mixed_precision:
                with autocast():
                    predicted_noise = model(noisy_batch, t)
                    loss = F.mse_loss(predicted_noise, noise)
                    loss = loss / grad_accumulation_steps
                scaler.scale(loss).backward()
            else:
                predicted_noise = model(noisy_batch, t)
                loss = F.mse_loss(predicted_noise, noise)
                loss = loss / grad_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                if use_mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # EMA update AFTER optimizer.step()
                if ema:
                    ema.update()
            
            epoch_loss += loss.item() * grad_accumulation_steps
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': loss.item() * grad_accumulation_steps, 'lr': f'{current_lr:.2e}'})
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        # Update LR scheduler
        if scheduler:
            if lr_scheduler == 'plateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check early stopping
        should_stop, stop_reason, is_best = early_stopping(avg_loss)
        
        # Get statistics
        stats = early_stopping.get_stats()
        stats_str = ""
        if stats:
            stats_str = f" | Mean: {stats['mean']:.4f} | Std: {stats['std']:.6f} | Best: {stats['best']:.4f} | Patience: {stats['patience_counter']}/{early_stopping_patience}"
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | LR: {current_lr:.2e}{stats_str}")
        
        # Save best model
        if is_best:
            print(f"✨ New best loss: {avg_loss:.4f}")
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
                'schedule_type': schedule_type,
                'attention_resolutions': attention_resolutions,
                'use_bottleneck_attention': use_bottleneck_attention,
            }
            if ema:
                checkpoint['ema_shadow'] = ema.shadow
            
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pt'))
        
        # Check if we should stop
        if should_stop:
            print(f"\n⚠️  Early stopping triggered: {stop_reason}")
            print(f"Best loss achieved: {best_loss:.4f}")
            break
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Checkpoint and samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
                'schedule_type': schedule_type,
                'attention_resolutions': attention_resolutions,
                'use_bottleneck_attention': use_bottleneck_attention,
            }
            if ema:
                checkpoint['ema_shadow'] = ema.shadow
            
            torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            
            # Generate samples
            print(f"Generating samples...")
            if ema:
                ema.apply_shadow()
            generate_samples(model, schedule, device, 
                           os.path.join(output_dir, 'samples', f'epoch_{epoch+1}.png'),
                           num_images=4)
            if ema:
                ema.restore()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
    
    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()
    
    print(f"\n✅ Training complete! Models saved to {output_dir}")
    print(f"Best loss: {best_loss:.4f}")


# ============================================================================
# 5. SAMPLING
# ============================================================================

@torch.no_grad()
def sample_images(model, schedule, num_images=4, image_size=256, device='cuda'):
    """Generate images using DDPM sampling."""
    model.eval()
    x = torch.randn(num_images, 3, image_size, image_size, device=device)
    
    for t in tqdm(reversed(range(schedule.timesteps)), desc="Sampling", total=schedule.timesteps):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)
        
        alpha_t = extract(schedule.alphas, t_batch, x.shape)
        beta_t = extract(schedule.betas, t_batch, x.shape)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
        sqrt_one_minus_alpha_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
        
        posterior_mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise)
        
        if t > 0:
            noise = torch.randn_like(x)
            posterior_variance_t = extract(schedule.posterior_variance, t_batch, x.shape)
            x = posterior_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            x = posterior_mean
    
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x


def generate_samples(model, schedule, device, save_path, num_images=4):
    """Generate and save sample images."""
    images = sample_images(model, schedule, num_images, device=device)
    
    fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 4))
    if num_images == 1:
        axes = [axes]
    
    for idx, ax in enumerate(axes):
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved samples to {save_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Diffusion Model")
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory containing ImageDataset images')
    parser.add_argument('--output-dir', type=str, default='image_diffusion',
                       help='Output directory for models and samples')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (4 works well)')
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)  # UPDATED from 1e-4
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--gpu-id', type=int, default=None)
    parser.add_argument('--no-ema', action='store_true',
                       help='Disable EMA (use this flag to turn OFF EMA for testing)')
    parser.add_argument('--ema-decay', type=float, default=0.995)
    parser.add_argument('--schedule', type=str, default='linear', choices=['cosine', 'linear'])
    parser.add_argument('--attention', type=str, default=None,
                       help='Attention resolutions (e.g., "32" or "32,64")')
    parser.add_argument('--no-bottleneck-attention', action='store_true', default=True,
                       help='Disable bottleneck attention (RECOMMENDED for small datasets)')
    parser.add_argument('--no-mixed-precision', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--lr-scheduler', type=str, default='cosine_warm_restarts',
                       choices=['cosine', 'cosine_warm_restarts', 'plateau', 'none'])
    parser.add_argument('--lr-min', type=float, default=5e-5)  # UPDATED from 1e-6
    parser.add_argument('--early-stopping-patience', type=int, default=30)
    parser.add_argument('--early-stopping-delta', type=float, default=1e-5)
    parser.add_argument('--oscillation-window', type=int, default=10)
    parser.add_argument('--oscillation-threshold', type=float, default=0.0005)
    parser.add_argument('--augmentation', type=str, default='moderate',  # UPDATED from 'aggressive'
                       choices=['aggressive', 'moderate', 'minimal'],
                       help='Augmentation level (moderate recommended for stability)')
    
    args = parser.parse_args()
    
    attention_resolutions = None
    if args.attention:
        attention_resolutions = tuple(int(x.strip()) for x in args.attention.split(','))
    
    train_diffusion_model(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        image_size=args.image_size,
        gpu_id=args.gpu_id,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        schedule_type=args.schedule,
        attention_resolutions=attention_resolutions,
        use_bottleneck_attention=not args.no_bottleneck_attention,
        use_mixed_precision=not args.no_mixed_precision,
        resume_checkpoint=args.resume,
        use_lr_scheduler=(args.lr_scheduler != 'none'),
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_delta,
        oscillation_window=args.oscillation_window,
        oscillation_threshold=args.oscillation_threshold,
        augmentation_level=args.augmentation
    )
