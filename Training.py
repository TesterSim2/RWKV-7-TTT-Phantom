"""
RWKV-7 TTT Phantom Training Pipeline
Implements pre-training with meta-learning objective and compression stage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import os
import json
from pathlib import Path
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
from dataclasses import asdict

# Import our model
from rwkv7_ttt_phantom import RWKV7TTTPhantom, RWKV7PhantomConfig, GhostLinear

warnings.filterwarnings('ignore')


class TextDataset(Dataset):
    """Simple text dataset for training"""
    def __init__(self, data_path: str, config: RWKV7PhantomConfig, seq_length: int = 512):
        self.config = config
        self.seq_length = seq_length
        
        # Load tokenized data (expecting .bin and .idx files)
        self.data = np.memmap(data_path + '.bin', dtype=np.uint16, mode='r')
        with open(data_path + '.idx', 'rb') as f:
            self.indices = np.frombuffer(f.read(), dtype=np.int64)
        
    def __len__(self):
        return len(self.indices) - 1
    
    def __getitem__(self, idx):
        start = self.indices[idx]
        end = self.indices[idx + 1]
        
        # Sample a random chunk
        if end - start > self.seq_length + 1:
            start_pos = np.random.randint(start, end - self.seq_length - 1)
            data = self.data[start_pos:start_pos + self.seq_length + 1]
        else:
            # Pad if necessary
            data = np.zeros(self.seq_length + 1, dtype=np.int64)
            actual_data = self.data[start:end]
            data[:len(actual_data)] = actual_data
        
        x = torch.from_numpy(data[:-1].astype(np.int64))
        y = torch.from_numpy(data[1:].astype(np.int64))
        
        return x, y


class RWKV7PhantomLightning(pl.LightningModule):
    """PyTorch Lightning module for RWKV-7 TTT Phantom"""
    def __init__(self, config: RWKV7PhantomConfig, learning_rate: float = 6e-4,
                 warmup_steps: int = 1000, weight_decay: float = 0.01,
                 beta1: float = 0.9, beta2: float = 0.99, adam_eps: float = 1e-8):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.model = RWKV7TTTPhantom(config)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
        
        # Meta-learning weight for TTT loss
        self.meta_loss_weight = config.ttt_loss_weight
        
        # Track training metrics
        self.train_loss_tracker = []
        self.val_loss_tracker = []
    
    def forward(self, x, state=None):
        return self.model(x, state)
    
    def compute_ttt_meta_loss(self, logits: torch.Tensor, next_logits: torch.Tensor,
                              state_before: List[torch.Tensor], state_after: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute meta-learning loss that encourages TTT gradient steps to improve next-token prediction
        Similar to MAML but applied to hidden states
        """
        if not self.config.ttt_enabled:
            return torch.tensor(0.0, device=logits.device)
        
        meta_loss = 0.0
        count = 0
        
        # For each layer with TTT
        for i in range(self.config.n_layer):
            tm_state_before = state_before[i * 3 + 2]
            tm_state_after = state_after[i * 3 + 2]
            
            # Compute improvement in state quality
            state_diff = (tm_state_after - tm_state_before).abs().mean()
            
            # Encourage meaningful but stable updates
            meta_loss += state_diff
            count += 1
        
        # Also measure improvement in next-token prediction
        with torch.no_grad():
            improvement = F.softmax(next_logits, dim=-1).max(dim=-1)[0] - \
                         F.softmax(logits, dim=-1).max(dim=-1)[0]
            meta_loss -= improvement.mean() * 0.1  # Reward improvement
        
        return meta_loss / count if count > 0 else torch.tensor(0.0, device=logits.device)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        B, T = x.shape
        
        # Initialize or get state
        state = self.model.init_state(B)
        
        # Forward pass with state tracking for meta-learning
        if self.config.ttt_enabled and np.random.random() < 0.1:  # Meta-learn on 10% of batches
            # Split sequence for meta-learning
            split_point = T // 2
            x1, x2 = x[:, :split_point], x[:, split_point:]
            y1, y2 = y[:, :split_point], y[:, split_point:]
            
            # First half
            logits1, state_mid = self.model(x1, state)
            loss1 = F.cross_entropy(logits1.reshape(-1, logits1.size(-1)), y1.reshape(-1))
            
            # Second half (benefits from TTT)
            state_mid_copy = [s.clone() for s in state_mid]
            logits2, state_final = self.model(x2, state_mid)
            loss2 = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), y2.reshape(-1))
            
            # Meta loss
            meta_loss = self.compute_ttt_meta_loss(logits1, logits2, state_mid_copy, state_final)
            
            # Total loss
            loss = loss1 + loss2 + self.meta_loss_weight * meta_loss
            
            # Log metrics
            self.log('train/loss', loss, prog_bar=True)
            self.log('train/base_loss', loss1 + loss2)
            self.log('train/meta_loss', meta_loss)
        else:
            # Standard forward pass
            logits, _ = self.model(x, state, return_state=False)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            self.log('train/loss', loss, prog_bar=True)
        
        # Track perplexity
        self.log('train/perplexity', torch.exp(loss), prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        logits, _ = self.model(x, return_state=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        # Metrics
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/perplexity', torch.exp(loss), prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Separate parameters by weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Don't decay biases, layer norms, embeddings
            if 'bias' in name or 'ln' in name or name.endswith('.weight') and 'emb' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Special handling for TTT parameters (higher LR)
        ttt_params = []
        if self.config.ttt_enabled:
            for name, param in self.model.named_parameters():
                if 'ttt' in name and param.requires_grad:
                    ttt_params.append(param)
        
        # Create optimizer groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        if ttt_params:
            optim_groups.append({
                'params': ttt_params,
                'weight_decay': 0.0,
                'lr': self.learning_rate * 2  # Higher LR for TTT components
            })
        
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.adam_eps
        )
        
        # Cosine annealing with warmup
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=100000, eta_min=self.learning_rate * 0.1),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]
    
    def optimizer_step(self, *args, **kwargs):
        # Linear warmup
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.warmup_steps))
            for pg in self.optimizers()[0].param_groups:
                pg['lr'] = lr_scale * self.learning_rate
        
        super().optimizer_step(*args, **kwargs)


class CompressionCallback(pl.Callback):
    """Callback to apply compression techniques after initial training"""
    def __init__(self, config: RWKV7PhantomConfig, compression_epoch: int = 50):
        self.config = config
        self.compression_epoch = compression_epoch
        self.compressed = False
    
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.compression_epoch and not self.compressed:
            print("\n=== Applying compression techniques ===")
            self.apply_compression(pl_module.model)
            self.compressed = True
    
    def apply_compression(self, model: RWKV7TTTPhantom):
        """Apply SVD compression and sparsity to model"""
        with torch.no_grad():
            # SVD compression for weight matrices
            if self.config.use_svd_compression:
                for name, module in model.named_modules():
                    if isinstance(module, GhostLinear):
                        self._compress_ghost_linear(module)
                    elif isinstance(module, nn.Linear) and module.weight.shape[0] > 256:
                        self._compress_linear_svd(module)
            
            # Apply sparsity masks
            if self.config.use_sparsity:
                self._apply_sparsity(model)
    
    def _compress_ghost_linear(self, module: GhostLinear):
        """Compress GhostLinear module using SVD"""
        # Compress core weights
        U, S, V = torch.svd(module.core_weight.data)
        rank = int(module.core_weight.shape[0] * self.config.svd_rank_ratio)
        module.core_weight.data = (U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].T).contiguous()
        
        # Compress ghost projection
        U, S, V = torch.svd(module.ghost_proj.data)
        rank = int(module.ghost_proj.shape[0] * self.config.svd_rank_ratio)
        module.ghost_proj.data = (U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].T).contiguous()
    
    def _compress_linear_svd(self, module: nn.Linear):
        """Compress linear layer using SVD"""
        U, S, V = torch.svd(module.weight.data)
        rank = int(min(module.weight.shape) * self.config.svd_rank_ratio)
        module.weight.data = (U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].T).contiguous()
    
    def _apply_sparsity(self, model: RWKV7TTTPhantom):
        """Apply structured sparsity to FFN layers"""
        for block in model.blocks:
            # Sparsify channel mixing weights
            key_weight = block.channel_mixing.key.core_weight.data
            value_weight = block.channel_mixing.value.core_weight.data
            
            # Compute importance scores
            key_importance = key_weight.abs().sum(dim=1)
            value_importance = value_weight.abs().sum(dim=0)
            
            # Create sparsity masks
            k_threshold = torch.quantile(key_importance, self.config.sparsity_ratio)
            v_threshold = torch.quantile(value_importance, self.config.sparsity_ratio)
            
            key_weight[key_importance < k_threshold] = 0
            value_weight[:, value_importance < v_threshold] = 0


def train_model(
    data_path: str,
    output_dir: str,
    config: Optional[RWKV7PhantomConfig] = None,
    batch_size: int = 16,
    learning_rate: float = 6e-4,
    num_epochs: int = 100,
    val_split: float = 0.05,
    use_wandb: bool = True,
    project_name: str = "rwkv7-ttt-phantom"
):
    """Main training function"""
    
    # Setup config
    if config is None:
        config = RWKV7PhantomConfig()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Initialize wandb
    if use_wandb:
        wandb.init(project=project_name, config=asdict(config))
    
    # Create datasets
    dataset = TextDataset(data_path, config)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = RWKV7PhantomLightning(config, learning_rate=learning_rate)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename='rwkv7-phantom-{epoch:02d}-{val_loss:.3f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Add compression callback if enabled
    if config.use_svd_compression or config.use_sparsity:
        callbacks.append(CompressionCallback(config, compression_epoch=int(num_epochs * 0.8)))
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,  # Use mixed precision
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,  # Effective batch size = 64
        log_every_n_steps=10,
        val_check_interval=0.25,  # Validate 4 times per epoch
        enable_progress_bar=True,
        logger=wandb.init() if use_wandb else None
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    torch.save(model.model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    print(f"\nTraining completed! Model saved to {output_dir}")
    
    return model


def load_pretrained_rwkv7(checkpoint_path: str, config: RWKV7PhantomConfig) -> RWKV7TTTPhantom:
    """Load pretrained RWKV-7 weights and adapt to Phantom architecture"""
    model = RWKV7TTTPhantom(config)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Adapt state dict to our architecture
    new_state_dict = {}
    for k, v in state_dict.items():
        # Handle naming differences
        if k.startswith('blocks.'):
            # Map RWKV-7 weights to our GhostRNN decomposition
            if 'att.receptance.weight' in k or 'att.key.weight' in k or \
               'att.value.weight' in k or 'att.output.weight' in k:
                # Split into core and ghost components
                base_name = k.replace('.weight', '')
                core_size = int(v.shape[0] * config.core_ratio)
                
                # Use SVD to initialize core and ghost
                if len(v.shape) == 2:
                    U, S, V = torch.svd(v)
                    core_weight = (U[:core_size, :] @ torch.diag(S[:core_size]) @ V[:, :core_size].T)
                    
                    # Initialize ghost projection
                    ghost_proj = U[core_size:, :core_size]
                    
                    new_state_dict[base_name + '.core_weight'] = core_weight
                    new_state_dict[base_name + '.ghost_proj'] = ghost_proj
                    new_state_dict[base_name + '.bias'] = torch.zeros(v.shape[0])
                else:
                    new_state_dict[k] = v
            else:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v
    
    # Load adapted weights
    model.load_state_dict(new_state_dict, strict=False)
    
    return model


# Example usage
if __name__ == "__main__":
    # Configuration for a small model suitable for A100 40GB
    config = RWKV7PhantomConfig(
        n_layer=12,
        n_embd=768,
        n_head=12,
        head_size=64,
        vocab_size=65536,
        ghost_ratio=0.75,
        core_ratio=0.25,
        ttt_enabled=True,
        ttt_lr=0.01,
        ttt_steps=1,
        max_depth_iter=1,
        use_svd_compression=True,
        svd_rank_ratio=0.5,
        use_sparsity=True,
        sparsity_ratio=0.3
    )
    
    print("RWKV-7 TTT Phantom Training Configuration:")
    print(json.dumps(asdict(config), indent=2))
    
    # Train model (example - requires actual data)
    # model = train_model(
    #     data_path="path/to/minipile",
    #     output_dir="./output/rwkv7-phantom",
    #     config=config,
    #     batch_size=16,
    #     learning_rate=6e-4,
    #     num_epochs=100
    # )
