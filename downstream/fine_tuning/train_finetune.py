"""
Full Fine-tuning for SSL Pretrained ViT on Multiple Datasets (Accelerate version)

Supported datasets:
- imagenet (ImageNet-1k): 1000 classes
- cifar10: 10 classes
- cifar100: 100 classes
- inat19 (iNaturalist 2019): 1010 species
- flowers (Oxford Flowers 102): 102 classes
- cars (Stanford Cars): 196 classes
- galaxy10: 10 galaxy morphology classes

Uses DeiT-style augmentation with Mixup/CutMix/RandAugment/RandomErasing.

Usage:
    accelerate launch --num_processes=8 train_finetune.py \
        --checkpoint /path/to/ckpt.pt --model vit_base_patch16_224 --dataset cifar100
"""

import warnings
warnings.filterwarnings("ignore", message=".*Metadata Warning.*")
warnings.filterwarnings("ignore", message=".*Corrupt EXIF data.*")

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from timm.data.mixup import Mixup
from sklearn.metrics import roc_auc_score
import tqdm
import os
from accelerate import Accelerator
from accelerate.utils import set_seed

# Import shared utilities (absolute imports)
from downstream.model_zoo import load_custom_checkpoint, VIT_CONFIGS, VISREG_CKPTS, DINOV2_CONFIGS
from downstream.dataset_utils import (
    DATASET_CONFIGS, 
    create_dataset, 
    get_num_classes,
    is_multi_label,
)


class ViTClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 1000, drop_path: float = 0.1, 
                 pretrained: str = None):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Determine timm model name and whether to load pretrained weights
        pretrained_timm = False
        
        self.is_dinov2 = model_name.startswith("dinov2_")
        
        if self.is_dinov2:
            # DINOv2: use timm pretrained weights directly
            if model_name not in DINOV2_CONFIGS:
                raise ValueError(f"Unknown DINOv2 model: {model_name}. Available: {list(DINOV2_CONFIGS.keys())}")
            timm_name, default_drop_path, _ = DINOV2_CONFIGS[model_name]
            drop_path = default_drop_path
            pretrained_timm = True
        elif pretrained == "dino_v1":
            # DINOv1: use timm pretrained weights
            if model_name == "vit_s":
                timm_name = "vit_small_patch16_224.dino"
            elif model_name == "vit_b":
                timm_name = "vit_base_patch16_224.dino"
            else:
                raise ValueError(f"DINOv1 weights only available for vit_s and vit_b, not {model_name}")
            pretrained_timm = True
            drop_path = 0.0  # DINOv1 doesn't use drop path
        elif model_name in VIT_CONFIGS:
            # Standard ViT (for custom checkpoints)
            timm_name, default_drop_path, _ = VIT_CONFIGS[model_name]
            drop_path = drop_path if drop_path != 0.1 else default_drop_path
        else:
            timm_name = model_name  # Use as-is if it's already a timm name
        
        # DINOv2 supports dynamic image sizes via positional embedding interpolation
        self.backbone = timm.create_model(
            timm_name, 
            pretrained=pretrained_timm, 
            num_classes=0,  # Remove default head
            drop_path_rate=drop_path,
            dynamic_img_size=self.is_dinov2,  # Enable dynamic input size for DINOv2
        )
        # SyncBatchNorm + Linear head (common for fine-tuning)
        self.head = nn.Sequential(
            nn.SyncBatchNorm(self.backbone.num_features),
            nn.Linear(self.backbone.num_features, num_classes),
        )
        # Initialize linear layer (DINO style: trunc_normal with std=0.01)
        nn.init.trunc_normal_(self.head[1].weight, std=0.01)
        nn.init.zeros_(self.head[1].bias)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
    
    def get_num_layers(self):
        if hasattr(self.backbone, 'blocks'):
            return len(self.backbone.blocks)
        return 12  # Default for ViT-B
    
    def needs_checkpoint_loading(self):
        return self.pretrained is None and not self.model_name.startswith("dinov2_")




def get_layer_id(name, num_layers):
    if 'cls_token' in name or 'pos_embed' in name or 'patch_embed' in name:
        return 0
    elif 'blocks.' in name:
        # Find blocks.X pattern anywhere in the name
        # e.g., "backbone.blocks.0.attn.qkv.weight" -> block 0
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part == 'blocks' and i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1]) + 1
        return num_layers
    else:
        return num_layers


def create_optimizer(model, lr, weight_decay, layer_decay, num_layers):
    # Layer-wise learning rate decay
    lr_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]
    
    # Group parameters by layer
    layer_params = {i: {'decay': [], 'no_decay': []} for i in range(num_layers + 1)}
    layer_params['head'] = {'decay': [], 'no_decay': []}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Head parameters get full LR
        if 'head' in name:
            if 'bias' in name:
                layer_params['head']['no_decay'].append(param)
            else:
                layer_params['head']['decay'].append(param)
        else:
            layer_id = get_layer_id(name, num_layers)
            # No weight decay for bias, norm layers, and positional embeddings
            if 'bias' in name or 'norm' in name or 'gamma' in name or 'beta' in name or 'pos_embed' in name or 'cls_token' in name:
                layer_params[layer_id]['no_decay'].append(param)
            else:
                layer_params[layer_id]['decay'].append(param)
    
    # Create param groups
    param_groups = []
    for layer_id in range(num_layers + 1):
        layer_lr = lr * lr_scales[layer_id]
        if layer_params[layer_id]['decay']:
            param_groups.append({
                'params': layer_params[layer_id]['decay'],
                'lr': layer_lr,
                'weight_decay': weight_decay,
            })
        if layer_params[layer_id]['no_decay']:
            param_groups.append({
                'params': layer_params[layer_id]['no_decay'],
                'lr': layer_lr,
                'weight_decay': 0.0,
            })
    
    # Head parameters (full LR)
    if layer_params['head']['decay']:
        param_groups.append({
            'params': layer_params['head']['decay'],
            'lr': lr,
            'weight_decay': weight_decay,
        })
    if layer_params['head']['no_decay']:
        param_groups.append({
            'params': layer_params['head']['no_decay'],
            'lr': lr,
            'weight_decay': 0.0,
        })
    
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=0.0):
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1e-6,
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps,
        eta_min=min_lr
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_steps]
    )


def train_one_lr(args, base_lr, accelerator, num_classes, num_workers, train_ds, val_ds, lr_suffix="", multi_label=False):
    set_seed(args.seed)
    
    # Create fresh model for this LR
    model = ViTClassifier(args.model, num_classes=num_classes, drop_path=args.drop_path,
                          pretrained=args.pretrained)
    
    # Load custom checkpoint if not using pretrained weights
    if model.needs_checkpoint_loading():
        load_custom_checkpoint(model.backbone, args.checkpoint)
    
    num_layers = model.get_num_layers()
    
    # Linear LR scaling
    effective_batch_size = args.batch_size * accelerator.num_processes
    lr_scale = effective_batch_size / args.base_batch_size
    scaled_lr = base_lr * lr_scale
    scaled_min_lr = args.min_lr * lr_scale
    
    # Create optimizer
    optimizer = create_optimizer(model, scaled_lr, args.weight_decay, args.layer_decay, num_layers)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=True,
        prefetch_factor=accelerator.num_processes * 2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=accelerator.num_processes * 2,
    )
    
    # Create scheduler
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, scaled_min_lr)
    
    # Prepare with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # Loss function and mixup setup
    mixup_fn = None
    if multi_label:
        # Multi-label: BCEWithLogitsLoss, no mixup
        criterion = nn.BCEWithLogitsLoss()
    elif args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
            mode='batch', label_smoothing=args.label_smoothing, num_classes=num_classes
        )
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Training loop
    best_acc = 0.0
    prev_best_ckpt = None
    
    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"Training with LR={base_lr:.1e} (scaled: {scaled_lr:.2e})")
        print(f"{'='*70}")
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_main_process)
        
        epoch_loss = 0.0
        num_batches = 0
        
        for imgs, labels in pbar:
            optimizer.zero_grad()
            if mixup_fn is not None:
                imgs, labels = mixup_fn(imgs, labels)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            num_batches += 1
            current_lr = optimizer.param_groups[-1]['lr']
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})
        
        avg_loss = epoch_loss / num_batches
        
        # Validation
        model.eval()
        if multi_label:
            # Multi-label: compute both Accuracy and AUC-ROC
            all_outputs = []
            all_labels_list = []
            with torch.no_grad():
                for imgs, labels in tqdm.tqdm(val_loader, desc="Validating", leave=False, disable=not accelerator.is_main_process):
                    outputs = model(imgs)
                    gathered_outputs, gathered_labels = accelerator.gather_for_metrics((outputs, labels))
                    all_outputs.append(gathered_outputs.cpu())
                    all_labels_list.append(gathered_labels.cpu())
            
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels_cat = torch.cat(all_labels_list, dim=0)
            
            # Compute accuracy (threshold=0.5)
            preds = (torch.sigmoid(all_outputs) > 0.5).float()
            ml_acc = (preds == all_labels_cat).float().mean().item()
            
            # Compute mean AUC-ROC (skip classes with no positive samples)
            valid_classes = []
            for c in range(num_classes):
                if all_labels_cat[:, c].sum() > 0 and all_labels_cat[:, c].sum() < len(all_labels_cat):
                    valid_classes.append(c)
            if valid_classes:
                ml_auc = roc_auc_score(
                    all_labels_cat[:, valid_classes].numpy(),
                    torch.sigmoid(all_outputs[:, valid_classes]).numpy(),
                    average='macro'
                )
            else:
                ml_auc = 0.0
            
            # Use AUC as primary metric for best model selection
            acc = ml_acc
            metric_str = f"Acc={ml_acc*100:.2f}%, AUC={ml_auc*100:.2f}%"
        else:
            # Single-label: compute accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in tqdm.tqdm(val_loader, desc="Validating", leave=False, disable=not accelerator.is_main_process):
                    outputs = model(imgs)
                    preds = outputs.argmax(dim=1)
                    all_preds, all_labels = accelerator.gather_for_metrics((preds, labels))
                    correct += (all_preds == all_labels).sum().item()
                    total += all_labels.size(0)
            acc = correct / total
            metric_str = f"Acc={acc*100:.2f}%"
        
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, {metric_str}", end="")
            if is_best:
                print(f" (NEW BEST!)")
            else:
                print()
        
        # Save best checkpoint
        if is_best:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if prev_best_ckpt is not None and os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                unwrapped_model = accelerator.unwrap_model(model)
                save_dict = {
                    'epoch': epoch + 1,
                    'model': unwrapped_model.state_dict(),
                    'best_acc': best_acc,
                    'base_lr': base_lr,
                    'args': vars(args),
                }
                ckpt_name = f"{args.dataset}{lr_suffix}_best_acc{best_acc*100:.2f}.pt"
                ckpt_path = os.path.join(args.output_dir, ckpt_name)
                torch.save(save_dict, ckpt_path)
                prev_best_ckpt = ckpt_path
    
    metric_name = "AUC" if multi_label else "Acc"  # Primary metric for model selection
    if accelerator.is_main_process:
        print(f"LR={base_lr:.1e} finished. Best {metric_name}: {best_acc*100:.2f}%\n")
    
    return best_acc, base_lr


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SSL pretrained ViT on multiple datasets")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help=f"Checkpoint key from VISREG_CKPTS or direct path. Available keys: {list(VISREG_CKPTS.keys())}")
    parser.add_argument("--pretrained", type=str, default=None, choices=["dino_v1"],
                        help="Pretrained weights: 'dino_v1' (for vit_s/vit_b). For DINOv2, use --model dinov2_vit_*")
    parser.add_argument("--model", type=str, default="vit_b", 
                        help=f"Model architecture. Short names: {list(VIT_CONFIGS.keys())}, DINOv2: {list(DINOV2_CONFIGS.keys())}")
    parser.add_argument("--dataset", type=str, default="imagenet", 
                        choices=list(DATASET_CONFIGS.keys()),
                        help=f"Dataset to fine-tune on. Options: {list(DATASET_CONFIGS.keys())}")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: derived from checkpoint path)")
    
    # Training hyperparameters (DINO defaults)
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate (for batch_size=256)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--base_batch_size", type=int, default=256, help="Reference batch size for LR scaling")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--layer_decay", type=float, default=0.75, help="Layer-wise LR decay")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--drop_path", type=float, default=0.1, help="Drop path rate")
    
    # Optional settings
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing (DeiT default: 0.1, set 0 to disable)")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (DINOv2 supports dynamic sizes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm (0 = disabled)")
    
    # DeiT-style augmentation
    parser.add_argument("--mixup", type=float, default=0.8, help="Mixup alpha (0 to disable)")
    parser.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha (0 to disable)")
    parser.add_argument("--mixup_prob", type=float, default=1.0, help="Probability of applying mixup/cutmix")
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5, help="Probability of switching between mixup and cutmix")
    
    # Grid LR search
    parser.add_argument("--grid_lr", action="store_true", help="Enable grid search over learning rates")
    parser.add_argument("--grid_lr_values", type=str, default="3e-4,2e-4,1e-4,5e-4,6e-4,7e-4",
                        help="Comma-separated list of base LRs to search (before scaling)")
    args = parser.parse_args()
    
    # Determine if using pretrained model (DINOv1 or DINOv2)
    is_dinov2 = args.model.startswith("dinov2_")
    uses_pretrained = args.pretrained is not None or is_dinov2
    
    # DINOv2 supports dynamic image sizes (default 224 is fine, but 518 was original training size)
    if is_dinov2 and args.img_size == 224:
        print(f"Note: DINOv2 supports dynamic sizes. Using {args.img_size}x{args.img_size} (original training: 518x518)")
    
    # Validate: need either checkpoint OR pretrained weights
    if args.checkpoint is None and not uses_pretrained:
        raise ValueError(
            "Must specify either --checkpoint (custom checkpoint) or "
            "--pretrained dino_v1 (with --model vit_s/vit_b) or "
            "--model dinov2_vit_* (DINOv2 pretrained)"
        )
    
    # Get dataset configuration
    num_classes = get_num_classes(args.dataset)
    multi_label = is_multi_label(args.dataset)

    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    
    set_seed(args.seed)
    
    # Determine output directory
    if args.checkpoint:
        # Resolve checkpoint key to actual path if needed
        if args.checkpoint in VISREG_CKPTS:
            ckpt_path = Path(VISREG_CKPTS[args.checkpoint])
        else:
            ckpt_path = Path(args.checkpoint)
        # Derive output directory from checkpoint path
        run_dir = ckpt_path.parent.parent
        if args.output_dir is None:
            args.output_dir = str(run_dir / f"finetune_{args.dataset}")
        elif not os.path.isabs(args.output_dir):
            args.output_dir = str(run_dir / args.output_dir)
    else:
        # For pretrained models, use default output directory
        if args.output_dir is None:
            pretrained_name = args.model if is_dinov2 else f"{args.model}_{args.pretrained}"
            args.output_dir = f"downstream/fine_tuning/outputs/{pretrained_name}/finetune_{args.dataset}"
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
        label_type = "multi-label" if multi_label else "single-label"
        print(f"Dataset: {args.dataset} ({num_classes} classes, {label_type})")
        if args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}")
        elif is_dinov2:
            print(f"Using DINOv2 pretrained weights: {args.model}")
        elif args.pretrained:
            print(f"Using pretrained weights: {args.pretrained}")
        print(f"Model: {args.model}")
    
    # Prepare datasets (shared across LR runs)
    train_ds = create_dataset(args.dataset, "train", args.img_size, use_deit_aug=True, combine_train_val=True)
    val_ds = create_dataset(args.dataset, "val", args.img_size, use_deit_aug=False)
    num_workers = max(1, os.cpu_count() // accelerator.num_processes)
    
    # Print configuration
    if accelerator.is_main_process:
        effective_batch_size = args.batch_size * accelerator.num_processes
        print(f"\n{'='*70}")
        print(f"Fine-tuning Configuration")
        print(f"{'='*70}")
        print(f"  Dataset:        {args.dataset} ({num_classes} classes)")
        print(f"  Image size:     {args.img_size}x{args.img_size}")
        print(f"  Epochs:         {args.epochs}")
        print(f"  Batch size:     {args.batch_size} x {accelerator.num_processes} GPUs = {effective_batch_size}")
        print(f"  Layer decay:    {args.layer_decay}")
        print(f"  Weight decay:   {args.weight_decay}")
        print(f"  Warmup epochs:  {args.warmup_epochs}")
        print(f"  Drop path:      {args.drop_path}")
        print(f"  Label smooth:   {args.label_smoothing}")
        print(f"  Grad clip:      {args.grad_clip}")
        if multi_label:
            print(f"  Mixup/CutMix:   DISABLED (multi-label dataset)")
        else:
            print(f"  Mixup alpha:    {args.mixup}")
            print(f"  CutMix alpha:   {args.cutmix}")
        if args.grid_lr:
            print(f"  Grid LR search: ENABLED")
            print(f"  LR values:      {args.grid_lr_values}")
        else:
            print(f"  Base LR:        {args.lr}")
        print(f"{'='*70}")
    
    if args.grid_lr:
        # Grid LR search mode
        lr_values = [float(x.strip()) for x in args.grid_lr_values.split(",")]
        results = []
        
        if accelerator.is_main_process:
            print(f"\n>>> Starting Grid LR Search over {len(lr_values)} values: {lr_values}")
        
        for i, lr in enumerate(lr_values):
            if accelerator.is_main_process:
                print(f"\n{'#'*70}")
                print(f"# Grid Search [{i+1}/{len(lr_values)}]: LR = {lr:.1e}")
                print(f"{'#'*70}")
            
            lr_suffix = f"_lr{lr:.0e}".replace(".", "p")
            best_acc, used_lr = train_one_lr(
                args, lr, accelerator, num_classes, num_workers, train_ds, val_ds, lr_suffix, multi_label
            )
            results.append((lr, best_acc))
        
        # Find best LR
        best_lr, best_acc = max(results, key=lambda x: x[1])
        metric_name = "AUC" if multi_label else "Acc"
        
        if accelerator.is_main_process:
            print(f"\n{'='*70}")
            print(f"GRID LR SEARCH COMPLETE")
            print(f"{'='*70}")
            print(f"Results:")
            for lr, acc in results:
                marker = " <-- BEST" if lr == best_lr else ""
                print(f"  LR={lr:.1e}: {metric_name}={acc*100:.2f}%{marker}")
            print(f"{'='*70}")
            print(f"Best LR: {best_lr:.1e} with {metric_name}: {best_acc*100:.2f}%")
            print(f"Checkpoints saved to: {args.output_dir}")
            print(f"{'='*70}")
    else:
        # Single LR mode
        best_acc, _ = train_one_lr(
            args, args.lr, accelerator, num_classes, num_workers, train_ds, val_ds, "", multi_label
        )
        
        metric_name = "AUC" if multi_label else "accuracy"
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Training complete!")
            print(f"Best {metric_name}: {best_acc*100:.2f}%")
            print(f"Checkpoints saved to: {args.output_dir}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
