import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
from PIL import Image
from pathlib import Path
from accelerate.utils import set_seed
# Import shared model utilities (absolute imports)
from downstream.model_zoo import create_backbone
from downstream.dataset_utils import DATASET_CONFIGS


class ViTEncoder(nn.Module):
    def __init__(self, model_name="vit_b", pretrained=False, checkpoint=None, verbose=True):
        super().__init__()
        
        # Use shared model loading
        self.backbone, self.embed_dim, self.has_cls = create_backbone(
            model_name=model_name,
            pretrained=pretrained if pretrained else None,
            checkpoint=checkpoint,
            verbose=verbose,
        )

    def forward(self, x, layers=1):
        # x: [B, 3, H, W]
        # Linear approach: last layer (layers=1)
        # +ms approach: last 4 layers (layers=4)
        
        indices = list(range(-layers, 0))
        _, intermediates = self.backbone.forward_intermediates(
            x, 
            indices=indices,
            norm=True,
            return_prefix_tokens=False
        )
        
        all_patch_feats = []
        for p_feat in intermediates:
            if isinstance(p_feat, tuple):
                p_feat = p_feat[0] # [B, N, C]
                
            if p_feat.ndim == 3: # [B, N, C]
                B, N, C = p_feat.shape
                H = W = int(N**0.5)
                p_feat = p_feat.transpose(1, 2).reshape(B, C, H, W)
            
            all_patch_feats.append(p_feat)
            
        # Concatenate along channel dimension
        patch_feats = torch.cat(all_patch_feats, dim=1)
        return patch_feats

class LinearSegmentationHead(nn.Module):
    def __init__(self, embed_dim, num_classes=150):
        super().__init__()
        self.head = nn.Sequential(
            nn.SyncBatchNorm(embed_dim),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        )
        
    def forward(self, x, target_size=None):
        # x: [B, C, h, w]
        x = self.head(x)
        # Only interpolate if target_size is specified (for inference)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class ADE20KDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, split="train", transform=None):
        # Import here to avoid circular imports
        if root is None:
            root = DATASET_CONFIGS["ade20k"]["root"]
        self.root = root
        self.transform = transform
        
        # Map split names
        split_dir = "training" if split == "train" else "validation"
        
        self.img_dir = os.path.join(root, "images", split_dir)
        self.ann_dir = os.path.join(root, "annotations", split_dir)
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        ann_name = img_name.replace('.jpg', '.png')
        
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.ann_dir, ann_name)).convert("L")
        
        if self.transform:
            image, mask = self.transform(image, mask)
            
        mask = torch.from_numpy(np.array(mask)).long()
        mask = mask - 1  # 0 (Background) -> -1 (ignored)
        return image, mask

def multiscale_inference(encoder, head, image, num_layers, base_size, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    B, C, H, W = image.shape
    
    # Accumulate logits
    total_logits = None
    
    for scale in scales:
        # Scale image
        new_size = int(base_size * scale)
        # Make divisible by patch size (16)
        new_size = (new_size // 16) * 16
        if new_size < 16:
            new_size = 16
            
        scaled_img = F.interpolate(image, size=(new_size, new_size), mode='bilinear', align_corners=False)
        
        # Forward pass
        patch_feats = encoder(scaled_img, layers=num_layers)
        logits = head(patch_feats, target_size=(H, W))
        
        # Horizontal flip
        scaled_img_flip = torch.flip(scaled_img, dims=[3])
        patch_feats_flip = encoder(scaled_img_flip, layers=num_layers)
        logits_flip = head(patch_feats_flip, target_size=(H, W))
        logits_flip = torch.flip(logits_flip, dims=[3])
        
        # Average
        logits = (logits + logits_flip) / 2
        
        if total_logits is None:
            total_logits = logits
        else:
            total_logits = total_logits + logits
    
    # Average over scales
    total_logits = total_logits / len(scales)
    return total_logits

def get_transforms(img_size=512, split="train"):
    if split == "train":
        def transform(img, mask):
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.5, 1.0), ratio=(3./4., 4./3.))
            img = transforms.functional.resized_crop(img, i, j, h, w, (img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC)
            mask = transforms.functional.resized_crop(mask, i, j, h, w, (img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
            
            if torch.rand(1) > 0.5:
                img = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)
                
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            return img, mask
    else:
        def transform(img, mask):
            img = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC)(img)
            mask = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)(mask)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            return img, mask
    return transform

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint key (e.g., vit_b_dim_256)")
    parser.add_argument("--pretrained", type=str, default=None, choices=[None, "dino_v1", "ijepa"], 
                        help="Pre-trained weights to load. For DINOv2/MoCov3/MAE, use --model directly.")
    parser.add_argument("--model", type=str, default="vit_b",
                        help="Model architecture (e.g., vit_s/b/l/h, dinov2_vit_s/b/l/g, mocov3_vit_s/b, mae_vit_b/l/h, ibot_vit_s/b/l, data2vec_vit_b/l)")
    parser.add_argument("--setup", type=str, default="linear", choices=["linear", "ms"], 
                        help="Linear uses last layer, ms uses last 4 layers concatenated")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=None, help="Default 512 for linear, 640 for ms")
    parser.add_argument("--no_tta", action="store_true", help="Disable multiscale TTA for ms setup")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    # Check for built-in pretrained models
    is_dinov2 = args.model.startswith("dinov2_")
    is_mocov3 = args.model.startswith("mocov3_")
    is_mae = args.model.startswith("mae_")
    is_ibot = args.model.startswith("ibot_")
    is_data2vec = args.model.startswith("data2vec_")
    has_builtin_weights = is_dinov2 or is_mocov3 or is_mae or is_ibot or is_data2vec

    if args.checkpoint is None and args.pretrained is None and not has_builtin_weights:
        raise ValueError("Either --checkpoint or --pretrained must be specified (or use dinov2_*/mocov3_*/mae_*/ibot_*/data2vec_* model).")
    
    # Defaults from paper/image
    # Detect patch size for proper image size selection
    # Note: iBOT and data2vec 1.0 use patch16 for all models
    is_patch14 = (args.model in ["vit_h", "vit_l"] or 
                  args.model.startswith("dinov2_") or 
                  args.model == "mae_vit_h" or
                  args.pretrained == "ijepa")
    
    # BEiT-based models (data2vec 1.0) don't support dynamic image sizes - must use 224
    is_beit_based = is_data2vec
    
    if args.img_size is None:
        if is_beit_based:
            args.img_size = 224  # BEiT doesn't support dynamic sizes
            print(f"Note: {args.model} uses BEiT architecture which only supports 224x224 images.")
        elif args.setup == "ms":
            args.img_size = 644 if is_patch14 else 640  # 644 = 14*46, 640 = 16*40
        else:
            args.img_size = 518 if is_patch14 else 512  # 518 = 14*37, 512 = 16*32
    num_layers = 4 if args.setup == "ms" else 1
    
    accelerator = Accelerator(mixed_precision="bf16")
    
    # Model - use shared model loading (only print on main process)
    encoder = ViTEncoder(model_name=args.model, pretrained=args.pretrained, checkpoint=args.checkpoint, verbose=accelerator.is_main_process)
    head = LinearSegmentationHead(encoder.embed_dim * num_layers, num_classes=150) 
    
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Data
    train_ds = ADE20KDataset(split="train", transform=get_transforms(args.img_size, "train"))
    val_ds = ADE20KDataset(split="validation", transform=get_transforms(args.img_size, "val"))
    
    cpu_count = os.cpu_count()
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=cpu_count // accelerator.num_processes)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=cpu_count // accelerator.num_processes)
    
    # Optimizer (DINOv2 uses SGD with momentum)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(head.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    else:
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    
    # Polynomial LR decay (DINOv2 style)
    # Note: Using single-GPU step count intentionally - this results in slower LR decay
    # which empirically works better for linear segmentation probes
    total_steps = len(train_ds) // args.bs * args.epochs
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=0.9)
    
    # Prepare
    encoder, head, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        encoder, head, optimizer, train_loader, val_loader, scheduler
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1) 
    best_miou = 0.0
    
    for epoch in range(args.epochs):
        head.train()
        total_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(head):
                optimizer.zero_grad()
                patch_feats = encoder(images, layers=num_layers)
                logits = head(patch_feats, target_size=(args.img_size, args.img_size))
                
                loss = criterion(logits, masks)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Validation (mIoU)
        head.eval()
        all_preds = []
        all_masks = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating", disable=not accelerator.is_main_process):
                if args.setup == "ms" and not args.no_tta:
                    # Multiscale TTA for +ms setup
                    logits = multiscale_inference(encoder, head, images, num_layers, args.img_size)
                else:
                    # Standard inference for linear setup
                    patch_feats = encoder(images, layers=num_layers)
                    logits = head(patch_feats, target_size=(args.img_size, args.img_size))
                preds = logits.argmax(dim=1)
                
                # Use gather_for_metrics to correctly handle padding samples across GPUs
                # and move to CPU immediately to save VRAM
                all_preds.append(accelerator.gather_for_metrics(preds).cpu())
                all_masks.append(accelerator.gather_for_metrics(masks).cpu())
                
        if accelerator.is_main_process:
            # Concatenate on CPU
            all_preds = torch.cat(all_preds, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            
            # Optimized mIoU calculation using confusion matrix
            num_classes = 150
            # Filter out ignored index (-1)
            valid_mask = all_masks != -1
            all_preds = all_preds[valid_mask]
            all_masks = all_masks[valid_mask]
            
            # Compute confusion matrix: [num_classes, num_classes]
            # Fast way: flatten to 1D index
            indices = all_masks * num_classes + all_preds
            conf_mat = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
            
            # IoU = TP / (TP + FP + FN)
            intersection = conf_mat.diag()
            union = conf_mat.sum(0) + conf_mat.sum(1) - intersection
            
            iou = intersection / union.float().clamp(min=1e-6)
            # Only count classes that actually appeared in the validation set
            mask_counts = conf_mat.sum(1)
            valid_classes = mask_counts > 0
            miou = iou[valid_classes].mean().item()
            
            if miou > best_miou:
                best_miou = miou
            
            accelerator.print(f"Epoch {epoch+1} mIoU: {miou:.4f}")
    
    if accelerator.is_main_process:
        accelerator.print(f"\nBest mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    main()
