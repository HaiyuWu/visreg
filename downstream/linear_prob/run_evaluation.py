import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Import shared utilities (absolute imports)
from downstream.model_zoo import create_backbone
from downstream.dataset_utils import (
    DATASET_CONFIGS,
    create_dataset,
    get_num_classes,
    is_multi_label,
    DEFAULT_EVAL_DATASETS,
    OOD_EVAL_DATASETS,
)


class ViTEncoder(nn.Module):
    def __init__(self, model_name="vit_l", proj_dim=128, pretrained=False, checkpoint=None):
        super().__init__()
        
        self.is_dinov2 = model_name.startswith("dinov2_")
        
        # Use shared model loading
        self.backbone, self.embed_dim, self.has_cls = create_backbone(
            model_name=model_name,
            pretrained=pretrained if pretrained else None,
            checkpoint=checkpoint,
        )
        
        # DINOv2 models may have register tokens in addition to CLS
        self.num_prefix_tokens = getattr(self.backbone, 'num_prefix_tokens', 1 if self.has_cls else 0)
        
        if self.is_dinov2:
            print(f"Loaded DINOv2 model: embed_dim={self.embed_dim}, num_prefix_tokens={self.num_prefix_tokens}")

    def forward(self, x):
        # Feature extraction strategy (official protocols):
        # - DINO/DINOv2/models with CLS: Last 4 layers, CLS token only -> 4 * embed_dim
        # - I-JEPA (no CLS): Last 4 layers, avgpool only -> 4 * embed_dim
        _, intermediates = self.backbone.forward_intermediates(
            x, 
            indices=[-4, -3, -2, -1],  # Last 4 layers
            return_prefix_tokens=self.has_cls,
            norm=True,
        )
        features = []
        for layer_out in intermediates:
            if self.has_cls:
                # CLS token only per layer (official DINO/DINOv2 protocol)
                _, prefix_tokens = layer_out
                layer_feat = prefix_tokens[:, 0, :]  # CLS token [B, C]
            else:
                # No CLS token (e.g., I-JEPA): avgpool only (official I-JEPA protocol)
                # Paper: "concatenation of the last four layers of the average-pooled patch representations"
                patch_tokens = layer_out  # [B, C, H, W]
                layer_feat = patch_tokens.mean(dim=(2, 3))  # [B, C]
            features.append(layer_feat)
        return torch.cat(features, dim=-1)  # [B, 4 * embed_dim]

# --- Linear Classifier Container (Trains multiple LRs in parallel) ---

class MultiLRClassifiers(nn.Module):
    def __init__(self, in_dim, num_classes, lrs):
        super().__init__()
        self.classifiers = nn.ModuleList([
            nn.Sequential(nn.SyncBatchNorm(in_dim, affine=True), nn.Linear(in_dim, num_classes))
            for _ in lrs
        ])
        for c in self.classifiers:
            c[1].weight.data.normal_(mean=0.0, std=0.01)
            c[1].bias.data.zero_()
    
    def forward(self, x):
        return [c(x) for c in self.classifiers]


@torch.no_grad()
def extract_test_features(accelerator, encoder, loader):
    all_feats, all_labels = [], []
    encoder.eval()
    for x, y in tqdm.tqdm(loader, desc="Caching Test Features", disable=not accelerator.is_main_process):
        feat = encoder(x)
        all_feats.append(accelerator.gather(feat))
        all_labels.append(accelerator.gather(y))
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

def train_and_eval_simclr_protocol(accelerator, args, encoder, train_loader, val_feats, val_labels, 
                                     test_feats, test_labels, num_classes, is_multi_label=False):
    # DINOv2 Protocol: Official LR search space
    base_lrs = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
    if args.quick:
        base_lrs = [1e-2, 5e-3, 1e-3]
    
    # DINOv2 Protocol: Linear Scaling Rule (batch_size * gpus / 256)
    total_bs = args.batch_size * accelerator.num_processes
    lrs = [lr * total_bs / 256.0 for lr in base_lrs]
    
    # Use unwrapped model to access attributes when running with DDP
    unwrapped_encoder = accelerator.unwrap_model(encoder)
    # Feature dimension: 4 layers × embed_dim (CLS-only or avgpool-only)
    feat_dim = unwrapped_encoder.embed_dim * 4
    
    # Setup Parallel Classifiers
    multi_classifier = MultiLRClassifiers(feat_dim, num_classes, lrs).to(accelerator.device)
    # DINOv2 Protocol: Use SGD with Momentum (0.9) and weight_decay=0 for linear heads
    optimizers = [torch.optim.SGD(c.parameters(), lr=lr, momentum=0.9, weight_decay=0) for c, lr in zip(multi_classifier.classifiers, lrs)]
    
    # DINOv2 Protocol: Cosine Annealing Scheduler
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0) for opt in optimizers]
    
    # Prepare classifier and optimizers for distributed training
    multi_classifier = accelerator.prepare(multi_classifier)
    optimizers = [accelerator.prepare(opt) for opt in optimizers]
    
    # Selection of loss function
    criterion = F.binary_cross_entropy_with_logits if is_multi_label else F.cross_entropy
    
    for epoch in range(args.epochs):
        multi_classifier.train()
        for x, y in train_loader:
            with torch.no_grad(): 
                feat = encoder(x)
            
            outputs = multi_classifier(feat)
            for opt, out in zip(optimizers, outputs):
                opt.zero_grad()
                loss = criterion(out, y)
                accelerator.backward(loss)
                opt.step()
        
        # DINOv2 Protocol: Step schedulers after each epoch
        for s in schedulers:
            s.step()

    # Eval on VALIDATION set to select best LR
    multi_classifier.eval()
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=args.batch_size, shuffle=False)
    
    val_outputs = [[] for _ in range(len(lrs))]
    
    if accelerator.is_main_process:
        with torch.no_grad():
            for feat, y in val_loader:
                feat, y = feat.to(accelerator.device), y.to(accelerator.device)
                unwrapped_classifier = accelerator.unwrap_model(multi_classifier)
                logits_list = unwrapped_classifier(feat)
                for i, logits in enumerate(logits_list):
                    if is_multi_label:
                        val_outputs[i].append(torch.sigmoid(logits).cpu())
                    else:
                        val_outputs[i].append(logits.cpu())
    
        y_val_true = val_labels.cpu().numpy()
        best_idx = 0
        best_val_score = -1.0
        
        for i, lr in enumerate(lrs):
            y_pred_raw = torch.cat(val_outputs[i], dim=0).numpy()
            
            if is_multi_label:
                aucs = []
                for c in range(num_classes):
                    if len(np.unique(y_val_true[:, c])) > 1:
                        aucs.append(roc_auc_score(y_val_true[:, c], y_pred_raw[:, c]))
                score = np.mean(aucs) if aucs else 0.0
            else:
                score = (y_pred_raw.argmax(axis=1) == y_val_true).mean()
            
            if score > best_val_score:
                best_val_score = score
                best_idx = i
        
        print(f"  Best LR selected on val: {lrs[best_idx]:.1e} (val acc: {best_val_score:.4f})")

        # Final evaluation on TEST set using the best classifier
        test_loader = DataLoader(TensorDataset(test_feats, test_labels), batch_size=args.batch_size, shuffle=False)
        test_outputs = []
        
        with torch.no_grad():
            for feat, y in test_loader:
                feat = feat.to(accelerator.device)
                unwrapped_classifier = accelerator.unwrap_model(multi_classifier)
                # Only use the best classifier
                logits = unwrapped_classifier.classifiers[best_idx](feat)
                if is_multi_label:
                    test_outputs.append(torch.sigmoid(logits).cpu())
                else:
                    test_outputs.append(logits.cpu())
        
        y_test_true = test_labels.cpu().numpy()
        y_pred_best = torch.cat(test_outputs, dim=0).numpy()
        
        if is_multi_label:
            aucs = []
            for c in range(num_classes):
                if len(np.unique(y_test_true[:, c])) > 1:
                    aucs.append(roc_auc_score(y_test_true[:, c], y_pred_best[:, c]))
            auc_roc = np.mean(aucs) if aucs else 0.0
            
            y_pred_bin = (y_pred_best > 0.5).astype(float)
            acc = (y_pred_bin == y_test_true).all(axis=1).mean()
            precision = precision_score(y_test_true, y_pred_bin, average='macro', zero_division=0)
            recall = recall_score(y_test_true, y_pred_bin, average='macro', zero_division=0)
            
            results = {
                "acc": acc,
                "val_acc": best_val_score,
                "auc_roc": auc_roc,
                "precision": precision,
                "recall": recall,
                "best_lr": lrs[best_idx]
            }
            print(f"  TEST: AUC-ROC={auc_roc:.4f} | Acc={acc:.4f} | Prec={precision:.4f} | Rec={recall:.4f}")
        else:
            acc = (y_pred_best.argmax(axis=1) == y_test_true).mean()
            results = {
                "acc": acc,
                "val_acc": best_val_score,
                "best_lr": lrs[best_idx]
            }
            print(f"  TEST Acc={acc:.4f}")
            
        return results
    else:
        return None


def train_online_and_eval(accelerator, args, encoder, train_loader, test_feats, test_labels, num_classes, is_multi_label=False, model_type="default"):
    # Use DINOv2 LR grid for all models (unified evaluation protocol)
    base_lrs = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.3, 0.5]
    
    if args.quick:
        base_lrs = [0.1, 0.05, 0.01]
    
    # Linear Scaling Rule (batch_size * gpus / 256)
    total_bs = args.batch_size * accelerator.num_processes
    lrs = [lr * total_bs / 256.0 for lr in base_lrs]
    
    # Use unwrapped model to access attributes when running with DDP
    unwrapped_encoder = accelerator.unwrap_model(encoder)
    # Feature dimension: 4 layers × embed_dim (CLS-only or avgpool-only)
    feat_dim = unwrapped_encoder.embed_dim * 4
    
    # Setup Parallel Classifiers
    multi_classifier = MultiLRClassifiers(feat_dim, num_classes, lrs).to(accelerator.device)
    # DINOv2 Protocol: Use SGD with Momentum (0.9) and weight_decay=0 for linear heads
    optimizers = [torch.optim.SGD(c.parameters(), lr=lr, momentum=0.9, weight_decay=0) for c, lr in zip(multi_classifier.classifiers, lrs)]
    
    # DINOv2 Protocol: Cosine Annealing Scheduler
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0) for opt in optimizers]
    
    # Prepare classifier and optimizers for distributed training
    multi_classifier = accelerator.prepare(multi_classifier)
    optimizers = [accelerator.prepare(opt) for opt in optimizers]
    
    # Selection of loss function
    criterion = F.binary_cross_entropy_with_logits if is_multi_label else F.cross_entropy
    
    for epoch in range(args.epochs):
        multi_classifier.train()
        for x, y in train_loader:
            with torch.no_grad(): 
                feat = encoder(x)
            
            outputs = multi_classifier(feat)
            for opt, out in zip(optimizers, outputs):
                opt.zero_grad()
                loss = criterion(out, y)
                accelerator.backward(loss)
                opt.step()
        
        # DINOv2 Protocol: Step schedulers after each epoch
        for s in schedulers:
            s.step()

    # Eval
    multi_classifier.eval()
    # No need to prepare this TensorDataset loader as test_feats are already gathered/static
    test_loader = DataLoader(TensorDataset(test_feats, test_labels), batch_size=args.batch_size, shuffle=False)
    
    # We want to find the best head and then calculate detailed metrics for it
    # We'll use AUC-ROC as the selector for multi-label, and Accuracy for single-label
    performance_scores = [0.0] * len(lrs)
    all_outputs = [[] for _ in range(len(lrs))]
    
    # We only need to run evaluation on the main process since test_feats are mirrored
    if accelerator.is_main_process:
        with torch.no_grad():
            for feat, y in test_loader:
                feat, y = feat.to(accelerator.device), y.to(accelerator.device)
                # Unwrap to get the original model list
                unwrapped_classifier = accelerator.unwrap_model(multi_classifier)
                logits_list = unwrapped_classifier(feat)
                for i, logits in enumerate(logits_list):
                    if is_multi_label:
                        all_outputs[i].append(torch.sigmoid(logits).cpu())
                    else:
                        all_outputs[i].append(logits.cpu())
    
        y_true = test_labels.cpu().numpy()
        best_idx = 0
        best_val = -1.0
        
        results = {}
        
        for i, lr in enumerate(lrs):
            y_pred_raw = torch.cat(all_outputs[i], dim=0).numpy()
            
            if is_multi_label:
                # For multi-label, calculate macro AUC-ROC as the primary selector
                aucs = []
                for c in range(num_classes):
                    if len(np.unique(y_true[:, c])) > 1:
                        aucs.append(roc_auc_score(y_true[:, c], y_pred_raw[:, c]))
                score = np.mean(aucs) if aucs else 0.0
            else:
                # For single-label, calculate accuracy as the primary selector
                score = (y_pred_raw.argmax(axis=1) == y_true).mean()
            
            performance_scores[i] = score
            if score > best_val:
                best_val = score
                best_idx = i

        # Final detailed metrics for the best performing head
        y_pred_best = torch.cat(all_outputs[best_idx], dim=0).numpy()
        
        if is_multi_label:
            # AU-ROC
            aucs = []
            for c in range(num_classes):
                if len(np.unique(y_true[:, c])) > 1:
                    aucs.append(roc_auc_score(y_true[:, c], y_pred_best[:, c]))
            auc_roc = np.mean(aucs) if aucs else 0.0
            
            # Accuracy (Exact Match), Precision, Recall (at 0.5 threshold)
            y_pred_bin = (y_pred_best > 0.5).astype(float)
            acc = (y_pred_bin == y_true).all(axis=1).mean()
            precision = precision_score(y_true, y_pred_bin, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred_bin, average='macro', zero_division=0)
            
            results = {
                "acc": acc,
                "auc_roc": auc_roc,
                "precision": precision,
                "recall": recall,
                "best_lr": lrs[best_idx]
            }
            print(f"  Best Head (LR={lrs[best_idx]:.1e}) | AUC-ROC={auc_roc:.4f} | Acc={acc:.4f} | Prec={precision:.4f} | Rec={recall:.4f}")
        else:
            acc = (y_pred_best.argmax(axis=1) == y_true).mean()
            results = {
                "acc": acc,
                "best_lr": lrs[best_idx]
            }
            print(f"  Best Head (LR={lrs[best_idx]:.1e}) | Acc={acc:.4f}")
            
        return results
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None, choices=[None, "dino_v1", "ijepa"], 
                        help="Pre-trained weights to load (e.g., dino_v1, ijepa). For DINOv2/MoCov3, use --model directly.")
    parser.add_argument("--model", type=str, default="vit_l",
                        help="Model architecture. Options: vit_s/b/l/h, dinov2_vit_s/b/l/g, mocov3_vit_s/b, mae_vit_b/l/h, ibot_vit_s/b/l, data2vec_vit_b/l")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--datasets", type=str, default="default", help="Comma-separated list of datasets. 'default' runs only datasets with train/val/test splits (SimCLR protocol). 'all' runs everything. 'hf' runs HuggingFace datasets only.")
    parser.add_argument("--shots", type=int, nargs="+", default=[-1], help="Shot settings; -1 = full train set (e.g. --shots 1 10 -1).")
    parser.add_argument("--output", type=str, default="downstream/results_summary.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Models with built-in pretrained weights
    is_dinov2 = args.model.startswith("dinov2_")
    is_mocov3 = args.model.startswith("mocov3_")
    is_mae = args.model.startswith("mae_")
    is_ibot = args.model.startswith("ibot_")
    is_data2vec = args.model.startswith("data2vec_")
    has_builtin_weights = is_dinov2 or is_mocov3 or is_mae or is_ibot or is_data2vec
    
    if args.checkpoint is None and args.pretrained is None and not has_builtin_weights:
        raise ValueError("Either --checkpoint or --pretrained must be specified (or use a dinov2_*/mocov3_*/mae_*/ibot_*/data2vec_* model which has pretrained weights).")

    # I-JEPA requires ViT-H model
    if args.pretrained == "ijepa" and args.model != "vit_h":
        print(f"Note: I-JEPA requires vit_h model. Overriding --model {args.model} -> vit_h")
        args.model = "vit_h"
    
    # Models with built-in weights don't need --pretrained flag
    if has_builtin_weights and args.pretrained is not None:
        print(f"Note: {args.model} has pretrained weights built-in. Ignoring --pretrained {args.pretrained}")

    accelerator = Accelerator(mixed_precision="bf16")

    # Filter datasets based on input
    if args.datasets.lower() == "default":
        target_datasets = DEFAULT_EVAL_DATASETS
    elif args.datasets.lower() == "ood":
        target_datasets = OOD_EVAL_DATASETS
    elif args.datasets.lower() == "all":
        target_datasets = list(DATASET_CONFIGS.keys())
    elif args.datasets.lower() == "hf":
        # Only HuggingFace datasets
        target_datasets = [d for d in DEFAULT_EVAL_DATASETS if DATASET_CONFIGS[d].get("type") == "huggingface"]
    elif args.datasets.lower() == "original":
        # Only original datasets
        target_datasets = [d for d in DEFAULT_EVAL_DATASETS if DATASET_CONFIGS[d].get("type") == "original"]
    else:
        target_datasets = [d.strip() for d in args.datasets.split(",") if d.strip() in DATASET_CONFIGS]
        if not target_datasets:
            if accelerator.is_main_process:
                print(f"Error: No valid datasets found in '{args.datasets}'. Available: {list(DATASET_CONFIGS.keys())}")
            return

    if accelerator.is_main_process:
        print(f"Running evaluation on: {', '.join(target_datasets)}")
        print(f"Model: {args.model}")
        if args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}")
        elif is_dinov2:
            print(f"Using DINOv2 pretrained weights (built-in)")
        elif is_mocov3:
            print(f"Using MoCo v3 pretrained weights (auto-download)")
        elif is_mae:
            print(f"Using MAE pretrained weights (auto-download)")
        elif args.pretrained:
            print(f"Loading pretrained: {args.pretrained}")
    
    # Create encoder using shared model loading
    encoder = ViTEncoder(
        model_name=args.model,
        pretrained=args.pretrained,
        checkpoint=args.checkpoint,
    ).to(accelerator.device)
    
    encoder = accelerator.prepare(encoder)

    # Determine model type for LR grid selection
    if is_mocov3:
        model_type = "mocov3"
    elif is_mae:
        model_type = "mae"
    elif args.pretrained == "ijepa":
        model_type = "ijepa"
    elif is_dinov2:
        model_type = "dinov2"
    else:
        model_type = "default"
    
    if accelerator.is_main_process:
        print(f"Using LR grid for: {model_type}")

    shot_settings = args.shots
    final_results = {s: {} for s in shot_settings}

    for name in target_datasets:
        # Re-seed per dataset to make results order-invariant
        set_seed(args.seed)
        if accelerator.is_main_process:
            print(f"\n--- Processing {name.upper()} ---")

        num_classes = get_num_classes(name)
        is_multilabel = is_multi_label(name)

        # Load train and test datasets using unified factory function
        train_ds = create_dataset(name, "train", img_size=224, use_deit_aug=False)
        test_ds = create_dataset(name, "test", img_size=224, use_deit_aug=False)

        if accelerator.is_main_process:
            print(f"  Loaded {name}: {len(train_ds)} train, {len(test_ds)} test samples")

        # Cache test features
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = accelerator.prepare(test_loader)
        test_feats, test_labels = extract_test_features(accelerator, encoder, test_loader)

        for s in shot_settings:
            shot_name = "all" if s == -1 else f"{s}-shot"
            if accelerator.is_main_process:
                print(f"Evaluating {name} @ {shot_name}...")

            # For few-shot, subsample train_ds
            if s > 0:
                indices = []
                labels = [train_ds[i][1] for i in range(len(train_ds))]
                if isinstance(labels[0], torch.Tensor):
                    labels = [l.argmax().item() if l.dim() > 0 else l.item() for l in labels]
                labels = np.array(labels)
                for c in range(num_classes):
                    c_indices = np.where(labels == c)[0]
                    if len(c_indices) > 0:
                        selected = np.random.choice(c_indices, min(s, len(c_indices)), replace=False)
                        indices.extend(selected.tolist())
                curr_train_ds = torch.utils.data.Subset(train_ds, indices)
            else:
                curr_train_ds = train_ds

            train_loader = DataLoader(curr_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
            train_loader = accelerator.prepare(train_loader)

            # Train on train split, evaluate on test split
            metrics = train_online_and_eval(
                accelerator, args, encoder, train_loader,
                test_feats, test_labels,
                num_classes, is_multi_label=is_multilabel, model_type=model_type
            )

            if accelerator.is_main_process:
                if metrics is not None:
                    final_results[s][name] = metrics
                    print(f"Result: {metrics['acc']*100:.2f}%")

            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        with open(args.output, "w") as f:
            json.dump(final_results, f, indent=2)

        print("\n" + "="*100)
        print(" FINAL RESULTS (Accuracy Table)")
        print("="*100)
        header = "| shots | " + " | ".join(f"{n:10}" for n in target_datasets) + " | avg. |"
        print(header)
        print("|" + "-"*len(header) + "|")
        for s in shot_settings:
            shot_label = "all" if s == -1 else f"{s:3}"
            row_vals = [final_results[s].get(n, {}).get("acc", 0.0) for n in target_datasets]
            avg = sum(row_vals) / len(row_vals) if row_vals else 0
            print(f"| {shot_label:5} | " + " | ".join(f"{v*100:10.2f}" for v in row_vals) + f" | {avg*100:6.2f} |")
        print("="*100)
        
        if "chestxray" in target_datasets:
            print("\nDetailed Medical Metrics (ChestXray):")
            print("-" * 65)
            print(f"{'Shots':<10} | {'AU-ROC':<10} | {'Acc':<10} | {'Prec':<10} | {'Rec':<10}")
            print("-" * 65)
            for s in shot_settings:
                m = final_results[s].get("chestxray", {})
                if m:
                    shot_label = "all" if s == -1 else str(s)
                    print(f"{shot_label:<10} | {m.get('auc_roc', 0):.4f}     | {m.get('acc', 0):.4f}     | {m.get('precision', 0):.4f}     | {m.get('recall', 0):.4f}")
            print("-" * 65)

if __name__ == "__main__":
    main()