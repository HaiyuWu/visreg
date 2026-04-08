import warnings
warnings.filterwarnings("ignore", message=".*Metadata Warning.*")
warnings.filterwarnings("ignore", message=".*Corrupt EXIF data.*")

from pathlib import Path
from visreg.utils import get_parameter_groups, fmt_lr, strip_compile_prefix
from visreg.data import build_dataset, build_kornia_aug_pipeline
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import math
import timm
import hydra, tqdm, wandb
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LinearLR, LambdaLR, SequentialLR
from torchvision.ops import MLP
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed


class ViTEncoderSmall(nn.Module):
    def __init__(self, proj_dim=128, img_size=128, dynamic_img_size=False):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.1,
            img_size=img_size,
            dynamic_img_size=dynamic_img_size,
        )
        self.embed_dim = self.backbone.embed_dim
        self.proj = MLP(self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            V = len(x)
            B = x[0].shape[0]
            h0, w0 = x[0].shape[-2], x[0].shape[-1]
            all_same_size = all(c.shape[-2] == h0 and c.shape[-1] == w0 for c in x)
            
            if all_same_size:
                x_stacked = torch.stack(x, dim=1)
                N, V = x_stacked.shape[:2]
                emb = self.backbone(x_stacked.flatten(0, 1))
                return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)

            by_size = {}
            for i, crop in enumerate(x):
                key = (int(crop.shape[-2]), int(crop.shape[-1]))
                by_size.setdefault(key, []).append(i)

            emb_per_crop = [None] * V
            proj_per_crop = [None] * V
            for _, idxs in by_size.items():
                batch = torch.cat([x[i] for i in idxs], dim=0)
                e = self.backbone(batch)
                p = self.proj(e)
                e_chunks = e.split(B, dim=0)
                p_chunks = p.split(B, dim=0)
                for j, i in enumerate(idxs):
                    emb_per_crop[i] = e_chunks[j]
                    proj_per_crop[i] = p_chunks[j]

            emb_vb = torch.stack(emb_per_crop, dim=0)
            emb_cat = emb_vb.transpose(0, 1).reshape(B * V, -1)
            proj = torch.stack(proj_per_crop, dim=0)
            return emb_cat, proj
        else:
            N, V = x.shape[:2]
            emb = self.backbone(x.flatten(0, 1))
            return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


@hydra.main(version_base=None, config_path="configs", config_name="imagenette")
def main(cfg: DictConfig):
    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    
    set_seed(0 + accelerator.process_index)
    
    if cfg.get("proj_dim") is None:
        OmegaConf.update(cfg, "proj_dim", 128, force_add=True)
    proj_dim = cfg.proj_dim
    
    method = HydraConfig.get().runtime.choices["loss"]
    if accelerator.is_main_process:
        num_proj = cfg.loss.get("num_projections", 0)
        run_name = f"{method}_lr{fmt_lr(cfg.lr)}_lamb{cfg.lamb}_projdim{proj_dim}_numproj{num_proj}"
        wandb.init(
            project="SSL-Imagenette-Toy",
            name=run_name,
            config=dict(cfg),
            id=cfg.wandb_id,
            resume="must" if cfg.wandb_id else None,
        )

    n_global = cfg.n_global
    n_local = cfg.get("n_local", 0) or 0
    global_img_size = cfg.global_img_size
    local_img_size = cfg.get("local_img_size", global_img_size)
    V_total = n_global + n_local
    
    save_ckpt = cfg.save_ckpt
    resume_path = cfg.resume
    if resume_path is not None:
        resume_path = Path(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    
    ckpt_dir = None
    if save_ckpt:
        ckpt_dir = Path(cfg.ckpt_dir)
        if accelerator.is_main_process:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()
    
    aug_cfg = OmegaConf.to_container(cfg.aug, resolve=True) if cfg.get("aug") else {}
    backend = str(aug_cfg.get("backend", "torchvision")).lower()
    train_ds = build_dataset(
        cfg.dataset,
        split="train",
        n_global=n_global,
        n_local=n_local,
        global_img_size=global_img_size,
        local_img_size=local_img_size,
        multicrop_backend=backend,
    )
    test_ds = build_dataset(cfg.dataset, split="validation", global_img_size=global_img_size)
    num_classes = train_ds.num_classes
    
    train_shuffle = not isinstance(train_ds, IterableDataset)
    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=train_shuffle, drop_last=True, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=8)

    if hasattr(train_ds, "num_samples"):
        steps_per_epoch = train_ds.num_samples // cfg.bs
        if accelerator.state.num_processes > 1:
            steps_per_epoch = steps_per_epoch // accelerator.state.num_processes
    else:
        steps_per_epoch = len(train_loader)

    net = ViTEncoderSmall(proj_dim=proj_dim, img_size=global_img_size, dynamic_img_size=False)
    probe = nn.Sequential(nn.LayerNorm(net.embed_dim), nn.Linear(net.embed_dim, num_classes))

    if accelerator.state.num_processes > 1:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    
    reg_loss_fn = instantiate(cfg.loss)
    
    if accelerator.is_main_process:
        print(f"Dataset: {cfg.dataset} ({num_classes} classes)")
        print(f"Method: {method.upper()} | proj_dim={proj_dim}")
        print(f"Multi-crop: {n_global} global ({global_img_size}px) + {n_local} local ({local_img_size}px)")

    net_param_groups = get_parameter_groups(net, weight_decay=5e-2, no_decay_head=True)
    for g in net_param_groups:
        g["lr"] = cfg.lr
    probe_param_groups = [{"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}]
    opt = torch.optim.AdamW(net_param_groups + probe_param_groups)
    
    net, probe, opt, train_loader, test_loader = accelerator.prepare(net, probe, opt, train_loader, test_loader)
    reg_loss_fn = reg_loss_fn.to(accelerator.device)

    kornia_aug = None
    if backend == "kornia":
        kornia_aug = build_kornia_aug_pipeline(
            cj_brightness=float(aug_cfg.get("cj_brightness", 0.4)),
            cj_contrast=float(aug_cfg.get("cj_contrast", 0.4)),
            cj_saturation=float(aug_cfg.get("cj_saturation", 0.2)),
            cj_hue=float(aug_cfg.get("cj_hue", 0.1)),
        ).to(accelerator.device)

    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    total_steps = steps_per_epoch * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.001, total_iters=warmup_steps)
    min_factor = 1.0 / cfg.final_lr_div
    cosine_steps = max(int(total_steps - warmup_steps), 1)

    def cosine_factor(step: int) -> float:
        t = min(max(int(step), 0), cosine_steps)
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * t / cosine_steps))

    s2 = LambdaLR(opt, lr_lambda=cosine_factor)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    
    best_acc = float("-inf")
    start_epoch = 0
    
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=accelerator.device, weights_only=False)
        accelerator.unwrap_model(net).load_state_dict(strip_compile_prefix(ckpt["net_state_dict"]))
        accelerator.unwrap_model(probe).load_state_dict(strip_compile_prefix(ckpt["probe_state_dict"]))
        opt.load_state_dict(ckpt["opt_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", float("-inf"))
        if accelerator.is_main_process:
            print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")
        accelerator.wait_for_everyone()
    
    for epoch in range(start_epoch, cfg.epochs):
        net.train()
        probe.train()
        step_count, inv_sum, jepa_sum, probe_sum, reg_sum = 0, 0.0, 0.0, 0.0, 0.0
        
        progress_bar = tqdm.tqdm(
            train_loader, total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{cfg.epochs}",
            disable=not accelerator.is_main_process
        )
        
        for batch_idx, (vs, y) in enumerate(progress_bar):
            global_step = epoch * steps_per_epoch + batch_idx
            if kornia_aug is not None:
                if isinstance(vs, torch.Tensor) and vs.dim() == 5:
                    vs = [vs[:, i] for i in range(vs.shape[1])]
                vs = [kornia_aug(v.to(accelerator.device, non_blocking=True).float().div_(255.0)) for v in vs]
            emb, proj = net(vs)
            
            global_mean = proj[:n_global].mean(0, keepdim=True)
            inv_loss = (proj - global_mean).square().mean()
            
            if accelerator.use_distributed:
                proj_local = proj.transpose(0, 1).contiguous()
                proj_gathered = torch.cat(torch.distributed.nn.all_gather(proj_local), dim=0).transpose(0, 1).contiguous()
            else:
                proj_gathered = proj

            reg_loss = reg_loss_fn(proj_gathered)
            jepa_loss = reg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
            
            y_rep = y.repeat_interleave(V_total)
            probe_loss = F.cross_entropy(probe(emb.detach()), y_rep)
            loss = jepa_loss + probe_loss

            accelerator.backward(loss)
            if cfg.clip_grad_norm > 0:
                accelerator.clip_grad_norm_(net.parameters(), cfg.clip_grad_norm)
            opt.step()
            opt.zero_grad()
            scheduler.step()

            step_count += 1
            inv_sum += inv_loss.item()
            jepa_sum += jepa_loss.item()
            probe_sum += probe_loss.item()
            reg_sum += reg_loss.item()

            if accelerator.is_main_process:
                wandb.log({
                    "train/probe": probe_loss.item(),
                    "train/jepa": jepa_loss.item(),
                    "train/reg": reg_loss.item(),
                    "train/inv": inv_loss.item(),
                }, step=global_step)

        net.eval()
        probe.eval()
        correct = torch.tensor(0, device=accelerator.device, dtype=torch.long)
        total = torch.tensor(0, device=accelerator.device, dtype=torch.long)
        
        with torch.inference_mode():
            for vs, y in test_loader:
                emb, _ = net(vs)
                correct += (probe(emb).argmax(1) == y).sum()
                total += y.numel()

        correct = accelerator.reduce(correct, reduction="sum").item()
        total = accelerator.reduce(total, reduction="sum").item()
        acc = correct / max(total, 1)
        epoch_end_step = (epoch + 1) * steps_per_epoch - 1
        
        if accelerator.is_main_process:
            if acc > best_acc:
                best_acc = acc
            print(f"Epoch {epoch+1}: Acc={acc:.4f} | Jepa={jepa_sum/step_count:.4f}, Reg={reg_sum/step_count:.4f}, Inv={inv_sum/step_count:.4f}")
            wandb.log({"test/acc": acc, "test/epoch": epoch}, step=epoch_end_step)

            if save_ckpt and ckpt_dir is not None:
                ckpt = {
                    "epoch": epoch + 1,
                    "best_acc": best_acc,
                    "net_state_dict": accelerator.get_state_dict(accelerator.unwrap_model(net)),
                    "probe_state_dict": accelerator.get_state_dict(accelerator.unwrap_model(probe)),
                    "opt_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                accelerator.save(ckpt, ckpt_dir / "latest.pt")

    if accelerator.is_main_process:
        print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")
        wandb.finish()


if __name__ == "__main__":
    main()
