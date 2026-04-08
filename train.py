from pathlib import Path
from visreg.utils import get_parameter_groups, fmt_lr, setup_run_dir, safe_token, strip_compile_prefix
from visreg.models import ViTEncoder
from visreg.data import build_dataset, build_kornia_aug_pipeline
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import math
import time
import hydra, tqdm, wandb
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LinearLR, LambdaLR, SequentialLR
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
import warnings
warnings.filterwarnings("ignore", message=".*Metadata Warning.*")
warnings.filterwarnings("ignore", message=".*Corrupt EXIF data.*")

# When using accelerate: pass --num_processes explicitly to avoid hang from saved config.
#   Single GPU:  uv run accelerate launch --num_processes 1 train.py --config-name=...
#   Multi-GPU:   uv run accelerate launch --num_processes 8 train.py --config-name=...
# Without accelerate (single GPU):  uv run train.py --config-name=...


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    
    set_seed(0 + accelerator.process_index)
    
    model_name = cfg.model
    method = HydraConfig.get().runtime.choices["loss"]
    wandb_run_id = cfg.wandb_id
    if accelerator.is_main_process:
        project_name = f"SSL-ImageNet1K-{model_name.upper().replace('_', '-')}"
        run_name = f"{method}_lr{fmt_lr(cfg.lr)}_lamb{cfg.lamb}_projdim{cfg.proj_dim}"
        wandb.init(
            project=project_name,
            name=run_name,
            config=dict(cfg),
            id=wandb_run_id,
            resume="must" if wandb_run_id else None,
        )

    n_global = int(cfg.n_global)
    n_local = int(cfg.n_local)
    global_img_size = cfg.global_img_size
    local_img_size = cfg.local_img_size
    V_total = n_global + n_local
    
    save_ckpt = cfg.save_ckpt
    
    resume_path = cfg.resume
    if resume_path is not None:
        resume_path = Path(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    
    aug_cfg = OmegaConf.to_container(cfg.aug, resolve=True) if cfg.get("aug") else {}
    backend = str(aug_cfg.get("backend", "torchvision")).lower()
    dataset_kw = dict(
        split="train",
        n_global=n_global,
        n_local=n_local,
        global_img_size=global_img_size,
        local_img_size=local_img_size,
        multicrop_backend=backend,
    )
    train_ds = build_dataset(cfg.dataset, **dataset_kw)
    num_classes = train_ds.num_classes

    test_kw = dict(split="validation", global_img_size=global_img_size)
    test_ds = build_dataset(cfg.dataset, **test_kw)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=8)

    # Let Accelerate handle distributed data sharding via prepare().
    is_iterable = isinstance(train_ds, IterableDataset)
    num_workers = int(cfg.get("num_workers", 8))
    prefetch_factor = int(cfg.get("prefetch_factor", 4))
    train_loader = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=not is_iterable, drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
    )

    pretrained = cfg.pretrained
    proj_gelu = cfg.proj_gelu
    net = ViTEncoder(model_name=model_name, proj_dim=cfg.proj_dim, pretrained=pretrained, proj_gelu=proj_gelu)
    proj_dim = net.proj_dim  # Get actual proj_dim from model
    OmegaConf.update(cfg, "proj_dim", proj_dim, force_add=True)  # Update config for logging
    
    if accelerator.is_main_process:
        print(f"Model: {model_name} | embed_dim={net.embed_dim} | proj_dim={proj_dim} | pretrained={pretrained} | proj_gelu={proj_gelu}")
    
    # Setup checkpoint directory (after we know proj_dim)
    run_dir = None
    ckpt_dir = None
    if save_ckpt:
        run_dir, ckpt_dir = setup_run_dir(
            cfg, method=method, n_global=n_global, n_local=n_local,
            is_main_process=accelerator.is_main_process
        )
        accelerator.wait_for_everyone()
    
    probe_input_dim = net.embed_dim * 2
    probe = nn.Sequential(nn.LayerNorm(probe_input_dim), nn.Linear(probe_input_dim, num_classes))

    if accelerator.state.num_processes > 1:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    
    reg_loss_fn = instantiate(cfg.loss)
    
    if accelerator.is_main_process:
        print(f"Initialized {method.upper()} loss")
        print(f"Invariance: locals align to global mean (n_global={n_global}, detached)")

    lr = cfg.lr

    if accelerator.is_main_process:
        print(f"World size: {accelerator.state.num_processes} GPUs")
    
    net_param_groups = get_parameter_groups(net, weight_decay=5e-2, no_decay_head=True)
    for g in net_param_groups:
        g["lr"] = lr
        
    probe_param_groups = [{"params": probe.parameters(), "lr": 2e-2, "weight_decay": 1e-6}]
    opt = torch.optim.AdamW(net_param_groups + probe_param_groups)
    
    net, probe, opt, train_loader, test_loader = accelerator.prepare(
        net, probe, opt, train_loader, test_loader
    )
    
    reg_loss_fn = reg_loss_fn.to(accelerator.device)

    kornia_aug = None
    if backend == "kornia":
        kornia_aug = build_kornia_aug_pipeline(
            cj_brightness=float(aug_cfg.get("cj_brightness", 0.4)),
            cj_contrast=float(aug_cfg.get("cj_contrast", 0.4)),
            cj_saturation=float(aug_cfg.get("cj_saturation", 0.2)),
            cj_hue=float(aug_cfg.get("cj_hue", 0.1)),
        ).to(accelerator.device)

    # Compute steps_per_epoch AFTER prepare() so len(train_loader) reflects per-GPU sharding.
    if hasattr(train_ds, "num_samples"):
        steps_per_epoch = train_ds.num_samples // cfg.bs
        if accelerator.state.num_processes > 1:
            steps_per_epoch = steps_per_epoch // accelerator.state.num_processes
    else:
        steps_per_epoch = len(train_loader)

    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    total_steps = steps_per_epoch * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.001, total_iters=warmup_steps)
    
    final_lr_div = cfg.final_lr_div
    min_factor = 1.0 / final_lr_div
    cosine_steps = max(int(total_steps - warmup_steps), 1)

    def cosine_factor(step: int) -> float:
        t = min(max(int(step), 0), cosine_steps)
        progress = t / cosine_steps
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))

    s2 = LambdaLR(opt, lr_lambda=cosine_factor)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    
    best_acc = float("-inf")
    start_epoch = 0
    
    if resume_path is not None:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {resume_path}")
        
        ckpt = torch.load(resume_path, map_location=accelerator.device, weights_only=False)
        
        net_unwrapped = accelerator.unwrap_model(net)
        probe_unwrapped = accelerator.unwrap_model(probe)
        net_unwrapped.load_state_dict(strip_compile_prefix(ckpt["net_state_dict"]))
        probe_unwrapped.load_state_dict(strip_compile_prefix(ckpt["probe_state_dict"]))
        
        opt.load_state_dict(ckpt["opt_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        
        start_epoch = ckpt.get("epoch", ckpt.get("epoch_idx", 0) + 1)
        best_acc = ckpt.get("best_acc", float("-inf"))
        
        if accelerator.is_main_process:
            print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")
        
        accelerator.wait_for_everyone()
    
    for epoch in range(start_epoch, cfg.epochs):
        net.train()
        probe.train()
        step_count = 0
        inv_sum = 0.0
        jepa_sum = 0.0
        probe_sum = 0.0
        reg_sum = 0.0
        
        progress_bar = tqdm.tqdm(
            train_loader,
            total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{cfg.epochs}",
            disable=not accelerator.is_main_process
        )

        profile_data_gpu = cfg.get("profile_data_gpu", False)
        t_after_prev_step = time.perf_counter()

        for batch_idx, (vs, y) in enumerate(progress_bar):
            t_data = time.perf_counter() - t_after_prev_step

            global_step = epoch * steps_per_epoch + batch_idx
            t_before_gpu = time.perf_counter()
            if kornia_aug is not None:
                if isinstance(vs, torch.Tensor) and vs.dim() == 5:
                    vs = [vs[:, i] for i in range(vs.shape[1])]
                vs = [kornia_aug(v.to(accelerator.device, non_blocking=True).float().div_(255.0)) for v in vs]
            emb, proj = net(vs)
            
            global_mean = proj[:n_global].mean(0, keepdim=True)
            inv_loss = (proj - global_mean).square().mean()
            
            if accelerator.use_distributed:
                proj_local_batch = proj.transpose(0, 1).contiguous()
                proj_global_batch = torch.cat(torch.distributed.nn.all_gather(proj_local_batch), dim=0)
                proj_gathered = proj_global_batch.transpose(0, 1).contiguous()
            else:
                proj_gathered = proj

            reg_loss = reg_loss_fn(proj_gathered)
            
            jepa_loss = reg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
            
            y_rep, yhat = y.repeat_interleave(V_total), probe(emb.detach())
            probe_loss = F.cross_entropy(yhat, y_rep)
            
            loss = jepa_loss + probe_loss

            accelerator.backward(loss)
            if profile_data_gpu:
                torch.cuda.synchronize()
            t_gpu = time.perf_counter() - t_before_gpu
            t_after_prev_step = time.perf_counter()

            if cfg.clip_grad_norm > 0:
                accelerator.clip_grad_norm_(net.parameters(), cfg.clip_grad_norm)
            
            opt.step()
            opt.zero_grad()
            scheduler.step()

            if profile_data_gpu and accelerator.is_main_process and (batch_idx + 1) % 100 == 0:
                wandb.log(
                    {"profile/data_time_sec": t_data, "profile/gpu_time_sec": t_gpu},
                    step=global_step,
                )
                print(f"  [profile] global_step={global_step} (epoch {epoch+1} batch {batch_idx+1}/{steps_per_epoch}): data_time={t_data:.3f}s gpu_time={t_gpu:.3f}s")

            step_count += 1
            inv_sum += inv_loss.item()
            jepa_sum += jepa_loss.item()
            probe_sum += probe_loss.item()
            reg_sum += reg_loss.item()

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "train/probe": probe_loss.item(),
                        "train/jepa": jepa_loss.item(),
                        "train/reg": reg_loss.item(),
                        "train/method": method,
                        "train/inv": inv_loss.item(),
                    },
                    step=global_step,
                )

        epoch_end_step = (epoch + 1) * steps_per_epoch - 1

        net.eval()
        probe.eval()
        correct = torch.tensor(0, device=accelerator.device, dtype=torch.long)
        total = torch.tensor(0, device=accelerator.device, dtype=torch.long)

        with torch.inference_mode():
            for vs, y in test_loader:
                emb, _ = net(vs)
                preds = probe(emb).argmax(1)
                correct += (preds == y).sum()
                total += y.numel()

        correct = accelerator.reduce(correct, reduction="sum").item()
        total = accelerator.reduce(total, reduction="sum").item()
        acc = correct / max(total, 1)

        if accelerator.is_main_process:
            if acc > best_acc:
                best_acc = acc
            print(f"Epoch {epoch+1}: Test Accuracy = {acc:.4f}")
            wandb.log({"test/acc": acc, "test/epoch": epoch}, step=epoch_end_step)

        if accelerator.is_main_process:
            if step_count > 0:
                wandb.log(
                    {
                        "epoch/train/probe_avg": probe_sum / step_count,
                        "epoch/train/jepa_avg": jepa_sum / step_count,
                        "epoch/train/inv_avg": inv_sum / step_count,
                        "epoch/train/reg_avg": reg_sum / step_count,
                        "epoch/idx": epoch,
                    },
                    step=epoch_end_step,
                )
                print(
                    f"Epoch {epoch+1} avgs - "
                    f"Acc: {acc:.4f} | "
                    f"Jepa: {jepa_sum/step_count:.4f}, "
                    f"Inv: {inv_sum/step_count:.4f}, "
                    f"Reg: {reg_sum/step_count:.4f}, "
                    f"Probe: {probe_sum/step_count:.4f}"
                )
            
            if save_ckpt and ckpt_dir is not None:
                net_unwrapped = accelerator.unwrap_model(net)
                probe_unwrapped = accelerator.unwrap_model(probe)
                
                ckpt = {
                    "epoch": epoch + 1,
                    "acc": acc,
                    "best_acc": best_acc,
                    "net_state_dict": accelerator.get_state_dict(net_unwrapped),
                    "probe_state_dict": accelerator.get_state_dict(probe_unwrapped),
                    "opt_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                }

                # Save checkpoint with accuracy in filename
                ckpt_path = ckpt_dir / f"epoch_{epoch+1}_acc{safe_token(f'{acc:.6f}')}.pt"
                accelerator.save(ckpt, ckpt_path)
                
                # Always save latest.pt for easy resume
                latest_path = ckpt_dir / "latest.pt"
                accelerator.save(ckpt, latest_path)
                
                # Periodic checkpoints
                if (epoch + 1) % 100 == 0:
                    periodic_path = ckpt_dir / f"periodic_epoch_{epoch+1}.pt"
                    accelerator.save(ckpt, periodic_path)
                    print(f"Saved periodic checkpoint: {periodic_path}")

                if acc >= best_acc:
                    (run_dir / "best.txt").write_text(ckpt_path.name)
    
    if accelerator.is_main_process:
        print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")
        wandb.finish()


if __name__ == "__main__":
    main()
