import logging
from pathlib import Path
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def safe_token(x: object) -> str:
    s = str(x)
    return (
        s.replace("/", "_")
        .replace(" ", "")
        .replace(":", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


def strip_compile_prefix(state_dict: dict) -> dict:
    prefix = "_orig_mod."
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def fmt_lr(x: float) -> str:
    x = float(x)
    if x == 0.0:
        return "0"
    if abs(x) < 1e-2 or abs(x) >= 1e2:
        s = f"{x:.0e}"
        return s.replace("e-0", "e-").replace("e+0", "e+")
    return f"{x:g}"


def build_run_name(cfg, *, method: str, n_global: int, n_local: int) -> str:
    num_proj = cfg.loss.get("num_projections", 0)
    return (
        f"{safe_token(method)}_{safe_token(cfg.model)}"
        f"_bs{safe_token(cfg.bs)}"
        f"_lamb{safe_token(cfg.lamb)}"
        f"_lr{safe_token(fmt_lr(cfg.lr))}"
        f"_projdim{safe_token(cfg.proj_dim)}"
        f"_numproj{safe_token(num_proj)}"
        f"_ng{safe_token(n_global)}"
        f"_nl{safe_token(n_local)}"
    )


def setup_run_dir(cfg, *, method: str, n_global: int, n_local: int, is_main_process: bool = True) -> tuple[Path, Path]:
    run_name = build_run_name(cfg, method=method, n_global=n_global, n_local=n_local)
    runs_root = Path(cfg.ckpt_dir)
    run_dir = runs_root / method.lower() / run_name
    ckpt_dir = run_dir / "checkpoints"
    
    if is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"
        OmegaConf.save(cfg, config_path)
        logger.info("Run directory: %s", run_dir)
        logger.info("Checkpoints will be saved to: %s", ckpt_dir)
    
    return run_dir, ckpt_dir


def get_parameter_groups(model, weight_decay=0.05, no_decay_head=False, *, skip_proj: bool = False):
    no_decay_keywords = {'bias', 'LayerNorm', 'layernorm', 'layer_norm', 'norm'}
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        skip_decay = False
        if name.endswith('.bias') or len(param.shape) == 1:
            skip_decay = True
        elif any(nd in name.lower() for nd in no_decay_keywords):
            skip_decay = True
        elif skip_proj and "proj" in name.lower():
            skip_decay = True
        elif no_decay_head and 'head' in name.lower():
            skip_decay = True
        if skip_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

