"""
Shared model loading utilities for downstream evaluation tasks.
Supports: DINOv1, DINOv2, MoCo v3, MAE, I-JEPA, iBOT, data2vec, and VISReg checkpoints.
"""

import logging
import os
import urllib.request
import torch
import torch.nn as nn
import timm
from huggingface_hub import hf_hub_download
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_CKPT_BASE = os.environ.get(
    "VISREG_CKPT_DIR",
    "./checkpoints",
)

# --- Model Configurations ---

VIT_CONFIGS = {
    "vit_s": ("vit_small_patch8_224", 0.1, 1024),
    "vit_b": ("vit_base_patch16_224", 0.1, 1536),
    "vit_l": ("vit_large_patch14_224", 0.1, 2048),
    "vit_h": ("vit_huge_patch14_224", 0.0, 2560),
}

DINOV2_CONFIGS = {
    "dinov2_vit_s": ("vit_small_patch14_dinov2.lvd142m", 0.0, 3072),
    "dinov2_vit_b": ("vit_base_patch14_dinov2.lvd142m", 0.0, 6144),
    "dinov2_vit_l": ("vit_large_patch14_dinov2.lvd142m", 0.0, 8192),
    "dinov2_vit_g": ("vit_giant_patch14_dinov2.lvd142m", 0.0, 12288),
}

IJEPA_CKPT_URL = "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"
IJEPA_CKPT_PATH = os.path.join(_CKPT_BASE, "ijepa/ijepa_vit_h14_in1k.pth.tar")

MOCOV3_CKPTS = {
    "mocov3_vit_s": {
        "url": "https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar",
        "path": os.path.join(_CKPT_BASE, "mocov3/mocov3_vit_s_300ep.pth.tar"),
        "timm_model": "vit_small_patch16_224",
        "embed_dim": 384,
    },
    "mocov3_vit_b": {
        "url": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
        "path": os.path.join(_CKPT_BASE, "mocov3/mocov3_vit_b_300ep.pth.tar"),
        "timm_model": "vit_base_patch16_224",
        "embed_dim": 768,
    },
}

MAE_CKPTS = {
    "mae_vit_b": {
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        "path": os.path.join(_CKPT_BASE, "mae/mae_pretrain_vit_base.pth"),
        "timm_model": "vit_base_patch16_224",
        "embed_dim": 768,
    },
    "mae_vit_l": {
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
        "path": os.path.join(_CKPT_BASE, "mae/mae_pretrain_vit_large.pth"),
        "timm_model": "vit_large_patch16_224",
        "embed_dim": 1024,
    },
    "mae_vit_h": {
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
        "path": os.path.join(_CKPT_BASE, "mae/mae_pretrain_vit_huge.pth"),
        "timm_model": "vit_huge_patch14_224",
        "embed_dim": 1280,
    },
}

# iBOT checkpoints (from bytedance/ibot, ImageNet-1K pretrained)
IBOT_CKPTS = {
    "ibot_vit_s": {
        "url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth",
        "path": os.path.join(_CKPT_BASE, "ibot/ibot_vit_s_16.pth"),
        "timm_model": "vit_small_patch16_224",
        "embed_dim": 384,
    },
    "ibot_vit_b": {
        "url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth",
        "path": os.path.join(_CKPT_BASE, "ibot/ibot_vit_b_16.pth"),
        "timm_model": "vit_base_patch16_224",
        "embed_dim": 768,
    },
    "ibot_vit_l": {
        "url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/checkpoint_teacher.pth",
        "path": os.path.join(_CKPT_BASE, "ibot/ibot_vit_l_16.pth"),
        "timm_model": "vit_large_patch16_224",
        "embed_dim": 1024,
    },
}

# data2vec 1.0 vision checkpoints (from data2vec_vision repo, BEiT-based, ImageNet-1K pretrained)
# Source: https://github.com/facebookresearch/data2vec_vision/tree/main/beit
DATA2VEC_CKPTS = {
    "data2vec_vit_b": {
        "url": "https://dl.fbaipublicfiles.com/fairseq/data2vec/data2vec_vision/base_800/checkpoint-799.pth",
        "path": os.path.join(_CKPT_BASE, "data2vec/data2vec_vit_b_800ep.pth"),
        "timm_model": "beit_base_patch16_224",
        "embed_dim": 768,
        "epochs": 800,
    },
    "data2vec_vit_l": {
        "url": "https://dl.fbaipublicfiles.com/fairseq/data2vec/data2vec_vision/large_1600/checkpoint-799.pth",
        "path": os.path.join(_CKPT_BASE, "data2vec/data2vec_vit_l_1600ep.pth"),
        "timm_model": "beit_large_patch16_224",
        "embed_dim": 1024,
        "epochs": 1600,
    },
}

VISREG_CKPTS = {
    "visreg_vit_b": "visreg-vit-b-inet1k.pth",
    "visreg_vit_l": "visreg-vit-l-inet1k.pth",
}

# --- Checkpoint Loading Functions ---

def load_mocov3_checkpoint(model, ckpt_url: str, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        logger.info(f"Downloading MoCo v3 checkpoint to {ckpt_path}...")
        urllib.request.urlretrieve(ckpt_url, ckpt_path)
        logger.info("Download complete.")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get('state_dict', ckpt)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.base_encoder.'):
            new_state_dict[k.replace('module.base_encoder.', '')] = v
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"MoCo v3 checkpoint loaded. Missing: {msg.missing_keys[:5]}... Unexpected: {msg.unexpected_keys[:5]}...")
    return model


def load_mae_checkpoint(model, ckpt_url: str, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        logger.info(f"Downloading MAE checkpoint to {ckpt_path}...")
        urllib.request.urlretrieve(ckpt_url, ckpt_path)
        logger.info("Download complete.")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get('model', ckpt)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('decoder') or k == 'mask_token':
            continue
        new_state_dict[k] = v
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"MAE checkpoint loaded. Missing: {msg.missing_keys[:5]}... Unexpected: {msg.unexpected_keys[:5]}...")
    return model


def load_ibot_checkpoint(model, ckpt_url: str, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        logger.info(f"Downloading iBOT checkpoint to {ckpt_path}...")
        urllib.request.urlretrieve(ckpt_url, ckpt_path)
        logger.info("Download complete.")
    
    # weights_only=False needed for older checkpoints
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # iBOT stores teacher weights in 'state_dict' key
    state_dict = ckpt.get('state_dict', ckpt)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present, and skip head/mlp_head
        k = k.replace('module.', '')
        if k.startswith('head.') or k.startswith('mlp_head.'):
            continue
        new_state_dict[k] = v
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"iBOT checkpoint loaded. Missing: {msg.missing_keys[:5]}... Unexpected: {msg.unexpected_keys[:5]}...")
    return model


def load_data2vec_checkpoint(model, ckpt_url: str, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        logger.info(f"Downloading data2vec checkpoint to {ckpt_path}...")
        urllib.request.urlretrieve(ckpt_url, ckpt_path)
        logger.info("Download complete.")
    
    # weights_only=False needed for older checkpoints with numpy arrays
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # data2vec vision stores weights in 'model' key (BEiT format)
    state_dict = ckpt.get('model', ckpt)
    
    # Clean up state dict and handle key mappings
    new_state_dict = {}
    shared_rel_pos_bias = None
    
    for k, v in state_dict.items():
        # Skip head weights and mask token
        if k.startswith('head.') or k.startswith('lm_head.') or k == 'mask_token':
            continue
        # Remove 'module.' prefix if present
        if k.startswith('module.'):
            k = k[7:]
        
        # Rename norm -> fc_norm (final layer norm)
        if k == 'norm.weight':
            new_state_dict['fc_norm.weight'] = v
            continue
        if k == 'norm.bias':
            new_state_dict['fc_norm.bias'] = v
            continue
        
        # Save shared relative position bias for later
        if k == 'rel_pos_bias.relative_position_bias_table':
            shared_rel_pos_bias = v
            new_state_dict[k] = v  # Also keep the shared one
            continue
        if k == 'rel_pos_bias.relative_position_index':
            continue  # Skip index, timm regenerates it
        
        new_state_dict[k] = v
    
    # Copy shared relative position bias to each block (timm BEiT expects per-block)
    if shared_rel_pos_bias is not None:
        # Get actual number of blocks from model
        num_blocks = len(model.blocks)
        for i in range(num_blocks):
            block_key = f'blocks.{i}.attn.relative_position_bias_table'
            new_state_dict[block_key] = shared_rel_pos_bias.clone()
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    # Filter out relative_position_index from missing (it's regenerated)
    missing = [k for k in msg.missing_keys if 'relative_position_index' not in k]
    logger.info(f"data2vec checkpoint loaded. Missing: {len(missing)} keys, Unexpected: {len(msg.unexpected_keys)} keys")
    if missing:
        logger.info(f"  Missing: {missing[:3]}...")
    if msg.unexpected_keys:
        logger.info(f"  Unexpected: {msg.unexpected_keys[:3]}...")
    return model


def load_ijepa_checkpoint(model, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        logger.info(f"Downloading I-JEPA checkpoint to {ckpt_path}...")
        urllib.request.urlretrieve(IJEPA_CKPT_URL, ckpt_path)
        logger.info("Download complete.")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if 'target_encoder' in ckpt:
        state_dict = ckpt['target_encoder']
    elif 'encoder' in ckpt:
        state_dict = ckpt['encoder']
    else:
        state_dict = ckpt
    
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"I-JEPA checkpoint loaded. Missing: {msg.missing_keys[:5]}... Unexpected: {msg.unexpected_keys[:5]}...")
    return model


def load_custom_checkpoint(model, ckpt_key_or_path: str, verbose: bool = True):
    if ckpt_key_or_path in VISREG_CKPTS:
        filename = VISREG_CKPTS[ckpt_key_or_path]
        if verbose:
            logger.info(f"Downloading {filename} from HuggingFace...")
        ckpt_path = hf_hub_download(repo_id="BooBooWu/visreg", filename=filename)
    else:
        ckpt_path = ckpt_key_or_path
    if verbose:
        logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("net_state_dict", ckpt)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # Strip torch.compile's _orig_mod. prefix first
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        # Handle both "backbone." prefix and "module." prefix
        if k.startswith("backbone."):
            new_state_dict[k[9:]] = v
        elif k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    if verbose:
        logger.info(f"Custom checkpoint loaded. Missing: {msg.missing_keys[:5]}... Unexpected: {msg.unexpected_keys[:5]}...")
    return model


def get_model_type(model_name: str, pretrained: Optional[str] = None) -> str:
    if model_name.startswith("dinov2_"):
        return "dinov2"
    elif model_name.startswith("mocov3_"):
        return "mocov3"
    elif model_name.startswith("mae_"):
        return "mae"
    elif model_name.startswith("ibot_"):
        return "ibot"
    elif model_name.startswith("data2vec_"):
        return "data2vec"
    elif pretrained == "ijepa":
        return "ijepa"
    else:
        return "default"


def get_available_models() -> Dict[str, list]:
    return {
        "vit": list(VIT_CONFIGS.keys()),
        "dinov2": list(DINOV2_CONFIGS.keys()),
        "mocov3": list(MOCOV3_CKPTS.keys()),
        "mae": list(MAE_CKPTS.keys()),
        "ibot": list(IBOT_CKPTS.keys()),
        "data2vec": list(DATA2VEC_CKPTS.keys()),
        "visreg": list(VISREG_CKPTS.keys()),
    }


# --- Main Backbone Creation Function ---

def create_backbone(
    model_name: str,
    pretrained: Optional[str] = None,
    checkpoint: Optional[str] = None,
    verbose: bool = True,
) -> tuple:
    is_dinov2 = model_name.startswith("dinov2_")
    is_mocov3 = model_name.startswith("mocov3_")
    is_mae = model_name.startswith("mae_")
    is_ibot = model_name.startswith("ibot_")
    is_data2vec = model_name.startswith("data2vec_")
    
    # Determine timm model name and settings
    if is_dinov2:
        if model_name not in DINOV2_CONFIGS:
            raise ValueError(f"Unknown DINOv2 model: {model_name}. Available: {list(DINOV2_CONFIGS.keys())}")
        timm_name, drop_path, _ = DINOV2_CONFIGS[model_name]
        pretrained_timm = True
        no_embed_class = False
        
    elif is_mocov3:
        if model_name not in MOCOV3_CKPTS:
            raise ValueError(f"Unknown MoCo v3 model: {model_name}. Available: {list(MOCOV3_CKPTS.keys())}")
        mocov3_cfg = MOCOV3_CKPTS[model_name]
        timm_name = mocov3_cfg["timm_model"]
        drop_path = 0.0
        pretrained_timm = False
        no_embed_class = False
        
    elif is_mae:
        if model_name not in MAE_CKPTS:
            raise ValueError(f"Unknown MAE model: {model_name}. Available: {list(MAE_CKPTS.keys())}")
        mae_cfg = MAE_CKPTS[model_name]
        timm_name = mae_cfg["timm_model"]
        drop_path = 0.0
        pretrained_timm = False
        no_embed_class = False
        
    elif is_ibot:
        if model_name not in IBOT_CKPTS:
            raise ValueError(f"Unknown iBOT model: {model_name}. Available: {list(IBOT_CKPTS.keys())}")
        ibot_cfg = IBOT_CKPTS[model_name]
        timm_name = ibot_cfg["timm_model"]
        drop_path = 0.0
        pretrained_timm = False
        no_embed_class = False
        
    elif is_data2vec:
        if model_name not in DATA2VEC_CKPTS:
            raise ValueError(f"Unknown data2vec model: {model_name}. Available: {list(DATA2VEC_CKPTS.keys())}")
        data2vec_cfg = DATA2VEC_CKPTS[model_name]
        timm_name = data2vec_cfg["timm_model"]
        drop_path = 0.0
        pretrained_timm = False
        no_embed_class = False
        
    else:
        if model_name not in VIT_CONFIGS:
            all_models = (list(VIT_CONFIGS.keys()) + list(DINOV2_CONFIGS.keys()) + 
                         list(MOCOV3_CKPTS.keys()) + list(MAE_CKPTS.keys()) + 
                         list(IBOT_CKPTS.keys()) + list(DATA2VEC_CKPTS.keys()))
            raise ValueError(f"Unknown model: {model_name}. Available: {all_models}")
        timm_name, drop_path, _ = VIT_CONFIGS[model_name]
        
        if pretrained == "dino_v1":
            if model_name == "vit_s":
                timm_name = "vit_small_patch16_224.dino"
            elif model_name == "vit_b":
                timm_name = "vit_base_patch16_224.dino"
            else:
                raise ValueError(f"DINOv1 weights not available via timm for {model_name}")
            pretrained_timm = True
        elif pretrained == "ijepa":
            if model_name != "vit_h":
                raise ValueError(f"I-JEPA pretrained weights require vit_h, got {model_name}")
            pretrained_timm = False
        else:
            pretrained_timm = bool(pretrained) if pretrained not in [None, False, "false", "False"] else False
        
        no_embed_class = (pretrained == "ijepa")
    
    # Create backbone
    # Note: BEiT models (used by data2vec) don't support dynamic_img_size
    model_kwargs = dict(
        pretrained=pretrained_timm,
        num_classes=0,
        drop_path_rate=drop_path,
    )
    if is_data2vec:
        # data2vec uses shared relative position bias
        model_kwargs["use_rel_pos_bias"] = True
        model_kwargs["use_shared_rel_pos_bias"] = True
    else:
        model_kwargs["dynamic_img_size"] = True
        model_kwargs["no_embed_class"] = no_embed_class
    
    backbone = timm.create_model(timm_name, **model_kwargs)
    
    # Load custom weights
    if checkpoint:
        backbone = load_custom_checkpoint(backbone, checkpoint, verbose=verbose)
    elif pretrained == "ijepa":
        backbone = load_ijepa_checkpoint(backbone, IJEPA_CKPT_PATH)
    elif is_mocov3:
        mocov3_cfg = MOCOV3_CKPTS[model_name]
        backbone = load_mocov3_checkpoint(backbone, mocov3_cfg["url"], mocov3_cfg["path"])
    elif is_mae:
        mae_cfg = MAE_CKPTS[model_name]
        backbone = load_mae_checkpoint(backbone, mae_cfg["url"], mae_cfg["path"])
    elif is_ibot:
        ibot_cfg = IBOT_CKPTS[model_name]
        backbone = load_ibot_checkpoint(backbone, ibot_cfg["url"], ibot_cfg["path"])
    elif is_data2vec:
        data2vec_cfg = DATA2VEC_CKPTS[model_name]
        backbone = load_data2vec_checkpoint(backbone, data2vec_cfg["url"], data2vec_cfg["path"])
    
    embed_dim = backbone.embed_dim
    
    # Determine if model has CLS token
    if pretrained == "ijepa":
        has_cls = False
    else:
        has_cls = hasattr(backbone, 'cls_token') and backbone.cls_token is not None
    
    return backbone, embed_dim, has_cls

