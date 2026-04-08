from .vit_encoder import ViTEncoder, VIT_CONFIGS

MODEL_REGISTRY = {
    "vit_encoder": ViTEncoder,
}


def build_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = ["ViTEncoder", "VIT_CONFIGS", "MODEL_REGISTRY", "build_model"]

