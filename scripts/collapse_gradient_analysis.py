"""
Collapse Gradient Analysis: VICReg vs Barlow Twins vs SWD vs SIGReg vs VISReg
==============================================================================

This script analyzes how gradients behave as features collapse (scale → 0).

Key Insight:
- Complete collapse: all embeddings become identical → variance → 0
- We measure gradient magnitude at various collapse levels to understand
  which methods provide stronger "recovery" signals when collapse occurs.

Methods:
1. VICReg: Variance-Invariance-Covariance Regularization
2. Barlow Twins: Cross-correlation matrix → Identity
3. SWD: Sliced Wasserstein Distance
4. SIGReg: Characteristic function matching via RFF
5. VISReg (Ours): Variance-Invariance-Sketching regularization

Usage:
    python scripts/collapse_gradient_analysis.py --out-dir figures/collapse
    python scripts/collapse_gradient_analysis.py --out-dir figures/collapse --mode detailed
"""

from __future__ import annotations
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
from typing import Callable

from visreg.losses import VISReg, SIGReg, SlicedWasserstein as SWD, VICReg, BarlowTwins


# ==============================================================================
# 2. Data Generation & Gradient Analysis
# ==============================================================================

def generate_features(V: int, B: int, D: int, device: torch.device, seed: int = 42) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    z = torch.randn((V, B, D), device=device, dtype=torch.float32, generator=g)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
    return z


def compute_gradient_norm(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
) -> float:
    z = z.clone().detach().requires_grad_(True)
    loss = loss_fn(z)
    loss.backward()

    # Mean gradient norm per sample
    grad_norm = z.grad.reshape(-1, z.shape[-1]).norm(dim=1).mean()
    return float(grad_norm.detach().cpu().item())


def analyze_collapse_gradients(
    methods: dict,
    z0: torch.Tensor,
    collapse_scales: np.ndarray,
    mc_samples: int = 10,
    seed: int = 42,
) -> dict:
    results = {}

    for name, loss_fn in methods.items():
        means, stds = [], []

        for i, scale in enumerate(collapse_scales):
            samples = []
            for k in range(mc_samples):
                torch.manual_seed(seed + 1000 * i + k)
                z = (z0 * float(scale)).clone()
                grad_norm = compute_gradient_norm(loss_fn, z)
                samples.append(grad_norm)

            means.append(np.mean(samples))
            stds.append(np.std(samples))

        results[name] = {"mean": np.array(means), "std": np.array(stds)}

    return results


# ==============================================================================
# 3. Visualization
# ==============================================================================

def plot_collapse_analysis(
    scales: np.ndarray,
    results: dict,
    output_path: Path,
    title: str = "Gradient Behavior During Collapse",
):
    font_size = 18
    font_name = 'serif'

    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.size'] = font_size
    plt.rcParams.update({
        'mathtext.fontset': 'custom',
        'mathtext.rm': font_name,
        'mathtext.it': font_name + ':italic',
        'mathtext.bf': font_name + ':bold',
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })

    # Professional color palette (consistent with Nature figure style)
    VISREG_COLOR = '#1a5f7a'      # Deep teal (matches VISReg in other figures)
    VISREG_STAR_COLOR = '#F5EA1D'  # VISReg star fill (requested)
    SWD_COLOR = '#e69f00'       # Warm orange
    SIGREG_COLOR = '#c73e1d'    # Burnt orange/red
    VICREG_COLOR = '#7b2d8e'    # Purple
    BARLOW_COLOR = '#2d6a4f'    # Forest green
    COLLAPSE_COLOR = '#8b1a1a'  # Dark red for collapse region
    GRID_COLOR = '#d0d0d0'

    colors = {
        "VICReg": VICREG_COLOR,
        "Barlow Twins": BARLOW_COLOR,
        "SWD": SWD_COLOR,
        "SIGReg": SIGREG_COLOR,
        "VISReg (Ours)": VISREG_COLOR,
    }

    markers = {
        "VICReg": "^",          # Triangle up
        "Barlow Twins": "D",    # Diamond
        "SWD": "o",             # Circle
        "SIGReg": "s",          # Square
        "VISReg (Ours)": r"$\star$",  # Star
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Proxy handles so the legend shows "line + marker" (matching bench_cost.py)
    proxy_handles = {}

    for name, data in results.items():
        mean = data["mean"]
        std = data["std"]
        color = colors.get(name, "#000000")
        marker = markers.get(name, "o")
        lw = 2.0 if "VISReg" in name else 1.5

        # Subsample for cleaner markers (every 5th point)
        idx = np.arange(0, len(scales), max(1, len(scales) // 10))

        ax.plot(scales, mean, '-', color=color, linewidth=lw, zorder=5)
        is_visreg = "VISReg" in name
        base_ms = 7
        ms = base_ms * 2.0 if is_visreg else base_ms
        ax.plot(
            scales[idx],
            mean[idx],
            linestyle="None",
            marker=marker,
            color=color,
            markerfacecolor=(VISREG_STAR_COLOR if is_visreg else "white"),
            markeredgecolor=color,
            markeredgewidth=(0.6 if is_visreg else 1.5),
            markersize=ms,
            zorder=10,
        )
        proxy_handles[name] = mlines.Line2D(
            [],
            [],
            color=color,
            linewidth=lw,
            marker=marker,
            markersize=ms,
            markerfacecolor=(VISREG_STAR_COLOR if is_visreg else "white"),
            markeredgecolor=color,
            markeredgewidth=(0.6 if is_visreg else 1.5),
            linestyle="-",
            label=name,
        )
        ax.fill_between(scales, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Feature scale (r)", fontweight='medium')
    ax.set_ylabel(r"Gradient magnitude $\|\nabla \mathcal{L}\|$", fontweight='medium')

    # Add collapse region annotation
    collapse_threshold = 0.01
    ax.axvspan(scales.min(), collapse_threshold, alpha=0.15, color=COLLAPSE_COLOR, zorder=1)

    # Position text in collapse region
    ylim = ax.get_ylim()
    ax.text(scales.min() * 3, 0.1, "Collapse region",
            fontsize=font_size - 2, color=COLLAPSE_COLOR, alpha=0.8,
            ha="left", va="top", fontstyle='italic')

    # Add reference line at healthy gradient level
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax.grid(True, which="both", alpha=0.3, color=GRID_COLOR,
            linestyle='-', linewidth=0.5)
    # Legend order (top-to-bottom): Barlow Twins, VISReg (Ours), VICReg, SWD, SIGReg
    order = ["Barlow Twins", "VISReg (Ours)", "SWD", "VICReg", "SIGReg"]
    legend = ax.legend([proxy_handles[o] for o in order if o in proxy_handles],
              [o for o in order if o in proxy_handles],
              loc='upper right', frameon=True, fancybox=False,
              edgecolor='#888888', framealpha=0.95, fontsize=14,
              handlelength=0.8, handletextpad=0.3, borderpad=0.2, labelspacing=0.2)
    legend.set_zorder(100)

    # Set axis limits for clean appearance
    ax.set_xlim(scales.min() * 0.8, scales.max() * 1.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"✓ Saved: {output_path}")

    # Also save PDF for paper
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"✓ Saved: {pdf_path}")

    plt.close(fig)


def plot_detailed_analysis(
    scales: np.ndarray,
    results: dict,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    font_size = 18
    font_name = 'serif'
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.size'] = font_size
    plt.rcParams.update({
        'mathtext.fontset': 'custom',
        'mathtext.rm': font_name,
        'mathtext.it': font_name + ':italic',
        'mathtext.bf': font_name + ':bold',
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size - 4,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
    })

    # Colors matching main figure
    VISREG_COLOR = '#1a5f7a'
    SWD_COLOR = '#e69f00'
    SIGREG_COLOR = '#c73e1d'
    VICREG_COLOR = '#7b2d8e'
    BARLOW_COLOR = '#2d6a4f'
    COLLAPSE_COLOR = '#8b1a1a'
    GRID_COLOR = '#d0d0d0'

    method_colors = {
        "VICReg": VICREG_COLOR,
        "Barlow Twins": BARLOW_COLOR,
        "SWD": SWD_COLOR,
        "SIGReg": SIGREG_COLOR,
        "VISReg (Ours)": VISREG_COLOR,
    }

    # 1. Gradient ratio plot (relative to VISReg)
    if "VISReg (Ours)" in results:
        fig, ax = plt.subplots(figsize=(6, 4))
        visreg_mean = results["VISReg (Ours)"]["mean"]

        for name, data in results.items():
            if name == "VISReg (Ours)":
                continue
            ratio = visreg_mean / (data["mean"] + 1e-10)
            color = method_colors.get(name, "#333333")
            ax.plot(scales, ratio, label=f"VISReg / {name}",
                    linewidth=1.8, color=color)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_xlabel("Feature scale (r)", fontweight='medium')
        ax.set_ylabel("Gradient ratio", fontweight='medium')
        ax.legend(loc='best', frameon=True, fancybox=False,
                  edgecolor='#888888', framealpha=0.95, fontsize=font_size - 4)
        ax.grid(True, which="both", alpha=0.3, color=GRID_COLOR, linewidth=0.5)

        fig.tight_layout()
        fig.savefig(output_dir / "gradient_ratio.png", dpi=200, bbox_inches='tight')
        fig.savefig(output_dir / "gradient_ratio.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved: {output_dir / 'gradient_ratio.png'}")

    # 2. Individual method plots (2x3 grid for 5 methods)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(results.items()):
        if idx >= 6:
            break
        ax = axes[idx]
        mean = data["mean"]
        std = data["std"]
        color = method_colors.get(name, "#333333")

        ax.plot(scales, mean, linewidth=1.8, color=color)
        ax.fill_between(scales, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Feature scale (r)")
        ax.set_ylabel(r"$\|\nabla \mathcal{L}\|$")
        ax.set_title(name, fontweight='medium', color=color)
        ax.grid(True, which="both", alpha=0.3, color=GRID_COLOR, linewidth=0.5)

        # Add collapse region
        ax.axvspan(scales.min(), 0.01, alpha=0.06, color=COLLAPSE_COLOR)

    # Hide unused axes
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "individual_methods.png", dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / "individual_methods.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {output_dir / 'individual_methods.png'}")


# ==============================================================================
# 4. Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze gradient behavior during feature collapse for SSL methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/collapse_gradient_analysis.py --out-dir figures/collapse
    python scripts/collapse_gradient_analysis.py --mode detailed --mc 20
        """
    )
    parser.add_argument("--out-dir", type=str, default="figures/collapse",
                        help="Output directory for figures")
    parser.add_argument("--mode", choices=["basic", "detailed"], default="basic",
                        help="Analysis mode: basic (single plot) or detailed (multiple plots)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--V", type=int, default=2, help="Number of views")
    parser.add_argument("--B", type=int, default=64, help="Batch size")
    parser.add_argument("--D", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--mc", type=int, default=10, help="Monte Carlo samples")
    parser.add_argument("--r-min", type=float, default=-5, help="Log10 of min scale")
    parser.add_argument("--r-max", type=float, default=1, help="Log10 of max scale")
    parser.add_argument("--r-steps", type=int, default=50, help="Number of scale steps")
    parser.add_argument("--num-proj", type=int, default=64, help="Number of projections for SIGReg/VISReg")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"Collapse Gradient Analysis")
    print(f"=" * 60)
    print(f"Device: {device}")
    print(f"Config: V={args.V}, B={args.B}, D={args.D}")
    print(f"Scale range: 10^{args.r_min} to 10^{args.r_max} ({args.r_steps} steps)")
    print(f"MC samples: {args.mc}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)

    # Generate base features
    print("\n[1/3] Generating base features...")
    z0 = generate_features(args.V, args.B, args.D, device, args.seed)
    print(f"  Shape: {z0.shape}, Norm: {z0.norm(dim=-1).mean():.4f}")

    # Initialize methods
    print("\n[2/3] Initializing SSL methods...")
    methods = {
        "VICReg": VICReg().to(device),
        "Barlow Twins": BarlowTwins().to(device),
        "SWD": SWD(num_projections=args.num_proj).to(device),
        "SIGReg": SIGReg(num_projections=args.num_proj).to(device),
        "VISReg (Ours)": VISReg(num_projections=args.num_proj).to(device),
    }

    for name, m in methods.items():
        # Quick sanity check
        with torch.no_grad():
            loss = m(z0.clone())
        print(f"  {name}: loss @ r=1.0 = {loss.item():.4f}")

    # Define collapse scales
    scales = np.logspace(args.r_min, args.r_max, args.r_steps)

    # Run analysis
    print(f"\n[3/3] Analyzing gradient behavior across {len(scales)} collapse levels...")
    results = analyze_collapse_gradients(
        methods={name: m for name, m in methods.items()},
        z0=z0,
        collapse_scales=scales,
        mc_samples=args.mc,
        seed=args.seed,
    )

    # Save numerical results
    np.savez(
        output_dir / "collapse_gradients.npz",
        scales=scales,
        **{f"{name}_mean": data["mean"] for name, data in results.items()},
        **{f"{name}_std": data["std"] for name, data in results.items()},
    )
    print(f"\n✓ Saved numerical data: {output_dir / 'collapse_gradients.npz'}")

    # Generate plots
    print("\nGenerating plots...")
    plot_collapse_analysis(
        scales=scales,
        results=results,
        output_path=output_dir / "collapse_gradient_comparison.png",
        title=f"Gradient Response During Collapse (D={args.D}, B={args.B})",
    )

    if args.mode == "detailed":
        plot_detailed_analysis(scales, results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary: Gradient Magnitude at Severe Collapse (r ≈ 10⁻⁴)")
    print("=" * 60)
    collapse_idx = np.argmin(np.abs(scales - 1e-4))
    for name, data in results.items():
        grad = data["mean"][collapse_idx]
        print(f"  {name:20s}: {grad:.6f}")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
