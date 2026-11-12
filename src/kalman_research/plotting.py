"""Shared figure style so every plot in the paper looks like one system."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
# Categorical colours, assigned in this fixed order (never by rank).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
TRUTH = "#0b0b0b"  # ground truth is always black
NEUTRAL = "#9a9992"  # raw measurements / reference bands

# Sequential: one hue, light to dark. Diverging: two hues around a grey midpoint.
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#2a78d6", "#0f3566"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#dedcd6", "#eb6834"])


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def save(fig, out_dir: Path, name: str) -> str:
    """Save ``fig`` as ``<out_dir>/figures/<name>.png`` and return the relative path."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png")
    plt.close(fig)
    return f"figures/{name}.png"
