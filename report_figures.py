"""Lightweight, data-driven summary figures for the decision-support report.

These charts are built directly from the locked report ``ctx`` (numbers only) and
returned as inline SVG strings, so the report writer can embed them without pulling in
scanpy/torch or touching the trained model. This is deliberately separate from
``generate_figures.py`` (which renders model-bound UMAPs/heatmaps for the chat tool):
the report path must stay light and deterministic.

Public entry point:  build_summary_figures(ctx) -> {"burden": svg|None,
                                                     "perturbation": svg|None,
                                                     "enrichment": svg|None}
Every figure is best-effort: any missing data or a matplotlib import failure yields
None for that figure, and the template simply omits it.
"""
from __future__ import annotations

from io import StringIO
from typing import Any
import re

# Report palette (kept in sync with templates/decision_support_report.html.j2)
BLUE = "#123b62"
TEAL = "#19766f"
RED = "#a93333"
MUTED = "#5b6776"
LINE = "#d5dde6"
DONUT_BG = "#e3e9ef"


def _new_axes(w_in: float, h_in: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "svg.fonttype": "none",
        "axes.edgecolor": LINE, "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    return plt, fig, ax


def _svg(plt, fig, width_mm: float) -> str:
    """Serialise a figure to a self-contained, width-constrained inline SVG string."""
    buf = StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    s = re.sub(r"<\?xml.*?\?>", "", s, flags=re.S)
    s = re.sub(r"<!DOCTYPE.*?>", "", s, flags=re.S)
    m = re.search(r'width="([0-9.]+)pt" height="([0-9.]+)pt"', s)
    style = f"width:{width_mm}mm;height:auto;display:block;margin:0 auto"
    if m:
        w0, h0 = float(m.group(1)), float(m.group(2))
        height_mm = round(width_mm * h0 / w0, 1)
        style = f"width:{width_mm}mm;height:{height_mm}mm;display:block;margin:0 auto"
        s = re.sub(r'(<svg[^>]*?)\swidth="[0-9.]+pt"\sheight="[0-9.]+pt"', r"\1", s, count=1)
    s = s.replace("<svg ", f'<svg style="{style}" ', 1)
    return s.strip()


def _num(value: Any):
    """Best-effort float from a number or a string like '2.2e-64' / '72.3%'."""
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


def _focus_set(ctx: dict) -> set:
    return {str(t).strip().casefold() for t in (ctx.get("focus_targets") or []) if str(t).strip()}


def _is_focus(label: str, focus: set) -> bool:
    if not focus:
        return False
    parts = {p for p in re.split(r"\s*\+\s*|\s*,\s*", str(label).casefold()) if p}
    return bool(parts & focus)


# --------------------------------------------------------------------------- burden
def burden_figure(ctx: dict, width_mm: float = 60.0):
    burden = ctx.get("burden") or {}
    frac = _num(burden.get("fraction"))
    n_hr = burden.get("n_high_risk")
    n_tot = burden.get("n_cells")
    if frac is None or not n_tot:
        return None
    pct = frac * 100
    try:
        plt, fig, ax = _new_axes(3.8, 2.9)
        ax.pie([max(frac, 1e-4), max(1 - frac, 0)], colors=[RED, DONUT_BG],
               startangle=90, counterclock=False,
               wedgeprops=dict(width=0.34, edgecolor="white", linewidth=1.5))
        ax.text(0, 0.12, f"{pct:.1f}%", ha="center", va="center",
                fontsize=25, fontweight="bold", color=BLUE)
        ax.text(0, -0.24, "model-labelled\nhigh-risk", ha="center", va="center",
                fontsize=9.5, color=MUTED)
        scope = "dataset-wide" if ctx.get("analysis_scope") == "cohort" else "in the selected sample"
        sub = f"{int(n_hr):,} high-risk  of  {int(n_tot):,} cells ({scope})" if n_hr is not None \
            else f"of {int(n_tot):,} cells ({scope})"
        ax.text(0, -1.4, sub, ha="center", va="center", fontsize=9.5, color=MUTED)
        ax.set(aspect="equal")
        ax.axis("off")
        return _svg(plt, fig, width_mm)
    except Exception:
        return None


# --------------------------------------------------------------- perturbation bars
def perturbation_figure(ctx: dict, width_mm: float = 165.0):
    rows = [r for r in (ctx.get("target_rows") or []) if _num(r.get("reduction")) is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: _num(r.get("reduction")))
    labels = [str(r.get("target", "")) for r in rows]
    vals = [_num(r.get("reduction")) for r in rows]
    focus = _focus_set(ctx)
    colors = [RED if _is_focus(lbl, focus) else BLUE for lbl in labels]
    try:
        h = max(1.7, 0.34 * len(rows) + 0.7)
        plt, fig, ax = _new_axes(7.0, h)
        y = range(len(rows))
        ax.barh(list(y), vals, color=colors, height=0.62, edgecolor="white")
        for yi, v in zip(y, vals):
            ax.text(v - 1.5, yi, f"{v:.1f}%", va="center", ha="right",
                    color="white", fontsize=9.5, fontweight="bold")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=8, color=MUTED)
        ax.set_xlabel("Model-labelled high-risk reduction (single target)", fontsize=9.5, color=MUTED)
        ax.set_title("Target-network perturbation", loc="left", color=BLUE,
                     fontsize=12, fontweight="bold", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=LINE, linewidth=0.7)
        if focus:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(fc=RED, label="Report focus"),
                               Patch(fc=BLUE, label="Other model target")],
                      loc="lower right", frameon=False, fontsize=8.5)
        return _svg(plt, fig, width_mm)
    except Exception:
        return None


# ---------------------------------------------------------------- enrichment bars
def enrichment_figure(ctx: dict, width_mm: float = 165.0):
    paths = ctx.get("activated_pathways") or []
    items = []
    for p in paths:
        q = _num(p.get("q_value"))
        name = p.get("name")
        if q is not None and q > 0 and name:
            import math
            items.append((str(name), -math.log10(q), p.get("q_value")))
    if not items:
        return None
    items = sorted(items, key=lambda t: t[1])  # ascending -> largest at top after barh
    labels = [t[0] for t in items]
    vals = [t[1] for t in items]
    qraw = [t[2] for t in items]
    try:
        h = max(1.8, 0.34 * len(items) + 0.7)
        plt, fig, ax = _new_axes(7.0, h)
        y = range(len(items))
        ax.barh(list(y), vals, color=TEAL, height=0.66, edgecolor="white")
        vmax = max(vals) or 1
        for yi, v, q in zip(y, vals, qraw):
            ax.text(v - vmax * 0.01, yi, f"q={q}", va="center", ha="right",
                    color="white", fontsize=8.5, fontweight="bold")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xlim(0, vmax * 1.06)
        ax.set_xlabel("Enrichment strength  −log10(q)   (bundled manuscript table)",
                      fontsize=9.5, color=MUTED)
        ax.set_title("High-risk marker program — pathway enrichment", loc="left",
                     color=BLUE, fontsize=12, fontweight="bold", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=LINE, linewidth=0.7)
        return _svg(plt, fig, width_mm)
    except Exception:
        return None


def build_summary_figures(ctx: dict) -> dict:
    """Return inline-SVG summary figures for the report (any of which may be None)."""
    if not isinstance(ctx, dict):
        return {"burden": None, "perturbation": None, "enrichment": None}
    return {
        "burden": burden_figure(ctx),
        "perturbation": perturbation_figure(ctx),
        "enrichment": enrichment_figure(ctx),
    }
