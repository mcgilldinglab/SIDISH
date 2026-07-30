"""Live SIDISH figure generation, modularised so both the agent's show_figures tool
and build_report can request specific plots. Figures are saved (matplotlib Agg) into
figures/ with the filenames the report payload expects; the report writer inlines
whatever exists and drops the rest.

Order matters: the UMAP embedding must be computed on RAW counts (before get_MarkerGenes
normalises adata in place), so generate() runs embedding-based plots first, then the
marker heatmap, then survival/perturbation.

Usage:  python generate_figures.py [PATIENT_ID]
        from code:  import generate_figures as G; G.generate("CID3946", kinds=["highrisk_umap"])
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, scanpy as sc
from pathlib import Path

import sidish_tools as T

HERE = Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

# figure kind -> saved filename (kept in sync with the payload's figure paths)
FILES = {
    "highrisk_umap":   "{pid}_umap_highrisk.png",
    "celltype_umap":   "{pid}_umap_celltype.png",
    "distribution":    "{pid}_highrisk_distribution.png",
    "celltype_bar":    "{pid}_celltype_proportions.png",
    "marker_heatmap":  "{pid}_marker_heatmap.png",
    "survival":        "{pid}_survival.png",
    "perturbation_bar":"{pid}_perturbation_single.png",
    "transition_umap": "{pid}_perturbation_umap.png",
}


def _save(path):
    plt.gcf().savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return str(path)


def ensure_embedding():
    """Compute the UMAP embedding once (on raw counts) and add SIDISH_ labels/colours."""
    sdh = T.CONTEXT["sdh"]
    ad = T._adata()
    if ad is None:
        return None
    if "X_umap" not in ad.obsm or "SIDISH_" not in ad.obs:
        sdh.get_embedding(celltype=True)      # latent -> neighbors -> umap (no leiden)
        sdh.set_adata()                        # SIDISH_ (red/grey) labels + colours
        T.CONTEXT["adata"] = sdh.adata
    return T.CONTEXT["adata"]


# ---------------- individual figures (return saved path or None) ----------------
def highrisk_umap(pid):
    ad = ensure_embedding()
    sc.pl.umap(ad, color=["SIDISH_"], title="SIDISH high-risk cells", show=False,
               size=8, frameon=False)
    return _save(FIG / FILES["highrisk_umap"].format(pid=pid))


def celltype_umap(pid):
    ad = ensure_embedding()
    if "celltype_major" not in ad.obs.columns:
        return None
    sc.pl.umap(ad, color=["celltype_major"], title="Cell types", show=False,
               size=8, frameon=False)
    return _save(FIG / FILES["celltype_umap"].format(pid=pid))


def distribution(pid):
    """Per-sample high-risk vs background cell counts across the cohort."""
    ad = T._adata()
    obs = ad.obs
    hr = obs["SIDISH"].astype(str) == "h"
    tab = obs.assign(hr=hr).groupby("patient")["hr"].agg(["sum", "count"])
    tab["bg"] = tab["count"] - tab["sum"]
    tab = tab.sort_values("sum", ascending=True)
    plt.figure(figsize=(7, 4))
    y = np.arange(len(tab))
    plt.barh(y, tab["bg"], color="#c9ced6", label="Background")
    plt.barh(y, tab["sum"], left=tab["bg"], color="#c0392b", label="High-risk")
    plt.yticks(y, tab.index)
    plt.xlabel("cells"); plt.legend(loc="lower right", frameon=False)
    plt.title("High-risk vs background cells per sample")
    # highlight the requested patient
    if pid in list(tab.index):
        plt.gca().get_yticklabels()[list(tab.index).index(pid)].set_fontweight("bold")
    return _save(FIG / FILES["distribution"].format(pid=pid))


def celltype_bar(pid):
    pf = T.patient_features(pid)
    comp = pf.get("celltype_composition", {})
    if not comp:
        return None
    ks = list(comp.keys()); vs = [comp[k] * 100 for k in ks]
    plt.figure(figsize=(6, 3.4))
    plt.barh(ks[::-1], vs[::-1], color="#c0392b")
    plt.xlabel("% of high-risk cells"); plt.title(f"{pid} high-risk composition")
    return _save(FIG / FILES["celltype_bar"].format(pid=pid))


def marker_heatmap(pid):
    ad = T._adata()
    mk = T.marker_genes(logfc_threshold=1.5, top=15)      # normalises adata in place (once)
    present = [g for g in mk["genes"][:12] if g in ad.var_names]
    if not present:
        return None
    grp = ad.obs["SIDISH"].astype(str)
    X = ad[:, present].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    means = np.vstack([X[(grp == g).values].mean(0) for g in ["h", "b"]])
    z = (means - means.mean(0)) / (means.std(0) + 1e-9)
    plt.figure(figsize=(7, 2.6))
    plt.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.yticks([0, 1], ["High-risk", "Background"])
    plt.xticks(range(len(present)), present, rotation=90, fontsize=8)
    plt.colorbar(label="z(mean expr)"); plt.title("High-risk marker genes")
    return _save(FIG / FILES["marker_heatmap"].format(pid=pid))


def survival(pid):
    sdh = T.CONTEXT["sdh"]
    if not hasattr(sdh, "upregulated_genes"):
        T.marker_genes(logfc_threshold=1.5)
    sdh.plot_KM(penalizer=10, data_name="TCGA-BRCA")
    return _save(FIG / FILES["survival"].format(pid=pid))


def perturbation_bar(pid, patient=None, **pert_kwargs):
    pert = T.perturbation(patient=patient, **pert_kwargs)
    single = pert.get("single", [])
    if not single:
        return None
    gs = [s["target"] for s in single]; rs = [s["reduction"] for s in single]
    scope = f"{patient}" if patient else "cohort"
    plt.figure(figsize=(6, 3.4))
    plt.barh(gs[::-1], rs[::-1], color="#1f4e79")
    plt.xlabel("% high-risk reduction (single KO)")
    plt.title(f"{pid} top single-gene knockouts ({scope})")
    return _save(FIG / FILES["perturbation_bar"].format(pid=pid))


def transition_umap(pid, gene=None, patient=None):
    sdh = T.CONTEXT["sdh"]
    if not hasattr(sdh, "delta_change"):
        sdh.delta_change = {}
    if gene is None:
        pert = T.perturbation(patient=patient)
        gene = pert["single"][0]["target"] if pert.get("single") else "COL6A1"
    sdh.plot_perturbation_UMAP_differential([gene])
    return _save(FIG / FILES["transition_umap"].format(pid=pid))


_KINDS = {
    "highrisk_umap": highrisk_umap, "celltype_umap": celltype_umap,
    "distribution": distribution, "celltype_bar": celltype_bar,
    "marker_heatmap": marker_heatmap, "survival": survival,
    "perturbation_bar": perturbation_bar, "transition_umap": transition_umap,
}
# sensible order: raw-count plots -> normalised heatmap -> survival -> perturbation
DEFAULT_ORDER = ["highrisk_umap", "celltype_umap", "distribution", "celltype_bar",
                 "marker_heatmap", "survival", "perturbation_bar", "transition_umap"]


def generate(pid="CID3946", kinds=None, patient=None):
    """Generate the requested figure kinds (default: all) in a safe order.
    patient= makes the perturbation plots patient-specific. Returns {kind: path-or-error}."""
    if not T._live() and not T.CONFIG.get("force_mock"):   # self-sufficient: load the model
        T.init_sidish()
    kinds = kinds or DEFAULT_ORDER
    ordered = [k for k in DEFAULT_ORDER if k in kinds]
    out = {}
    for k in ordered:
        try:
            if k in ("perturbation_bar", "transition_umap"):
                out[k] = _KINDS[k](pid, patient=patient)
            else:
                out[k] = _KINDS[k](pid)
        except Exception as e:
            out[k] = f"error: {e}"
    return out


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "CID3946"
    t0 = time.time()
    print(T.init_sidish(), flush=True)
    res = generate(pid)
    for k, v in res.items():
        print(f"  {k:16s} {v}", flush=True)
    print(f"[{time.time()-t0:.0f}s] done", flush=True)
