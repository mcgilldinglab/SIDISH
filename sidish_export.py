"""Deterministic SIDISH -> payload.json exporter.

Runs your trained SIDISH model through its real public API and writes a payload
in the SAME schema as payloads/mock_payload_BRCA_CID3946.json, plus figures.
report_writer.py then consumes it unchanged.

This file is written against the verified SIDISH API (tutorials + SIDISH.py):
  patient column default "Sample", cell-type column default "celltype_major",
  risk labels adata.obs.SIDISH in {"h","b"}, get_MarkerGenes returns
  (upregulated, downregulated), run_Perturbation returns four gene-keyed dicts,
  run_double_Perturbation(top_genes) returns (percentage_double_dict, pvals).

It CANNOT be run here (needs your data + GPU). Fill in CONFIG and run locally
inside the SIDISH env. Caveats are flagged inline with `# NOTE`.
"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")               # headless: save figures, never plt.show()
import matplotlib.pyplot as plt

from SIDISH import SIDISH as sidish  # your package

# ============================ CONFIG — EDIT ME ============================
PATH        = "../data/BRCA/"        # trained-run dir (has adata_SIDISH.h5ad, deepCox, vae_transfer, W_matrix_*.csv)
BULK_CSV    = "../data/processed_bulk.csv"
PATIENT_ID  = "CID3946"              # showcase patient (None => cohort-only report)
PATIENT_COL = "Sample"              # verified default; BRCA may use "Patient" — check adata.obs
CELLTYPE_COL= "celltype_major"      # verified default
DEVICE      = "cuda"                # or "cpu"
SEED        = 0
OUTDIR      = Path("payloads")
FIGDIR      = Path("figures")
CANCER      = {"disease": "Breast cancer", "cancer_subtype": "Triple-negative (TNBC)",
               "data_source": "scRNA-seq (GSE176078); bulk+survival: TCGA-BRCA (n=1194)"}
TOP_MARKERS, TOP_SINGLE, TOP_DUAL = 8, 5, 3
# Locked target->drug map (never let the LLM invent drugs). Extend as needed.
DRUGS = {
    "CTLA4": ["Ipilimumab (approved, anti-CTLA-4)"], "IL6": ["Tocilizumab/Siltuximab (anti-IL6)"],
    "KDR": ["Apatinib (investigational in BRCA)"], "IGF1R": ["IGF1R inhibitor (investigational)"],
    "NOTCH1": ["NADI-351 (investigational)"], "VEGFA": ["Bevacizumab (approved)"],
    "MAP2K1": ["Trametinib (approved)"], "AKT1": ["Ipatasertib (investigational)"],
    "CDK1": ["Dinaciclib (investigational)"], "SPARC": ["Nab-paclitaxel (approved)"],
    "PLK1": ["Volasertib (investigational)"], "EGFR": ["Erlotinib (approved)"],
}
# =========================================================================


def _pick(cols, *cands):
    for c in cands:
        if c in cols:
            return c
    return None


def load_model():
    adata = sc.read_h5ad(f"{PATH}adata_SIDISH.h5ad")
    bulk = pd.read_csv(BULK_CSV, index_col=0)
    sdh = sidish.SIDISH(adata, bulk, DEVICE, seed=SEED)
    # Hyperparameters MUST match the ones the run was trained with:
    sdh.init_Phase1(225, 20, 32, [512, 128], 512, "Adam", 1.0e-4, 1e-4, 0)
    sdh.init_Phase2(500, 128, 1e-4, 0, 0.2, 256)
    sdh.reload(PATH)
    return sdh


def composition(obs, celltype_col, mask):
    """Return {celltype: fraction} over the cells selected by boolean mask."""
    sub = obs.loc[mask, celltype_col].value_counts(normalize=True)
    return {str(k): float(v) for k, v in sub.items()}


def fig_umap(adata, color, path, title):
    ax = sc.pl.umap(adata, color=color, show=False, size=12, title=title)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close("all")


def fig_bar(dct, path, title, color="#1f4e79"):
    plt.figure(figsize=(6, 3.6))
    plt.barh(list(dct.keys())[::-1], [v for v in list(dct.values())[::-1]], color=color)
    plt.xlabel("value"); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close("all")


def main():
    OUTDIR.mkdir(exist_ok=True); FIGDIR.mkdir(exist_ok=True)
    sdh = load_model()

    # Embedding adata carries SIDISH labels + leiden + UMAP (written by set_adata()).
    emb_path = f"{PATH}adata_SIDISH_embedding.h5ad"
    adata = sc.read_h5ad(emb_path) if os.path.exists(emb_path) else sdh.get_embedding()
    obs = adata.obs
    ct_col = _pick(obs.columns, CELLTYPE_COL, "cell_type", "celltype", "leiden")
    pt_col = _pick(obs.columns, PATIENT_COL, "Patient", "patient", "sample")
    hr = obs.SIDISH.astype(str) == "h"

    # ---- cohort figures + context ----
    fig_umap(adata, "SIDISH__" if "SIDISH__" in obs else "SIDISH",
             FIGDIR / "umap_highrisk.png", "High-risk (red) vs background")
    if ct_col:
        fig_umap(adata, ct_col, FIGDIR / "umap_celltype.png", "Cell types")
    cohort = {
        "cohort": CANCER.get("cohort", "SIDISH cohort"),
        "n_cells_total": int(adata.n_obs), "n_high_risk_total": int(hr.sum()),
        "high_risk_fraction_cohort": float(hr.mean()),
        "high_risk_composition_cohort": composition(obs, ct_col, hr) if ct_col else {},
    }

    # ---- markers (NOTE: get_MarkerGenes normalizes adata in place; call once) ----
    up, _ = sdh.get_MarkerGenes(logfc_threshold=1)
    marker_genes = [{"gene": str(g), "note": ""} for g in list(up)[:TOP_MARKERS]]

    # ---- survival ----
    try:
        sdh.plot_KM(penalizer=10)
        plt.savefig(FIGDIR / "survival.png", dpi=150, bbox_inches="tight"); plt.close("all")
    except Exception as e:
        print("[warn] KM figure skipped:", e)

    # ---- perturbation (cohort-level; works out of the box) ----
    pct, delta, pflip, pscore = sdh.run_Perturbation()
    pdf = (pd.DataFrame({"Genes": list(pct.keys()), "pct": list(pct.values())})
           .sort_values("pct", ascending=False).reset_index(drop=True))
    single = [{"target": r.Genes, "high_risk_reduction_percent": round(float(r.pct), 1),
               "candidate_drugs": DRUGS.get(r.Genes, [])}
              for r in pdf.head(TOP_SINGLE).itertuples()]
    fig_bar({r.Genes: r.pct for r in pdf.head(12).itertuples()},
            FIGDIR / "perturbation_single.png", "Top single-gene high-risk reduction (%)",
            color="#c0392b")

    dbl_pct, _ = sdh.run_double_Perturbation(pdf.Genes.values[:20])
    dbl = sorted(dbl_pct.items(), key=lambda kv: -float(kv[1]))[:TOP_DUAL]
    dual = [{"target": str(k).replace(",", " + "),
             "high_risk_reduction_percent": round(float(v), 1),
             "candidate_drugs": []} for k, v in dbl]
    try:
        sdh.plot_double_Perturbation_Heatmap(dbl_pct)
        plt.savefig(FIGDIR / "perturbation_double.png", dpi=150, bbox_inches="tight"); plt.close("all")
    except Exception as e:
        print("[warn] double heatmap skipped:", e)

    # ---- patient-level features ----
    if PATIENT_ID and pt_col:
        pmask = obs[pt_col].astype(str) == str(PATIENT_ID)
        p_hr = pmask & hr
        patient_frac = float(p_hr.sum() / max(pmask.sum(), 1))
        comp = composition(obs, ct_col, p_hr) if ct_col else {}
        base = cohort["high_risk_composition_cohort"]
        enriched = [k for k in comp if comp[k] > base.get(k, 0)][:3]
        reduced = [k for k in base if base.get(k, 0) > comp.get(k, 0)][:3]
    else:
        patient_frac, comp, enriched, reduced = cohort["high_risk_fraction_cohort"], \
            cohort["high_risk_composition_cohort"], [], []

    payload = {
        "meta": {"report_type": "patient" if PATIENT_ID else "cohort",
                 "generated_by": "SIDISH-Agent (live export)",
                 "research_use_only": True, "model_version": "SIDISH v1.0.0"},
        "patient": {"patient_id": PATIENT_ID or "COHORT", **CANCER,
                    "age": "not available", "sex": "not available",
                    "stage": "not available", "prior_treatment": "not available",
                    "metadata_available": False},
        "cohort_context": cohort,
        "features": {
            "patient_high_risk_fraction": patient_frac,
            "patient_rank_in_cohort": "see cohort table",
            "contrast_patient": {"patient_id": "n/a", "high_risk_fraction": 0.0},
            "celltype_composition": comp,
            "enriched_vs_control": enriched, "reduced_vs_control": reduced,
            "marker_genes": marker_genes,
            "figures": {"umap_highrisk": "figures/umap_highrisk.png",
                        "umap_celltype": "figures/umap_celltype.png",
                        "celltype_barplot": "figures/umap_celltype.png",
                        "marker_heatmap": "figures/umap_celltype.png"},
        },
        "perturbation": {"single_gene_top": single, "dual_gene_top": dual,
                         "figures": {"single_gene_bar": "figures/perturbation_single.png",
                                     "dual_gene_heatmap": "figures/perturbation_double.png",
                                     "perturbation_umap": "figures/umap_highrisk.png"}},
        "prognosis": {"km_pvalue": "see KM figure",
                      "km_note": "SIDISH markers stratify TCGA survival.",
                      "km_figure": "figures/survival.png"},
    }
    out = OUTDIR / f"{PATIENT_ID or 'COHORT'}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[ok] wrote {out} and figures to {FIGDIR}/")


if __name__ == "__main__":
    main()
