"""SIDISH agent tools — LIVE against the GitHub SIDISH package.

Each function is ONE capability the agent's LLM may call. They call the real
GitHub SIDISH API (get_MarkerGenes, plot_KM, get_percentille, run_Perturbation,
run_double_Perturbation) on your trained BREAST_CANCER run. The LLM never runs
the math; it decides which tools to call and reads their return dicts.

LIVE REQUIREMENTS (once):
  1. The SIDISH GitHub package importable as `SIDISH` (folder with __init__.py).
  2. Deps: torch, torch_geometric, pyro-ppl, shap, lifelines, joblib, scanpy.
  3. A trained BRCA run at PATH (adata_SIDISH.h5ad, vae_transfer, deepCox).
  4. PPI files for perturbation at ../data/PPI/ :
       hippie_current.txt, 9606.protein.links.v11.5.txt, 9606.protein.info.v11.5.txt
  5. GPU recommended (run_Perturbation sweeps every gene).

Call init_sidish(...) once (the load_context tool). If it isn't called, tools
fall back to the paper-seeded mock so the agent still demos.
"""
import json, os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS: tolerate duplicate libomp runtimes

HERE = Path(__file__).parent
CONTEXT = {"sdh": None, "adata": None, "mode": "uninitialized", "cache": {},
           "case_id": None, "source_dataset": None}

# Point the tools at your trained BRCA run. init_sidish() with no args uses these.
CONFIG = {
    "run_dir":    str(HERE / "BREAST_CANCER") + "/",                 # adata_SIDISH.h5ad, vae_transfer, deepCox
    "adata_path": str(HERE / "BREAST_CANCER" / "adata_SIDISH.h5ad"),
    "bulk_csv":   str(HERE / "BREAST_CANCER" / "bulk_result.csv"),   # cols: duration, event, <genes...>
    "device":     os.environ.get("SIDISH_DEVICE", "cpu"),           # Apple Silicon has no CUDA: "cpu" or "mps"
    "percentile": 0.90,
    "force_mock": False,          # backwards-compatible alias for explicit demo mode
    "demo_mode": os.environ.get("SIDISH_DEMO_MODE", "").lower() in {"1", "true", "yes"},
    "model_version": os.environ.get("SIDISH_MODEL_VERSION", "SIDISH v1.0.0"),
}


def _demo_enabled() -> bool:
    return bool(CONFIG.get("force_mock") or CONFIG.get("demo_mode"))


def _mock_or_raise(capability: str):
    """Fail closed unless the caller explicitly enabled the isolated demo mode."""
    if _demo_enabled():
        return True
    raise RuntimeError(
        f"SIDISH is not live; {capability} was not run. "
        "Clinician mode never substitutes mock results. Set SIDISH_DEMO_MODE=1 only in a labelled demo."
    )

def _load_mock():
    for p in [HERE / "payloads" / "mock_payload_BRCA_CID3946.json", HERE / "mock_payload_BRCA_CID3946.json"]:
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError("mock payload not found")
_MOCK = _load_mock()

# BRCA init parameters (from your BREAST.ipynb — differ from the LUNG tutorial)
BRCA_PHASE1 = (225, 20, 16, [512, 256, 64], 256, "Adam", 1.0e-3, 1e-4, 0, "Dense")
BRCA_PHASE2 = (1000, 64, 1e-6, 0, 0.2, 512)

# CMap / LINCS-informed target -> perturbagen mapping (offline connectivity fallback).
# Each entry: {"drug", "moa" (mechanism of action), "status"}. Used by drug_perturbation()
# and to fill the per-target `drugs` field. On a lab server with a CLUE_API_KEY, the live
# L1000 signature-connectivity query supersedes this table.
CMAP_DRUGS = {
    # --- CAF / ECM / cytoskeleton program (the high-risk drivers in this BRCA run) ---
    "MDK":     [{"drug": "iMDK", "moa": "midkine inhibitor", "status": "preclinical"}],
    "SPARC":   [{"drug": "Nab-paclitaxel", "moa": "SPARC-avid albumin-bound taxane", "status": "approved"}],
    "SPARCL1": [{"drug": "Nab-paclitaxel", "moa": "SPARC-family / ECM", "status": "approved (context)"}],
    "FN1":     [{"drug": "Cilengitide", "moa": "integrin/fibronectin-axis inhibitor", "status": "investigational"}],
    "POSTN":   [{"drug": "anti-periostin mAb", "moa": "periostin neutralization", "status": "preclinical"}],
    "LOX":     [{"drug": "beta-aminopropionitrile (BAPN)", "moa": "lysyl oxidase inhibitor", "status": "preclinical"}],
    "LOXL2":   [{"drug": "Simtuzumab", "moa": "anti-LOXL2 mAb", "status": "investigational"}],
    "COL1A1":  [{"drug": "Simtuzumab (LOXL2 axis)", "moa": "collagen cross-linking", "status": "investigational"}],
    "COL6A1":  [{"drug": "Simtuzumab (LOXL2/collagen axis)", "moa": "collagen deposition", "status": "investigational"}],
    "TIMP1":   [{"drug": "Marimastat (MMP/TIMP axis)", "moa": "matrix metalloproteinase inhibitor", "status": "investigational"}],
    "MMP2":    [{"drug": "Marimastat", "moa": "MMP inhibitor", "status": "investigational"}],
    "IGFBP7":  [{"drug": "Ganitumab (IGF axis)", "moa": "IGF-1R pathway mAb", "status": "investigational"}],
    "IGFBP4":  [{"drug": "IGF-1R inhibitor", "moa": "IGF pathway", "status": "investigational"}],
    # --- angiogenesis / RTK ---
    "VEGFA":   [{"drug": "Bevacizumab", "moa": "anti-VEGFA mAb", "status": "approved"}],
    "KDR":     [{"drug": "Apatinib", "moa": "VEGFR2 TKI", "status": "investigational (BRCA)"}],
    "PDGFRB":  [{"drug": "Imatinib", "moa": "PDGFR TKI", "status": "approved"}],
    "PDGFRA":  [{"drug": "Olaratumab", "moa": "anti-PDGFRA mAb", "status": "investigational"}],
    "IGF1R":   [{"drug": "Ganitumab / Linsitinib", "moa": "IGF-1R inhibitor", "status": "investigational"}],
    "EGFR":    [{"drug": "Erlotinib / Cetuximab", "moa": "EGFR inhibitor", "status": "approved"}],
    # --- immune / signaling ---
    "CTLA4":   [{"drug": "Ipilimumab", "moa": "anti-CTLA-4 checkpoint mAb", "status": "approved"}],
    "IL6":     [{"drug": "Tocilizumab / Siltuximab", "moa": "anti-IL6(R)", "status": "approved"}],
    "NOTCH1":  [{"drug": "gamma-secretase inhibitor (e.g. RO4929097)", "moa": "NOTCH pathway", "status": "investigational"}],
    "MAP2K1":  [{"drug": "Trametinib", "moa": "MEK1/2 inhibitor", "status": "approved"}],
    "AKT1":    [{"drug": "Ipatasertib", "moa": "AKT inhibitor", "status": "investigational"}],
    "CDK1":    [{"drug": "Dinaciclib", "moa": "CDK inhibitor", "status": "investigational"}],
    "PLK1":    [{"drug": "Volasertib", "moa": "PLK1 inhibitor", "status": "investigational"}],
}
# Legacy flat map kept for back-compat (perturbation single `drugs` field).
DRUGS = {g: [f"{d['drug']} ({d['status']})" for d in ds] for g, ds in CMAP_DRUGS.items()}


def _cmap_compounds(gene: str) -> list:
    """CMap-mapped candidate compounds for a target gene (offline table)."""
    return CMAP_DRUGS.get(str(gene), [])


# ---------------------------------------------------------------- init_sidish
def _pick_device(pref=None):
    """Resolve a usable torch device string; degrade cuda -> mps/cpu when no CUDA present."""
    pref = (pref or CONFIG.get("device") or "cpu").strip()
    if pref.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return pref
            return "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            return "cpu"
    return pref


def init_sidish(adata_path: str = None, bulk_path: str = None, path: str = None,
                device: str = None, seed: int = 0) -> dict:
    """Load and reload a trained BRCA SIDISH run for LIVE analysis.
    With no arguments it uses CONFIG paths (unless CONFIG['force_mock'] is True).
    adata_path: '<PATH>/adata_SIDISH.h5ad' ; bulk_path: bulk+survival csv ;
    path: the trained-run directory ('<PATH>/')."""
    explicit = bool(adata_path or bulk_path or path)
    adata_path = adata_path or CONFIG["adata_path"]
    bulk_path  = bulk_path  or CONFIG["bulk_csv"]
    path       = path       or CONFIG["run_dir"]
    device     = _pick_device(device)
    if _demo_enabled() and not explicit:
        CONTEXT["mode"] = "mock"
        CONTEXT["case_id"] = "DEMO-CID3946"
        return {"mode": "mock", "demo": True,
                "note": "Explicit demo mode: paper-seeded values, never a clinical case."}
    import scanpy as sc, pandas as pd
    from SIDISH import SIDISH as SIDISHModel
    adata = sc.read_h5ad(adata_path)
    bulk = pd.read_csv(bulk_path)                        # cols: duration, event, <genes...> (NO index_col)
    if "patient" not in adata.obs:                       # patient = cell-index prefix
        adata.obs["patient"] = adata.obs.index.str.split("_").str[0]
    sdh = SIDISHModel(adata, bulk, device, seed=seed)
    sdh.init_Phase1(*BRCA_PHASE1)
    sdh.init_Phase2(*BRCA_PHASE2)
    sdh.reload(path)
    sdh.get_percentille(CONFIG["percentile"])            # sets percentile_cells (perturbation needs it)
    CONTEXT.update(sdh=sdh, adata=adata, mode="live", cache={},
                   source_dataset=str(Path(adata_path).resolve()))
    return {"mode": "live", "device": device, "n_cells": int(adata.n_obs),
            "n_patients": int(adata.obs["patient"].nunique()),
            "has_celltype": "celltype_major" in adata.obs.columns,
            "model_version": CONFIG["model_version"],
            "source_dataset": CONTEXT["source_dataset"]}


def _live():
    return CONTEXT["mode"] == "live" and CONTEXT["sdh"] is not None


def _adata():
    """adata from CONTEXT, falling back to the model's own adata (for notebook handoff)."""
    if CONTEXT.get("adata") is not None:
        return CONTEXT["adata"]
    sdh = CONTEXT.get("sdh")
    return getattr(sdh, "adata", None) if sdh is not None else None


# ------------------------------------------------------------------ load_dataset
# SIDISH was trained (manuscript) on breast, lung, pancreatic. For these we can reuse
# the published bulk+survival (and trained run if present); other diseases need an upload.
DISEASE_ALIASES = {
    "breast": "breast", "brca": "breast", "tnbc": "breast", "breast cancer": "breast",
    "lung": "lung", "nsclc": "lung", "luad": "lung", "lusc": "lung", "lung cancer": "lung",
    "pancreatic": "pancreatic", "pancreas": "pancreatic", "pdac": "pancreatic",
    "pancreatic cancer": "pancreatic",
}
DISEASE_RUNS = {
    "breast":     {"run_dir": str(HERE / "BREAST_CANCER") + "/",
                   "bulk": str(HERE / "BREAST_CANCER" / "bulk_result.csv")},
    "lung":       {"run_dir": str(HERE / "LUNG_CANCER") + "/",
                   "bulk": str(HERE / "LUNG_CANCER" / "bulk_result.csv")},
    "pancreatic": {"run_dir": str(HERE / "PANCREAS_CANCER") + "/",
                   "bulk": str(HERE / "PANCREAS_CANCER" / "bulk_result.csv")},
}


def _validate_bulk(path: str) -> tuple:
    """A SIDISH bulk table is a CSV with columns: duration, event, <gene1>, <gene2>, ...
    (no index column, survival time in 'duration', 0/1 censoring in 'event')."""
    import pandas as pd
    try:
        b = pd.read_csv(path, nrows=25)
    except Exception as e:
        return False, f"could not read bulk csv: {e}"
    cols = [str(c) for c in b.columns]
    if len(cols) < 3:
        return False, "need at least duration, event and one gene column"
    if cols[0].lower() != "duration" or cols[1].lower() != "event":
        return False, (f"first two columns must be exactly 'duration','event' (got {cols[:2]}). "
                       "Save with no index column.")
    try:
        ev = set(pd.unique(b["event"].dropna()))
        if not ev.issubset({0, 1, 0.0, 1.0, True, False}):
            return False, f"'event' must be 0/1 censoring indicators (got values like {list(ev)[:4]})"
    except Exception:
        pass
    return True, f"ok: {len(cols) - 2} gene columns; first cols {cols[:3]}"


def load_dataset(adata_path: str, cancer_type: str = None, bulk_path: str = None,
                 device: str = None) -> dict:
    """Ingest a user's PREPROCESSED single-cell dataset and prepare SIDISH.

    Routing:
      * If the disease is one SIDISH was trained on (breast / lung / pancreatic), reuse the
        published bulk+survival for that type — no upload needed.
      * Otherwise the user MUST provide bulk_path: a bulk RNA-seq + survival table in SIDISH
        format (CSV columns: duration, event, then one column per gene; no index column).
    If a trained run + SIDISH labels already exist for that type, this reloads immediately;
    otherwise it reports that SIDISH must be TRAINED on the data (a GPU/lab-server job)."""
    import numpy as np
    import scanpy as sc
    from sidish_data_router import route_case_data
    if not Path(adata_path).exists():
        return {"error": f"single-cell file not found: {adata_path}"}
    ad = sc.read_h5ad(adata_path)
    route = route_case_data(cancer_type or "", user_bulk_path=bulk_path)
    if not route.ready:
        return {"status": "need_bulk" if route.needs_user_bulk and not bulk_path else "data_not_ready",
                "cancer_type": route.cancer_type, "n_cells": int(ad.n_obs),
                "routing": route.to_dict(), "message": route.message}

    run_dir, bulk = route.run_dir, route.bulk_path
    has_labels = "SIDISH" in ad.obs.columns
    reference_adata = Path(run_dir or "") / "adata_SIDISH.h5ad" if run_dir else None
    same_reference = bool(reference_adata and reference_adata.exists()
                          and Path(adata_path).resolve() == reference_adata.resolve())
    trained = bool(run_dir and (Path(run_dir) / "vae_transfer").exists()
                   and (Path(run_dir) / "deepCox").exists())
    if trained and has_labels and same_reference:
        res = init_sidish(adata_path=adata_path, bulk_path=bulk,
                          path=str(Path(run_dir)) + os.sep, device=device)
        return {"status": "loaded_reference", "cancer_type": route.cancer_type,
                "bulk": bulk, "run_dir": run_dir, "routing": route.to_dict(), **res}

    # A new scRNA-seq dataset must be trained with the selected bulk-survival reference.
    # Reloading weights and recalculating a percentile on the new cells would force an
    # apparent high-risk fraction and is therefore prohibited.
    return {"status": "needs_training", "cancer_type": route.cancer_type,
            "bulk": bulk, "run_dir": run_dir, "n_cells": int(ad.n_obs),
            "has_SIDISH_labels": has_labels, "routing": route.to_dict(),
            "message": (f"The single-cell dataset is ready for a new SIDISH training run using "
                        f"{route.bulk_source}. Training is required before case-level results; "
                        "the system will not apply a newly fitted percentile to untrained cells.")}


# ---------------------------------------------------------- highrisk_overview
def highrisk_overview() -> dict:
    """Cohort high-risk fraction and per-patient high-risk percentages."""
    if not _live():
        _mock_or_raise("high-risk overview")
        cc = _MOCK["cohort_context"]
        return {"mode": "mock", "demo": True, "scope": "cohort-level association",
                "cohort_high_risk_fraction": cc["high_risk_fraction_cohort"],
                "top_patient": _MOCK["patient"]["patient_id"],
                "top_patient_fraction": _MOCK["features"]["patient_high_risk_fraction"]}
    obs = _adata().obs
    hr = obs["SIDISH"].astype(str) == "h"
    per = obs.assign(hr=hr).groupby("patient")["hr"].mean().sort_values(ascending=False)
    return {"mode": "live", "scope": "cohort-level association",
            "n_cells": int(len(obs)), "n_high_risk": int(hr.sum()),
            "cohort_high_risk_fraction": float(hr.mean()),
            "per_patient": {k: round(float(v), 4) for k, v in per.items()}}


# ------------------------------------------------------------ patient_features
def patient_features(patient_id: str) -> dict:
    """High-risk burden + cell-type composition of one patient's high-risk cells."""
    if not _live():
        _mock_or_raise("sample features")
        f = _MOCK["features"]
        return {"mode": "mock", "demo": True, "scope": "sample-specific observation",
                "patient_id": patient_id,
                "high_risk_fraction": f["patient_high_risk_fraction"],
                "celltype_composition": f["celltype_composition"],
                "enriched": f["enriched_vs_control"], "reduced": f["reduced_vs_control"]}
    obs = _adata().obs
    pm = obs["patient"].astype(str) == str(patient_id)
    hr = obs["SIDISH"].astype(str) == "h"
    p_hr = pm & hr
    has_celltype = "celltype_major" in obs.columns
    comp = ({str(k): float(v) for k, v in
             obs.loc[p_hr, "celltype_major"].dropna().value_counts(normalize=True).items()
             if v > 0} if has_celltype else {})
    base = ({str(k): float(v) for k, v in
             obs.loc[hr, "celltype_major"].dropna().value_counts(normalize=True).items()
             if v > 0} if has_celltype else {})
    return {"mode": "live", "scope": "sample-specific observation", "patient_id": patient_id,
            "n_cells": int(pm.sum()), "n_high_risk": int(p_hr.sum()),
            "high_risk_fraction": float(p_hr.sum() / max(pm.sum(), 1)),
            "celltype_available": has_celltype, "celltype_field": ("celltype_major" if has_celltype else None),
            "celltype_composition": comp,
            "enriched": [k for k in comp if comp[k] > base.get(k, 0)][:3],
            "reduced": [k for k in base if base.get(k, 0) > comp.get(k, 0)][:3]}


def cohort_features() -> dict:
    """High-risk burden and cell-type composition across every cell in the dataset."""
    if not _live():
        _mock_or_raise("dataset-wide features")
        overview = highrisk_overview()
        features = _MOCK["features"]
        return {"mode": "mock", "demo": True, "scope": "cohort-level association",
                "analysis_scope": "cohort", "n_cells": overview.get("n_cells", 0),
                "n_high_risk": overview.get("n_high_risk", 0),
                "high_risk_fraction": overview.get("cohort_high_risk_fraction", 0),
                "celltype_composition": features.get("celltype_composition", {}),
                "enriched": [], "reduced": []}
    obs = _adata().obs
    high_risk = obs["SIDISH"].astype(str) == "h"
    has_celltype = "celltype_major" in obs.columns
    composition = ({
        str(key): float(value)
        for key, value in obs.loc[high_risk, "celltype_major"].dropna().value_counts(normalize=True).items()
        if value > 0
    } if has_celltype else {})
    return {"mode": "live", "scope": "cohort-level association",
            "analysis_scope": "cohort", "n_cells": int(len(obs)),
            "n_high_risk": int(high_risk.sum()),
            "high_risk_fraction": float(high_risk.mean()),
            "celltype_available": has_celltype,
            "celltype_field": ("celltype_major" if has_celltype else None),
            "celltype_composition": composition, "enriched": [], "reduced": []}


# ----------------------------------------------------------------- marker_genes
def marker_genes(logfc_threshold: float = 1.5, pval_threshold: float = 0.05, top: int = 8) -> dict:
    """Marker genes for high-risk cells via SIDISH.get_MarkerGenes (wilcoxon).

    get_MarkerGenes normalizes adata IN PLACE, so we run it EXACTLY ONCE and cache the
    full ranked table; any later threshold just re-filters that table (no re-normalization)."""
    if not _live():
        _mock_or_raise("marker-gene analysis")
        return {"mode": "mock", "demo": True, "scope": "cohort-level association",
                "genes": [m["gene"] for m in _MOCK["features"]["marker_genes"]]}
    import numpy as np, scanpy as sc
    sdh = CONTEXT["sdh"]
    ranked = CONTEXT["cache"].get("ranked_deg")
    if ranked is None:
        sdh.get_MarkerGenes(logfc_threshold=logfc_threshold, pval_threshold=pval_threshold)  # normalizes ONCE
        ranked = sc.get.rank_genes_groups_df(sdh.adata, group="h", key="SIDISH_deg")
        CONTEXT["cache"]["ranked_deg"] = ranked
    pcol = "pvals_adj" if "pvals_adj" in ranked.columns else "pvals"
    up = ranked[(ranked["logfoldchanges"] > logfc_threshold)
                & (ranked[pcol] < pval_threshold)]["names"].tolist()
    down = ranked[(ranked["logfoldchanges"] < -logfc_threshold)
                  & (ranked[pcol] < pval_threshold)]["names"].tolist()
    CONTEXT["cache"]["up_genes"] = up
    CONTEXT["cache"]["down_genes"] = down
    sdh.upregulated_genes = np.array(up)                  # keep survival_km aligned to this threshold
    sdh.downregulated_genes = np.array(down)
    CONTEXT["cache"].pop("km", None)                      # invalidate KM cache (depends on up-genes)
    return {"mode": "live", "scope": "cohort-level association",
            "multiple_testing_column": pcol, "genes": up[:top],
            "downregulated": down[:top], "n_up": len(up),
            "n_down": len(down), "n_total": len(up)}


# ------------------------------------------------------------------ survival_km
def survival_km(penalizer: float = 10.0) -> dict:
    """Log-rank p-value for the marker signature on bulk (replicates plot_KM math)."""
    if not _live():
        _mock_or_raise("survival association")
        return {"mode": "mock", "demo": True, "scope": "cohort-level association",
                "km_pvalue": _MOCK["prognosis"]["km_pvalue"]}
    if "km" in CONTEXT["cache"]:
        return CONTEXT["cache"]["km"]
    import numpy as np
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    sdh = CONTEXT["sdh"]
    if not hasattr(sdh, "upregulated_genes"):
        marker_genes()
    DEG = np.append(sdh.upregulated_genes, ["duration", "event"])
    res = sdh.bulk.filter(DEG)
    cph = CoxPHFitter(penalizer=penalizer).fit(res, duration_col="duration", event_col="event")
    coef = cph.summary.T.filter(sdh.upregulated_genes).iloc[0].values.reshape(-1, 1)
    risk = res.iloc[:, :-2].values @ coef
    grp = np.where(risk >= np.median(risk), "High-Risk", "Background")
    res = res.copy(); res["risk"] = grp
    hi, lo = res[res.risk == "High-Risk"], res[res.risk == "Background"]
    lr = logrank_test(lo["duration"], hi["duration"], lo["event"], hi["event"])
    out = {"mode": "live", "scope": "cohort-level association",
           "km_pvalue": f"{lr.p_value:.2e}", "n_samples": int(len(res)),
           "n_events": int(res["event"].sum()),
           "validation_status": "exploratory_same_cohort",
           "limitation": ("The Cox coefficients, median split, and log-rank association are "
                          "estimated in the available bulk cohort; this is not independent validation.")}
    CONTEXT["cache"]["km"] = out
    return out


# ----------------------------------------------------------------- perturbation
def _candidate_genes(sdh, scope, genes, n_genes):
    """Resolve which genes to perturb: an explicit list, 'all' (genome-wide, GPU/lab
    only), or the default = top `n_genes` upregulated high-risk marker genes."""
    var = set(map(str, sdh.adata.var.index))
    if genes:
        return [g for g in genes if str(g) in var]
    if scope == "all":
        return list(map(str, sdh.adata.var.index))
    up = CONTEXT["cache"].get("up_genes")
    if not up:                                            # reuse markers already computed at ANY
        marker_genes()                                    # threshold — avoids re-normalizing adata
        up = CONTEXT["cache"].get("up_genes") or []
    up = [g for g in up if str(g) in var]
    return up[:int(n_genes)]


def perturbation(genes: list = None, scope: str = "markers", n_genes: int = 50,
                 n_jobs: int = 4, top_single: int = 5, top_dual: int = 3,
                 dual_top_n: int = 6, patient: str = None) -> dict:
    """In-silico single- and dual-gene knockouts using SIDISH's perturbation engine.

    Scope (agent-controllable):
      * default: top `n_genes` (50) high-risk marker genes — tractable on CPU.
      * genes=[...]: perturb exactly this set (e.g. a clinician's gene panel).
      * scope='all': every gene (genome-wide, like the paper) — run on GPU/lab server.
      * patient='CID...': restrict the knockout to THAT patient's cells, so the
        high-risk reduction is patient-specific (default = whole cohort).
    Requires the PPI files (./PPI or ../data/PPI). Result is cached per (patient,scope)."""
    if not _live():
        _mock_or_raise("in-silico perturbation")
        p = _MOCK["perturbation"]
        return {"mode": "mock", "demo": True, "scope": "computational hypothesis",
                "single": [{"target": x["target"], "reduction": x["high_risk_reduction_percent"],
                            "drugs": x.get("candidate_drugs", [])} for x in p["single_gene_top"]],
                "dual": [{"target": x["target"], "reduction": x["high_risk_reduction_percent"]}
                         for x in p["dual_gene_top"]]}
    ckey = "pert::%s::%s::%s" % (patient or "cohort", scope,
                                 ",".join(genes) if genes else n_genes)
    if ckey in CONTEXT["cache"]:
        return CONTEXT["cache"][ckey]
    import os, shutil, tempfile, numpy as np, scanpy as sc, pandas as pd
    from SIDISH.in_silico_perturbation import InSilicoPerturbation
    sdh = CONTEXT["sdh"]
    cand = _candidate_genes(sdh, scope, genes, n_genes)
    if not cand:
        return {"error": "no candidate genes to perturb (run marker_genes first?)"}

    orig_path, tmpdir = sdh.path, None
    try:
        if patient:                                       # write a patient-only adata_SIDISH.h5ad
            full = sc.read_h5ad(orig_path + "adata_SIDISH.h5ad")
            if "patient" in full.obs.columns:
                pmask = np.asarray(full.obs["patient"].astype(str) == str(patient))
            else:
                pmask = np.asarray(full.obs.index.str.split("_").str[0].astype(str) == str(patient))
            if int(pmask.sum()) == 0:
                return {"error": f"patient {patient} not found in the cohort"}
            sub = full[pmask].copy()
            if int((sub.obs["SIDISH"].astype(str) == "h").sum()) == 0:
                return {"error": f"patient {patient} has no high-risk cells to perturb"}
            tmpdir = tempfile.mkdtemp(prefix="sidish_pt_")
            sub.write_h5ad(os.path.join(tmpdir, "adata_SIDISH.h5ad"))
            sdh.path = tmpdir + os.sep

        # SIDISH re-reads a fresh (raw-count) adata; percentile_cells stays the COHORT threshold
        # (set on raw counts at init) so patient labels are on the same scale.
        sdh.adata = sc.read_h5ad(sdh.path + "adata_SIDISH.h5ad")
        sdh.genes = list(cand)
        engine = InSilicoPerturbation(sdh.adata)
        engine.genes = list(cand)                         # SCOPE the single-gene sweep
        engine.setup_ppi_network(threshold=0.7)
        sdh.optimized_results = engine.run_parallel_processing(sdh.adata, n_jobs=n_jobs)
        pc, dc, pf, ps = sdh.analyze_perturbation_effects()  # sets sdh.percent_change / p_flip

        df = pd.DataFrame({"g": list(pc), "pct": list(pc.values())}).sort_values("pct", ascending=False)
        single = [{"target": r.g, "reduction": round(float(r.pct), 1),
                   "p_flip": float(pf.get(r.g, 1.0)),
                   "p_score": float(ps.get(r.g, 1.0)),
                   "high_to_background": int(sdh.h_to_b_dict.get(r.g, 0)),
                   "background_to_high": int(sdh.b_to_h_dict.get(r.g, 0)),
                   "affected_cells": int(sdh.h_to_b_dict.get(r.g, 0) +
                                         sdh.b_to_h_dict.get(r.g, 0)),
                   "drugs": DRUGS.get(r.g, [])}
                  for r in df.head(top_single).itertuples()]

        # Keep the exact before/after labels in memory for an inline chat figure. They are
        # deliberately excluded from the persisted evidence payload because the arrays can be
        # large and are reproducible from the hashed case inputs.
        for row in single:
            idx = list(cand).index(row["target"])
            perturbed = sdh.annotateCells(
                sdh.optimized_results[idx].copy(), sdh.percentile_cells,
                mode="no", perturbation=True)
            CONTEXT["cache"][f"transition::target::{row['target']}"] = {
                "obs_names": list(map(str, sdh.adata.obs_names)),
                "before": list(map(str, sdh.adata.obs["SIDISH"])),
                "after": list(map(str, perturbed.obs["SIDISH"])),
                "risk_delta": [float(x) for x in perturbed.obs["perturbation_score"]],
            }

        dual, k = [], min(int(dual_top_n), len(df))
        try:                                              # double sweep over strongest single hits
            dbl_pct, _ = sdh.run_double_Perturbation(df.g.values[:k], top_n=k)
            dbl = sorted(dbl_pct.items(), key=lambda kv: -float(kv[1]))[:top_dual]
            dual = [{"target": str(kk).replace("+", " + "), "reduction": round(float(v), 1)}
                    for kk, v in dbl]
        except Exception as e:
            dual = [{"error": f"double perturbation failed: {e}"}]
    finally:
        if tmpdir:                                        # restore cohort state + clean up
            sdh.path = orig_path
            if CONTEXT.get("adata") is not None:
                sdh.adata = CONTEXT["adata"]
            shutil.rmtree(tmpdir, ignore_errors=True)

    out = {"mode": "live", "n_perturbed": len(cand), "patient": patient or "cohort",
           "evidence_scope": ("patient-specific model perturbation" if patient
                              else "cohort-level association"),
           "perturbation_model": "target plus PPI-neighbour network ablation",
           "limitation": ("This implementation modifies the nominated target and connected PPI "
                          "neighbours; it is not a literal isolated single-gene knockout."),
           "gene_scope": ("explicit" if genes else scope), "single": single, "dual": dual}
    CONTEXT["cache"][ckey] = out
    CONTEXT["cache"]["pert"] = out                        # last result (drug_perturbation reads this)
    return out


# ------------------------------------------------------------- drug_perturbation
def _clue_connectivity(up_genes, down_genes, api_key, n=10):
    """Live CMap/L1000 connectivity query against clue.io (lab-server path).
    Submits the high-risk up/down signature and returns the most NEGATIVELY connected
    perturbagens (drugs predicted to reverse the high-risk program). Best-effort:
    any failure raises so the caller falls back to the offline table."""
    import json as _json, urllib.request as _u
    payload = {"name": "SIDISH_highrisk", "data": {
        "upGenes": [str(g) for g in list(up_genes)[:150]],
        "downGenes": [str(g) for g in list(down_genes)[:150]]}}
    req = _u.Request("https://api.clue.io/api/gene_expression/query?user_key=" + api_key,
                     data=_json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with _u.urlopen(req, timeout=60) as r:              # noqa: S310 (trusted host, key-gated)
        res = _json.loads(r.read().decode())
    return {"mode": "live", "method": "CMap L1000 connectivity (clue.io)",
            "note": "Perturbagens predicted to reverse the high-risk signature (negative connectivity).",
            "connections": res.get("result", res)}


def _reverse_signature_drugs(top: int = 8) -> list:
    """CMap-style reverse-signature scoring (offline proxy): rank drugs whose target is
    UP-regulated in the high-risk program — inhibiting them opposes (reverses) it. Scored
    by the target's strength among the high-risk markers."""
    up = CONTEXT["cache"].get("up_genes") or []
    up_rank = {g: i for i, g in enumerate(up)}
    down = set(CONTEXT["cache"].get("down_genes") or [])
    rows = []
    for target, drugs in CMAP_DRUGS.items():
        if target in up_rank:
            score, direction = 1.0 / (1 + up_rank[target]), "inhibit target (up in high-risk)"
        elif target in down:
            score, direction = -0.5, "target down in high-risk"
        else:
            continue
        for d in drugs:
            rows.append({"drug": d["drug"], "target": target, "moa": d["moa"],
                         "status": d["status"], "connectivity_score": round(score, 3),
                         "rationale": direction})
    rows.sort(key=lambda r: -r["connectivity_score"])
    return rows[:top]


def drug_perturbation(targets: list = None, top_targets: int = 6, method: str = "targets") -> dict:
    """Map the high-risk program to candidate drugs via CMap/LINCS.
      method='targets' (default): map the top knockout targets -> compounds (clue.io if
        CLUE_API_KEY is set, else a curated CMap-derived table).
      method='reverse_signature': rank drugs predicted to REVERSE the high-risk up/down
        signature (offline target-based connectivity proxy; live L1000 with CLUE_API_KEY).
    All results are research-use hypotheses."""
    if method == "reverse_signature":
        if not _live():
            _mock_or_raise("drug perturbation")
        if not CONTEXT["cache"].get("up_genes"):
            marker_genes()
        api_key = os.environ.get("CLUE_API_KEY") or os.environ.get("SIDISH_CLUE_API_KEY")
        if api_key:
            try:
                return _clue_connectivity(
                    CONTEXT["cache"].get("up_genes") or [],
                    CONTEXT["cache"].get("down_genes") or [],
                    api_key,
                    n=max(top_targets, 8),
                )
            except Exception as exc:
                live_note = f"Live CMap query failed ({exc}); using the bundled offline proxy."
        else:
            live_note = "Set CLUE_API_KEY for live L1000 signature connectivity."
        return {"mode": "live" if _live() else "mock",
                "evidence_scope": "computational hypothesis",
                "method": "CMap reverse-signature (offline proxy)",
                "note": "Drugs whose target is up-regulated in high-risk cells (inhibition reverses the "
                        f"program). {live_note}",
                "ranked_drugs": _reverse_signature_drugs(top=max(top_targets, 8)),
                "limitation": ("The offline score is a target-rank heuristic, not a measured drug "
                               "perturbation or patient response prediction.")}
    # Resolve which targets to map: explicit list -> top perturbation hits -> top markers.
    if not targets:
        pert = CONTEXT["cache"].get("pert")
        if pert and pert.get("single"):
            targets = [s["target"] for s in pert["single"]][:top_targets]
        else:
            up = CONTEXT["cache"].get("up_genes")
            targets = list(up)[:top_targets] if up else []
    if not targets:
        return {"error": "no targets — run perturbation or marker_genes first, or pass targets=[...]"}

    api_key = os.environ.get("CLUE_API_KEY") or os.environ.get("SIDISH_CLUE_API_KEY")
    note = "offline curated CMap/LINCS target->perturbagen mapping; set CLUE_API_KEY for live L1000 connectivity."
    if api_key:
        try:
            up = CONTEXT["cache"].get("up_genes") or targets
            down = getattr(CONTEXT.get("sdh"), "downregulated_genes", [])
            return _clue_connectivity(up, down, api_key)
        except Exception as e:
            note = f"live CMap query failed ({e}); using offline curated mapping."

    by_target = [{"target": str(g), "compounds": _cmap_compounds(g)} for g in targets]
    mapped = [r for r in by_target if r["compounds"]]
    if not _live():
        _mock_or_raise("drug perturbation")
    return {"mode": "live" if _live() else "mock",
            "evidence_scope": "computational hypothesis",
            "method": "CMap target-based (offline)", "note": note,
            "knowledge_base_version": "SIDISH bundled target-to-compound mapping 2026-07-28",
            "n_targets": len(targets), "n_mapped": len(mapped), "by_target": by_target,
            "limitation": ("Target-to-compound links may be indirect, context-dependent, or approved "
                           "only in other indications; they are literature leads, not response predictions.")}


# -------------------------------------------------------------- pathway_enrichment
def pathway_enrichment(top_n: int = 8, q_threshold: float = 0.05) -> dict:
    """Enriched biological pathways of the high-risk marker program (the guide's 'pathway
    analysis'). Reads the bundled over-representation table (BREAST_PATHWAY.txt: GO/BP
    enrichment of the high-risk markers) and keeps pathways overlapping the CURRENT
    up-marker genes, ranked by FDR q-value. Also returns the suppressed (down) gene program.
    All numbers come from the enrichment table / SIDISH — never invented."""
    if not _live():
        _mock_or_raise("pathway analysis")
        pw = _MOCK.get("pathways", {})
        return {"mode": "mock", "demo": True, "scope": "cohort-level association",
                "activated": pw.get("activated", []),
                "suppressed_genes": pw.get("suppressed_genes", [])}
    import os, pandas as pd
    # ensure markers (up + down) are available
    if not CONTEXT["cache"].get("up_genes"):
        marker_genes()
    up = set(CONTEXT["cache"].get("up_genes") or [])
    down = list(CONTEXT["cache"].get("down_genes") or [])

    # Preferred path: recompute enrichment for the current marker set from a
    # versioned GMT collection supplied by the deployment.
    gmt_path = os.environ.get("SIDISH_GENESETS_GMT")
    if gmt_path and Path(gmt_path).is_file():
        from scipy.stats import hypergeom
        from statsmodels.stats.multitest import multipletests
        universe = set(map(str, _adata().var_names))
        sets = []
        with Path(gmt_path).open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    sets.append((parts[0], set(parts[2:]) & universe))
        raw = []
        N, n = len(universe), len(up & universe)
        for name, members in sets:
            overlap = up & members
            if N and n and len(overlap) >= 2:
                pval = float(hypergeom.sf(len(overlap) - 1, N, len(members), n))
                raw.append((name, pval, sorted(overlap), len(members)))
        if raw:
            adjusted = multipletests([r[1] for r in raw], method="fdr_bh")[1]
            rows = [{"name": r[0], "source": Path(gmt_path).name,
                     "q_value": f"{float(q):.1e}", "n_genes": len(r[2]),
                     "genes": r[2][:12]}
                    for r, q in zip(raw, adjusted) if q <= q_threshold]
            rows.sort(key=lambda x: float(x["q_value"]))
            return {"mode": "live", "scope": "cohort-level association",
                    "method": "Current-marker hypergeometric enrichment with BH-FDR correction",
                    "gene_set_source": str(Path(gmt_path).resolve()),
                    "activated": rows[:top_n], "suppressed_genes": down[:10],
                    "suppressed_note": "Downregulated genes are descriptive; run a separate enrichment if needed.",
                    "limitation": "Pathways depend on the configured gene-set collection and expression universe."}

    pth = next((c for c in [os.path.join(CONFIG["run_dir"], "BREAST_PATHWAY.txt"),
                            str(HERE / "BREAST_CANCER" / "BREAST_PATHWAY.txt")]
                if os.path.exists(c)), None)
    if pth is None:
        return {"mode": "live", "activated": [], "suppressed_genes": down[:10],
                "note": "No bundled pathway enrichment table found; run enrichment (e.g. Enrichr) on the lab server."}
    df = pd.read_csv(pth, sep="\t")
    qcol = next((c for c in df.columns if "FDR B&H" in c), None) or \
           next((c for c in df.columns if str(c).lower().startswith("p-value")), df.columns[4])
    gcol = next((c for c in df.columns if "Hit in Query" in c), df.columns[-1])
    rows = []
    for _, r in df.iterrows():
        try:
            q = float(r[qcol])
        except (ValueError, TypeError):
            continue
        genes = [g.strip() for g in str(r[gcol]).split(",") if g.strip()]
        overlap = [g for g in genes if g in up] if up else genes
        if q <= q_threshold and (not up or len(overlap) >= 3):
            rows.append({"name": str(r["Name"]).strip(), "source": str(r["Category"]).strip(),
                         "q_value": f"{q:.1e}", "_q": q, "n_genes": len(overlap),
                         "genes": overlap[:12]})
    rows = sorted(rows, key=lambda x: x["_q"])[:top_n]
    for x in rows:
        x.pop("_q", None)
    return {"mode": "live", "scope": "cohort-level association",
            "method": "Overlap with the bundled manuscript GO enrichment table",
            "activated": rows, "suppressed_genes": down[:10],
            "suppressed_note": "Genes reduced in high-risk cells; recompute enrichment for the current gene set.",
            "limitation": ("Displayed q-values belong to the original manuscript enrichment table. "
                           "They are not newly estimated for a different case or marker threshold.")}


# ------------------------------------------------------------ pathway_perturbation
def pathway_perturbation(pathway: str = None, genes: list = None, top_pathways: int = 3,
                         n_jobs: int = 4, patient: str = None) -> dict:
    """Knock out a WHOLE gene program at once (all genes of a pathway simultaneously) and
    measure the high-risk reduction — UNAGI-style pathway perturbation. Give a pathway name
    (matched against the enrichment table), an explicit gene list, or neither to sweep the
    top enriched pathways. Numbers come from SIDISH's engine."""
    if not _live():
        _mock_or_raise("pathway perturbation")
        return {"mode": "mock", "demo": True, "evidence_scope": "computational hypothesis",
                "pathways": [{"pathway": "extracellular matrix organization", "reduction": 61.0}]}
    import numpy as np
    import scanpy as sc
    from scipy.stats import binomtest, wilcoxon
    from SIDISH.in_silico_perturbation import InSilicoPerturbation
    from SIDISH.gene_perturbation_utils import GenePerturbationUtils
    sdh = CONTEXT["sdh"]

    # resolve the pathway -> gene sets to test
    enr = pathway_enrichment(top_n=max(top_pathways, 8))
    acts = enr.get("activated", [])
    if genes:
        sets = [{"pathway": (pathway or "custom gene set"), "genes": list(genes)}]
    elif pathway:
        hit = next((a for a in acts if pathway.lower() in a["name"].lower()), None)
        if not hit:
            return {"error": f"pathway '{pathway}' not found in the enrichment table",
                    "available": [a["name"] for a in acts]}
        sets = [{"pathway": hit["name"], "genes": hit["genes"]}]
    else:
        sets = [{"pathway": a["name"], "genes": a["genes"]} for a in acts[:top_pathways]]

    base = sc.read_h5ad(sdh.path + "adata_SIDISH.h5ad")   # fresh raw counts
    if patient:
        if "patient" in base.obs.columns:
            pmask = base.obs["patient"].astype(str) == str(patient)
        else:
            pmask = base.obs.index.str.split("_").str[0].astype(str) == str(patient)
        if int(pmask.sum()) == 0:
            return {"error": f"patient {patient} not found in the cohort"}
        base = base[pmask].copy()
    var = set(map(str, base.var.index))
    engine = InSilicoPerturbation(base)
    engine.setup_ppi_network(threshold=0.7)
    n_h = int((base.obs["SIDISH"].astype(str) == "h").sum())
    if n_h == 0:
        return {"error": f"{patient or 'cohort'} has no high-risk cells to perturb"}

    results = []
    for s in sets:
        gene_set = [g for g in s["genes"] if str(g) in var]
        if not gene_set:
            continue
        adata_p = base.copy()
        for g in gene_set:                                    # knock out every gene in the program
            dn, ind = engine.ppi_handler.get_neighbors(g)
            neigh = dn + ind
            ndf = engine.ppi_handler.ppi_df[
                engine.ppi_handler.ppi_df["Source"].isin(neigh) |
                engine.ppi_handler.ppi_df["Target"].isin(neigh)]
            if not ndf.empty:
                adata_p = GenePerturbationUtils.adjust_expression(adata_p, g, ndf)
            else:
                adata_p.X = GenePerturbationUtils.knockout_gene(adata_p, g).tocsr()
        adata_p = sdh.annotateCells(adata_p, sdh.percentile_cells, mode="no", perturbation=True)
        h_to_b = int(((base.obs["SIDISH"].astype(str) == "h") &
                      (adata_p.obs["SIDISH"].astype(str) == "b")).sum())
        b_to_h = int(((base.obs["SIDISH"].astype(str) == "b") &
                      (adata_p.obs["SIDISH"].astype(str) == "h")).sum())
        flips = h_to_b + b_to_h
        p_flip = float(binomtest(h_to_b, flips, p=0.5, alternative="greater").pvalue) \
            if flips else 1.0
        delta = adata_p.obs["perturbation_score"].to_numpy()
        p_score = 1.0 if np.allclose(delta, 0) else float(
            wilcoxon(delta, alternative="greater").pvalue)
        results.append({"pathway": s["pathway"], "n_genes": len(gene_set),
                        "reduction": round(((h_to_b - b_to_h) / max(n_h, 1)) * 100, 1),
                        "high_to_background": h_to_b,
                        "background_to_high": b_to_h,
                        "affected_cells": flips, "p_flip": p_flip, "p_score": p_score,
                        "genes": gene_set[:10]})
        CONTEXT["cache"][f"transition::pathway::{s['pathway']}"] = {
            "obs_names": list(map(str, base.obs_names)),
            "before": list(map(str, base.obs["SIDISH"])),
            "after": list(map(str, adata_p.obs["SIDISH"])),
            "risk_delta": [float(x) for x in delta],
        }
    results.sort(key=lambda r: -r["reduction"])
    return {"mode": "live", "patient": patient or "cohort",
            "evidence_scope": ("patient-specific model perturbation" if patient
                               else "cohort-level association"),
            "method": "simultaneous target-plus-PPI-network ablation for pathway member genes",
            "pathways": results,
            "limitation": "Computational network intervention; not a pharmacologic response prediction."}


# ----------------------------------------------------------------------- mechanism
def _ppi_handler():
    """Cached PPI network handler (build once; the STRING/HIPPIE parse is the slow part)."""
    h = CONTEXT["cache"].get("ppi_handler")
    if h is None:
        import scanpy as sc
        from SIDISH.ppi_network_handler import PPINetworkHandler
        sdh = CONTEXT["sdh"]
        ad = sc.read_h5ad(sdh.path + "adata_SIDISH.h5ad")
        h = PPINetworkHandler(ad)
        h.load_network(0.7)
        CONTEXT["cache"]["ppi_handler"] = h
    return h


def _pathway_gene_sets():
    """Cached parse of BREAST_PATHWAY.txt into {name, q, genes} for overlap enrichment."""
    ps = CONTEXT["cache"].get("pathway_sets")
    if ps is None:
        import os, pandas as pd
        pth = next((c for c in [os.path.join(CONFIG["run_dir"], "BREAST_PATHWAY.txt"),
                                str(HERE / "BREAST_CANCER" / "BREAST_PATHWAY.txt")]
                    if os.path.exists(c)), None)
        ps = []
        if pth:
            df = pd.read_csv(pth, sep="\t")
            qcol = next((c for c in df.columns if "FDR B&H" in c), df.columns[4])
            gcol = next((c for c in df.columns if "Hit in Query" in c), df.columns[-1])
            for _, r in df.iterrows():
                try:
                    q = float(r[qcol])
                except (ValueError, TypeError):
                    continue
                genes = {g.strip() for g in str(r[gcol]).split(",") if g.strip()}
                ps.append({"name": str(r["Name"]).strip(), "q": q, "genes": genes})
        CONTEXT["cache"]["pathway_sets"] = ps
    return ps


def mechanism(target: str = None, drug: str = None, top_pathways: int = 5, top_tfs: int = 5) -> dict:
    """Explain WHY a perturbation/drug works (the 'story' the doctors want). Resolves the
    affected gene network (target + PPI neighbours — the genes the knockout actually changes),
    then runs PATHWAY enrichment + TF enrichment on it, and returns a grounded mechanistic
    narrative. Give a target gene, a drug name, or neither (uses the top perturbation hit)."""
    if not _live():
        _mock_or_raise("mechanism analysis")
        return {"mode": "mock", "demo": True, "target": "COL6A1",
                "narrative": "Perturbing COL6A1 disrupts extracellular-matrix organization."}
    import mechanism_refs as MR
    sdh = CONTEXT["sdh"]
    # resolve the target gene
    if drug and not target:
        target = next((g for g, ds in CMAP_DRUGS.items()
                       if any(drug.lower() in d["drug"].lower() for d in ds)), None)
    if not target:
        pert = CONTEXT["cache"].get("pert")
        if pert and pert.get("single"):
            target = pert["single"][0]["target"]
        else:
            up = CONTEXT["cache"].get("up_genes") or (marker_genes() and CONTEXT["cache"].get("up_genes"))
            target = up[0] if up else None
    if not target:
        return {"error": "no target — run perturbation first or pass target=/drug="}

    var = set(map(str, sdh.adata.var.index))
    h = _ppi_handler()
    direct, indirect = h.get_neighbors(target)
    affected = [g for g in ([target] + list(direct) + list(indirect)) if str(g) in var]
    aset = set(affected)

    pathways = []
    for p in _pathway_gene_sets():
        ov = aset & p["genes"]
        if len(ov) >= 2:
            pathways.append({"pathway": p["name"], "q_value": f"{p['q']:.1e}",
                             "overlap": len(ov), "genes": sorted(ov)[:8]})
    pathways.sort(key=lambda x: (-x["overlap"], x["q_value"]))
    pathways = pathways[:top_pathways]

    tf_rows = []
    for tf, reg in MR.TF_REGULONS.items():
        ov = aset & set(reg)
        if ov:
            tf_rows.append({"tf": tf, "n_targets": len(ov), "targets": sorted(ov)[:8]})
    tf_rows.sort(key=lambda x: -x["n_targets"])
    tf_rows = tf_rows[:top_tfs]
    tfs_perturbed = sorted(aset & MR.HUMAN_TFS)

    drugs = [d["drug"] for d in _cmap_compounds(target)]
    top_path = pathways[0]["pathway"] if pathways else "extracellular-matrix / stromal programs"
    top_tf = tf_rows[0]["tf"] if tf_rows else (tfs_perturbed[0] if tfs_perturbed else None)
    drug_str = drugs[0] if drugs else None
    narrative = (
        f"Perturbing <strong>{target}</strong>"
        + (f" (e.g. {drug_str})" if drug_str else "")
        + f" alters a connected network of {len(affected)} genes that is enriched for "
        + f"<strong>{top_path}</strong>"
        + (f", with overlap involving transcription factor <strong>{top_tf}</strong> targets" if top_tf else "")
        + f". The model therefore nominates {target} and this connected program for orthogonal "
        + f"validation; the overlap does not establish causal pathway control or drug response."
    )
    return {"mode": "live", "evidence_scope": "computational hypothesis",
            "target": target, "drugs": drugs, "n_affected": len(affected),
            "affected_genes": affected[:25], "pathways": pathways, "tf_enrichment": tf_rows,
            "tfs_perturbed": tfs_perturbed, "narrative": narrative,
            "method": "affected genes = target + PPI neighbours; pathway-table and curated-regulon overlap",
            "limitation": "Overlap counts are descriptive and do not establish enrichment significance or causality."}


# ------------------------------------------------------------------ show_figures
def show_figures(patient_id: str = "cohort", kinds: list = None,
                 patient_specific: bool = False) -> dict:
    """Generate SIDISH plots and return their file paths, so the agent can show them.
    kinds (default all): highrisk_umap, celltype_umap, distribution, celltype_bar,
    marker_heatmap, survival, perturbation_bar, transition_umap."""
    if not _live():
        _mock_or_raise("figure generation")
        figs = _MOCK["features"]["figures"]
        return {"mode": "mock", "demo": True, "figures": figs}
    import generate_figures as G
    made = G.generate(
        patient_id,
        kinds=kinds,
        patient=patient_id if patient_specific else None,
    )
    return {"mode": "live", "patient_id": patient_id, "figures": made,
            "patient_specific": patient_specific,
            "note": "PNG files saved under figures/; reference these paths in the report."}


# ----------------------------------------------------------------- build_report
def build_report(patient_id: str, use_llm: bool = False, with_figures: bool = True,
                 patient_specific: bool = False, make_pdf: bool = False,
                 survival_only: bool = False, keep_figures: list = None) -> dict:
    """Assemble the payload from tool outputs and render the 4-section report.
    with_figures generates the live plots first; patient_specific runs the perturbation
    on only this patient's cells; make_pdf also writes a PDF; survival_only (or
    keep_figures=['survival',...]) drops all figures except the ones named."""
    if not _live() and not _demo_enabled():               # self-sufficient; clinician mode fails closed
        try:
            init_sidish()
        except Exception as e:
            raise RuntimeError(f"could not load the live SIDISH model; report not generated: {e}") from e
    if _live() and with_figures:
        try:
            import generate_figures as G
            G.generate(patient_id, patient=patient_id if patient_specific else None)
        except Exception as e:
            print(f"[warn] figure generation failed: {e}")
    feats = patient_features(patient_id)
    marks = marker_genes()
    pert = perturbation(patient=patient_id if patient_specific else None)
    km = survival_km()

    payload = json.loads(json.dumps(_MOCK))               # schema skeleton
    payload["patient"]["patient_id"] = patient_id
    if _live():
        overview = highrisk_overview()
        per = overview.get("per_patient", {})
        ordered = sorted(per, key=per.get, reverse=True)
        rank = ordered.index(str(patient_id)) + 1 if str(patient_id) in ordered else None
        contrast_id = min(per, key=per.get) if per else None
        payload["meta"] = {
            "report_type": "research-use clinician decision support",
            "generated_by": "SIDISH-Agent live analysis",
            "research_use_only": True,
            "demo": False,
            "model_version": CONFIG["model_version"],
        }
        payload["patient"].update({
            "age": "not available", "sex": "not available", "stage": "not available",
            "prior_treatment": "not available", "metadata_available": False,
            "data_source": CONTEXT.get("source_dataset") or "configured SIDISH single-cell dataset",
        })
        payload["cohort_context"].update({
            "n_cells_total": overview.get("n_cells"),
            "n_high_risk_total": overview.get("n_high_risk"),
            "high_risk_fraction_cohort": overview.get("cohort_high_risk_fraction"),
        })
        payload["features"]["patient_high_risk_fraction"] = feats["high_risk_fraction"]
        payload["features"]["n_cells"] = feats.get("n_cells")
        payload["features"]["n_high_risk"] = feats.get("n_high_risk")
        payload["features"]["scope"] = feats.get("scope")
        payload["features"]["patient_rank_in_cohort"] = (
            f"rank {rank} of {len(ordered)} by model-defined high-risk-cell fraction"
            if rank else "not available")
        payload["features"]["contrast_patient"] = {
            "patient_id": contrast_id or "not available",
            "high_risk_fraction": per.get(contrast_id) if contrast_id else None,
            "note": "lowest observed burden in this analysed cohort" if contrast_id else "not available",
        }
        payload["features"]["celltype_composition"] = feats["celltype_composition"]
        payload["features"]["enriched_vs_control"] = feats["enriched"]
        payload["features"]["reduced_vs_control"] = feats["reduced"]
        payload["features"]["marker_genes"] = [{"gene": g, "note": ""} for g in marks["genes"]]
        payload["features"]["marker_genes_down"] = marks.get("downregulated", [])
        payload["pathways"] = pathway_enrichment()        # activated pathways + suppressed genes
        payload["perturbation"]["single_gene_top"] = [
            {"target": x["target"], "high_risk_reduction_percent": x["reduction"],
             "candidate_drugs": x.get("drugs", [])} for x in pert["single"]]
        payload["perturbation"]["evidence_scope"] = pert.get("evidence_scope")
        payload["perturbation"]["method"] = pert.get("perturbation_model")
        payload["perturbation"]["limitation"] = pert.get("limitation")

        def _dual_drugs(pair):                            # union of CMap drugs for both genes
            ds = []
            for g in str(pair).replace(" + ", "+").split("+"):
                ds += DRUGS.get(g.strip(), [])
            return ds
        payload["perturbation"]["dual_gene_top"] = [
            {"target": x["target"], "high_risk_reduction_percent": x["reduction"],
             "candidate_drugs": _dual_drugs(x["target"])}
            for x in pert["dual"] if "error" not in x]
        payload["prognosis"]["km_pvalue"] = km["km_pvalue"]
        payload["prognosis"]["scope"] = km.get("scope")
        payload["prognosis"]["km_note"] = km.get("limitation")
        payload["perturbation"]["drug_mapping"] = drug_perturbation()   # CMap target->compound block
        try:
            payload["mechanism"] = mechanism()                          # why the top target works
        except Exception as e:
            payload["mechanism"] = {"error": str(e)}
        payload.pop("cost", None)                                        # not part of a clinical evidence report

    pdir = HERE / "payloads"; pdir.mkdir(exist_ok=True)
    ppath = pdir / f"{patient_id}.json"; ppath.write_text(json.dumps(payload, indent=2))
    keep = keep_figures or (["survival"] if survival_only else None)
    suffix = "_textonly" if (survival_only and not keep_figures) else ""
    import report_writer
    out = report_writer.build(str(ppath), use_llm=use_llm, make_pdf=make_pdf,
                              keep_figures=keep, out_suffix=suffix)
    res = {"status": "ok", "report_path": out, "mode": CONTEXT["mode"]}
    if make_pdf:
        res["pdf_path"] = str(out).rsplit(".html", 1)[0] + ".pdf"
    return res


# ------------------ tool registry + JSON schemas the LLM sees ------------------
REGISTRY = {"init_sidish": init_sidish, "load_dataset": load_dataset,
            "highrisk_overview": highrisk_overview, "patient_features": patient_features,
            "cohort_features": cohort_features,
            "marker_genes": marker_genes, "survival_km": survival_km,
            "perturbation": perturbation, "drug_perturbation": drug_perturbation,
            "pathway_enrichment": pathway_enrichment, "pathway_perturbation": pathway_perturbation,
            "mechanism": mechanism,
            "show_figures": show_figures, "build_report": build_report}

def _s(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}

TOOLS_SPEC = [
    _s("init_sidish", "Load+reload the already-trained BRCA SIDISH run for live analysis. Omit paths for demo mode.",
       {"adata_path": {"type": "string"}, "bulk_path": {"type": "string"},
        "path": {"type": "string"}, "device": {"type": "string"}}),
    _s("load_dataset",
       "Ingest a user's preprocessed single-cell dataset. If the disease is breast/lung/pancreatic, "
       "reuses the published bulk+survival; otherwise the user must supply bulk_path (SIDISH format: "
       "columns duration,event,<genes...>). Reports if SIDISH must be trained (GPU/lab job).",
       {"adata_path": {"type": "string"}, "cancer_type": {"type": "string"},
        "bulk_path": {"type": "string"}, "device": {"type": "string"}}, ["adata_path"]),
    _s("highrisk_overview", "Cohort high-risk fraction and per-patient high-risk % (the per-sample distribution).", {}),
    _s("patient_features", "High-risk burden and cell-type composition for one patient.",
       {"patient_id": {"type": "string"}}, ["patient_id"]),
    _s("cohort_features", "High-risk burden and cell-type composition across the entire dataset.", {}),
    _s("marker_genes", "Up- and down-regulated marker genes for high-risk cells (SIDISH.get_MarkerGenes).",
       {"logfc_threshold": {"type": "number"}}),
    _s("pathway_enrichment",
       "Enriched biological pathways of the high-risk marker program (GO/BP over-representation), "
       "plus the suppressed (down-regulated) gene set. Computes adjusted p-values from a configured "
       "gene-set database; otherwise labels the bundled manuscript table as reference evidence.",
       {"top_n": {"type": "integer"}}),
    _s("survival_km", "Kaplan-Meier log-rank p-value for the marker signature.",
       {"penalizer": {"type": "number"}}),
    _s("perturbation",
       "In-silico single- and dual-gene knockouts. By default perturbs the top ~50 high-risk "
       "marker genes (fast). Pass genes=[...] for a specific panel, scope='all' for genome-wide "
       "(GPU/lab only), or n_genes to change how many top markers to sweep.",
       {"genes": {"type": "array", "items": {"type": "string"},
                  "description": "Explicit gene symbols to knock out; overrides scope."},
        "scope": {"type": "string", "enum": ["markers", "all"],
                  "description": "'markers' = top n_genes high-risk markers (default); 'all' = every gene."},
        "n_genes": {"type": "integer", "description": "How many top marker genes to sweep (default 50)."},
        "patient": {"type": "string",
                    "description": "Restrict the knockout to this patient's cells (patient-specific effect)."},
        "n_jobs": {"type": "integer"}}),
    _s("drug_perturbation",
       "Map the high-risk program to candidate compounds via CMap/LINCS. method='targets' maps "
       "top knockout targets to drugs; method='reverse_signature' ranks drugs predicted to "
       "reverse the high-risk up/down signature. Returns hypotheses only, with mapping limitations.",
       {"targets": {"type": "array", "items": {"type": "string"},
                    "description": "Gene targets to map; defaults to the top perturbation hits."},
        "method": {"type": "string", "enum": ["targets", "reverse_signature"]},
        "top_targets": {"type": "integer"}}),
    _s("pathway_perturbation",
       "Knock out a WHOLE pathway/gene program at once and measure high-risk reduction. Pass a "
       "pathway name, an explicit gene list, or neither to sweep the top enriched pathways.",
       {"pathway": {"type": "string"}, "genes": {"type": "array", "items": {"type": "string"}},
        "top_pathways": {"type": "integer"},
        "patient": {"type": "string",
                    "description": "Restrict the model perturbation to this patient's cells."}}),
    _s("mechanism",
       "Explain WHY a drug/target works: resolves the affected gene network (target + PPI "
       "neighbours), runs pathway + TF enrichment on it, and returns a mechanistic narrative. "
       "Pass a target gene, a drug name, or neither (top perturbation hit).",
       {"target": {"type": "string"}, "drug": {"type": "string"},
        "top_pathways": {"type": "integer"}, "top_tfs": {"type": "integer"}}),
    _s("show_figures",
       "Generate SIDISH plots and return their file paths so they can be shown/referenced. "
       "kinds: highrisk_umap, celltype_umap, distribution, celltype_bar, marker_heatmap, "
       "survival, perturbation_bar, transition_umap (default = all).",
       {"patient_id": {"type": "string"},
        "kinds": {"type": "array", "items": {"type": "string"}},
        "patient_specific": {"type": "boolean",
                             "description": "Use patient-restricted perturbation results in perturbation figures."}}),
    _s("build_report", "Assemble and render the 4-section clinical report for a patient (auto-generates figures).",
       {"patient_id": {"type": "string"}, "use_llm": {"type": "boolean"},
        "patient_specific": {"type": "boolean",
                             "description": "Run the perturbation on only this patient's cells."},
        "make_pdf": {"type": "boolean", "description": "Also write a PDF alongside the HTML."},
        "survival_only": {"type": "boolean",
                          "description": "Drop all figures except the survival (KM) plot (text-focused variant)."}},
       ["patient_id"]),
]
