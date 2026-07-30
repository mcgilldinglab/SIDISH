"""SIDISH workflow engine — YAML-config driven, with preflight validation.

Mirrors UNAGI's cli.py -> workflow.py pattern so an external agent (or the CLI) can run
SIDISH reproducibly from a config file instead of ad-hoc Python. Two workflows:

  * analysis  : reload a TRAINED SIDISH run and produce a structured clinician
                decision-support report
                (markers -> pathways -> perturbation -> drugs -> survival -> report).
  * training  : train SIDISH on a NEW single-cell dataset (long, GPU/lab-server;
                driven as a durable job by skills/.../scripts/sidish_job.py).

Public API (used by sidish_cli.py and the job orchestrator):
    run_workflow_from_config(config_path, validate_only=False) -> dict
    preflight(config, repo_root) -> dict            (raises PreflightError on hard failures)
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path

HERE = Path(__file__).parent


class PreflightError(Exception):
    """Raised when a config or its inputs fail validation before any heavy work."""


# ------------------------------------------------------------------ config I/O
def load_config(config_path: str) -> dict:
    import yaml
    p = Path(config_path).expanduser()
    if not p.exists():
        raise PreflightError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise PreflightError(f"config must be a YAML mapping: {p}")
    return cfg


def _resolve(path: str) -> Path:
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else (HERE / p)


# ------------------------------------------------------------------ preflight
def _validate_bulk(path: Path) -> tuple[bool, str]:
    from sidish_data_router import validate_bulk_survival
    checked = validate_bulk_survival(path)
    detail = (f"{checked.n_samples} samples, {checked.n_genes} gene columns"
              if checked.ok else "; ".join(checked.errors))
    return checked.ok, detail


def preflight(cfg: dict, repo_root: Path = HERE) -> dict:
    """Validate a config and its inputs. Returns a report dict; raises PreflightError
    on hard failures so no heavy work starts on a broken setup."""
    mode = cfg.get("mode", "analysis")
    checks, errors, warnings = [], [], []

    def ok(name, cond, msg="", warn=False):
        checks.append({"check": name, "ok": bool(cond), "detail": msg})
        if not cond:
            (warnings if warn else errors).append(f"{name}: {msg}")

    data = cfg.get("data", {})
    model = cfg.get("model", cfg.get("training", {}))

    # --- single-cell input ---
    adata_path = data.get("adata_path")
    ok("adata_path set", bool(adata_path), "data.adata_path is required")
    if adata_path:
        ok("adata_path exists", _resolve(adata_path).exists(), str(_resolve(adata_path)))

    # --- bulk + survival ---
    from sidish_data_router import normalize_cancer_type, route_case_data
    cancer = normalize_cancer_type(data.get("cancer_type") or "")
    bulk = data.get("bulk_csv")
    route = route_case_data(cancer, str(_resolve(bulk)) if bulk else None)
    ok("bulk source", route.ready, route.message)
    if route.validation:
        ok("bulk_csv format", route.validation.ok,
           (f"{route.validation.n_samples} samples, {route.validation.n_genes} genes"
            if route.validation.ok else "; ".join(route.validation.errors)))
    if (mode == "training" and adata_path and _resolve(adata_path).is_file()
            and route.ready and route.bulk_path):
        try:
            import anndata as ad
            import pandas as pd
            single = ad.read_h5ad(_resolve(adata_path), backed="r")
            single_genes = list(map(str, single.var_names))
            single.file.close()
            bulk_genes = list(map(str, pd.read_csv(route.bulk_path, nrows=0).columns[2:]))
            common = set(single_genes).intersection(bulk_genes)
            overlap = len(common) / max(len(bulk_genes), 1)
            aligned = (len(single_genes) == len(set(single_genes))
                       and len(common) >= 500 and overlap >= 0.80)
            ok("single-cell/bulk gene alignment", aligned,
               f"{len(common)} common genes; {overlap:.1%} of bulk genes represented")
        except Exception as exc:
            ok("single-cell/bulk gene alignment", False, f"could not check alignment: {exc}")

    # --- device ---
    device = str(model.get("device") or "cpu")
    if device.startswith("cuda"):
        try:
            import torch
            ok("cuda available", torch.cuda.is_available(),
               "config asks for cuda but none is available; will fall back to cpu", warn=True)
        except Exception:
            ok("cuda check", False, "torch not importable to verify cuda", warn=True)
    else:
        ok("device", True, device)

    if mode in {"analysis", "case_analysis"}:
        run_dir = _resolve(data.get("run_dir", "")) if data.get("run_dir") else None
        ok("run_dir set", bool(run_dir),
           str(run_dir) if run_dir else "data.run_dir (trained run) is required for analysis")
        if run_dir:
            ok("trained VAE present", (run_dir / "vae_transfer").exists(), str(run_dir / "vae_transfer"))
            ok("trained DeepCox present", (run_dir / "deepCox").exists(), str(run_dir / "deepCox"))
        # perturbation needs PPI
        if cfg.get("preflight", {}).get("require_ppi", True):
            ppi_dir = next((d for d in [os.environ.get("SIDISH_PPI_DIR"), str(HERE / "PPI"),
                                        str(HERE / ".." / "data" / "PPI")]
                            if d and (Path(d) / "hippie_current.txt").exists()), None)
            ok("PPI files", bool(ppi_dir),
               ppi_dir if ppi_dir else ("PPI files (hippie_current.txt, STRING links/info) not found "
                                        "for perturbation; set SIDISH_PPI_DIR or place them in ./PPI"),
               warn=True)
        # LLM (only if report.use_llm)
        if cfg.get("report", {}).get("use_llm"):
            has_llm = bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL")) \
                or bool(cfg.get("llm", {}).get("base_url"))
            ok("LLM configured", has_llm,
               "report.use_llm is true but no LLM endpoint is set (OPENAI_BASE_URL/OPENAI_API_KEY "
               "or llm.base_url); will fall back to the deterministic report", warn=True)

    if mode == "training":
        ok("training params", bool(model.get("iterations") or model.get("max_iter")),
           "training.iterations is required", warn=True)
        ok("output path", bool(cfg.get("training", {}).get("path") or cfg.get("output_dir")),
           "training.path (where to save the run) is recommended", warn=True)

    report = {"mode": mode, "ok": len(errors) == 0, "checks": checks,
              "errors": errors, "warnings": warnings}
    if errors:
        raise PreflightError("; ".join(errors))
    return report


# ------------------------------------------------------------------ run
def _apply_to_tools(cfg: dict):
    """Push config values into sidish_tools.CONFIG and go live."""
    import sidish_tools as T
    data, model = cfg.get("data", {}), cfg.get("model", {})
    if data.get("run_dir"):
        T.CONFIG["run_dir"] = str(_resolve(data["run_dir"])) + ("" if str(data["run_dir"]).endswith("/") else "/")
    if data.get("adata_path"):
        T.CONFIG["adata_path"] = str(_resolve(data["adata_path"]))
    from sidish_data_router import route_case_data
    supplied_bulk = str(_resolve(data["bulk_csv"])) if data.get("bulk_csv") else None
    route = route_case_data(data.get("cancer_type", ""), supplied_bulk)
    if not route.ready:
        raise PreflightError(route.message)
    T.CONFIG["bulk_csv"] = str(route.bulk_path)
    if model.get("device"):
        T.CONFIG["device"] = str(model["device"])
    if model.get("percentile") is not None:
        T.CONFIG["percentile"] = float(model["percentile"])
    T.CONFIG["force_mock"] = False
    # LLM env from config (optional)
    llm = cfg.get("llm", {})
    if llm.get("base_url"):
        os.environ["OPENAI_BASE_URL"] = str(llm["base_url"])
    if llm.get("api_key"):
        os.environ["OPENAI_API_KEY"] = str(llm["api_key"])
    if llm.get("model"):
        os.environ["SIDISH_LLM_MODEL"] = str(llm["model"])
    return T


def run_analysis(cfg: dict) -> dict:
    import matplotlib; matplotlib.use("Agg")
    T = _apply_to_tools(cfg)
    boot = T.init_sidish()
    if boot.get("mode") != "live":
        return {"status": "error", "detail": "could not load a live SIDISH model", "boot": boot}

    an = cfg.get("analysis", {})
    rep = cfg.get("report", {})
    patient = an.get("patient")
    if not patient:                                   # default to the top high-risk patient
        per = T.highrisk_overview().get("per_patient", {})
        patient = max(per, key=per.get) if per else "cohort"

    from decision_report_writer import render_case_report
    from sidish_case_analysis import run_reference_case
    from sidish_case_store import CaseStore
    from sidish_contracts import CaseMetadata

    case_cfg = cfg.get("case", {})
    case_id = str(case_cfg.get("case_id") or f"{patient}-analysis")
    store = CaseStore(str(_resolve(case_cfg["root"])) if case_cfg.get("root") else None)
    result_path = store.case_dir(case_id) / "result.json"
    if result_path.exists():
        result = store.get(case_id)
    else:
        cancer = str(cfg.get("data", {}).get("cancer_type") or "unspecified")
        result = store.create(CaseMetadata(
            case_id=case_id, cancer_type=cancer,
            disease_label=str(case_cfg.get("disease_label") or cancer),
            specimen_id=str(patient), patient_id=str(patient), research_use_only=True,
        ))
    result = run_reference_case(
        result, patient_id=str(patient), adata_path=T.CONFIG["adata_path"],
        bulk_path=T.CONFIG["bulk_csv"], run_dir=T.CONFIG["run_dir"],
        device=T.CONFIG["device"],
        patient_specific=bool(an.get("patient_specific", True)),
        include_figures=not bool(rep.get("survival_only", False)),
    )
    store.save(result)
    reports = render_case_report(result_path, use_llm=bool(rep.get("use_llm", False)),
                                 make_pdf=bool(rep.get("make_pdf", False)))
    return {"status": "ok", "patient": patient, "case_id": case_id,
            "result_path": str(result_path), "reports": reports}


def run_training(cfg: dict) -> dict:
    """Kick off SIDISH training on a NEW dataset. Heavy (GPU/lab). Returns a descriptor;
    the actual multi-hour run is executed here when invoked inside a durable job."""
    import matplotlib; matplotlib.use("Agg")
    import scanpy as sc, pandas as pd
    from SIDISH import SIDISH
    t = cfg.get("training", {})
    data = cfg.get("data", {})
    device = str(t.get("device") or "cpu")
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    from sidish_data_router import route_case_data
    supplied_bulk = str(_resolve(data["bulk_csv"])) if data.get("bulk_csv") else None
    route = route_case_data(data.get("cancer_type", ""), supplied_bulk)
    if not route.ready:
        raise PreflightError(route.message)
    adata = sc.read_h5ad(_resolve(data["adata_path"]))
    bulk = pd.read_csv(route.bulk_path)
    patient_column = data.get("patient_column")
    if patient_column:
        if patient_column == "__index_prefix__":
            adata.obs["patient"] = adata.obs_names.astype(str).str.split("_").str[0]
        elif patient_column in adata.obs.columns:
            adata.obs["patient"] = adata.obs[patient_column].astype(str)
        else:
            raise PreflightError(f"configured patient column {patient_column!r} is absent from AnnData.obs")
    if data.get("align_genes", True):
        bulk_genes = list(map(str, bulk.columns[2:]))
        single_genes = set(map(str, adata.var_names))
        common_genes = [gene for gene in bulk_genes if gene in single_genes]
        overlap = len(common_genes) / max(len(bulk_genes), 1)
        if not adata.var_names.is_unique or len(common_genes) < 500 or overlap < 0.80:
            raise PreflightError(
                "single-cell/bulk gene alignment failed: require unique single-cell gene names, "
                ">=500 common genes, and >=80% of routed bulk genes"
            )
        adata = adata[:, common_genes].copy()
        bulk = bulk[["duration", "event", *common_genes]].copy()
    out_dir = str(_resolve(t.get("path", "RUNS/new_run"))) + "/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sdh = SIDISH(adata, bulk, device, seed=int(cfg.get("random_seed", 0)))
    p1 = t.get("phase1") or [225, 20, 16, [512, 256, 64], 256, "Adam", 1.0e-3, 1e-4, 0, "Dense"]
    p2 = t.get("phase2") or [1000, 64, 1e-6, 0, 0.2, 512]
    sdh.init_Phase1(*p1)
    sdh.init_Phase2(*p2)
    iterations = int(t.get("iterations", 10))
    if not 1 <= iterations <= 100:
        raise PreflightError("training.iterations must be an integer between 1 and 100")
    sdh.train(iterations, float(t.get("percentile", 0.90)),
              float(t.get("steepness", 1.0)), out_dir, show=False)
    result_path = cfg.get("case", {}).get("result_path")
    if result_path:
        from sidish_contracts import SIDISHCaseResult
        rp = _resolve(result_path)
        if rp.exists():
            case_result = SIDISHCaseResult.load(rp)
            case_result.status = "training_complete"
            case_result.metadata.training_iterations = iterations
            case_result.record("training_completed", detail={
                "run_dir": out_dir, "device": device, "iterations": iterations})
            case_result.save(rp)
    return {"status": "ok", "run_dir": out_dir, "device": device,
            "iterations": iterations,
            "note": "trained run written; point an analysis config at this run_dir"}


def run_case_analysis(cfg: dict) -> dict:
    """Run a complete case-isolated analysis and write the structured result bundle."""
    from sidish_contracts import SIDISHCaseResult
    from sidish_case_analysis import run_reference_case
    from decision_report_writer import render_case_report

    case = cfg.get("case", {})
    result_path = _resolve(case.get("result_path", "")) if case.get("result_path") else None
    if not result_path or not result_path.exists():
        raise PreflightError("case.result_path must point to an existing SIDISHCaseResult JSON")
    result = SIDISHCaseResult.load(result_path)
    data, model = cfg.get("data", {}), cfg.get("model", {})
    analysis, report = cfg.get("analysis", {}), cfg.get("report", {})
    patient = analysis.get("patient")
    if not patient:
        raise PreflightError("analysis.patient is required for case_analysis")
    try:
        result.status = "analysis_running"
        result.record("analysis_started", detail={"config_mode": "case_analysis"})
        result.save(result_path)
        result = run_reference_case(
            result, patient_id=str(patient), adata_path=str(_resolve(data["adata_path"])),
            bulk_path=str(_resolve(data["bulk_csv"])), run_dir=str(_resolve(data["run_dir"])) + os.sep,
            device=str(model.get("device") or "cpu"),
            patient_specific=bool(analysis.get("patient_specific", True)),
            include_figures=bool(analysis.get("include_figures", True)),
        )
        result.save(result_path)
        outputs = render_case_report(
            result_path, use_llm=bool(report.get("use_llm", False)),
            make_pdf=bool(report.get("make_pdf", True)))
        return {"status": "ok", "case_id": result.metadata.case_id,
                "result_path": str(result_path), "reports": outputs}
    except Exception as exc:
        result.status = "analysis_failed"
        result.errors.append(str(exc))
        result.record("analysis_failed", detail={"error": str(exc)})
        result.save(result_path)
        raise


def run_workflow_from_config(config_path: str, validate_only: bool = False) -> dict:
    cfg = load_config(config_path)
    report = preflight(cfg)                            # raises PreflightError on hard failures
    if validate_only:
        return {"workflow": "preflight", "config": str(config_path), **report}
    mode = cfg.get("mode", "analysis")
    if mode == "training":
        result = run_training(cfg)
    elif mode == "case_analysis":
        result = run_case_analysis(cfg)
    else:
        result = run_analysis(cfg)
    return {"workflow": mode, "config": str(config_path), "preflight": report, "result": result}
