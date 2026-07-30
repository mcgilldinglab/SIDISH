"""SIDISH workflow CLI — mirrors UNAGI's `python -m UNAGI.cli`.

    python sidish_cli.py preflight -c configs/sidish_workflow_template.yaml
    python sidish_cli.py run       -c configs/sidish_workflow_template.yaml
    python sidish_cli.py report    -c configs/sidish_report.example.yaml
    python sidish_cli.py report    --patient CID3946 --pdf        # config-free shortcut
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import sys

from sidish_env import load_env
load_env()

from sidish_workflow import run_workflow_from_config, PreflightError


def _report_shortcut(args) -> int:
    """Config-free structured report for the bundled breast reference."""
    import matplotlib; matplotlib.use("Agg")
    import sidish_tools as T
    from sidish_case_store import CaseStore
    from sidish_case_analysis import run_reference_case
    from sidish_contracts import CaseMetadata
    from decision_report_writer import render_case_report
    boot = T.init_sidish()
    patient = args.patient
    if not patient:
        per = T.highrisk_overview().get("per_patient", {})
        patient = max(per, key=per.get) if per else "cohort"
    case_id = f"{patient}-reference"
    store = CaseStore()
    result_path = store.case_dir(case_id) / "result.json"
    if result_path.exists():
        result = store.get(case_id)
    else:
        result = store.create(CaseMetadata(
            case_id=case_id, cancer_type="breast", disease_label="Breast cancer",
            specimen_id=patient, patient_id=patient, research_use_only=True, demo=False))
    result = run_reference_case(result, patient_id=patient,
                                patient_specific=True, include_figures=not args.survival_only)
    store.save(result)
    outputs = render_case_report(result_path, use_llm=args.use_llm, make_pdf=args.pdf)
    print(json.dumps({"patient": patient, "mode": boot.get("mode"),
                      "case_result": str(result_path), "reports": outputs}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sidish", description="SIDISH workflow CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("preflight", help="Validate a workflow YAML and its inputs")
    pf.add_argument("-c", "--config", required=True)

    rn = sub.add_parser("run", help="Run a workflow (analysis or training) from a YAML config")
    rn.add_argument("-c", "--config", required=True)

    rp = sub.add_parser("report", help="Build a structured decision-support report (from a config or --patient)")
    rp.add_argument("-c", "--config", default=None)
    rp.add_argument("--patient", default=None)
    rp.add_argument("--use-llm", action="store_true")
    rp.add_argument("--pdf", action="store_true")
    rp.add_argument("--survival-only", action="store_true")
    rp.add_argument("--patient-specific", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = run_workflow_from_config(args.config, validate_only=True)
        elif args.command == "run":
            result = run_workflow_from_config(args.config, validate_only=False)
        else:  # report
            if args.config:
                result = run_workflow_from_config(args.config, validate_only=False)
            else:
                return _report_shortcut(args)
    except PreflightError as exc:
        print(f"[SIDISH preflight error] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"[SIDISH error] {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
