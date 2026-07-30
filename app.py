"""SIDISH chat-first clinician decision-support workspace."""
from __future__ import annotations

from html import escape
from pathlib import Path
import hmac
import os

from sidish_env import load_env
load_env()

import streamlit as st

from sidish_agent_session import SIDISHAgentSession
from sidish_case_analysis import (
    change_selected_patient, change_training_iterations, configure_patient_selection,
    prepare_uploaded_case,
)
from sidish_case_store import CaseStore
from sidish_contracts import CaseMetadata, utc_now
from sidish_data_router import normalize_cancer_type
from sidish_policy import DISCLAIMER


HERE = Path(__file__).resolve().parent
store = CaseStore()
st.set_page_config(page_title="SIDISH", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg, #f7fbff 0%, #ffffff 38%); }
  [data-testid="stSidebar"] { background: #f0f6fa; border-right: 1px solid #dce8f0; }
  .sidish-hero { padding: 1.2rem 1.35rem; border-radius: 18px; color: white;
    background: linear-gradient(120deg,#0d3b66,#126e82); margin-bottom: 1rem;
    box-shadow: 0 10px 28px rgba(13,59,102,.14); }
  .sidish-hero h2 { margin:0 0 .25rem 0; color:white; }
  .sidish-hero p { margin:0; opacity:.9; }
  .status-card { background:white; border:1px solid #dce6ee; border-radius:14px;
    padding:1rem; min-height:116px; box-shadow:0 3px 12px rgba(28,55,76,.05); }
  .status-card.pass { border-top:5px solid #169873; }
  .status-card.warning { border-top:5px solid #e5a11a; }
  .status-card.fail { border-top:5px solid #d64545; }
  .status-card.not_assessed { border-top:5px solid #8293a1; }
  .status-label { font-size:.72rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
  .status-label.pass { color:#11775b; } .status-label.warning { color:#a96d00; }
  .status-label.fail { color:#b22e2e; } .status-label.not_assessed { color:#647582; }
  .status-title { font-weight:750; color:#17364f; margin:.35rem 0; }
  .status-detail { font-size:.86rem; color:#536879; line-height:1.35; }
  .scope-card { border-radius:12px; padding:.8rem .9rem; margin:.35rem 0; border-left:5px solid; }
  .scope-sample { background:#eaf8f3; border-color:#169873; }
  .scope-patient { background:#fff6df; border-color:#e5a11a; }
  .scope-cohort { background:#eaf3fb; border-color:#2b78b8; }
  .scope-hypothesis { background:#f4effb; border-color:#7a57a8; }
  .scope-title { font-weight:750; color:#17364f; }
  .muted { color:#607484; font-size:.88rem; }
  div[data-testid="stChatMessage"] { border:1px solid #e0e9ef; border-radius:14px; padding:.35rem; }
</style>
""", unsafe_allow_html=True)


def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _auth_gate():
    expected = os.environ.get("SIDISH_APP_PASSWORD")
    if not expected:
        st.warning("Local prototype mode: no application password configured. Bind only to 127.0.0.1.")
        return
    if st.session_state.get("authenticated"):
        return
    st.title("SIDISH")
    st.caption("Clinician decision support for single-cell risk-state analysis")
    supplied = st.text_input("Application password", type="password")
    if st.button("Sign in", type="primary") and hmac.compare_digest(supplied, expected):
        st.session_state["authenticated"] = True
        _rerun()
    st.stop()


def _schema(case_id: str) -> dict:
    path = store.case_dir(case_id) / "inputs" / "single_cell_schema.json"
    if not path.exists():
        return {"columns": {}}
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def _qc_cards(result):
    if not result.qc:
        st.info("Quality checks will appear after the single-cell file is validated.")
        return
    for start in range(0, len(result.qc), 3):
        cols = st.columns(3)
        for col, check in zip(cols, result.qc[start:start + 3]):
            status = check.status.value
            label = {"pass": "Pass", "warning": "Review", "fail": "Fail",
                     "not_assessed": "Pending"}.get(status, status)
            col.markdown(
                f'<div class="status-card {status}"><div class="status-label {status}">{label}</div>'
                f'<div class="status-title">{escape(check.name)}</div>'
                f'<div class="status-detail">{escape(str(check.detail))}</div></div>',
                unsafe_allow_html=True)


def _evidence_cards(result):
    st.caption("Evidence type is not a score. It tells you where a finding comes from and whether it applies to one selected patient or the entire dataset.")
    definitions = [
        ("Sample observation", "Measured or counted in the selected patient's cells.", "scope-sample"),
        ("Patient model perturbation", "A patient-restricted simulation; useful for forming a validation hypothesis.", "scope-patient"),
        ("Dataset-wide observation", "Measured or counted across every cell in the uploaded single-cell dataset.", "scope-cohort"),
        ("Dataset-wide model perturbation", "A modelled change across every cell in the uploaded dataset; it is not a patient-specific response prediction.", "scope-patient"),
        ("Cohort association", "Calculated across the study cohort; biological context, not an individual prediction.", "scope-cohort"),
        ("Computational hypothesis", "A target, pathway, mechanism, or drug lead requiring independent confirmation.", "scope-hypothesis"),
    ]
    with st.expander("How to read the evidence types"):
        for title, detail, css in definitions:
            st.markdown(f'<div class="scope-card {css}"><div class="scope-title">{title}</div>'
                        f'<div class="muted">{detail}</div></div>', unsafe_allow_html=True)
    if not result.evidence:
        st.info("No findings have been calculated yet. Ask SIDISH a question in the chat to build evidence as you go.")
        return
    for item in result.evidence.values():
        css = ("scope-sample" if item.scope.value.startswith("sample") else
               "scope-patient" if (item.scope.value.startswith("patient") or
                                    item.scope.value.startswith("dataset-wide model")) else
               "scope-cohort" if (item.scope.value.startswith("cohort") or
                                   item.scope.value.startswith("dataset-wide")) else "scope-hypothesis")
        st.markdown(f'<div class="scope-card {css}"><div class="scope-title">{escape(item.title)}</div>'
                    f'<div class="muted">{escape(item.scope.value)} · quality: {escape(item.quality.value)}</div></div>',
                    unsafe_allow_html=True)


def _report_downloads(result):
    reports = [a for a in result.artifacts if a.kind in {"report_html", "report_pdf"}]
    if not reports:
        return
    st.success("A draft report is available from this conversation.")
    cols = st.columns(min(2, len(reports)))
    for col, artifact in zip(cols, reports[-2:]):
        path = Path(artifact.path)
        if path.is_file():
            col.download_button(
                f"Download {path.suffix[1:].upper()}", path.read_bytes(), file_name=path.name,
                mime="application/pdf" if path.suffix == ".pdf" else "text/html",
                key=f"download-{active_case}-{path.suffix}")
    with st.expander("Scientific and clinical review sign-off"):
        with st.form(f"signoff-{active_case}"):
            scientific = st.text_input("Scientific reviewer", value=result.signoff.scientific_reviewer or "")
            clinical = st.text_input("Clinical reviewer", value=result.signoff.clinical_reviewer or "")
            comments = st.text_area("Review comments", value=result.signoff.comments or "")
            reviewed = st.form_submit_button("Record review sign-off")
        if reviewed:
            if not scientific or not clinical:
                st.error("Both reviewers are required.")
            else:
                result.signoff.scientific_reviewer = scientific
                result.signoff.clinical_reviewer = clinical
                result.signoff.comments = comments
                result.signoff.reviewed_at = utc_now()
                result.signoff.status = "reviewed"
                result.record("review_signed", actor="user")
                store.save(result)
                st.success("Review recorded in the case audit trail.")


_auth_gate()

with st.sidebar:
    st.markdown("## 🧬 SIDISH")
    st.caption("Case workspace")
    cases = store.list_cases()
    labels = {f"{c['case_id']} · {c['status'].replace('_', ' ')}": c["case_id"] for c in cases}
    requested_case = st.session_state.pop("select_case_next", None)
    if requested_case:
        requested_label = next((label for label, cid in labels.items() if cid == requested_case), None)
        if requested_label:
            st.session_state["case_selector"] = requested_label
    selected_label = st.selectbox("Active case", ["Create a new case"] + list(labels),
                                  key="case_selector")
    active_case = labels.get(selected_label)

    with st.expander("Create a case", expanded=not cases):
        with st.form("create_case"):
            case_id = st.text_input("Pseudonymous case ID")
            disease_label = st.text_input("Disease label")
            cancer_choice = st.selectbox("Cancer reference", ["breast", "lung", "pancreatic", "other"])
            other_cancer = st.text_input("Other disease name") if cancer_choice == "other" else ""
            subtype = st.text_input("Subtype (optional)")
            sc_file = st.file_uploader("Preprocessed single-cell data (.h5ad)", type=["h5ad"])
            bulk_file = (st.file_uploader("Matched bulk + survival (.csv)", type=["csv"])
                         if cancer_choice == "other" else None)
            submitted = st.form_submit_button("Upload and inspect", type="primary")
        if submitted:
            if not case_id or not disease_label or sc_file is None:
                st.error("Case ID, disease label, and single-cell file are required.")
            elif cancer_choice == "other" and bulk_file is None:
                st.error("Other diseases require a matched bulk-survival CSV.")
            else:
                try:
                    cancer = normalize_cancer_type(other_cancer if cancer_choice == "other" else cancer_choice)
                    result = store.create(CaseMetadata(
                        case_id=case_id, cancer_type=cancer, disease_label=disease_label,
                        cancer_subtype=subtype or None, research_use_only=True))
                    sc_path = store.save_upload(result.metadata.case_id, sc_file.name,
                                                sc_file.getvalue(), "single_cell")
                    bulk_path = (store.save_upload(result.metadata.case_id, bulk_file.name,
                                                   bulk_file.getvalue(), "bulk_survival")
                                 if bulk_file else None)
                    result, route = prepare_uploaded_case(store.get(result.metadata.case_id), str(sc_path),
                                                          str(bulk_path) if bulk_path else None)
                    store.save(result)
                    st.session_state["select_case_next"] = result.metadata.case_id
                    st.success(route["message"])
                    _rerun()
                except FileExistsError:
                    st.error("That case ID already exists.")
                except Exception as exc:
                    st.error(f"Case creation failed: {exc}")

if not active_case:
    st.markdown('<div class="sidish-hero"><h2>Ask better questions of single-cell data</h2>'
                '<p>Create a pseudonymous case, choose one patient or the entire dataset, then analyse it conversationally.</p></div>',
                unsafe_allow_html=True)
    st.info("Breast, lung, and pancreatic cases use their locked manuscript bulk-survival reference. Other diseases require a matched bulk-survival CSV.")
    st.stop()

result = store.get(active_case)
patient_label = ("Entire single-cell dataset" if result.metadata.analysis_scope == "cohort"
                 else result.metadata.patient_id or "Patient not selected")
st.markdown(f'<div class="sidish-hero"><h2>{escape(result.metadata.case_id)}</h2>'
            f'<p>{escape(result.metadata.disease_label)} · {escape(patient_label)} · '
            f'{escape(result.status.replace("_", " "))}</p></div>', unsafe_allow_html=True)
st.caption("Use pseudonymous identifiers only. Do not enter names, medical-record numbers, dates of birth, or direct identifiers.")

overview_tab, chat_tab = st.tabs(["Case & data", "Chat with SIDISH"])

with overview_tab:
    if result.status == "patient_selection_required":
        st.subheader("Choose what to analyse")
        st.write("Choose the `AnnData.obs` column that identifies patients or samples. Then analyse either one identifier or every cell across the uploaded dataset.")
        schema = _schema(active_case)
        columns = list(schema.get("columns", {}))
        if columns:
            preferred = next((c for c in ("patient", "patient_id", "sample", "Patient") if c in columns), columns[0])
            patient_column = st.selectbox("Patient/sample column", columns,
                                          index=columns.index(preferred), key=f"patient-column-{active_case}")
            patients = schema["columns"][patient_column]
            scope_choice = st.radio(
                "Analysis scope", ["One patient/sample", "Entire single-cell dataset"],
                horizontal=True, key=f"analysis-scope-{active_case}")
            analysis_scope = "patient" if scope_choice == "One patient/sample" else "cohort"
            selected_patient = (st.selectbox("Patient/sample to analyse", patients,
                                             key=f"patient-choice-{active_case}")
                                if analysis_scope == "patient" else None)
            training_iterations = st.number_input(
                "SIDISH training iterations", min_value=1, max_value=100,
                value=int(result.metadata.training_iterations), step=1,
                help="The number of iterative SIDISH refinement cycles. Fewer iterations finish sooner; the selected value is saved with the case.",
                key=f"training-iterations-{active_case}")
            st.caption(f"{len(patients):,} distinct identifiers found in this column.")
            if analysis_scope == "cohort":
                st.info(f"SIDISH will analyse all cells across all {len(patients):,} identifiers. Perturbation results will be dataset-wide, not patient-specific.")
            if st.button("Use this analysis scope", type="primary"):
                try:
                    sc_path = store.case_dir(active_case) / "inputs" / "single_cell.h5ad"
                    bulk_path = store.case_dir(active_case) / "inputs" / "bulk_survival.csv"
                    result, _ = configure_patient_selection(
                        result, sc_path, patient_column, selected_patient,
                        str(bulk_path) if bulk_path.exists() else None,
                        device=os.environ.get("SIDISH_TRAINING_DEVICE", "cuda:1"),
                        analysis_scope=analysis_scope,
                        training_iterations=int(training_iterations))
                    store.save(result)
                    st.success("Analysis scope selected. Open Chat with SIDISH and ask your first question.")
                    _rerun()
                except Exception as exc:
                    st.error(f"Could not save the analysis scope: {exc}")
    elif result.metadata.patient_column:
        with st.expander("Change analysis scope or patient"):
            schema = _schema(active_case)
            patients = schema.get("columns", {}).get(result.metadata.patient_column, [])
            if patients:
                new_scope_choice = st.radio(
                    "Analysis scope", ["One patient/sample", "Entire single-cell dataset"],
                    index=1 if result.metadata.analysis_scope == "cohort" else 0,
                    horizontal=True, key=f"change-scope-{active_case}")
                new_scope = "patient" if new_scope_choice == "One patient/sample" else "cohort"
                current_index = (patients.index(result.metadata.patient_id)
                                 if result.metadata.patient_id in patients else 0)
                new_patient = (st.selectbox(
                    f"Patient/sample from `{result.metadata.patient_column}`", patients,
                    index=current_index, key=f"change-patient-{active_case}")
                               if new_scope == "patient" else None)
                unchanged = (new_scope == result.metadata.analysis_scope and
                             (new_scope == "cohort" or new_patient == result.metadata.patient_id))
                st.caption("Changing the scope reuses the trained model. Scope-dependent findings and previous reports are cleared.")
                if new_scope == "cohort":
                    st.info(f"The next analyses will include every cell across all {len(patients):,} identifiers.")
                if st.button("Apply analysis scope", disabled=unchanged):
                    try:
                        sc_path = store.case_dir(active_case) / "inputs" / "single_cell.h5ad"
                        result = change_selected_patient(
                            result, sc_path, new_patient, analysis_scope=new_scope)
                        store.save(result)
                        label = new_patient if new_scope == "patient" else "the entire dataset"
                        st.success(f"Analysis scope changed to {label}.")
                        _rerun()
                    except Exception as exc:
                        st.error(f"Could not change the analysis scope: {exc}")
        if result.status == "training_required":
            with st.expander("Training settings", expanded=True):
                st.write("Choose the number of iterative SIDISH refinement cycles before starting the training job.")
                revised_iterations = st.number_input(
                    "Training iterations", min_value=1, max_value=100,
                    value=int(result.metadata.training_iterations), step=1,
                    key=f"revise-training-iterations-{active_case}")
                iteration_unchanged = int(revised_iterations) == result.metadata.training_iterations
                if st.button("Save training iterations", disabled=iteration_unchanged):
                    try:
                        sc_path = store.case_dir(active_case) / "inputs" / "single_cell.h5ad"
                        result = change_training_iterations(
                            result, sc_path, int(revised_iterations))
                        store.save(result)
                        st.success(f"SIDISH will run for {int(revised_iterations)} iterations.")
                        _rerun()
                    except Exception as exc:
                        st.error(f"Could not update the training iterations: {exc}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Data status", result.status.replace("_", " ").title())
    c2.metric("Analysis scope", patient_label)
    c3.metric("Training iterations", result.metadata.training_iterations)
    c4.metric("Cancer reference", result.metadata.cancer_type.title())
    c5.metric("Findings available", len(result.evidence))
    if result.errors:
        st.error("\n".join(result.errors))
    if result.warnings:
        st.warning("\n".join(result.warnings))
    st.subheader("Data quality at a glance")
    st.caption("Green checks passed. Amber checks deserve review but may still be usable. Red checks must be resolved before SIDISH analysis.")
    _qc_cards(result)
    st.subheader("What the evidence means")
    _evidence_cards(result)

with chat_tab:
    st.subheader(f"Chat with SIDISH about {patient_label}")
    st.caption("Ask one question at a time. SIDISH runs only the required calculation and reuses cached results during this session.")
    st.info(DISCLAIMER)
    if not store.read_chat(active_case):
        with st.chat_message("assistant"):
            if result.status == "patient_selection_required":
                st.markdown("Select one patient or the entire dataset in **Case & data**, then return here.")
            elif result.status == "training_required":
                st.markdown(f"Your data passed initial checks. The current scope is **{patient_label}**, and training is configured for **{result.metadata.training_iterations} iterations**. Ask a question; if it needs a trained model, I will explain why and ask before starting training.")
            else:
                st.markdown("Ask me about high-risk cells, cell types, markers, pathways, target or drug perturbations, or say **generate the final report**.")
    for event in store.read_chat(active_case):
        if event["role"] in {"user", "assistant"}:
            with st.chat_message(event["role"]):
                st.markdown(event["content"])
                for figure in event.get("figures", []):
                    try:
                        safe_figure = store.assert_case_path(active_case, figure)
                        if safe_figure.is_file():
                            st.image(str(safe_figure), width="stretch")
                    except (PermissionError, OSError, ValueError):
                        st.warning("A saved chat figure is unavailable or outside this case workspace.")
                if event.get("table"):
                    st.dataframe(event["table"], width="stretch", hide_index=True)

    prompt = st.chat_input("Ask SIDISH about the selected analysis scope…",
                           disabled=result.status == "patient_selection_required")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            status_box = st.status("SIDISH is preparing the requested analysis…", expanded=True)
            def progress(message: str):
                status_box.write(message)
            try:
                agent = SIDISHAgentSession(active_case, store=store, progress=progress)
                reply = agent.send(prompt)
                status_box.update(label="SIDISH finished this step", state="complete", expanded=False)
                st.markdown(reply["content"])
                for figure in reply.get("figures", []):
                    safe_figure = store.assert_case_path(active_case, figure)
                    if safe_figure.is_file():
                        st.image(str(safe_figure), width="stretch")
                if reply.get("table"):
                    st.dataframe(reply["table"], width="stretch", hide_index=True)
            except Exception as exc:
                status_box.update(label="SIDISH could not complete this step", state="error")
                st.error(str(exc))
        result = store.get(active_case)
    if result.status in {"training_queued", "training_running"}:
        if st.button("Refresh training status"):
            _rerun()
    _report_downloads(result)
