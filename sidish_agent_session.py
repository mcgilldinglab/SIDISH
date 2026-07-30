"""Case-bound conversational SIDISH agent with governed, on-demand tools."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import json
import os
import re
import subprocess
import sys
import time

from sidish_case_store import CaseStore
from sidish_case_analysis import change_training_iterations
from sidish_on_demand import run_capability
from sidish_policy import audit_text


SYSTEM = """You are SIDISH, a conversational research-use clinical decision-support assistant.
You are bound to one pseudonymous case and its selected analysis scope: either one patient/sample
or the entire single-cell dataset. Use a tool for every case fact and state that scope clearly.
Preserve evidence scope verbatim. Never turn cohort evidence into a patient claim, diagnose,
predict treatment response or absolute prognosis, or select treatment. Run only the capability
needed for the question. If a user names a gene, pathway, target, drug, or method, pass that exact
entity to the tool; never silently substitute a default sweep. A drug tool maps target/signature
hypotheses and does not simulate pharmacologic response. Perturbation outputs are hypotheses for
validation. Generate a report only when explicitly asked. Explain results plainly and concisely.
If training is required, explain that analysis needs a trained case model and ask for confirmation;
never start training merely because an analysis question was asked."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None


def _default_model() -> LLMConfig:
    """Select a configured provider; local Ollama is the no-key final option."""
    provider = os.environ.get("SIDISH_LLM_PROVIDER", "auto").strip().lower()
    override = os.environ.get("SIDISH_LLM_MODEL")
    if provider == "auto":
        if os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API_KEY") != "ollama":
            provider = "openai"
        elif os.environ.get("OPENAI_BASE_URL"):
            provider = "compatible"
        else:
            provider = "ollama"
    if provider == "gemini":
        return LLMConfig("gemini", override or os.environ.get("SIDISH_GEMINI_MODEL", "gemini-2.0-flash"),
                         os.environ.get("GEMINI_API_KEY", ""),
                         os.environ.get("GEMINI_BASE_URL",
                                        "https://generativelanguage.googleapis.com/v1beta/openai/"))
    if provider == "ollama":
        return LLMConfig("ollama", override or os.environ.get("SIDISH_OLLAMA_MODEL", "qwen3:8b"),
                         "ollama", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    if provider == "compatible":
        return LLMConfig("compatible", override or "qwen3:8b",
                         os.environ.get("OPENAI_API_KEY", "local"), os.environ.get("OPENAI_BASE_URL"))
    return LLMConfig("openai", override or os.environ.get("SIDISH_OPENAI_MODEL", "gpt-4o-mini"),
                     os.environ.get("OPENAI_API_KEY", ""), os.environ.get("OPENAI_BASE_URL"))


def _schema(name: str, description: str, properties: dict | None = None,
            required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {},
                           "required": required or [], "additionalProperties": False}}}


def _intent(text: str) -> str | None:
    low = text.lower()
    if re.search(r"\b(generate|create|make|render)\b.{0,20}\b(final )?report\b", low):
        return "report"
    if "pathway" in low and re.search(r"\b(perturb|knock|simulate|ablat|effect)\w*\b", low):
        return "pathway"
    if re.search(r"\b(drug|compound|perturbagen|reverse[- ]signature)\b", low):
        return "drug"
    if re.search(r"\b(perturb|knockout|knock out|ablat|simulate)\w*\b", low) or \
            re.search(r"\btarget(s|ing)?\b", low):
        return "target"
    if re.search(r"\b(marker|survival|molecular program|pathway context|enrichment)\b", low):
        return "markers"
    if re.search(r"\b(high[- ]risk|hieric|risk cell|cell[- ]type)\b", low):
        return "highrisk"
    return None


def _requested_training_iterations(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\s+iterations?\b", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_params(text: str, capability: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    gene_stop = {"SIDISH", "UMAP", "RNA", "DNA", "VAE", "PPI", "HIGH", "RISK", "CELL"}
    genes = [g for g in re.findall(r"\b[A-Z][A-Z0-9-]{1,14}\b", text) if g not in gene_stop]
    if capability in {"target", "report"} and not genes:
        named_genes = re.findall(
            r"\b(?:gene|target|perturb|perturbation|focus(?:ing)?(?:\s+on)?|knockout|knock\s+out|ablate)\s+(?:gene\s+)?([A-Za-z][A-Za-z0-9-]{1,14})\b",
            text, flags=re.I)
        ignored = {"the", "this", "that", "gene", "target", "pathway", "patient"}
        genes = [gene for gene in named_genes if gene.lower() not in ignored]
    if capability == "target" and genes:
        params["genes"] = list(dict.fromkeys(genes))
    elif capability == "report":
        if genes:
            params["focus_targets"] = list(dict.fromkeys(genes))
        if re.search(r"\b(include|show)\s+(all|other)\s+perturbations?\b", text, flags=re.I):
            params["include_other_perturbations"] = True
    elif capability == "pathway":
        match = re.search(r"(?:perturb|simulate|ablate|knock(?:\s+out)?)\s+(?:the\s+)?(.+?)\s+pathway\b",
                          text, flags=re.I)
        if not match:
            match = re.search(r"\bpathway\s+(?:called\s+)?[\"']?(.+?)[\"']?(?:[?.]|$)", text,
                              flags=re.I)
        if match:
            params["pathway"] = match.group(1).strip(" \"'")
        if genes:
            params["genes"] = list(dict.fromkeys(genes))
    elif capability == "drug":
        if "reverse signature" in text.lower() or "reverse-signature" in text.lower():
            params["method"] = "reverse_signature"
        if genes:
            params["targets"] = list(dict.fromkeys(genes))
        quoted = re.search(r"[\"']([^\"']{2,60})[\"']", text)
        named = re.search(r"\b(?:drug|compound|perturbagen)\s+(?:called\s+|named\s+)?([A-Za-z0-9][A-Za-z0-9 ()+./-]{1,50})",
                          text, flags=re.I)
        if quoted:
            params["drug"] = quoted.group(1).strip()
        elif named and not genes:
            params["drug"] = re.split(r"\b(?:for|on|in|with|please)\b|[?.]", named.group(1),
                                      maxsplit=1, flags=re.I)[0].strip()
    return params


def _strip_hidden_reasoning(text: str) -> str:
    """Remove local-model reasoning tags before text reaches the clinician-facing chat."""
    cleaned = re.sub(r"<(think|analysis)>.*?</\1>", "", str(text or ""),
                     flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<(think|analysis)>.*$", "", cleaned,
                     flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _is_status_question(text: str) -> bool:
    low = text.lower()
    return bool(
        re.search(r"\b(?:what(?:'s| is) the )?(?:case|training|job|task) (?:status|progress)\b", low)
        or re.search(r"\b(?:is|has) (?:the )?(?:training|job|task).*(?:running|finished|done)\b", low)
        or re.search(r"\b(?:still running|done yet|how long.*(?:training|job|task))\b", low)
    )


def _is_confirmation(text: str) -> bool:
    return bool(re.fullmatch(r"\s*(yes|yes please|confirm|confirmed|proceed|go ahead|start|start training|train it)[.!]?\s*",
                             text.lower()))


class CaseTools:
    def __init__(self, store: CaseStore, case_id: str, progress: Callable[[str], None]):
        self.store, self.case_id, self.progress = store, case_id, progress
        gene_array = {"type": "array", "items": {"type": "string"},
                      "description": "Exact gene symbols requested by the user."}
        self.spec = [
            _schema("case_status", "Return selected case/patient, status, QC and available evidence."),
            _schema("high_risk_cells", "Run or return high-risk burden and composition for the selected patient or entire dataset."),
            _schema("marker_and_pathway_analysis", "Run or return cohort markers, pathways and survival context."),
            _schema("target_perturbation", "Run target-network perturbation in the selected patient or entire dataset. Pass named genes exactly.",
                    {"genes": gene_array, "scope": {"type": "string", "enum": ["markers", "all"]}}),
            _schema("pathway_perturbation", "Run a named pathway or custom gene-set perturbation in the selected analysis scope.",
                    {"pathway": {"type": "string"}, "genes": gene_array}),
            _schema("drug_perturbation", "Map a named drug, targets, or high-risk reverse signature to hypotheses; this does not simulate drug response.",
                    {"drug": {"type": "string"}, "targets": gene_array,
                     "method": {"type": "string", "enum": ["targets", "reverse_signature"]}}),
            _schema("generate_report", "Complete evidence and generate the draft HTML/PDF report. Optionally focus the perturbation section on named targets while retaining case, QC, burden, marker, pathway-context, provenance, limitations, and review sections.",
                    {"focus_targets": gene_array,
                     "include_other_perturbations": {"type": "boolean"},
                     "filter": {"type": "string", "description": "Backward-compatible plain-text target focus, such as ERBB2."}}),
        ]
        self.registry: dict[str, Callable[..., Any]] = {
            "case_status": self.case_status, "high_risk_cells": self.high_risk_cells,
            "marker_and_pathway_analysis": self.marker_and_pathway_analysis,
            "target_perturbation": self.target_perturbation,
            "pathway_perturbation": self.pathway_perturbation,
            "drug_perturbation": self.drug_perturbation, "generate_report": self.generate_report,
        }

    def _result(self):
        return self.store.get(self.case_id)

    def _run(self, capability: str, **params):
        return run_capability(self.case_id, capability, self.store, self.progress, params=params)

    def case_status(self):
        result = self._result()
        return {"case": asdict(result.metadata), "status": result.status,
                "qc": [asdict(q) for q in result.qc],
                "evidence": [{"title": e.title, "scope": e.scope.value,
                              "quality": e.quality.value} for e in result.evidence.values()],
                "warnings": result.warnings, "errors": result.errors}

    def high_risk_cells(self): return self._run("highrisk")
    def marker_and_pathway_analysis(self): return self._run("markers")
    def target_perturbation(self, genes=None, scope="markers"): return self._run("target", genes=genes, scope=scope)
    def pathway_perturbation(self, pathway=None, genes=None): return self._run("pathway", pathway=pathway, genes=genes)
    def drug_perturbation(self, drug=None, targets=None, method="targets"): return self._run("drug", drug=drug, targets=targets, method=method)
    def generate_report(self, focus_targets=None, include_other_perturbations=False,
                        filter=None):
        if filter and not focus_targets:
            candidates = re.findall(r"\b[A-Za-z][A-Za-z0-9-]{1,14}\b", str(filter))
            ignored = {"focus", "only", "target", "gene", "perturbation", "report", "on"}
            focus_targets = [item for item in candidates if item.lower() not in ignored]
        return self._run(
            "report", focus_targets=focus_targets,
            include_other_perturbations=include_other_perturbations)


class SIDISHAgentSession:
    def __init__(self, case_id: str, store: CaseStore | None = None, client=None,
                 model: str | None = None, progress: Callable[[str], None] | None = None):
        self.store = store or CaseStore()
        self.case_id = case_id
        self.progress = progress or (lambda _message: None)
        self.tools = CaseTools(self.store, case_id, self.progress)
        self.config = _default_model()
        if model:
            self.config = LLMConfig(self.config.provider, model, self.config.api_key, self.config.base_url)
        self.model = self.config.model
        self.client = client
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM}]
        for event in self.store.read_chat(case_id):
            if event.get("role") in {"user", "assistant"}:
                self.messages.append({"role": event["role"], "content": event.get("content", "")})

    def _client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url,
                                 timeout=float(os.environ.get("SIDISH_LLM_TIMEOUT", "30")),
                                 max_retries=0)
        return self.client

    def _completion(self, **kwargs):
        attempts = max(1, int(os.environ.get("SIDISH_LLM_RETRIES", "3")))
        for attempt in range(attempts):
            try:
                return self._client().chat.completions.create(**kwargs)
            except Exception as exc:
                retryable = "429" in str(exc) or "rate" in str(exc).lower() or \
                    "temporar" in str(exc).lower() or "timeout" in str(exc).lower()
                if not retryable or attempt + 1 == attempts:
                    raise
                time.sleep(min(2 ** attempt, 4))

    def _pending_training_request(self, result) -> dict | None:
        pending = None
        for event in result.audit:
            if event.get("action") == "training_consent_requested":
                pending = event.get("detail", {})
            elif event.get("action") == "training_job_submitted_from_chat":
                pending = None
        return pending

    def _request_training_consent(self, result, capability: str, params: dict) -> str:
        result.record("training_consent_requested", actor="user",
                      detail={"capability": capability, "parameters": params})
        self.store.save(result)
        entity = params.get("genes") or params.get("targets") or params.get("pathway") or params.get("drug")
        requested = f" for **{entity if isinstance(entity, str) else ', '.join(entity or [])}**" if entity else ""
        return (f"That {capability} analysis{requested} needs a trained SIDISH model for this case. "
                f"Training is configured for **{result.metadata.training_iterations} iterations** and fits "
                "the single-cell VAE and survival model, so it can take substantial time. "
                "Would you like me to start training now? Reply **yes** to confirm. Nothing has been queued yet.")

    def _queue_training(self, result) -> str:
        cfg = self.store.case_dir(self.case_id) / "jobs" / "training.yaml"
        if not cfg.exists():
            return "Training cannot start until the analysis scope and training settings are saved in Case & data."
        script = Path(__file__).resolve().parent / "skills" / "sidish-report-orchestrator" / "scripts" / "sidish_job.py"
        completed = subprocess.run(
            [sys.executable, str(script), "submit", "--config", str(cfg),
             "--job-name", f"{self.case_id}-training"], capture_output=True, text=True,
            check=True, shell=False)
        job = json.loads(completed.stdout)
        result.status = "training_queued"
        result.record("training_job_submitted_from_chat", actor="user", detail=job)
        self.store.save(result)
        return (f"Training is now confirmed for **{result.metadata.training_iterations} iterations**, and task "
                f"`{job['job_id']}` has started in the background. "
                "Ask **what is the training status?** at any time. Once it completes, repeat the "
                "analysis request and SIDISH will run only that capability in the selected scope.")

    def _training_status(self, result) -> str:
        event = next((e for e in reversed(result.audit)
                      if e["action"] == "training_job_submitted_from_chat"), None)
        if not event:
            return f"The case is currently `{result.status}`; no training task is recorded."
        job_id = event["detail"].get("job_id")
        script = Path(__file__).resolve().parent / "skills" / "sidish-report-orchestrator" / "scripts" / "sidish_job.py"
        completed = subprocess.run([sys.executable, str(script), "status", job_id],
                                   capture_output=True, text=True, shell=False)
        if completed.returncode:
            return f"Training task `{job_id}` is recorded, but its current status could not be read."
        status = json.loads(completed.stdout).get("status", result.status)
        if status == "running" and result.status == "training_queued":
            result.status = "training_running"
            self.store.save(result)
        elif status in {"failed", "cancelled"}:
            result.status = f"training_{status}"
            result.errors.append(f"Training task {job_id} ended with status {status}; inspect its log before retrying.")
            self.store.save(result)
        current = self.store.get(self.case_id)
        return (f"SIDISH training task `{job_id}` is **{status}**. It was configured for "
                f"**{current.metadata.training_iterations} iterations**. Current case status: `{current.status}`.")

    def _raw_cell_count(self, result) -> str:
        path = self.store.case_dir(self.case_id) / "inputs" / "single_cell.h5ad"
        if not path.is_file() or not result.metadata.patient_column:
            return "I cannot count the uploaded cells until a patient/sample column is selected."
        import anndata as ad
        data = ad.read_h5ad(path, backed="r")
        try:
            if result.metadata.analysis_scope == "cohort":
                count = int(data.n_obs)
            else:
                column = result.metadata.patient_column
                count = int((data.obs[column].astype(str) == str(result.metadata.patient_id)).sum())
        finally:
            if getattr(data, "file", None):
                data.file.close()
        if result.metadata.analysis_scope == "cohort":
            return (f"The uploaded single-cell file contains **{count:,} cells across the entire dataset**. "
                    "Evidence scope: **dataset-wide raw data count**. This does not require SIDISH "
                    "training and differs from the model-labelled high-risk-cell count.")
        return (f"The uploaded single-cell file contains **{count:,} cells** for **{result.metadata.patient_id}** "
                f"in column `{result.metadata.patient_column}`. Evidence scope: **sample-specific raw data count**. "
                "This does not require SIDISH training and differs from the model-labelled high-risk-cell count.")

    @staticmethod
    def _format(capability: str, value: dict, subject: str,
                analysis_scope: str = "patient") -> str:
        scope_text = ("dataset-wide model perturbation" if analysis_scope == "cohort"
                      else "patient-specific model perturbation")
        if capability == "highrisk":
            pct = 100 * float(value.get("high_risk_fraction", 0))
            comp = value.get("celltype_composition", {})
            top = max(comp, key=comp.get) if comp else "not available"
            observation_scope = ("dataset-wide observation" if analysis_scope == "cohort"
                                 else "sample-specific observation")
            composition_text = (f"The largest high-risk compartment is **{top}**. "
                                if value.get("celltype_available", bool(comp)) and comp else
                                "No cell-type annotation was supplied, so composition is not reported. ")
            return (f"For **{subject}**, SIDISH labelled **{value.get('n_high_risk', 0):,} of "
                    f"{value.get('n_cells', 0):,} cells ({pct:.1f}%)** as high-risk. {composition_text}"
                    f"Evidence scope: **{observation_scope}**. "
                    "This fraction is not an individual prognosis.")
        if capability == "target":
            rows = value.get("single", [])[:3]
            text = ", ".join(f"{r.get('target')} ({r.get('reduction')}% net high-risk reduction; "
                             f"{r.get('affected_cells', 0)} label flips)" for r in rows)
            prefix = ("Dataset-wide" if analysis_scope == "cohort" else "Patient-restricted")
            return (f"{prefix} target-network hypothesis: **{text or 'none available'}**. "
                    f"The inline UMAP shows modeled label transitions. Evidence scope: **{scope_text}**; "
                    "this is an experimental priority, not a response prediction.")
        if capability == "pathway":
            rows = value.get("pathways", [])[:3]
            text = ", ".join(f"{r.get('pathway')} ({r.get('reduction')}% net reduction; "
                             f"{r.get('affected_cells', 0)} label flips)" for r in rows)
            return (f"Pathway perturbation hypothesis: **{text or 'none available'}**. Evidence scope: "
                    f"**{scope_text}**; validate the pathway independently.")
        if capability == "drug":
            rows = value.get("by_target", [])[:6]
            names = [c.get("drug") for row in rows for c in row.get("compounds", [])]
            requested = f" for **{value['requested_drug']}**" if value.get("requested_drug") else ""
            return (f"Drug/target mapping{requested}: **{', '.join(names[:8]) or 'no mapped compounds'}**. "
                    "Evidence scope: **computational hypothesis**. SIDISH mapped target/signature evidence; "
                    "it did not simulate a pharmacologic response and this is not a drug recommendation.")
        if capability == "markers":
            genes = value.get("markers", {}).get("genes", [])[:8]
            return (f"The cohort high-risk marker program includes **{', '.join(genes)}**. Evidence scope: "
                    "**cohort-level association**, providing context rather than a patient-only measurement.")
        if capability == "report":
            focus = value.get("report_focus", "Complete available SIDISH evidence")
            return (f"The draft clinician decision-support report is ready below the conversation. Report focus: "
                    f"**{focus}**. It reflects "
                    "the validated evidence bundle and requires scientific and clinical review.")
        return json.dumps(value, default=str)

    def _deterministic(self, user_text: str) -> tuple[str, dict[str, Any]]:
        result = self.store.get(self.case_id)
        low = user_text.lower()
        if _is_status_question(user_text):
            subject = ("the entire dataset" if result.metadata.analysis_scope == "cohort"
                       else result.metadata.patient_id or "not selected")
            answer = self._training_status(result) if "training" in result.status else (
                f"Case **{self.case_id}** is `{result.status}` with analysis scope **{subject}**.")
            return answer, {}
        if result.status == "patient_selection_required":
            return ("Choose the patient/sample column and whether to analyse one patient/sample or the "
                    "entire dataset in **Case & data** before I run SIDISH."), {}
        raw_count = bool(re.search(r"\b(how many|number of|count)\b.{0,20}\bcells?\b|\bcells?\b.{0,20}\b(how many|count)\b", low)) \
            and not re.search(r"high[- ]risk|hieric", low)
        if raw_count:
            return self._raw_cell_count(result), {}
        capability = _intent(user_text)
        params = _extract_params(user_text, capability)
        if result.status == "training_required":
            iteration_note = ""
            requested_iterations = _requested_training_iterations(user_text)
            if requested_iterations is not None:
                single_cell = self.store.case_dir(self.case_id) / "inputs" / "single_cell.h5ad"
                result = change_training_iterations(result, single_cell, requested_iterations)
                self.store.save(result)
                iteration_note = (f"SIDISH training is now configured for **{requested_iterations} iterations**. ")
            pending = self._pending_training_request(result)
            if pending and _is_confirmation(user_text):
                return iteration_note + self._queue_training(result), {}
            if re.search(r"\b(start|begin|run)\s+(the\s+)?training\b", low):
                return iteration_note + self._queue_training(result), {}
            if capability:
                return iteration_note + self._request_training_consent(result, capability, params), {}
            if iteration_note:
                return (iteration_note + "You can now ask me to start training or request an analysis; "
                        "I will still ask for confirmation before queuing the job."), {}
            return ("I can discuss SIDISH and inspect non-model case metadata now. Model risk labels and "
                    "perturbations need training, which I will start only after you explicitly confirm.", {})
        if result.status in {"training_queued", "training_running"}:
            return self._training_status(result), {}
        if result.status in {"training_failed", "training_cancelled"}:
            return ("The SIDISH training task did not complete. Review the case error and task log before retrying; "
                    "I will not substitute results.", {})
        if capability is None:
            return ("I can discuss this case conversationally and run only what you ask: high-risk cells, markers, "
                    "a named gene or pathway perturbation, drug/target mapping, or the final report. What should we examine?", {})
        value = run_capability(self.case_id, capability, self.store, self.progress, params=params)
        artifacts = value.pop("_chat_artifacts", {})
        subject = ("the entire single-cell dataset" if result.metadata.analysis_scope == "cohort"
                   else result.metadata.patient_id or "selected patient")
        return self._format(capability, value, subject, result.metadata.analysis_scope), artifacts

    def _llm_answer(self, user_text: str, max_steps: int) -> tuple[str, dict[str, Any]]:
        artifacts: dict[str, Any] = {"figures": [], "table": None}
        for _ in range(max_steps):
            response = self._completion(model=self.model, messages=self.messages,
                                        tools=self.tools.spec, tool_choice="auto", temperature=0)
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return message.content or "", artifacts
            for call in message.tool_calls:
                try:
                    if call.function.name == "generate_report" and _intent(user_text) != "report":
                        raise PermissionError("report generation requires an explicit user request")
                    arguments = json.loads(call.function.arguments or "{}")
                    value = self.tools.registry[call.function.name](**arguments)
                    chat_artifacts = value.pop("_chat_artifacts", {}) if isinstance(value, dict) else {}
                    artifacts["figures"].extend(chat_artifacts.get("figures", []))
                    if chat_artifacts.get("table"):
                        artifacts["table"] = chat_artifacts["table"]
                except Exception as exc:
                    value = {"error": str(exc)}
                self.messages.append({"role": "tool", "tool_call_id": call.id,
                                      "name": call.function.name,
                                      "content": json.dumps(value, default=str)})
        return "I stopped after the maximum number of tool steps without a final answer.", artifacts

    def send(self, user_text: str, max_steps: int = 8) -> dict[str, Any]:
        self.store.append_chat(self.case_id, "user", user_text)
        self.messages.append({"role": "user", "content": user_text})
        result = self.store.get(self.case_id)
        use_gate = result.status in {"patient_selection_required", "training_required", "training_queued",
                                     "training_running", "training_failed", "training_cancelled"}
        metadata: dict[str, Any] = {"provider": self.config.provider, "model": self.model}
        if use_gate or _is_status_question(user_text):
            answer, artifacts = self._deterministic(user_text)
        else:
            try:
                answer, artifacts = self._llm_answer(user_text, max_steps)
            except Exception as exc:
                self.progress("The configured conversational model is unavailable; using SIDISH's safe local parser for this turn.")
                answer, artifacts = self._deterministic(user_text)
                metadata.update({"llm_fallback": True, "llm_error": type(exc).__name__})
        answer = _strip_hidden_reasoning(answer)
        findings = audit_text(answer)
        if findings:
            answer = ("I cannot release that wording because it crosses the SIDISH decision-support boundary. "
                      "I can describe the evidence, scope, and confirmation options instead.")
            artifacts = {}
        figures = list(dict.fromkeys(artifacts.get("figures", [])))
        table = artifacts.get("table")
        self.messages.append({"role": "assistant", "content": answer})
        metadata["policy_findings"] = len(findings)
        self.store.append_chat(self.case_id, "assistant", answer, metadata,
                               figures=figures, table=table)
        return {"role": "assistant", "content": answer, "figures": figures, "table": table,
                "metadata": metadata}
