"""Deterministic policy checks for SIDISH chat and report claims."""
from __future__ import annotations

from dataclasses import dataclass
import re


DISCLAIMER = (
    "SIDISH is research-use clinician decision support. Results are computational "
    "observations or hypotheses requiring orthogonal confirmation and clinician review; "
    "they do not diagnose disease, estimate absolute individual risk, select therapy, "
    "or replace guideline-based care."
)

PROHIBITED = {
    "treatment directive": [
        r"\b(?:start|give|administer|prescribe|initiate|switch to)\s+(?:the patient\s+)?"
        r"(?!(?:training|analysis|job|task|report)\b)[a-z0-9]",
        r"\b(?:i|we|sidish)\s+recommend(?:s)?\b",
        r"\b(?:i|we|sidish)\s+recommend(?:s)?\s+(?:treating|therapy|treatment|[a-z0-9-]+\s+for\s+the patient)",
        r"\bthe patient should (?:receive|take|stop|avoid)\b",
        r"\b(?:stop|avoid|discontinue)\s+(?:chemotherapy|radiation|immunotherapy|treatment|medication)",
        r"\b(?:best treatment|drug of choice|preferred therapy)\s+is\b",
        r"\buse\s+[a-z0-9-]+\s+to treat\b",
    ],
    "response prediction": [
        r"\bthe patient (?:will|is predicted to|should) respond\b",
        r"\bthe patient is (?:likely|unlikely) to respond\b",
        r"\bpredicted clinical response\b",
        r"\bguarantee(?:s|d)?\b",
    ],
    "diagnostic claim": [
        r"\bdiagnos(?:is|e|ed) (?:is )?confirmed\b",
        r"\bproves? (?:the )?diagnosis\b",
    ],
    "absolute prognosis": [
        r"\b(?:individual|patient) (?:survival|mortality|recurrence) (?:risk|probability)\b",
        r"\bhas a \d+(?:\.\d+)?% (?:chance|risk) of (?:death|recurrence|survival)\b",
        r"\b(?:this|the) patient has (?:a )?(?:poor|good|favourable|favorable) prognosis\b",
    ],
}


@dataclass
class PolicyFinding:
    category: str
    match: str


def audit_text(text: str) -> list[PolicyFinding]:
    findings: list[PolicyFinding] = []
    for category, patterns in PROHIBITED.items():
        for pattern in patterns:
            for hit in re.finditer(pattern, text or "", flags=re.IGNORECASE):
                findings.append(PolicyFinding(category, hit.group(0)))
    return findings


def enforce_text(text: str) -> str:
    findings = audit_text(text)
    if findings:
        summary = ", ".join(f"{f.category}: {f.match!r}" for f in findings[:8])
        raise ValueError(f"output requires clinical-safety review ({summary})")
    return text


def allowed_clinical_language() -> list[str]:
    return [
        "nominates a target or pathway for orthogonal validation",
        "suggests a pathology or molecular question for clinician consideration",
        "supports molecular tumour board discussion",
        "reports a model-defined cell fraction with its denominator and QC",
        "describes cohort association separately from the current sample",
    ]
