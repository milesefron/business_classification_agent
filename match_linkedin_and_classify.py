#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# ============================================================
# Models (Pydantic)
# NOTE: We intentionally treat URLs as plain strings because
# OpenAI Structured Outputs JSON Schema does not accept
# "format": "uri", and LinkedIn URLs returned may omit scheme.
# ============================================================

class AlternateCandidate(BaseModel):
    linkedin_url: Optional[str] = None
    display_name: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class ResolveResult(BaseModel):
    input_name: str
    normalized_name: Optional[str] = None

    linkedin_url: Optional[str] = None
    linkedin_display_name: Optional[str] = None
    linkedin_confidence: float = Field(ge=0.0, le=1.0)

    # 1–3 sentences describing what the org does, based on search snippets
    evidence_summary: str

    alternates: List[AlternateCandidate] = Field(default_factory=list)


class Classification(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class FinalResult(BaseModel):
    input_name: str
    normalized_name: Optional[str] = None

    linkedin_url: Optional[str] = None
    linkedin_display_name: Optional[str] = None
    linkedin_confidence: float = Field(ge=0.0, le=1.0)

    evidence_summary: str
    classification: Classification
    rationale: str

    alternates: List[AlternateCandidate] = Field(default_factory=list)


# ============================================================
# Vocab loading
# ============================================================

def load_vocab(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    labels = payload.get("labels")
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError('vocab JSON must be shaped like: {"labels": ["..."]}')
    if "Other" not in labels:
        labels.append("Other")
    return labels


# ============================================================
# Structured Outputs JSON Schemas
# IMPORTANT: Do NOT use JSON Schema "format" fields. OpenAI's
# Structured Outputs supports a restricted dialect and rejects
# "format": "uri". Keep URLs as plain strings.
# ============================================================

def resolve_schema() -> Dict[str, Any]:
    return {
        "name": "linkedin_company_resolver",
        "description": "Resolve a business name to the most likely LinkedIn company page and produce a short evidence summary.",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "input_name": {"type": "string"},
                "normalized_name": {"type": ["string", "null"]},

                "linkedin_url": {"type": ["string", "null"]},
                "linkedin_display_name": {"type": ["string", "null"]},
                "linkedin_confidence": {"type": "number", "minimum": 0, "maximum": 1},

                "evidence_summary": {"type": "string"},

                "alternates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "linkedin_url": {"type": ["string", "null"]},
                            "display_name": {"type": ["string", "null"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["linkedin_url", "display_name", "confidence"],
                    },
                },
            },
            "required": [
                "input_name",
                "normalized_name",
                "linkedin_url",
                "linkedin_display_name",
                "linkedin_confidence",
                "evidence_summary",
                "alternates",
            ],
        },
    }


def classify_schema(allowed_labels: List[str]) -> Dict[str, Any]:
    return {
        "name": "controlled_vocab_classifier",
        "description": "Choose exactly one label from the provided controlled vocabulary.",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "classification": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string", "enum": allowed_labels},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["label", "confidence"],
                },
                "rationale": {"type": "string"},
            },
            "required": ["classification", "rationale"],
        },
    }


# ============================================================
# Prompts
# ============================================================

def resolve_prompt(business_name: str) -> str:
    return f"""You are resolving a business/organization name to an official LinkedIn COMPANY profile.

Tasks:
1) Find the most likely official LinkedIn company page.
   - Prefer: https://www.linkedin.com/company/<slug>/
   - Avoid personal profiles, posts, job listings, and low-confidence aggregators.
2) Normalize the name (optional): fix casing, remove Inc/LLC/etc if helpful.
3) Provide a SHORT evidence_summary (1–3 sentences) describing what the org does.
   - Base this on snippets from LinkedIn / official website search results.
   - If you cannot find reliable evidence, say so explicitly in evidence_summary.

Output requirements:
- If uncertain, set linkedin_url null and linkedin_confidence low.
- Provide up to 3 alternates if there are multiple plausible matches.
- Output ONLY JSON. No markdown. No extra text.

Business name: {business_name}
"""


def classify_prompt(
    business_name: str,
    normalized_name: Optional[str],
    linkedin_url: Optional[str],
    evidence_summary: str,
    allowed_labels: List[str],
) -> str:
    labels_bullets = "\n".join(f"- {x}" for x in allowed_labels)
    return f"""You are classifying an organization into EXACTLY ONE label from a controlled vocabulary.

You must use ONLY the evidence provided below. Do NOT browse the web. Do NOT invent facts.
Output ONLY JSON. No markdown. No extra text.

Guidance for choosing labels:
- Technology: software, IT services, cloud/hosting, developer tools, hardware tech
- Finance and Banking: banks, investment, insurance, lending, fintech primarily in finance
- Consulting: advisory services, professional services firms, agencies primarily consulting
- Marketing and Sales: advertising, PR, marketing agencies, sales enablement, lead gen
- Healthcare and Pharmaceuticals: hospitals, clinics, biotech, pharma, medical devices
- Education: schools, universities, training providers, edtech primarily education
- Government and Public Sector: government agencies, public authorities, municipalities
- Retail and E-commerce: selling goods directly to consumers (online or physical)
- Operations and Supply Chain: logistics, warehousing, procurement, fulfillment, manufacturing ops focus
- Entrepreneurship: incubators/accelerators, startup studios, founder communities (only if that's the core)
- Other: if none clearly fit or evidence is too weak

Confidence guidance:
- 0.85–1.00 strong evidence
- 0.60–0.84 moderate evidence
- 0.30–0.59 weak evidence
- 0.00–0.29 unknown

Input:
- Original name: {business_name}
- Normalized name: {normalized_name}
- LinkedIn URL: {linkedin_url}
- Evidence summary: {evidence_summary}

Controlled vocabulary (choose one):
{labels_bullets}
"""


# ============================================================
# URL normalization (optional but useful)
# ============================================================

def normalize_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.strip()

    # Common missing-scheme cases:
    if u.startswith("www."):
        u = "https://" + u
    if u.startswith("linkedin.com/"):
        u = "https://" + u

    parsed = urlparse(u)
    if not parsed.scheme:
        u = "https://" + u
    return u


# ============================================================
# OpenAI helpers
# ============================================================

class OpenAIRequestError(RuntimeError):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(OpenAIRequestError),
)
def responses_structured(
    client: OpenAI,
    model: str,
    prompt: str,
    schema_wrapper: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calls OpenAI Responses API with Structured Outputs.

    IMPORTANT: The API expects text.format = {type, name, schema, strict}.
    """
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            tools=tools or [],
            include=include or [],
            temperature=0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_wrapper["name"],
                    "schema": schema_wrapper["schema"],
                    "strict": True,
                }
            },
        )
    except Exception as e:
        raise OpenAIRequestError(str(e)) from e

    raw = getattr(resp, "output_text", None)
    if not raw:
        raise OpenAIRequestError("No output_text returned from Responses API.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise OpenAIRequestError(f"Model did not return valid JSON: {e}\nRaw output:\n{raw}") from e


def pass1_resolve(
    client: OpenAI,
    model: str,
    search_tool_type: str,
    business_name: str,
) -> ResolveResult:
    data = responses_structured(
        client=client,
        model=model,
        prompt=resolve_prompt(business_name),
        schema_wrapper=resolve_schema(),
        tools=[{"type": search_tool_type}],
        include=["web_search_call.action.sources"],
    )
    resolved = ResolveResult.model_validate(data)

    # Normalize URLs post-hoc
    resolved.linkedin_url = normalize_url(resolved.linkedin_url)
    for alt in resolved.alternates:
        alt.linkedin_url = normalize_url(alt.linkedin_url)

    return resolved


def pass2_classify(
    client: OpenAI,
    model: str,
    business_name: str,
    resolved: ResolveResult,
    allowed_labels: List[str],
) -> Dict[str, Any]:
    return responses_structured(
        client=client,
        model=model,
        prompt=classify_prompt(
            business_name=business_name,
            normalized_name=resolved.normalized_name,
            linkedin_url=resolved.linkedin_url,
            evidence_summary=resolved.evidence_summary,
            allowed_labels=allowed_labels,
        ),
        schema_wrapper=classify_schema(allowed_labels),
        tools=[],  # IMPORTANT: no web tool in pass 2
        include=[],
    )


def run_2pass(
    client: OpenAI,
    model: str,
    search_tool_type: str,
    business_name: str,
    allowed_labels: List[str],
) -> FinalResult:
    resolved = pass1_resolve(client, model, search_tool_type, business_name)
    classified = pass2_classify(client, model, business_name, resolved, allowed_labels)

    final = FinalResult(
        input_name=resolved.input_name,
        normalized_name=resolved.normalized_name,
        linkedin_url=resolved.linkedin_url,
        linkedin_display_name=resolved.linkedin_display_name,
        linkedin_confidence=resolved.linkedin_confidence,
        evidence_summary=resolved.evidence_summary,
        classification=Classification.model_validate(classified["classification"]),
        rationale=classified["rationale"],
        alternates=resolved.alternates,
    )
    return final


# ============================================================
# CLI
# ============================================================

def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="2-pass LinkedIn company match + controlled vocab classification")
    ap.add_argument("name", help="Business name string to match/classify")
    ap.add_argument("--vocab", required=True, help='Path to vocab JSON like {"labels":[...]}')
    ap.add_argument("--out", default="-", help="Output JSON path (default: stdout)")

    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1"))
    ap.add_argument("--search-tool", default=os.getenv("OPENAI_SEARCH_TOOL", "web_search_preview"))
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY (set it in .env or env).", file=sys.stderr)
        return 2

    args = parse_args(argv)
    labels = load_vocab(args.vocab)

    client = OpenAI()

    try:
        final = run_2pass(
            client=client,
            model=args.model,
            search_tool_type=args.search_tool,
            business_name=args.name,
            allowed_labels=labels,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    payload = final.model_dump()

    if args.out in ("-", "stdout"):
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
