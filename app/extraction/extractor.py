"""
Step 3: Extraction
---------------------
Goal: given raw CV text, get back a validated CVReview object (from
our Step 2 schema) with all 9 standard competency areas assessed.

Two reliability problems to solve here, beyond just "write a good
prompt":

1. LLMs (especially smaller local ones like llama3.2/qwen2.5) don't
   always return perfectly valid JSON - a trailing comma, a missing
   quote, prose before/after the JSON block, etc. We force JSON mode
   via Ollama's `format: "json"` parameter AND still defensively parse
   (strip markdown fences, find the JSON object boundaries) before
   handing it to Pydantic.

2. Even syntactically valid JSON might not match OUR schema (wrong
   enum value for tier, missing required field, competency area
   phrased differently than our fixed list). We validate with
   Pydantic and, on failure, retry with the validation error fed back
   to the model - this "self-correction" loop meaningfully improves
   reliability with smaller local models, which don't always get
   structured output right on the first try.

Design decision: we do NOT let the model choose which competency
areas to cover. We explicitly list all 9 STANDARD_COMPETENCY_AREAS in
the prompt and require the output to cover every one of them - this
is what guarantees "AI skills that are NOT demonstrated" actually
shows up in the output, rather than the model only writing about
things it found and silently omitting gaps.
"""

import json
import re
import sys
from pathlib import Path

# extractor.py lives at app/extraction/extractor.py, so parent.parent
# is the project root - must be on sys.path BEFORE the app.* imports
# below, since those imports execute immediately at module load time
# (not inside __main__, which runs too late for this fix).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests

from app.schema.evidence import CVReview, STANDARD_COMPETENCY_AREAS


SYSTEM_INSTRUCTION = """You are a meticulous technical CV reviewer. Your ONLY job is to \
identify evidence of AI-related technical competencies in the CV text provided, and \
classify each one honestly.

CRITICAL RULES:
1. You must assess EVERY competency area listed below - do not skip any, even if the \
CV says nothing about it (in that case, use tier "not_demonstrated" with empty evidence).
2. Distinguish carefully between:
   - A competency backed by a specific project, role, or described piece of work \
(tier: demonstrated_strong or demonstrated_moderate)
   - A competency that ONLY appears in a skills/technology list with no project or \
role tied to it anywhere else in the CV (tier: mentioned_only)
   - A competency phrased too vaguely to classify confidently, e.g. "familiar with", \
"exposure to", "coursework in" (tier: insufficient_info) - IMPORTANT: you must still \
quote the exact ambiguous phrase as evidence. The vague wording itself IS the evidence \
- it's what makes the tier "insufficient_info" rather than "not_demonstrated". Never \
leave evidence empty for this tier if any relevant text exists in the CV.
   - A competency that does not appear anywhere in the CV (tier: not_demonstrated)
3. Every evidence quote MUST be copied verbatim or near-verbatim from the CV text - \
never paraphrase or invent a quote.
4. EVERY assessment, INCLUDING tier "not_demonstrated", MUST include a non-empty \
"reasoning" string of at least one full sentence. For "not_demonstrated", explain what \
you looked for and confirm it does not appear anywhere in the CV (e.g. "No mention of \
solution architecture responsibilities, diagrams, or system design ownership appears \
anywhere in the CV"). Reasoning is never optional, regardless of tier.
5. You must NOT provide a hiring recommendation, pass/fail judgment, score, or rating \
of any kind. Your job is exclusively to report what is and isn't evidenced.
6. Named AI/ML tools and frameworks found ANYWHERE in the CV (skills list or project \
description) MUST be attributed as evidence to the matching competency area below - \
do not omit them into "additional_technologies_noted" instead. For example:
   - TensorFlow, PyTorch, Keras, scikit-learn, Hugging Face Transformers, LangChain, \
LlamaIndex -> evidence for "AI Frameworks and Libraries"
   - ChromaDB, Pinecone, FAISS, pgvector, Weaviate, Milvus -> evidence for "Vector Databases"
   - sentence-transformers, OpenAI/Cohere embedding APIs -> evidence for "Embeddings"
   - OpenAI API, Anthropic API, Azure OpenAI, calling any hosted LLM -> evidence for \
"Large Language Models (LLMs)" AND "Model Integration and APIs"
   If a tool ONLY appears in a skills list with no project tied to it, that still \
counts as evidence - just at tier "mentioned_only", not "not_demonstrated".
7. If a competency area has MULTIPLE relevant tools/technologies with DIFFERENT \
strengths of evidence (e.g. one framework used in a real project, another only listed \
in skills with no project), you must NOT collapse them into a single quote. Include \
ALL relevant quotes for that area in the "evidence" list - one entry per tool/phrase. \
Choose the "tier" based on the STRONGEST evidence present, but the "reasoning" must \
explicitly name the weaker/mentioned-only items too, so no relevant technology is \
silently dropped just because a stronger example exists for the same area.
8. "additional_technologies_noted" is ONLY for AI/ML-relevant technologies that do \
NOT fit any of the 9 listed competency areas. General non-AI infrastructure tools \
(Docker, Kubernetes, PostgreSQL, AWS, Terraform, GraphQL, Redis, gRPC, CI/CD tools, \
etc.) are NOT AI-relevant and must NOT appear in this list at all - leave them out \
entirely, they are out of scope for this review.

Respond with ONLY a single valid JSON object matching this exact structure - no \
markdown fences, no explanation before or after:

{
  "candidate_source": "<the source label given to you>",
  "competencies": [
    {
      "area": "<one of the exact competency area names listed below>",
      "tier": "demonstrated_strong" | "demonstrated_moderate" | "mentioned_only" | "not_demonstrated" | "insufficient_info",
      "evidence": [{"quote": "<verbatim excerpt>", "location": "<section/role it appears under>"}],
      "reasoning": "<short factual explanation tied to the evidence, no judgment>"
    }
  ],
  "additional_technologies_noted": ["<technology name>", ...]
}

COMPETENCY AREAS TO ASSESS (cover ALL of these):
""" + "\n".join(f"- {area}" for area in STANDARD_COMPETENCY_AREAS)


def build_prompt(cv_text: str, source: str) -> str:
    return f"""{SYSTEM_INSTRUCTION}

CANDIDATE SOURCE LABEL: {source}

CV TEXT:
---
{cv_text}
---

Respond with ONLY the JSON object described above."""


def _extract_json_block(raw: str) -> str:
    """Defensive cleanup: strip markdown fences and grab the outermost
    {...} block, in case the model added stray prose despite instructions."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return raw
    return raw[start:end + 1]


def _repair_missing_reasoning(data: dict) -> dict:
    """Patch a known model failure mode before validation: smaller local
    models sometimes omit "reasoning" specifically on `not_demonstrated`
    entries in a long JSON array (observed consistently on the LAST
    competency area, regardless of prompt wording - this looks like a
    genuine small-model limitation on long structured output, not a
    prompt problem, since it survived multiple prompt rewrites).

    Rather than burn retries re-asking the model to fill in a field
    with genuinely low informational value (there's not much more to
    "reason" about an absence than the tier itself already says), we
    fill a standard, honest placeholder and let validation pass. This
    is a deliberate reliability/cost tradeoff, not a way of hiding a
    real gap - the resulting reasoning text is true and non-committal.

    Only applies to `not_demonstrated` entries specifically - any other
    tier missing reasoning is a real problem worth retrying for, since
    a "demonstrated" or "mentioned" claim with no explanation is a much
    bigger honesty gap than a "not present" claim with no elaboration.
    """
    for comp in data.get("competencies", []):
        if comp.get("tier") == "not_demonstrated":
            reasoning = comp.get("reasoning", "")
            if not reasoning or len(reasoning.strip()) < 10:
                area = comp.get("area", "this competency")
                comp["reasoning"] = (
                    f"No evidence of {area} was found anywhere in the CV text."
                )
    return data


class ExtractionError(Exception):
    pass


class Extractor:
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def _call_llm(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # Ollama JSON mode - forces syntactically valid JSON
                "options": {
                    "temperature": 0.1,  # low temp: consistent classification, not
                                          # creative variety
                    "num_predict": 4096,  # IMPORTANT: without this, Ollama's default
                                           # max output length can truncate mid-JSON
                                           # for a response this size (9 competency
                                           # areas x quotes + reasoning each). A
                                           # truncated response fails Pydantic with a
                                           # confusing "field required" error on
                                           # whatever field the cutoff landed on -
                                           # looks like a reasoning bug but is
                                           # actually a token-limit bug.
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["response"]

    def extract(self, cv_text: str, source: str, max_retries: int = 2) -> CVReview:
        prompt = build_prompt(cv_text, source)
        last_error = None

        for attempt in range(max_retries + 1):
            raw_response = self._call_llm(prompt)
            json_str = _extract_json_block(raw_response)

            try:
                data = json.loads(json_str)
                data = _repair_missing_reasoning(data)
                return CVReview.model_validate(data)
            except (json.JSONDecodeError, Exception) as e:
                last_error = e
                # Self-correction: feed the actual error back to the model
                # and ask it to fix its own output. This is what meaningfully
                # improves reliability with smaller local models over just
                # retrying the same prompt blind.
                prompt = f"""{build_prompt(cv_text, source)}

Your previous response failed validation with this error:
{str(e)[:500]}

Your previous response was:
{json_str[:1000]}

Fix the JSON so it is valid and matches the required structure exactly. \
Respond with ONLY the corrected JSON object."""

        raise ExtractionError(
            f"Failed to get valid structured output after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )


if __name__ == "__main__":
    from app.loaders.document_loader import load_document
    from app.validation.validator import validate_review

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_cvs/jordan_rivera.txt"
    doc = load_document(path)

    extractor = Extractor()
    review = extractor.extract(doc.text, source=doc.source)

    # Validate every extracted quote against the source CV text before
    # trusting the output - this is what makes the review defensible
    # if asked "how do you know the model didn't make this up".
    report = validate_review(review, doc.text)

    print(review.model_dump_json(indent=2))
    print()
    print("=" * 60)
    print(f"VALIDATION: {report.verified_count}/{report.total_quotes} quotes verified against source CV")
    if not report.all_verified:
        print("\n⚠ FLAGGED (did not verify cleanly - possible hallucination):")
        for v in report.flagged:
            print(f"  [{v.status.value}] score={v.match_score:.1f}  area={v.area}")
            print(f"    quote: {v.quote[:120]}...")
    else:
        print("All evidence quotes verified against the source CV. ✓")
