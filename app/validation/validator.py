"""
Step 4: Validation
---------------------
Goal: for every evidence quote the LLM claims came from the CV, verify
it actually appears there. This is the last line of defense against
hallucinated evidence - everything in Step 3 (schema constraints,
prompt tightening) reduces the CHANCE of a fabricated quote, but
doesn't guarantee it can't happen. This step actually checks.

Why not exact substring matching (`quote in cv_text`)? Because the
model's transcription of a quote is rarely byte-for-byte identical to
the source:
  - PDF extraction can introduce odd whitespace/line-break patterns
    the model doesn't reproduce exactly
  - The model may trim a trailing "..." or lightly normalize punctuation
  - It may quote across what was originally two lines as one continuous
    string

None of these are hallucination - they're minor transcription noise.
Exact matching would flag ALL of them as false positives, which trains
you to ignore the validator's warnings entirely (the boy-who-cried-wolf
problem for any check). So we use fuzzy string matching (rapidfuzz) and
a threshold: close enough to the source text counts as verified; a
quote with no good match anywhere in the CV is flagged as suspect.

Design decision: we validate against the RAW loaded CV text (not
against any chunked/processed version) - this is deliberately the
closest possible thing to "ground truth" for the check to be
meaningful.
"""

from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# validator.py lives at app/validation/validator.py, so parent.parent
# is the project root - must be on sys.path before the app.* imports
# below (which execute immediately at module load time).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rapidfuzz import fuzz

from app.schema.evidence import CVReview, CompetencyAssessment, EvidenceQuote


class VerificationStatus(str, Enum):
    VERIFIED = "verified"          # strong match found in source text
    PARTIAL_MATCH = "partial_match"  # some match, but below the confident threshold
    NOT_FOUND = "not_found"        # no reasonable match - likely hallucinated


@dataclass
class QuoteVerification:
    quote: str
    area: str
    status: VerificationStatus
    match_score: float  # 0-100, rapidfuzz partial_ratio score


@dataclass
class ValidationReport:
    verifications: list[QuoteVerification]

    @property
    def total_quotes(self) -> int:
        return len(self.verifications)

    @property
    def verified_count(self) -> int:
        return sum(1 for v in self.verifications if v.status == VerificationStatus.VERIFIED)

    @property
    def flagged(self) -> list[QuoteVerification]:
        """Quotes that did NOT verify cleanly - these need human review
        before the report should be trusted as-is."""
        return [v for v in self.verifications if v.status != VerificationStatus.VERIFIED]

    @property
    def all_verified(self) -> bool:
        return len(self.flagged) == 0


# Thresholds tuned for CV text: PDF extraction noise and light quote
# normalization are common and NOT hallucination, so the verified
# threshold is forgiving. Below the partial threshold, treat it as a
# real problem worth surfacing.
VERIFIED_THRESHOLD = 85
PARTIAL_THRESHOLD = 60


def _normalize(text: str) -> str:
    """Collapse whitespace so line-break/spacing differences between
    the source CV and the model's quote transcription don't cause
    false-positive mismatches."""
    return " ".join(text.split())


def verify_quote(quote: str, source_text: str) -> tuple[VerificationStatus, float]:
    norm_quote = _normalize(quote)
    norm_source = _normalize(source_text)

    # partial_ratio finds the best-matching substring of source_text
    # against quote, which is exactly what we want - the quote should
    # match SOME contiguous span of the CV, not the whole document.
    score = fuzz.partial_ratio(norm_quote, norm_source)

    if score >= VERIFIED_THRESHOLD:
        status = VerificationStatus.VERIFIED
    elif score >= PARTIAL_THRESHOLD:
        status = VerificationStatus.PARTIAL_MATCH
    else:
        status = VerificationStatus.NOT_FOUND

    return status, score


def validate_review(review: CVReview, source_text: str) -> ValidationReport:
    verifications = []

    for comp in review.competencies:
        for ev in comp.evidence:
            status, score = verify_quote(ev.quote, source_text)
            verifications.append(
                QuoteVerification(
                    quote=ev.quote,
                    area=comp.area,
                    status=status,
                    match_score=score,
                )
            )

    return ValidationReport(verifications=verifications)


if __name__ == "__main__":
    # Manual test using the real extraction output you got, plus one
    # deliberately fabricated quote to confirm detection works.
    from app.loaders.document_loader import load_document

    doc = load_document("data/sample_cvs/jordan_rivera.txt")

    test_review = CVReview(
        candidate_source=doc.source,
        competencies=[
            CompetencyAssessment(
                area="Python",
                tier="demonstrated_strong",
                evidence=[EvidenceQuote(
                    quote="Designed and built a Retrieval-Augmented Generation pipeline for internal document search, using ChromaDB as the vector store and sentence-transformers (all-MiniLM-L6-v2) for embeddings.",
                    location="Experience",
                )],
                reasoning="Real quote, should verify cleanly.",
            ),
            CompetencyAssessment(
                area="Machine Learning / Deep Learning",
                tier="demonstrated_strong",
                evidence=[EvidenceQuote(
                    quote="Led a team of 12 ML engineers building a proprietary deep learning framework from scratch.",
                    location="Experience",
                )],
                reasoning="FABRICATED - this sentence does not appear in the CV at all.",
            ),
        ],
    )

    report = validate_review(test_review, doc.text)

    print(f"Total quotes checked: {report.total_quotes}")
    print(f"Verified: {report.verified_count}")
    print(f"All verified: {report.all_verified}")
    print()
    for v in report.verifications:
        print(f"[{v.status.value.upper()}] score={v.match_score:.1f}  area={v.area}")
        print(f"    quote: {v.quote[:100]}...")
        print()
