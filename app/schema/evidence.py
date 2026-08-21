"""
Step 2: The Evidence Schema
------------------------------
This is the data model everything else builds on. Getting this right
matters more than any individual prompt or UI choice, because it's
what FORCES the LLM extraction step to be honest rather than vague.

Core design decision: CompetencyTier is a closed enum, not free text.
If we let the model just write a prose "level of competence" for each
area, it'll drift into hedging phrases that are hard to compare across
competencies ("solid experience", "some familiarity", "appears
competent") - none of which map cleanly to the exercise's actual
requirement: distinguish DEMONSTRATED from MENTIONED, and flag
INSUFFICIENT INFO explicitly rather than guessing.

Second core design decision: `evidence` is a list of EvidenceQuote
objects, each requiring an actual quote string. This is what makes the
"anti-hallucination" validation step (Step 4) possible at all - if the
schema only had a free-text "reasoning" field, there'd be nothing
concrete to check against the source CV text. Requiring a quote is what
turns "trust the model" into "verify the model."

Explicitly excluded from this schema on purpose: any kind of score,
rating, recommendation, or verdict field. The brief is explicit that
this system must NOT make a hiring/pass-fail decision - baking that
constraint into the schema itself (rather than just a prompt
instruction an LLM could ignore) is a stronger guarantee. There is
structurally nowhere to put a verdict even if the model tried to give
one - it would have to shoehorn it into `reasoning` prose, which is a
much easier thing to catch/strip in a review pass than a dedicated
"score" field silently produced and displayed.
"""

from enum import Enum
from pydantic import BaseModel, Field, model_validator


class CompetencyTier(str, Enum):
    DEMONSTRATED_STRONG = "demonstrated_strong"
    # Named project/role + specific technical detail (what was built,
    # how, and ideally an outcome).

    DEMONSTRATED_MODERATE = "demonstrated_moderate"
    # Named project/role, but shallow detail - e.g. "used X at Company Y"
    # with no elaboration on depth, scale, or the person's specific role.

    MENTIONED_ONLY = "mentioned_only"
    # Appears in a skills list / tech stack line with no project or
    # role tied to it anywhere else in the CV.

    NOT_DEMONSTRATED = "not_demonstrated"
    # The competency area does not appear anywhere in the CV, in any form.

    INSUFFICIENT_INFO = "insufficient_info"
    # Present, but phrased too ambiguously to classify into any tier
    # above with confidence (e.g. "familiar with X", "exposure to Y").


class EvidenceQuote(BaseModel):
    quote: str = Field(
        description="A short, near-verbatim excerpt from the CV supporting this assessment. "
                    "Must be checkable against the source text - not a paraphrase."
    )
    location: str = Field(
        default="",
        description="Where in the CV this appears, e.g. 'Experience > Nimbus Data Systems'",
    )


class CompetencyAssessment(BaseModel):
    area: str = Field(description="The AI competency area being assessed, e.g. 'Retrieval-Augmented Generation (RAG)'")
    tier: CompetencyTier
    evidence: list[EvidenceQuote] = Field(
        default_factory=list,
        description="Supporting quotes. Empty ONLY when tier is NOT_DEMONSTRATED.",
    )
    reasoning: str = Field(
        min_length=10,
        description="A short, factual explanation for why this tier was chosen, tied "
                    "directly to the evidence. No score, rating, or hiring judgment. "
                    "Cannot be empty - this is what makes the assessment checkable.",
    )

    @model_validator(mode="after")
    def evidence_required_unless_not_demonstrated(self) -> "CompetencyAssessment":
        # Structural enforcement, not just a prompt request: any tier that
        # claims something IS present in the CV must point to at least one
        # quote. Only NOT_DEMONSTRATED is allowed to have zero evidence,
        # since by definition there's nothing to quote. This turns a
        # possible silent gap (empty evidence list nobody notices) into a
        # hard validation failure that triggers the extractor's retry loop.
        if self.tier != CompetencyTier.NOT_DEMONSTRATED and not self.evidence:
            raise ValueError(
                f"tier '{self.tier}' requires at least one evidence quote "
                f"(only 'not_demonstrated' may have empty evidence)"
            )
        return self


class CVReview(BaseModel):
    candidate_source: str = Field(description="CV filename or identifying label")
    competencies: list[CompetencyAssessment]
    additional_technologies_noted: list[str] = Field(
        default_factory=list,
        description="Other AI-relevant technologies found in the CV that don't map "
                    "cleanly to the standard competency list (e.g. a specific "
                    "framework not in our fixed area list).",
    )

    def by_tier(self, tier: CompetencyTier) -> list[CompetencyAssessment]:
        return [c for c in self.competencies if c.tier == tier]


# The fixed set of competency areas the exercise asks us to always
# review, regardless of what the CV does or doesn't contain. Every one
# of these must appear in the output - even if the answer for that
# area is simply NOT_DEMONSTRATED - because the exercise explicitly
# asks for "relevant AI skills that are NOT demonstrated" too, not
# just the ones that are.
STANDARD_COMPETENCY_AREAS = [
    "Python",
    "Large Language Models (LLMs)",
    "Embeddings",
    "Vector Databases",
    "Retrieval-Augmented Generation (RAG)",
    "Machine Learning / Deep Learning",
    "AI Frameworks and Libraries",
    "Model Integration and APIs",
    "AI Solution Architecture",
]


if __name__ == "__main__":
    # Quick manual sanity check: build one by hand and print it as the
    # LLM extraction step (Step 3) will eventually produce it.
    example = CVReview(
        candidate_source="jordan_rivera.txt",
        competencies=[
            CompetencyAssessment(
                area="Retrieval-Augmented Generation (RAG)",
                tier=CompetencyTier.DEMONSTRATED_STRONG,
                evidence=[
                    EvidenceQuote(
                        quote="Designed and built a Retrieval-Augmented Generation pipeline "
                              "for internal document search, using ChromaDB as the vector "
                              "store and sentence-transformers for embeddings.",
                        location="Experience > Nimbus Data Systems",
                    )
                ],
                reasoning="Named project with specific tools (ChromaDB, sentence-transformers) "
                          "and a stated measurable outcome (lookup time reduction).",
            ),
            CompetencyAssessment(
                area="AI Frameworks and Libraries",
                tier=CompetencyTier.MENTIONED_ONLY,
                evidence=[EvidenceQuote(quote="TensorFlow, PyTorch", location="Skills")],
                reasoning="Listed in the skills section only; no project or role in the "
                          "CV ties either framework to actual work performed.",
            ),
            CompetencyAssessment(
                area="Machine Learning / Deep Learning",
                tier=CompetencyTier.INSUFFICIENT_INFO,
                evidence=[EvidenceQuote(quote="Familiar with machine learning concepts through coursework.", location="Experience > Alderly Tech (Junior Developer)")],
                reasoning="Too ambiguous to classify - 'familiar with concepts through "
                          "coursework' does not indicate applied experience, but is not "
                          "absent either.",
            ),
        ],
    )
    print(example.model_dump_json(indent=2))
