"""NLI Service - Natural Language Inference for Anti-Hallucination."""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class NLIResult:
    """NLI verification result."""

    label: Literal["entailment", "neutral", "contradiction"]
    score: float
    claim: str
    evidence: str


class NLIService:
    """Service for verifying claims against evidence (anti-hallucination)."""

    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli"):
        """Initialize NLI service.

        Args:
            model_name: HuggingFace model for NLI
        """
        self.model_name = model_name
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy load NLI pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline("text-classification", model=self.model_name)
                logger.info(f"Loaded NLI model: {self.model_name}")
            except ImportError:
                logger.error("transformers not installed. Install with: pip install transformers")
                raise

        return self._pipeline

    async def verify_claim(self, claim: str, evidence: str) -> NLIResult:
        """Verify if a claim is supported by evidence.

        Args:
            claim: Statement to verify
            evidence: Evidence text

        Returns:
            NLIResult with label and confidence
        """
        pipeline = self._get_pipeline()

        # Format input for NLI
        # Most NLI models expect: premise [SEP] hypothesis
        text = f"{evidence} [SEP] {claim}"

        # Get prediction
        result = pipeline(text, truncation=True, max_length=512)[0]

        # Map label
        label_map = {
            "ENTAILMENT": "entailment",
            "NEUTRAL": "neutral",
            "CONTRADICTION": "contradiction",
        }

        label = label_map.get(result['label'].upper(), result['label'].lower())

        return NLIResult(
            label=label,
            score=result['score'],
            claim=claim,
            evidence=evidence,
        )

    async def verify_multiple(self, claims: list[str], evidence: str) -> list[NLIResult]:
        """Verify multiple claims against evidence.

        Args:
            claims: List of claims to verify
            evidence: Evidence text

        Returns:
            List of NLIResults
        """
        results = []
        for claim in claims:
            result = await self.verify_claim(claim, evidence)
            results.append(result)

        return results
