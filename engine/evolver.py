"""Core evolution engine — orchestrates multi-agent sentence rewriting.

Architecture (K-Fish pattern adapted for writing):
  Round 1: All personas rewrite independently (no inter-agent communication)
  Round 2 (Delphi): Anonymized peer versions shown; each persona adopts or holds
  Aggregation: Synthesize best version with reasoning trace

Requires: anthropic SDK (pip install anthropic)
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

from engine.persona import Persona, get_all_personas, get_persona, PERSONAS


@dataclass(frozen=True)
class Rewrite:
    """A single persona's rewrite of a sentence."""
    persona_name: str
    original: str
    rewritten: str
    reasoning: str
    confidence: float
    round_num: int = 1


@dataclass
class EvolutionResult:
    """Complete evolution result for a sentence."""
    original: str
    issue_flags: list[str]
    round1_rewrites: list[Rewrite]
    round2_rewrites: list[Rewrite]
    final_sentence: str
    final_reasoning: str
    evolution_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_rewrite(raw: str, persona_name: str, original: str, round_num: int) -> Rewrite:
    """Parse a persona's response into a Rewrite object."""
    rewritten = original  # fallback
    reasoning = raw
    confidence = 0.5

    # Extract REWRITE line
    rewrite_match = re.search(r"REWRITE:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    if rewrite_match:
        rewritten = rewrite_match.group(1).strip().strip('"').strip("'")

    # Extract CONFIDENCE
    conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", raw, re.IGNORECASE)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except ValueError:
            pass

    # Everything else is reasoning
    reasoning_parts = []
    for line in raw.split("\n"):
        upper = line.strip().upper()
        if not upper.startswith("REWRITE:") and not upper.startswith("CONFIDENCE:"):
            reasoning_parts.append(line)
    reasoning = "\n".join(reasoning_parts).strip()

    return Rewrite(
        persona_name=persona_name,
        original=original,
        rewritten=rewritten,
        reasoning=reasoning,
        confidence=confidence,
        round_num=round_num,
    )


def _ab_score(original: str, evolved: str) -> str:
    """A/B scoring: verify the evolved sentence is actually better.

    Runs lightweight heuristic checks (no API calls):
    - If evolved is identical to original: "skip"
    - If evolved has MORE AI filler patterns: "reject"
    - If evolved is >50% longer (bloated): "reject"
    - If evolved lost >30% of content words: "reject"
    - Otherwise: "accept"
    """
    if original.strip() == evolved.strip():
        return "skip"

    # Check for AI filler in evolved that wasn't in original
    import re
    filler_patterns = [
        r"\bIt is worth noting that\b",
        r"\bFurthermore,\s",
        r"\bMoreover,\s",
        r"\bAdditionally,\s",
        r"\bcould potentially\b",
        r"\bmay possibly\b",
        r"\bIt is important to\b",
    ]
    orig_filler = sum(1 for p in filler_patterns if re.search(p, original, re.I))
    evolved_filler = sum(1 for p in filler_patterns if re.search(p, evolved, re.I))
    if evolved_filler > orig_filler:
        return "reject"

    # Bloat check
    orig_words = len(original.split())
    evolved_words = len(evolved.split())
    if orig_words > 0 and evolved_words > orig_words * 1.5:
        return "reject"

    # Content loss check — only reject if SUBSTANTIVE content is lost
    # Skip for short results (filler removal can produce short but correct output)
    orig_content = set(re.findall(r"\b[a-z]{4,}\b", original.lower()))
    evolved_content = set(re.findall(r"\b[a-z]{4,}\b", evolved.lower()))
    if len(orig_content) >= 5 and len(evolved_content & orig_content) < len(orig_content) * 0.5:
        return "reject"

    return "accept"


class SentenceEvolver:
    """Multi-agent sentence evolution engine."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        worker_model: str | None = None,
        max_tokens: int = 1024,
        aggregator_max_tokens: int = 2048,
        personas: list[str] | None = None,
        enable_delphi: bool = True,
        parallel: bool = True,
        api_key: str | None = None,
    ):
        if anthropic is None:
            raise ImportError(
                "anthropic SDK required. Install: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.worker_model = worker_model or model  # Haiku for workers, Sonnet for aggregator
        self.max_tokens = max_tokens
        self.aggregator_max_tokens = aggregator_max_tokens
        self.enable_delphi = enable_delphi
        self.parallel = parallel

        if personas:
            self.personas = [get_persona(name) for name in personas]
        else:
            self.personas = get_all_personas()

    def _call_persona(
        self,
        persona: Persona,
        sentence: str,
        issue_flags: list[str],
        round_num: int = 1,
        peer_versions: list[dict[str, str]] | None = None,
    ) -> Rewrite:
        """Call a single persona to rewrite a sentence."""
        user_msg = f"ORIGINAL SENTENCE:\n{sentence}\n\n"

        if issue_flags:
            user_msg += "AI STYLE CHECKER FLAGS:\n"
            for flag in issue_flags:
                user_msg += f"  - {flag}\n"
            user_msg += "\n"

        if round_num == 2 and peer_versions:
            user_msg += "PEER VERSIONS (anonymized, from Round 1):\n"
            for i, pv in enumerate(peer_versions):
                label = chr(65 + i)  # A, B, C, ...
                user_msg += f"\nVersion {label}:\n{pv['rewritten']}\n"
            user_msg += (
                "\nReview these peer versions. You may:\n"
                "- ADOPT elements you find superior (cite which version)\n"
                "- HOLD your original rewrite if you believe it's better (explain why)\n"
                "- SYNTHESIZE the best elements from multiple versions\n\n"
            )

        user_msg += "Now rewrite the sentence following your transformation strategy."

        response = self.client.messages.create(
            model=self.worker_model,
            max_tokens=self.max_tokens,
            system=persona.system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text
        return _parse_rewrite(raw, persona.name, sentence, round_num)

    def evolve(
        self,
        sentence: str,
        issue_flags: list[str] | None = None,
    ) -> EvolutionResult:
        """Evolve a single sentence through the full pipeline.

        Args:
            sentence: The original sentence to evolve.
            issue_flags: List of issues flagged by ai_style_checker.

        Returns:
            EvolutionResult with all rounds and final synthesis.
        """
        if issue_flags is None:
            issue_flags = []

        # ── Round 1: Independent rewrites ─────────────────────────
        round1: list[Rewrite] = []

        if self.parallel:
            with ThreadPoolExecutor(max_workers=min(len(self.personas), 5)) as pool:
                # Submit in persona order and collect in same order
                # (critical for Delphi peer-exclusion indexing)
                future_list = [
                    pool.submit(self._call_persona, p, sentence, issue_flags, 1)
                    for p in self.personas
                ]
                for i, future in enumerate(future_list):
                    try:
                        round1.append(future.result())
                    except Exception as e:
                        round1.append(Rewrite(
                            persona_name=self.personas[i].name,
                            original=sentence,
                            rewritten=sentence,
                            reasoning=f"Error: {e}",
                            confidence=0.0,
                            round_num=1,
                        ))
        else:
            for persona in self.personas:
                try:
                    rw = self._call_persona(persona, sentence, issue_flags, 1)
                    round1.append(rw)
                except Exception as e:
                    round1.append(Rewrite(
                        persona_name=persona.name,
                        original=sentence,
                        rewritten=sentence,
                        reasoning=f"Error: {e}",
                        confidence=0.0,
                        round_num=1,
                    ))

        # ── Round 2: Delphi (anonymized peer review) ──────────────
        round2: list[Rewrite] = []

        if self.enable_delphi and len(round1) >= 3:
            # Prepare anonymized versions (no persona names)
            peer_versions = [
                {"rewritten": rw.rewritten}
                for rw in round1
                if rw.confidence > 0.0
            ]

            if self.parallel:
                with ThreadPoolExecutor(max_workers=min(len(self.personas), 5)) as pool:
                    future_list = [
                        pool.submit(
                            self._call_persona, p, sentence, issue_flags, 2,
                            [pv for j, pv in enumerate(peer_versions) if j != i],
                        )
                        for i, p in enumerate(self.personas)
                    ]
                    for i, future in enumerate(future_list):
                        try:
                            round2.append(future.result())
                        except Exception as e:
                            round2.append(Rewrite(
                                persona_name=self.personas[i].name,
                                original=sentence,
                                rewritten=sentence,
                                reasoning=f"Error in Delphi: {e}",
                                confidence=0.0,
                                round_num=2,
                            ))
            else:
                for i, persona in enumerate(self.personas):
                    others = [pv for j, pv in enumerate(peer_versions) if j != i]
                    try:
                        rw = self._call_persona(persona, sentence, issue_flags, 2, others)
                        round2.append(rw)
                    except Exception as e:
                        round2.append(Rewrite(
                            persona_name=persona.name,
                            original=sentence,
                            rewritten=sentence,
                            reasoning=f"Error in Delphi: {e}",
                            confidence=0.0,
                            round_num=2,
                        ))

        # ── Aggregation ───────────────────────────────────────────
        final_sentence, final_reasoning = self._aggregate(
            sentence, issue_flags, round1, round2,
        )

        # ── A/B Scoring: verify evolution is actually better ──────
        ab_result = _ab_score(sentence, final_sentence)
        if ab_result == "reject":
            final_reasoning += "\n\n[A/B REJECTED: evolved version scored worse than original. Keeping original.]"
            final_sentence = sentence

        return EvolutionResult(
            original=sentence,
            issue_flags=issue_flags,
            round1_rewrites=round1,
            round2_rewrites=round2,
            final_sentence=final_sentence,
            final_reasoning=final_reasoning,
            metadata={
                "model": self.model,
                "personas": [p.name for p in self.personas],
                "delphi_enabled": self.enable_delphi,
                "round1_count": len(round1),
                "round2_count": len(round2),
                "ab_result": ab_result,
            },
        )

    def _aggregate(
        self,
        original: str,
        issue_flags: list[str],
        round1: list[Rewrite],
        round2: list[Rewrite],
    ) -> tuple[str, str]:
        """Synthesize final sentence from all rewrites.

        Uses an LLM aggregator call (like K-Fish's final synthesis).
        """
        # Use round2 if available, else round1
        final_round = round2 if round2 else round1

        # Sort by confidence
        ranked = sorted(final_round, key=lambda r: r.confidence, reverse=True)

        aggregator_prompt = f"""You are the Aggregator — a master editor synthesizing the best version
of a sentence from multiple expert rewrites.

ORIGINAL SENTENCE:
{original}

AI STYLE CHECKER FLAGS:
{chr(10).join(f'  - {f}' for f in issue_flags) if issue_flags else '  (none)'}

EXPERT REWRITES (sorted by confidence):
"""
        for rw in ranked:
            aggregator_prompt += f"""
--- {rw.persona_name} (confidence: {rw.confidence:.2f}, round {rw.round_num}) ---
Rewrite: {rw.rewritten}
Reasoning: {rw.reasoning[:200]}
"""

        aggregator_prompt += """
YOUR TASK:
1. Identify which rewrites addressed the flagged issues most effectively
2. Note where experts AGREE (strong signal) and DISAGREE (surface to user)
3. Synthesize the BEST final version, taking the strongest elements from each expert
4. If experts fundamentally disagree on direction, provide the top 2 options

Output format:
FINAL: [the best synthesized sentence]
REASONING: [why this version wins, which experts contributed what]
CONSENSUS: [what all experts agreed on]
DISAGREEMENTS: [where experts diverged — these are genuine choices for the author]
"""

        response = self.client.messages.create(
            model=self.model,  # aggregator uses the primary (stronger) model
            max_tokens=self.aggregator_max_tokens,
            system="You are a master editor synthesizing the best sentence from multiple expert rewrites.",
            messages=[{"role": "user", "content": aggregator_prompt}],
        )

        raw = response.content[0].text

        # Parse final sentence
        final_match = re.search(r"FINAL:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
        final_sentence = final_match.group(1).strip().strip('"').strip("'") if final_match else ranked[0].rewritten

        return final_sentence, raw


class OfflineEvolver:
    """Offline evolution using pre-defined transformation rules.

    No API calls — applies rule-based transformations for quick local use.
    Useful for CI pipelines or when API is unavailable.
    """

    # Common AI patterns and their replacements
    TRANSFORMS: list[tuple[re.Pattern[str], str, str]] = [
        (re.compile(r"\bIt is worth noting that\b", re.I), "", "removed filler"),
        (re.compile(r"\bIt is important to note that\b", re.I), "", "removed filler"),
        (re.compile(r"\bIt should be mentioned that\b", re.I), "", "removed filler"),
        (re.compile(r"\bIt is crucial to acknowledge that\b", re.I), "", "removed filler"),
        (re.compile(r"\bIt bears mentioning that\b", re.I), "", "removed filler"),
        (re.compile(r"\bFurthermore,\s*", re.I), "", "removed formulaic transition"),
        (re.compile(r"\bMoreover,\s*", re.I), "", "removed formulaic transition"),
        (re.compile(r"\bAdditionally,\s*", re.I), "", "removed formulaic transition"),
        (re.compile(r"\bConsequently,\s*", re.I), "", "removed formulaic transition"),
        (re.compile(r"\bin order to\b", re.I), "to", "simplified"),
        (re.compile(r"\ba large number of\b", re.I), "many", "simplified"),
        (re.compile(r"\bdue to the fact that\b", re.I), "because", "simplified"),
        (re.compile(r"\bin the event that\b", re.I), "if", "simplified"),
        (re.compile(r"\bat this point in time\b", re.I), "now", "simplified"),
        (re.compile(r"\bprior to\b", re.I), "before", "simplified"),
        (re.compile(r"\bsubsequent to\b", re.I), "after", "simplified"),
        (re.compile(r"\bwith regard to\b", re.I), "about", "simplified"),
        (re.compile(r"\bin the context of\b", re.I), "in", "simplified"),
        (re.compile(r"\bcould potentially\b", re.I), "could", "removed double hedge"),
        (re.compile(r"\bmay possibly\b", re.I), "may", "removed double hedge"),
        (re.compile(r"\bmight perhaps\b", re.I), "might", "removed double hedge"),
    ]

    def evolve(self, sentence: str, issue_flags: list[str] | None = None) -> EvolutionResult:
        """Apply rule-based transformations offline."""
        current = sentence
        applied: list[str] = []

        for pattern, replacement, desc in self.TRANSFORMS:
            new = pattern.sub(replacement, current)
            if new != current:
                applied.append(desc)
                current = new

        # Clean up double spaces and capitalisation after removal
        current = re.sub(r"\s+", " ", current).strip()
        if current and current[0].islower():
            current = current[0].upper() + current[1:]

        # A/B scoring: verify evolution is better
        final = current if applied else sentence
        ab_result = _ab_score(sentence, final)
        if ab_result == "reject":
            final = sentence

        return EvolutionResult(
            original=sentence,
            issue_flags=issue_flags or [],
            round1_rewrites=[Rewrite(
                persona_name="offline_rules",
                original=sentence,
                rewritten=current,
                reasoning=f"Applied {len(applied)} transforms: {', '.join(applied)}",
                confidence=0.6 if applied else 0.0,
                round_num=1,
            )],
            round2_rewrites=[],
            final_sentence=final,
            final_reasoning=f"Offline rule-based: {len(applied)} transforms applied (A/B: {ab_result})",
        )
