"""CLI entry point for sentence_evolver.

Usage:
    # Evolve a single sentence
    python cli.py "It is worth noting that digital transformation has become pivotal."

    # Evolve flagged sentences from ai_style_checker JSON output
    python cli.py --from-checker report.json

    # Evolve with specific personas only
    python cli.py "sentence" --personas compressor,clarifier,contrarian

    # Offline mode (no API calls, rule-based transforms)
    python cli.py "sentence" --offline

    # Skip Delphi round (faster, less thorough)
    python cli.py "sentence" --no-delphi

    # Pipe from ai_style_checker
    python -m ai_style_checker manuscript.qmd --format json | python cli.py --stdin

    # Use a specific model
    python cli.py "sentence" --model claude-sonnet-4-20250514
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine.evolver import SentenceEvolver, OfflineEvolver, EvolutionResult
from engine.persona import PERSONAS


def _print_result(result: EvolutionResult, verbose: bool = False) -> None:
    """Pretty-print an evolution result."""
    print("\n" + "=" * 70)
    print("  SENTENCE EVOLVER")
    print("=" * 70)
    print(f"\n  ORIGINAL:  {result.original}")
    print(f"  EVOLVED:   {result.final_sentence}")

    if result.issue_flags:
        print(f"\n  FLAGS: {', '.join(result.issue_flags)}")

    if verbose and result.round1_rewrites:
        print(f"\n  --- Round 1 ({len(result.round1_rewrites)} personas) ---")
        for rw in sorted(result.round1_rewrites, key=lambda r: -r.confidence):
            print(f"\n  [{rw.persona_name}] (conf: {rw.confidence:.2f})")
            print(f"    {rw.rewritten}")
            if rw.reasoning:
                # Print first 2 lines of reasoning
                lines = rw.reasoning.strip().split("\n")
                for line in lines[:2]:
                    if line.strip():
                        print(f"    > {line.strip()[:80]}")

    if verbose and result.round2_rewrites:
        print(f"\n  --- Round 2 Delphi ({len(result.round2_rewrites)} personas) ---")
        for rw in sorted(result.round2_rewrites, key=lambda r: -r.confidence):
            print(f"\n  [{rw.persona_name}] (conf: {rw.confidence:.2f})")
            print(f"    {rw.rewritten}")

    if verbose and result.final_reasoning:
        print(f"\n  --- Aggregator Reasoning ---")
        for line in result.final_reasoning.split("\n"):
            if line.strip():
                print(f"    {line.strip()[:100]}")

    print("\n" + "=" * 70 + "\n")


def _extract_flagged_sentences(checker_json: dict) -> list[tuple[str, list[str]]]:
    """Extract sentences with issues from ai_style_checker JSON output.

    Returns list of (sentence, [issue_messages]).
    """
    flagged: list[tuple[str, list[str]]] = []

    for result in checker_json.get("results", []):
        for issue in result.get("issues", []):
            context = issue.get("context", "")
            message = issue.get("message", "")
            severity = issue.get("severity", "info")
            if context and severity in ("error", "critical", "warning"):
                flagged.append((context, [message]))

    # Deduplicate by sentence
    seen: dict[str, list[str]] = {}
    for sent, flags in flagged:
        if sent in seen:
            seen[sent].extend(flags)
        else:
            seen[sent] = list(flags)

    return [(sent, flags) for sent, flags in seen.items()]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sentence_evolver",
        description="Multi-agent sentence rewriting with writer personas",
    )
    parser.add_argument(
        "sentence",
        nargs="?",
        type=str,
        help="Sentence to evolve (or use --from-checker / --stdin)",
    )
    parser.add_argument(
        "--from-checker",
        type=str,
        help="Path to ai_style_checker JSON report",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read ai_style_checker JSON from stdin",
    )
    parser.add_argument(
        "--personas",
        type=str,
        default=None,
        help="Comma-separated persona names (default: all 10)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline rule-based transforms (no API calls)",
    )
    parser.add_argument(
        "--no-delphi",
        action="store_true",
        help="Skip Delphi round 2 (faster)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all persona rewrites and reasoning",
    )
    parser.add_argument(
        "--format",
        choices=["console", "json"],
        default="console",
        help="Output format",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=10,
        help="Max sentences to evolve from checker report (default: 10)",
    )
    parser.add_argument(
        "--list-personas",
        action="store_true",
        help="List all available personas and exit",
    )
    args = parser.parse_args()

    # List personas
    if args.list_personas:
        print("\nAvailable Personas:\n")
        for name, persona in PERSONAS.items():
            print(f"  {name:20s}  [{persona.style}]")
            print(f"  {'':20s}  {persona.description}")
            print()
        return

    # Collect sentences to evolve
    sentences: list[tuple[str, list[str]]] = []

    if args.from_checker:
        data = json.loads(Path(args.from_checker).read_text(encoding="utf-8"))
        sentences = _extract_flagged_sentences(data)[:args.max_sentences]
    elif args.stdin:
        data = json.loads(sys.stdin.read())
        sentences = _extract_flagged_sentences(data)[:args.max_sentences]
    elif args.sentence:
        sentences = [(args.sentence, [])]
    else:
        parser.print_help()
        sys.exit(1)

    if not sentences:
        print("No sentences to evolve.", file=sys.stderr)
        sys.exit(1)

    # Create evolver
    if args.offline:
        evolver = OfflineEvolver()
    else:
        persona_names = args.personas.split(",") if args.personas else None
        evolver = SentenceEvolver(
            model=args.model,
            personas=persona_names,
            enable_delphi=not args.no_delphi,
        )

    # Evolve each sentence
    results: list[EvolutionResult] = []
    for i, (sentence, flags) in enumerate(sentences):
        print(f"\nEvolving sentence {i + 1}/{len(sentences)}...")
        result = evolver.evolve(sentence, flags)
        results.append(result)

        if args.format == "console":
            _print_result(result, verbose=args.verbose)

    # JSON output
    if args.format == "json":
        output = []
        for r in results:
            output.append({
                "original": r.original,
                "evolved": r.final_sentence,
                "issue_flags": r.issue_flags,
                "round1": [
                    {
                        "persona": rw.persona_name,
                        "rewrite": rw.rewritten,
                        "confidence": rw.confidence,
                        "reasoning": rw.reasoning[:200],
                    }
                    for rw in r.round1_rewrites
                ],
                "round2": [
                    {
                        "persona": rw.persona_name,
                        "rewrite": rw.rewritten,
                        "confidence": rw.confidence,
                    }
                    for rw in r.round2_rewrites
                ],
                "aggregator_reasoning": r.final_reasoning[:500],
            })
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
