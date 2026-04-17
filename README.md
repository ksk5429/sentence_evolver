# Sentence Evolver

Multi-agent sentence rewriting engine that evolves flagged sentences through 10 orthogonal writer personas using swarm intelligence and Delphi consensus.

## How It Works

```
Input: "It is worth noting that digital transformation has become pivotal."
       [flagged by ai_style_checker: "Zero-information filler phrase"]

     ┌──────────────┐
     │  10 Personas  │  Round 1: Independent rewrites
     │  (parallel)   │  No inter-agent communication
     └──────┬───────┘
            │
     ┌──────▼───────┐
     │    Delphi     │  Round 2: Anonymized peer versions shown
     │  (optional)   │  Each persona adopts, holds, or synthesizes
     └──────┬───────┘
            │
     ┌──────▼───────┐
     │  Aggregator   │  Synthesize best version with reasoning
     └──────┬───────┘
            │
Output: "Digital transformation has become pivotal."
        [reasoning: removed filler, 7 words saved, all personas agreed]
```

## The 10 Personas

Each persona is an **orthogonal transformation machine** (K-Fish pattern, Schoenegger 2024) — not style mimicry, but a fundamentally different reasoning approach to sentence revision.

| Persona | Strategy | Inspired By |
|---------|----------|-------------|
| **compressor** | Remove every word that doesn't earn its place | Strunk & White, Orwell |
| **precisionist** | Fix grammar, syntax, and convention violations | Chicago Manual of Style |
| **clarifier** | Make meaning land on first read | Steven Pinker, Sense of Style |
| **contrarian** | Challenge the claim, harden against peer review | Devil's Advocate |
| **concrete_thinker** | Replace abstraction with observable specifics | Hemingway |
| **architect** | Restructure information flow (topic-stress) | Joseph Williams |
| **voice_authenticator** | Remove robotic uniformity, inject human voice | Anti-AI Humanizer |
| **calibrator** | Match confidence level to evidence strength | Tetlock Superforecasting |
| **storyteller** | Add narrative momentum and reader engagement | Murakami / Narrative Craft |
| **domain_expert** | Fix domain-specific terminology and conventions | Reviewer 2 |

## Quick Start

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Evolve a single sentence
python cli.py "It is worth noting that the results underscore the importance of X."

# Verbose mode — see all persona rewrites
python cli.py "sentence" -v

# Offline mode (no API calls, rule-based transforms)
python cli.py "sentence" --offline

# List all personas
python cli.py --list-personas

# Use specific personas only (faster, cheaper)
python cli.py "sentence" --personas compressor,clarifier,contrarian

# Skip Delphi round (1 round instead of 2)
python cli.py "sentence" --no-delphi
```

## Integration with ai_style_checker

The intended workflow pipes flagged sentences from [ai_style_checker](https://github.com/ksk5429/ai_style_checker) into the evolver:

```bash
# Step 1: Check manuscript
python -m ai_style_checker manuscript.qmd --format json > flags.json

# Step 2: Evolve flagged sentences
python cli.py --from-checker flags.json -v

# Or pipe directly
python -m ai_style_checker manuscript.qmd --format json | python cli.py --stdin
```

## Offline Mode

For CI pipelines or when the API is unavailable, the `--offline` flag uses rule-based transforms (no LLM calls):

```bash
python cli.py "Furthermore, it is worth noting that X could potentially affect Y." --offline
# Output: "X could affect Y."
```

The offline evolver handles:
- Filler phrase removal ("It is worth noting that", "It should be mentioned")
- Formulaic transition removal ("Furthermore,", "Moreover,", "Additionally,")
- Verbose phrase simplification ("in order to" → "to", "due to the fact that" → "because")
- Double hedge reduction ("could potentially" → "could")

## Architecture

```
sentence_evolver/
├── cli.py                  # CLI entry point
├── engine/
│   ├── persona.py          # 10 writer personas (system prompts)
│   └── evolver.py          # SentenceEvolver (API) + OfflineEvolver (rules)
├── tests/
│   └── test_evolver.py     # 13 tests (all offline, no API needed)
├── pyproject.toml
└── README.md
```

## Design Principles

1. **Orthogonal reasoning** — personas use different decomposition strategies, not different "voices." This produces genuinely diverse rewrites rather than stylistic variations of the same fix.

2. **Delphi consensus** — in Round 2, personas see anonymized peer versions (labeled Version A/B/C, no persona names). This prevents identity leakage and forces engagement with the idea, not the source.

3. **Disagreement is signal** — when personas fundamentally disagree (compressor wants to cut, expander wants to add), the aggregator surfaces both options to the author rather than hiding the conflict.

4. **Offline fallback** — the rule-based evolver handles the most common AI patterns without any API calls, making it usable in CI pipelines.

## Related Projects

- [ai_style_checker](https://github.com/ksk5429/ai_style_checker) — Detect AI writing patterns (upstream: flags sentences)
- [publishing_engine](https://github.com/ksk5429/publishing_engine) — DOCX rendering for journal submissions
- [quant](https://github.com/ksk5429/quant) — K-Fish swarm intelligence engine (architectural inspiration)
- [kzero](https://github.com/ksk5429/kzero) — K-ZERO Council multi-agent deliberation

## Running Tests

```bash
python tests/test_evolver.py

# Or with pytest
pip install pytest
pytest tests/ -v
```

## License

Apache 2.0
