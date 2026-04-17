# Sentence Evolver

Multi-agent sentence rewriting engine that evolves flagged sentences through 10 orthogonal writer personas using swarm intelligence, Delphi consensus, and A/B validation.

## How It Works

```
Input: "It is worth noting that digital transformation has become pivotal."
       [flagged by ai_style_checker: "Zero-information filler phrase"]

     ┌──────────────┐
     │  10 Personas  │  Round 1: Independent rewrites (parallel)
     │  (orthogonal) │  No inter-agent communication
     └──────┬───────┘
            │
     ┌──────▼───────┐
     │    Delphi     │  Round 2: Anonymized peer versions shown
     │  (consensus)  │  Each persona adopts, holds, or synthesizes
     └──────┬───────┘
            │
     ┌──────▼───────┐
     │  Aggregator   │  Synthesize best version with reasoning
     └──────┬───────┘
            │
     ┌──────▼───────┐
     │  A/B Scoring  │  Verify evolution is actually BETTER
     │  (validation) │  Reject if more AI filler or content lost
     └──────┬───────┘
            │
Output: "Digital transformation has become pivotal."
```

## The 10 Personas

Each persona is an **orthogonal transformation machine** (K-Fish pattern, Schoenegger 2024) -- a fundamentally different reasoning approach, not style mimicry.

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

# Evolve a single sentence (all 10 personas + Delphi)
python cli.py "It is worth noting that the results underscore the importance of X."

# Verbose mode -- see all persona rewrites
python cli.py "sentence" -v

# Cost-optimized: Haiku for workers, Sonnet for aggregator (~80% cheaper)
python cli.py "sentence" --worker-model claude-haiku-4-5-20251001

# Offline mode (no API calls, rule-based transforms)
python cli.py "sentence" --offline

# Specific personas only (faster, cheaper)
python cli.py "sentence" --personas compressor,clarifier,contrarian

# Skip Delphi round (1 round instead of 2)
python cli.py "sentence" --no-delphi

# List all personas
python cli.py --list-personas
```

## A/B Scoring (Quality Gate)

Every evolution is validated before acceptance:

- Rejects if evolved version has **more AI filler** than original
- Rejects if evolved version is **>50% longer** (bloated)
- Rejects if evolved version **lost >50% of substantive content words**
- Accepts only if the evolution genuinely improves the sentence

This prevents the evolver from accidentally making text more robotic.

## Integration with ai_style_checker

```bash
# Step 1: Check manuscript
python -m ai_style_checker manuscript.qmd --format json > flags.json

# Step 2: Evolve flagged sentences
python cli.py --from-checker flags.json -v

# Or pipe directly
python -m ai_style_checker manuscript.qmd --format json | python cli.py --stdin
```

## Cost Analysis

| Mode | Calls/sentence | Cost/sentence | Notes |
|------|---------------|---------------|-------|
| Full (10 personas + Delphi) | 21 | ~$0.20 | All Sonnet |
| Optimized (Haiku workers) | 21 | ~$0.04 | `--worker-model claude-haiku-4-5-20251001` |
| Fast (3 personas, no Delphi) | 4 | ~$0.02 | `--personas compressor,clarifier,contrarian --no-delphi` |
| Offline | 0 | $0.00 | `--offline` (rule-based only) |

## Architecture

```
sentence_evolver/
├── cli.py                  # CLI entry point
├── engine/
│   ├── persona.py          # 10 writer personas (system prompts)
│   └── evolver.py          # SentenceEvolver (API) + OfflineEvolver + A/B scoring
├── tests/
│   └── test_evolver.py     # 13 tests (all offline, no API needed)
└── pyproject.toml
```

## Design Principles

1. **Orthogonal reasoning** -- personas use different decomposition strategies, not "voices." Produces genuinely diverse rewrites.

2. **Delphi consensus** -- Round 2 shows anonymized peer versions (Version A/B/C, no persona names). Prevents identity leakage.

3. **A/B validation** -- evolved sentences are checked against the original. If the evolution makes text worse, it's rejected.

4. **Disagreement is signal** -- when personas fundamentally disagree, the aggregator surfaces both options rather than hiding the conflict.

5. **Cost tiers** -- `--worker-model` uses Haiku for the 20 persona calls and Sonnet for the aggregator only.

## Ecosystem

| Repo | Purpose |
|------|---------|
| [ai_style_checker](https://github.com/ksk5429/ai_style_checker) | Detect AI writing patterns (12 checkers, 0-100 score) |
| **sentence_evolver** | Multi-agent sentence rewriting (this repo) |
| [publishing_engine](https://github.com/ksk5429/publishing_engine) | .qmd to publication DOCX (7 document types) |
| [manuscript_pipeline](https://github.com/ksk5429/manuscript_pipeline) | Orchestrator chaining all engines |
| [pdf_search_engine](https://github.com/ksk5429/pdf_search_engine) | Multi-source academic PDF search and download |

## License

Apache 2.0
