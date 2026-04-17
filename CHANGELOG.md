# Changelog

## 2026.04.2 (2026-04-17)

### Added
- A/B scoring loop: validates evolved sentences against originals
- `--worker-model` flag for cost optimization (Haiku workers, Sonnet aggregator)
- Aggregator system prompt for better output consistency
- Aggregator max_tokens increased to 2048

### Fixed
- Delphi peer-exclusion: build round1 in persona order (not as_completed arrival order)
- Error handling for `--from-checker` (FileNotFoundError, JSONDecodeError)
- Progress messages sent to stderr to keep JSON stdout clean

## 2026.04.1 (2026-04-17)

### Added
- Initial release with 10 writer personas (K-Fish orthogonal pattern)
- SentenceEvolver (Claude API) with Round 1 + Delphi Round 2 + Aggregator
- OfflineEvolver (rule-based, no API calls)
- 21 offline transform rules
- CLI with --from-checker, --stdin, --offline, --personas, --no-delphi
- 13 tests
