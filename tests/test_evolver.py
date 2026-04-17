"""Tests for sentence_evolver engine.

Tests the offline evolver (no API calls needed) and persona definitions.
Run: python tests/test_evolver.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.persona import PERSONAS, get_persona, get_all_personas, Persona
from engine.evolver import OfflineEvolver, _parse_rewrite, EvolutionResult


def test_all_personas_defined():
    """All 10 personas should be registered."""
    assert len(PERSONAS) == 10, f"Expected 10 personas, got {len(PERSONAS)}"
    expected = {
        "compressor", "precisionist", "clarifier", "contrarian",
        "concrete_thinker", "architect", "voice_authenticator",
        "calibrator", "storyteller", "domain_expert",
    }
    assert set(PERSONAS.keys()) == expected, f"Missing: {expected - set(PERSONAS.keys())}"


def test_persona_fields():
    """Each persona should have all required fields."""
    for name, persona in PERSONAS.items():
        assert persona.name == name
        assert len(persona.style) > 5, f"{name}: style too short"
        assert len(persona.system_prompt) > 100, f"{name}: system_prompt too short"
        assert len(persona.description) > 10, f"{name}: description too short"


def test_get_persona():
    """get_persona should return the correct persona."""
    p = get_persona("compressor")
    assert p.name == "compressor"
    assert "Strunk" in p.style


def test_get_persona_invalid():
    """get_persona should raise for unknown persona."""
    try:
        get_persona("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_get_all_personas():
    """get_all_personas should return all 10."""
    all_p = get_all_personas()
    assert len(all_p) == 10


def test_parse_rewrite():
    """_parse_rewrite should extract structured fields."""
    raw = """REWRITE: The model predicts capacity within 7%.
REMOVED: removed "It is worth noting that"
CONFIDENCE: 0.85"""
    rw = _parse_rewrite(raw, "compressor", "original text", 1)
    assert rw.rewritten == "The model predicts capacity within 7%."
    assert rw.confidence == 0.85
    assert rw.persona_name == "compressor"
    assert rw.round_num == 1


def test_parse_rewrite_fallback():
    """_parse_rewrite should handle missing fields gracefully."""
    raw = "Just some text without structured output"
    rw = _parse_rewrite(raw, "clarifier", "original", 1)
    assert rw.rewritten == "original"  # fallback to original
    assert rw.confidence == 0.5  # default


def test_offline_evolver_removes_filler():
    """OfflineEvolver should remove AI filler phrases."""
    evolver = OfflineEvolver()
    result = evolver.evolve(
        "It is worth noting that digital transformation has become pivotal.",
        ["Zero-information filler phrase"],
    )
    assert "worth noting" not in result.final_sentence.lower()
    assert len(result.final_sentence) < len(result.original)


def test_offline_evolver_removes_transitions():
    """OfflineEvolver should remove formulaic transitions."""
    evolver = OfflineEvolver()
    result = evolver.evolve("Furthermore, the results confirm our hypothesis.")
    assert not result.final_sentence.startswith("Furthermore")


def test_offline_evolver_simplifies():
    """OfflineEvolver should simplify verbose phrases."""
    evolver = OfflineEvolver()
    result = evolver.evolve("We did this in order to test the hypothesis.")
    assert "in order to" not in result.final_sentence
    assert "to test" in result.final_sentence


def test_offline_evolver_double_hedge():
    """OfflineEvolver should remove double hedges."""
    evolver = OfflineEvolver()
    result = evolver.evolve("The results could potentially indicate a trend.")
    assert "could potentially" not in result.final_sentence
    assert "could" in result.final_sentence.lower()


def test_offline_evolver_no_change():
    """OfflineEvolver should not change clean sentences."""
    evolver = OfflineEvolver()
    clean = "The model predicts monotonic capacity within 7%."
    result = evolver.evolve(clean)
    assert result.final_sentence == clean


def test_evolution_result_structure():
    """EvolutionResult should have all required fields."""
    evolver = OfflineEvolver()
    result = evolver.evolve("Furthermore, it is worth noting that X is Y.")
    assert isinstance(result, EvolutionResult)
    assert result.original != result.final_sentence
    assert len(result.round1_rewrites) >= 1
    assert isinstance(result.issue_flags, list)


# ── Runner ────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_all_personas_defined,
        test_persona_fields,
        test_get_persona,
        test_get_persona_invalid,
        test_get_all_personas,
        test_parse_rewrite,
        test_parse_rewrite_fallback,
        test_offline_evolver_removes_filler,
        test_offline_evolver_removes_transitions,
        test_offline_evolver_simplifies,
        test_offline_evolver_double_hedge,
        test_offline_evolver_no_change,
        test_evolution_result_structure,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            errors.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}: {type(e).__name__}: {e}")

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")

    if errors:
        print("\nFailures:")
        for name, msg in errors:
            print(f"  {name}: {msg}")

    return failed == 0


if __name__ == "__main__":
    print("Running sentence_evolver tests...\n")
    success = run_all()
    sys.exit(0 if success else 1)
