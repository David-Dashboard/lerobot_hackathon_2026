"""Target-selection strategies (pure)."""

from dataclasses import dataclass

import pytest

from trash_arm.decide import choose_next


@dataclass
class Det:
    label: str
    table_xy: tuple | None
    confidence: float = 0.5


def test_empty_returns_none():
    assert choose_next([]) is None


def test_unlocalized_are_ignored():
    assert choose_next([Det("a", None), Det("b", None)]) is None


def test_nearest_to_reference():
    items = [Det("far", (0.3, 0.0)), Det("near", (0.1, 0.0))]
    assert choose_next(items, strategy="nearest", reference=(0.0, 0.0)).label == "near"


def test_confidence_strategy():
    items = [Det("lo", (0.1, 0.0), 0.3), Det("hi", (0.3, 0.0), 0.9)]
    assert choose_next(items, strategy="confidence").label == "hi"


def test_most_isolated_strategy():
    # 'lonely' is far from the tight cluster -> most clearance.
    items = [
        Det("a", (0.10, 0.00)),
        Det("b", (0.11, 0.00)),
        Det("lonely", (0.30, 0.00)),
    ]
    assert choose_next(items, strategy="most_isolated").label == "lonely"


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        choose_next([Det("a", (0.1, 0.0))], strategy="nope")


def test_accepts_dict_detections():
    items = [{"table_xy": (0.3, 0.0)}, {"table_xy": (0.1, 0.0)}]
    chosen = choose_next(items, strategy="nearest", reference=(0.0, 0.0))
    assert chosen["table_xy"] == (0.1, 0.0)
