from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data.auto_labeled_provider import (
    FakeProvider,
    WordBox,
    get_provider,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "auto_labeled"
FAKE_JSON = FIXTURES / "fake_responses.json"


def test_fake_provider_returns_canned_boxes():
    provider = FakeProvider(responses_path=FAKE_JSON)
    boxes = provider.label_page(Path("page_wiki_001.png"))
    assert len(boxes) == 3
    assert boxes[0] == WordBox(x=12, y=20, w=80, h=24, text="भारत", conf=0.97)


def test_fake_provider_returns_empty_for_empty_page():
    provider = FakeProvider(responses_path=FAKE_JSON)
    assert provider.label_page(Path("page_empty.png")) == []


def test_fake_provider_raises_for_unknown_page():
    provider = FakeProvider(responses_path=FAKE_JSON)
    with pytest.raises(KeyError):
        provider.label_page(Path("nonexistent.png"))


def test_fake_provider_nfc_normalizes_text(tmp_path):
    f = tmp_path / "fake.json"
    f.write_text(json.dumps({
        "p.png": [{"x": 0, "y": 0, "w": 1, "h": 1, "text": "क\u094dष", "conf": 0.9}]
    }))
    provider = FakeProvider(responses_path=f)
    boxes = provider.label_page(Path("p.png"))
    assert boxes[0].text == "क्ष"  # NFC form


def test_get_provider_returns_fake_by_name():
    provider = get_provider("fake", fake_responses_path=FAKE_JSON)
    assert isinstance(provider, FakeProvider)


def test_get_provider_raises_for_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("nope")
