from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from scripts.data.auto_labeled_common import normalize_text


@dataclass(frozen=True)
class WordBox:
    x: int
    y: int
    w: int
    h: int
    text: str
    conf: float


class Provider(ABC):
    name: str = "abstract"

    @abstractmethod
    def label_page(self, image_path: Path) -> list[WordBox]:
        """Label a single page image. Return a list of word boxes.

        Implementations must NFC-normalize the returned text and must not
        return boxes with empty or whitespace-only text.
        """


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, responses_path: Path):
        self._responses: dict[str, list[dict]] = json.loads(
            Path(responses_path).read_text(encoding="utf-8")
        )

    def label_page(self, image_path: Path) -> list[WordBox]:
        key = Path(image_path).name
        if key not in self._responses:
            raise KeyError(f"fake provider has no response for: {key}")
        raw = self._responses[key]
        out: list[WordBox] = []
        for b in raw:
            text = normalize_text(b["text"])
            if not text:
                continue
            out.append(
                WordBox(
                    x=int(b["x"]),
                    y=int(b["y"]),
                    w=int(b["w"]),
                    h=int(b["h"]),
                    text=text,
                    conf=float(b["conf"]),
                )
            )
        return out


class AzureReadProvider(Provider):
    name = "azure"

    def __init__(self, endpoint: str, key: str):
        # Deferred import so this module stays importable without the Azure SDK.
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        self._client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

    def label_page(self, image_path: Path) -> list[WordBox]:
        with Path(image_path).open("rb") as f:
            poller = self._client.begin_analyze_document(
                model_id="prebuilt-read", body=f
            )
        result = poller.result()
        out: list[WordBox] = []
        for page in result.pages or []:
            for word in page.words or []:
                text = normalize_text(word.content or "")
                if not text:
                    continue
                poly = word.polygon  # flat list [x1,y1,x2,y2,x3,y3,x4,y4]
                if not poly:
                    continue
                xs = poly[0::2]
                ys = poly[1::2]
                x, y = int(min(xs)), int(min(ys))
                w, h = int(max(xs) - min(xs)), int(max(ys) - min(ys))
                out.append(
                    WordBox(x=x, y=y, w=w, h=h, text=text, conf=float(word.confidence))
                )
        return out


def get_provider(name: str, **kwargs) -> Provider:
    if name == "fake":
        return FakeProvider(responses_path=kwargs["fake_responses_path"])
    if name == "azure":
        import os
        endpoint = kwargs.get("azure_endpoint") or os.environ["AZURE_DOC_INTEL_ENDPOINT"]
        key = kwargs.get("azure_key") or os.environ["AZURE_DOC_INTEL_KEY"]
        return AzureReadProvider(endpoint=endpoint, key=key)
    raise ValueError(f"unknown provider: {name}")
