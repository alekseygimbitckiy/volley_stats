"""Optional jersey text OCR helpers."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, NamedTuple


ROOT = Path(__file__).resolve().parents[1]
DIGITS_RE = re.compile(r"\d{1,2}")
NAME_RE = re.compile(r"[A-ZА-ЯЁ]{3,}")
NAME_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"


class OCRIdentity(NamedTuple):
    kind: str
    value: str
    confidence: float
    raw_text: str


class JerseyOCR:
    def __init__(
        self,
        gpu: bool = False,
        min_confidence: float = 0.35,
        model_dir: str | Path = "external/easyocr",
        languages: list[str] | None = None,
        backend: str = "easyocr",
    ) -> None:
        self.min_confidence = min_confidence
        self.backend = backend
        if backend == "paddleocr":
            self.reader = PaddleOCRReader(gpu=gpu, model_dir=model_dir, languages=languages)
            return
        if backend != "easyocr":
            raise ValueError(f"Unsupported OCR backend: {backend}")

        import easyocr  # type: ignore

        model_path = resolve_project_path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        self.reader = easyocr.Reader(
            languages or ["en"],
            gpu=gpu,
            model_storage_directory=str(model_path),
            user_network_directory=str(model_path / "user_network"),
            verbose=False,
        )

    def read_number(self, cv2: Any, crop_bgr: Any) -> tuple[str | None, float | None]:
        best = self.read_best(cv2, crop_bgr, allowlist="0123456789", normalizer=normalize_number)
        if best is None:
            return None, None
        return best.value, best.confidence

    def read_name(self, cv2: Any, crop_bgr: Any, min_length: int = 3) -> tuple[str | None, float | None]:
        def normalize(value: str) -> str | None:
            return normalize_name(value, min_length=min_length)

        best = self.read_best(
            cv2,
            crop_bgr,
            allowlist=NAME_ALLOWLIST,
            normalizer=normalize,
        )
        if best is None:
            return None, None
        return best.value, best.confidence

    def read_identity(
        self,
        cv2: Any,
        crop_bgr: Any,
        prefer_name: bool = True,
        min_name_length: int = 3,
    ) -> OCRIdentity | None:
        name = self.read_best(
            cv2,
            crop_bgr,
            allowlist=NAME_ALLOWLIST,
            normalizer=lambda text: normalize_name(text, min_length=min_name_length),
            kind="name",
        )
        number = self.read_best(
            cv2,
            crop_bgr,
            allowlist="0123456789",
            normalizer=normalize_number,
            kind="number",
        )
        candidates = [candidate for candidate in (name, number) if candidate is not None]
        if not candidates:
            return None
        if prefer_name and name is not None and name.confidence >= self.min_confidence:
            return name
        return max(candidates, key=lambda candidate: candidate.confidence)

    def read_best(
        self,
        cv2: Any,
        crop_bgr: Any,
        allowlist: str,
        normalizer: Any,
        kind: str = "number",
    ) -> OCRIdentity | None:
        best_value = None
        best_raw = ""
        best_confidence = 0.0
        for image in prepare_ocr_images(cv2, crop_bgr):
            try:
                if self.backend == "paddleocr":
                    results = self.reader.readtext(image, allowlist=allowlist)
                else:
                    results = self.reader.readtext(
                        image,
                        detail=1,
                        paragraph=False,
                        allowlist=allowlist,
                        decoder="greedy",
                    )
            except Exception:
                continue
            for text, confidence in iter_ocr_text_scores(results):
                value = normalizer(text)
                if value and float(confidence) > best_confidence:
                    best_value = value
                    best_raw = text
                    best_confidence = float(confidence)
        if best_value is None or best_confidence < self.min_confidence:
            return None
        return OCRIdentity(kind=kind, value=best_value, confidence=best_confidence, raw_text=best_raw)


class PaddleOCRReader:
    def __init__(self, gpu: bool, model_dir: str | Path, languages: list[str] | None = None) -> None:
        import os

        cache_dir = resolve_project_path(model_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir / "paddlex"))
        os.environ.setdefault("PADDLE_HOME", str(cache_dir / "paddle"))
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
        os.environ.setdefault("FLAGS_json_format_model", "0")
        os.environ.setdefault("FLAGS_enable_pir_api", "0")

        from paddleocr import PaddleOCR  # type: ignore

        # PaddleOCR 3.x downloads PP-OCR models on first use. Disable document
        # orientation/unwarping models because jersey crops are already simple images.
        self.ocr = PaddleOCR(
            lang=choose_paddle_language(languages),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def readtext(self, image: Any, allowlist: str | None = None) -> list[tuple[str, float]]:
        if hasattr(self.ocr, "predict"):
            result = self.ocr.predict(image)
        else:
            result = self.ocr.ocr(image)
        return list(iter_ocr_text_scores(result))


def choose_paddle_language(languages: list[str] | None) -> str:
    languages = languages or ["en"]
    if "ru" in languages or "russian" in languages:
        return "ru"
    return "en"


def iter_ocr_text_scores(results: Any) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    collect_ocr_text_scores(results, pairs)
    return pairs


def collect_ocr_text_scores(value: Any, pairs: list[tuple[str, float]]) -> None:
    if value is None:
        return
    if hasattr(value, "json"):
        try:
            collect_ocr_text_scores(value.json, pairs)
            return
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            collect_ocr_text_scores(value.to_dict(), pairs)
            return
        except Exception:
            pass
    if isinstance(value, dict):
        texts = value.get("rec_texts") or value.get("texts")
        scores = value.get("rec_scores") or value.get("scores")
        if isinstance(texts, list):
            for idx, text in enumerate(texts):
                score = scores[idx] if isinstance(scores, list) and idx < len(scores) else 1.0
                pairs.append((str(text), float(score)))
        if "text" in value:
            pairs.append((str(value["text"]), float(value.get("score", value.get("confidence", 1.0)))))
        for nested in value.values():
            if isinstance(nested, (list, tuple, dict)):
                collect_ocr_text_scores(nested, pairs)
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 3 and isinstance(value[1], str):
            pairs.append((str(value[1]), float(value[2])))
            return
        if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], (float, int)):
            pairs.append((str(value[0]), float(value[1])))
            return
        if len(value) == 2 and isinstance(value[1], (list, tuple)) and len(value[1]) >= 2 and isinstance(value[1][0], str):
            pairs.append((str(value[1][0]), float(value[1][1])))
            return
        for nested in value:
            collect_ocr_text_scores(nested, pairs)


def prepare_ocr_images(cv2: Any, crop_bgr: Any) -> list[Any]:
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    height, width = crop_bgr.shape[:2]
    if height < 8 or width < 8:
        return []

    y1 = int(height * 0.12)
    y2 = int(height * 0.72)
    x1 = int(width * 0.10)
    x2 = int(width * 0.90)
    torso = crop_bgr[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]
    if torso.size == 0:
        torso = crop_bgr

    scale = max(2.0, 180.0 / max(1, torso.shape[1]))
    resized = cv2.resize(torso, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    inverse = cv2.bitwise_not(thresh)
    return [resized, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), cv2.cvtColor(inverse, cv2.COLOR_GRAY2BGR)]


def normalize_number(text: str) -> str | None:
    compact = re.sub(r"[^0-9]", "", text or "")
    match = DIGITS_RE.search(compact)
    if not match:
        return None
    value = match.group(0).lstrip("0")
    return value or "0"


def normalize_name(text: str, min_length: int = 3) -> str | None:
    compact = strip_diacritics(text or "")
    compact = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", compact).upper()
    match = NAME_RE.search(compact)
    if not match:
        return None
    value = match.group(0)
    if len(value) < min_length:
        return None
    return value


def strip_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()
