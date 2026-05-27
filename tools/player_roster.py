"""Roster matching for OCR player identities."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from jersey_ocr import OCRIdentity, normalize_name, normalize_number


ROOT = Path(__file__).resolve().parents[1]

CYRILLIC_TO_LATIN = str.maketrans(
    {
        "А": "A",
        "Б": "B",
        "В": "V",
        "Г": "G",
        "Д": "D",
        "Е": "E",
        "Ё": "E",
        "Ж": "ZH",
        "З": "Z",
        "И": "I",
        "Й": "I",
        "К": "K",
        "Л": "L",
        "М": "M",
        "Н": "N",
        "О": "O",
        "П": "P",
        "Р": "R",
        "С": "S",
        "Т": "T",
        "У": "U",
        "Ф": "F",
        "Х": "KH",
        "Ц": "TS",
        "Ч": "CH",
        "Ш": "SH",
        "Щ": "SCH",
        "Ъ": "",
        "Ы": "Y",
        "Ь": "",
        "Э": "E",
        "Ю": "YU",
        "Я": "YA",
    }
)


@dataclass
class RosterPlayer:
    player_id: str
    jersey_number: str | None
    names: list[str]
    name_keys: list[str]
    raw: dict[str, Any]


@dataclass
class RosterMatch:
    player: RosterPlayer
    kind: str
    confidence: float
    score: float
    matched_text: str


class PlayerRoster:
    def __init__(self, players: list[RosterPlayer], name_threshold: float = 0.78, name_margin: float = 0.05) -> None:
        self.players = players
        self.name_threshold = name_threshold
        self.name_margin = name_margin
        self.by_number: dict[str, list[RosterPlayer]] = {}
        for player in players:
            if player.jersey_number:
                self.by_number.setdefault(player.jersey_number, []).append(player)

    def match_identity(self, identity: OCRIdentity) -> RosterMatch | None:
        if identity.kind == "number":
            number = normalize_number(identity.value)
            if not number:
                return None
            players = self.by_number.get(number, [])
            if len(players) != 1:
                return None
            return RosterMatch(players[0], "number", identity.confidence, 1.0, number)

        name_key = normalize_name_key(identity.value)
        if not name_key:
            return None
        scored: list[tuple[float, RosterPlayer, str]] = []
        for player in self.players:
            for candidate in player.name_keys:
                score = SequenceMatcher(None, name_key, candidate).ratio()
                if name_key in candidate or candidate in name_key:
                    score = max(score, min(len(name_key), len(candidate)) / max(len(name_key), len(candidate)))
                scored.append((score, player, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_player, best_text = scored[0]
        second_score = 0.0
        for score, player, _text in scored[1:]:
            if player.player_id != best_player.player_id:
                second_score = score
                break
        if best_score < self.name_threshold:
            return None
        if second_score > 0 and best_score - second_score < self.name_margin:
            return None
        return RosterMatch(best_player, "name", identity.confidence, best_score, best_text)

    def player_by_id(self, player_id: str) -> RosterPlayer | None:
        for player in self.players:
            if player.player_id == player_id:
                return player
        return None


def load_roster(path: str | Path | None, name_threshold: float = 0.78, name_margin: float = 0.05) -> PlayerRoster | None:
    if not path:
        return None
    roster_path = resolve_project_path(path)
    if not roster_path.exists():
        raise FileNotFoundError(f"Roster file not found: {roster_path}")
    data = json.loads(roster_path.read_text(encoding="utf-8"))
    raw_players = data.get("players", data if isinstance(data, list) else [])
    players = []
    for item in raw_players:
        player_id = str(item.get("player_id") or "").strip()
        if not player_id:
            number = normalize_number(str(item.get("jersey_number") or item.get("number") or ""))
            first_name = str((item.get("names") or [item.get("name", "")])[0] or "")
            player_id = f"jersey_{number}" if number else f"name_{normalize_name_key(first_name) or 'unknown'}"
        jersey_number = normalize_number(str(item.get("jersey_number") or item.get("number") or ""))
        names = collect_names(item)
        name_keys = sorted({key for name in names for key in name_variants(name)})
        players.append(RosterPlayer(player_id=player_id, jersey_number=jersey_number, names=names, name_keys=name_keys, raw=item))
    return PlayerRoster(players=players, name_threshold=name_threshold, name_margin=name_margin)


def collect_names(item: dict[str, Any]) -> list[str]:
    values = []
    for key in ("name", "shirt_name", "last_name"):
        if item.get(key):
            values.append(str(item[key]))
    for key in ("names", "aliases"):
        if isinstance(item.get(key), list):
            values.extend(str(value) for value in item[key] if value)
    return values


def name_variants(value: str) -> list[str]:
    base = normalize_name_key(value)
    latin = normalize_name_key(transliterate_cyrillic(value))
    variants = [item for item in (base, latin) if item]
    normalized = normalize_name(value) or ""
    if normalized:
        variants.append(normalize_name_key(normalized))
    return variants


def normalize_name_key(value: str | None) -> str:
    text = strip_diacritics(value or "").upper().replace("Ё", "Е")
    text = transliterate_cyrillic(text)
    return re.sub(r"[^A-Z0-9]", "", text)


def strip_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def transliterate_cyrillic(value: str) -> str:
    return value.upper().translate(CYRILLIC_TO_LATIN)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()
