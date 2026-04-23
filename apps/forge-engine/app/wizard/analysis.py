from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

DatasetType = Literal[
    "raw_text",
    "instructions",
    "conversations",
    "qa_pairs",
    "code",
    "mixed",
    "unknown",
]

TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+|[^\sA-Za-zÀ-ÖØ-öø-ÿ0-9_]")
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")

CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".cpp",
    ".c",
    ".rs",
    ".go",
    ".swift",
    ".php",
    ".rb",
    ".scala",
    ".kt",
    ".sql",
    ".sh",
}

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".jsonl",
    ".json",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
} | CODE_EXTENSIONS

LANGUAGE_STOPWORDS: dict[str, set[str]] = {
    "english": {"the", "and", "is", "to", "of", "in", "that", "for", "with", "on"},
    "italian": {"il", "la", "di", "e", "che", "per", "con", "una", "sono", "del"},
    "spanish": {"el", "la", "de", "y", "que", "para", "con", "una", "los", "las"},
    "french": {"le", "la", "de", "et", "que", "pour", "avec", "une", "les", "des"},
    "german": {"der", "die", "und", "das", "ist", "für", "mit", "ein", "eine", "den"},
}


@dataclass(frozen=True, slots=True)
class DatasetAnalysis:
    data_path: str
    files_scanned: int
    documents_scanned: int
    estimated_tokens: int
    duplicate_ratio: float
    short_doc_ratio: float
    dominant_language: str
    dataset_type: DatasetType
    quality_score: float
    quality_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_path": self.data_path,
            "files_scanned": self.files_scanned,
            "documents_scanned": self.documents_scanned,
            "estimated_tokens": self.estimated_tokens,
            "duplicate_ratio": self.duplicate_ratio,
            "short_doc_ratio": self.short_doc_ratio,
            "dominant_language": self.dominant_language,
            "dataset_type": self.dataset_type,
            "quality_score": self.quality_score,
            "quality_notes": list(self.quality_notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetAnalysis":
        dataset_type = str(payload.get("dataset_type", "unknown"))
        if dataset_type not in _ALLOWED_DATASET_TYPES:
            dataset_type = "unknown"
        dataset_type_literal = cast(DatasetType, dataset_type)

        return cls(
            data_path=str(payload.get("data_path", "")),
            files_scanned=int(payload.get("files_scanned", 0)),
            documents_scanned=int(payload.get("documents_scanned", 0)),
            estimated_tokens=int(payload.get("estimated_tokens", 0)),
            duplicate_ratio=float(payload.get("duplicate_ratio", 0.0)),
            short_doc_ratio=float(payload.get("short_doc_ratio", 0.0)),
            dominant_language=str(payload.get("dominant_language", "unknown")),
            dataset_type=dataset_type_literal,
            quality_score=float(payload.get("quality_score", 0.0)),
            quality_notes=[str(item) for item in payload.get("quality_notes", [])],
        )


_ALLOWED_DATASET_TYPES: set[str] = {
    "raw_text",
    "instructions",
    "conversations",
    "qa_pairs",
    "code",
    "mixed",
    "unknown",
}


def analyze_dataset(
    data_path: str | Path,
    max_documents: int = 4000,
    max_chars_for_language: int = 200_000,
) -> DatasetAnalysis:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    files = _collect_files(path)
    seen_hashes: set[str] = set()
    type_counts: Counter[str] = Counter()

    documents_scanned = 0
    estimated_tokens = 0
    duplicate_docs = 0
    short_docs = 0
    parse_errors = 0
    language_chunks: list[str] = []
    language_chars_used = 0

    for file_path in files:
        if documents_scanned >= max_documents:
            break

        suffix = file_path.suffix.lower()
        try:
            if suffix == ".jsonl":
                lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                for raw_line in lines:
                    if documents_scanned >= max_documents:
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue

                    text, detected_type = _record_to_text_and_type(record)
                    if not text:
                        continue

                    tokens, is_duplicate = _consume_document(
                        text=text,
                        detected_type=detected_type,
                        type_counts=type_counts,
                        seen_hashes=seen_hashes,
                        language_chunks=language_chunks,
                        language_chars_used=language_chars_used,
                        max_chars_for_language=max_chars_for_language,
                    )
                    language_chars_used = _count_language_chars(language_chunks)

                    documents_scanned += 1
                    estimated_tokens += tokens
                    if tokens < 20:
                        short_docs += 1
                    if is_duplicate:
                        duplicate_docs += 1
            else:
                text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
                if not text:
                    continue

                detected_type = "code" if suffix in CODE_EXTENSIONS else "raw_text"
                if detected_type == "raw_text" and _looks_like_code(text):
                    detected_type = "code"

                tokens, is_duplicate = _consume_document(
                    text=text,
                    detected_type=detected_type,
                    type_counts=type_counts,
                    seen_hashes=seen_hashes,
                    language_chunks=language_chunks,
                    language_chars_used=language_chars_used,
                    max_chars_for_language=max_chars_for_language,
                )
                language_chars_used = _count_language_chars(language_chunks)

                documents_scanned += 1
                estimated_tokens += tokens
                if tokens < 20:
                    short_docs += 1
                if is_duplicate:
                    duplicate_docs += 1
        except OSError:
            parse_errors += 1

    duplicate_ratio = duplicate_docs / max(documents_scanned, 1)
    short_doc_ratio = short_docs / max(documents_scanned, 1)
    error_ratio = parse_errors / max(documents_scanned + parse_errors, 1)

    dataset_type = _resolve_dataset_type(type_counts, documents_scanned)
    dominant_language = detect_dominant_language(
        "\n".join(language_chunks),
        code_hint=(dataset_type == "code"),
    )
    quality_score = _quality_score(duplicate_ratio, short_doc_ratio, error_ratio)
    quality_notes = _quality_notes(duplicate_ratio, short_doc_ratio, error_ratio, dataset_type)

    return DatasetAnalysis(
        data_path=str(path.resolve()),
        files_scanned=len(files),
        documents_scanned=documents_scanned,
        estimated_tokens=estimated_tokens,
        duplicate_ratio=round(duplicate_ratio, 4),
        short_doc_ratio=round(short_doc_ratio, 4),
        dominant_language=dominant_language,
        dataset_type=dataset_type,
        quality_score=round(quality_score, 2),
        quality_notes=quality_notes,
    )


def detect_dominant_language(text: str, code_hint: bool = False) -> str:
    if code_hint:
        return "code-heavy"

    content = text.strip().lower()
    if not content:
        return "unknown"

    chinese_count = len(CHINESE_PATTERN.findall(content))
    if chinese_count / max(len(content), 1) > 0.15:
        return "chinese"

    words = WORD_PATTERN.findall(content)
    if not words:
        return "unknown"

    scores: dict[str, int] = {}
    for language, stopwords in LANGUAGE_STOPWORDS.items():
        scores[language] = sum(1 for word in words if word in stopwords)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_language, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if best_score < 3:
        return "unknown"
    if second_score > 0 and abs(best_score - second_score) <= 1:
        return "mixed-latin"
    return best_language


def _collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    return [
        file_path
        for file_path in sorted(path.rglob("*"))
        if file_path.is_file() and file_path.suffix.lower() in TEXT_EXTENSIONS
    ]


def _record_to_text_and_type(record: Any) -> tuple[str | None, str]:
    if not isinstance(record, dict):
        text = _coerce_text(record)
        return (text if text else None, "unknown")

    key_map = {str(key).lower(): key for key in record.keys()}
    keys = set(key_map.keys())

    conversation_keys = {"messages", "conversation", "dialogue", "turns", "chat"}
    for candidate_key in conversation_keys:
        if candidate_key in keys:
            original_key = key_map[candidate_key]
            text = _conversation_to_text(record[original_key])
            return (text if text else None, "conversations")

    has_instruction_pair = (
        ("instruction" in keys and ({"output", "response", "answer"} & keys))
        or ("prompt" in keys and ({"completion", "response", "output"} & keys))
        or ({"input", "output"} <= keys)
    )
    if has_instruction_pair:
        parts = []
        for field_name in ("instruction", "input", "prompt", "output", "response", "completion"):
            if field_name in key_map:
                parts.append(_coerce_text(record[key_map[field_name]]))
        joined = "\n".join(part for part in parts if part).strip()
        return (joined if joined else None, "instructions")

    has_qa_pair = (
        ({"question", "answer"} <= keys)
        or ({"query", "answer"} <= keys)
        or ({"question", "response"} <= keys)
    )
    if has_qa_pair:
        parts = []
        for field_name in ("question", "query", "answer", "response"):
            if field_name in key_map:
                parts.append(_coerce_text(record[key_map[field_name]]))
        joined = "\n".join(part for part in parts if part).strip()
        return (joined if joined else None, "qa_pairs")

    if ("code" in keys and "instruction" in keys) or ("code" in keys and "comment" in keys) or "source_code" in keys:
        code_value = record[key_map.get("code", key_map.get("source_code", ""))] if ("code" in keys or "source_code" in keys) else ""
        text = _coerce_text(code_value).strip()
        return (text if text else None, "code")

    if "text" in keys:
        text = _coerce_text(record[key_map["text"]]).strip()
        if text:
            return text, "raw_text"

    fallback = "\n".join(_coerce_text(value) for value in record.values()).strip()
    if not fallback:
        return None, "unknown"

    return fallback, "code" if _looks_like_code(fallback) else "unknown"


def _conversation_to_text(value: Any) -> str:
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict):
                role = _coerce_text(item.get("role", "")).strip()
                content = _coerce_text(item.get("content", item.get("text", ""))).strip()
                if role and content:
                    chunks.append(f"{role}: {content}")
                elif content:
                    chunks.append(content)
            else:
                text = _coerce_text(item).strip()
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()

    if isinstance(value, dict):
        return _coerce_text(value.get("content", value.get("text", ""))).strip()

    return _coerce_text(value).strip()


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_coerce_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_coerce_text(item) for item in value.values())
    return ""


def _consume_document(
    text: str,
    detected_type: str,
    type_counts: Counter[str],
    seen_hashes: set[str],
    language_chunks: list[str],
    language_chars_used: int,
    max_chars_for_language: int,
) -> tuple[int, bool]:
    normalized_type = _normalize_dataset_type(detected_type)
    type_counts[normalized_type] += 1

    if language_chars_used < max_chars_for_language:
        remaining = max_chars_for_language - language_chars_used
        language_chunks.append(text[:remaining])

    digest = hashlib.sha1(_normalize_for_hash(text).encode("utf-8")).hexdigest()
    is_duplicate = digest in seen_hashes
    if not is_duplicate:
        seen_hashes.add(digest)

    tokens = _estimate_token_count(text)
    return tokens, is_duplicate


def _normalize_dataset_type(value: str) -> DatasetType:
    if value in _ALLOWED_DATASET_TYPES:
        return cast(DatasetType, value)
    return "unknown"


def _resolve_dataset_type(type_counts: Counter[str], documents_scanned: int) -> DatasetType:
    if documents_scanned == 0:
        return "unknown"

    nonzero = {name: count for name, count in type_counts.items() if count > 0}
    if not nonzero:
        return "unknown"

    ranked = sorted(nonzero.items(), key=lambda item: item[1], reverse=True)
    top_type, top_count = ranked[0]
    top_ratio = top_count / documents_scanned

    if len(ranked) == 1:
        return _normalize_dataset_type(top_type)

    second_ratio = ranked[1][1] / documents_scanned

    if top_ratio >= 0.65 and top_type != "unknown":
        return _normalize_dataset_type(top_type)
    if top_ratio >= 0.45 and second_ratio >= 0.25:
        return "mixed"
    if top_type == "unknown" and top_ratio > 0.70:
        return "unknown"
    if top_type != "unknown" and top_ratio >= 0.40:
        return _normalize_dataset_type(top_type)
    return "mixed"


def _estimate_token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _normalize_for_hash(text: str) -> str:
    return " ".join(text.lower().split())


def _looks_like_code(text: str) -> bool:
    sample = text[:6000]
    if not sample.strip():
        return False

    code_keywords = (
        "def ",
        "class ",
        "import ",
        "public ",
        "private ",
        "function ",
        "return ",
        "const ",
        "let ",
        "var ",
        "#include",
        "SELECT ",
        "FROM ",
    )
    keyword_hits = sum(1 for keyword in code_keywords if keyword in sample)
    symbol_hits = sum(sample.count(symbol) for symbol in ("{", "}", ";", "=>", "()", "[]"))
    line_count = max(sample.count("\n"), 1)
    return keyword_hits >= 2 or (symbol_hits / line_count) > 1.2


def _quality_score(duplicate_ratio: float, short_doc_ratio: float, error_ratio: float) -> float:
    score = 100.0
    score -= duplicate_ratio * 55.0
    score -= short_doc_ratio * 25.0
    score -= error_ratio * 20.0
    return max(0.0, min(100.0, score))


def _quality_notes(
    duplicate_ratio: float,
    short_doc_ratio: float,
    error_ratio: float,
    dataset_type: DatasetType,
) -> list[str]:
    notes: list[str] = []
    if duplicate_ratio >= 0.20:
        notes.append("Molti duplicati: conviene ripulire prima dell'addestramento.")
    if short_doc_ratio >= 0.40:
        notes.append("Molti esempi molto corti: il modello impara meno contesto utile.")
    if error_ratio >= 0.10:
        notes.append("Sono presenti record non leggibili: controlla formati e codifica file.")
    if dataset_type == "unknown":
        notes.append("Formato dati non chiaro: usa JSONL strutturato o testo pulito.")
    if not notes:
        notes.append("Qualità generale accettabile per iniziare.")
    return notes


def _count_language_chars(chunks: list[str]) -> int:
    return sum(len(chunk) for chunk in chunks)
