from __future__ import annotations

import json
from pathlib import Path

from app.wizard.analysis import analyze_dataset, detect_dominant_language


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_classifies_raw_text_dataset(tmp_path: Path) -> None:
    _write(tmp_path / "notes.txt", "Questo e' un dataset di testo libero con frasi complete.")

    analysis = analyze_dataset(tmp_path)

    assert analysis.dataset_type == "raw_text"
    assert analysis.documents_scanned == 1
    assert analysis.estimated_tokens > 5


def test_classifies_instructions_dataset(tmp_path: Path) -> None:
    records = [
        {"instruction": "Riassumi", "output": "Breve riassunto"},
        {"prompt": "Traduci", "response": "Hello"},
    ]
    _write(tmp_path / "instructions.jsonl", "\n".join(json.dumps(item) for item in records))

    analysis = analyze_dataset(tmp_path)

    assert analysis.dataset_type == "instructions"
    assert analysis.documents_scanned == 2


def test_classifies_conversations_dataset(tmp_path: Path) -> None:
    records = [
        {
            "messages": [
                {"role": "user", "content": "Ciao"},
                {"role": "assistant", "content": "Ciao, come posso aiutarti?"},
            ]
        }
    ]
    _write(tmp_path / "chat.jsonl", "\n".join(json.dumps(item) for item in records))

    analysis = analyze_dataset(tmp_path)

    assert analysis.dataset_type == "conversations"


def test_classifies_qa_pairs_dataset(tmp_path: Path) -> None:
    records = [
        {"question": "Capitale d'Italia?", "answer": "Roma"},
        {"question": "2+2?", "answer": "4"},
    ]
    _write(tmp_path / "qa.jsonl", "\n".join(json.dumps(item) for item in records))

    analysis = analyze_dataset(tmp_path)

    assert analysis.dataset_type == "qa_pairs"


def test_classifies_code_dataset_from_extension(tmp_path: Path) -> None:
    code = "def add(a, b):\n    return a + b\n"
    _write(tmp_path / "module.py", code)

    analysis = analyze_dataset(tmp_path)

    assert analysis.dataset_type == "code"
    assert analysis.dominant_language == "code-heavy"


def test_classifies_mixed_dataset(tmp_path: Path) -> None:
    _write(tmp_path / "plain.txt", "Questo testo spiega il progetto in modo semplice.")
    _write(tmp_path / "app.py", "def greet(name):\n    return f'Hi {name}'\n")

    analysis = analyze_dataset(tmp_path)

    assert analysis.dataset_type == "mixed"


def test_returns_unknown_for_empty_folder(tmp_path: Path) -> None:
    analysis = analyze_dataset(tmp_path)

    assert analysis.dataset_type == "unknown"
    assert analysis.documents_scanned == 0
    assert analysis.estimated_tokens == 0


def test_duplicate_ratio_penalizes_quality(tmp_path: Path) -> None:
    duplicated = "Testo identico ripetuto per misurare i duplicati."
    _write(tmp_path / "a.txt", duplicated)
    _write(tmp_path / "b.txt", duplicated)

    analysis = analyze_dataset(tmp_path)

    assert analysis.duplicate_ratio >= 0.5
    assert analysis.quality_score < 80.0


def test_detect_dominant_language_italian() -> None:
    text = "il modello e la qualità dei dati sono importanti per il risultato finale"
    assert detect_dominant_language(text) == "italian"
