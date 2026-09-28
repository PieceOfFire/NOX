"""Сериализация проектов NOX и уже обработанных массивов спектра.

Здесь намеренно нет Tkinter: модуль можно проверить отдельно от интерфейса
и использовать позднее для пакетной обработки.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_FORMAT = "nmr-analysis-project"


def encode_signal(signal: np.ndarray | None) -> dict[str, object] | None:
    """Упаковать вещественный массив в JSON-совместимую запись."""
    if signal is None:
        return None
    values = np.asarray(signal, dtype="<f8")
    return {
        "dtype": "float64-le",
        "size": int(values.size),
        "data": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def decode_signal(payload: object, expected_size: int) -> np.ndarray | None:
    """Восстановить массив и проверить, что он относится к тому же спектру."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("неверный формат сохранённого скорректированного спектра")
    try:
        if payload.get("dtype") not in (None, "float64-le"):
            raise ValueError("неподдерживаемый тип сохранённого спектра")
        size = int(payload["size"])
        raw = base64.b64decode(str(payload["data"]), validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("не удалось прочитать сохранённый скорректированный спектр") from error
    values = np.frombuffer(raw, dtype="<f8").copy()
    if size != expected_size or values.size != expected_size:
        raise ValueError("размер сохранённого спектра не совпадает с исходным 1r")
    return values


def write_project(filename: Path, project: dict[str, object]) -> None:
    """Записать проект в компактный UTF-8 JSON."""
    with filename.open("w", encoding="utf-8") as file:
        json.dump(project, file, ensure_ascii=False, separators=(",", ":"))


def read_project(filename: str | Path) -> dict[str, Any]:
    """Прочитать проект и отсеять произвольные JSON-файлы."""
    with Path(filename).open(encoding="utf-8") as file:
        project = json.load(file)
    if not isinstance(project, dict) or project.get("format") != PROJECT_FORMAT:
        raise ValueError("это не проект NOX")
    return project
