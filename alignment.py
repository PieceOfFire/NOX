"""Алгоритмы совмещения 1D ЯМР-спектров по химическому сдвигу."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class AlignmentError(ValueError):
    """Спектры не содержат достаточно общей информации для совмещения."""


@dataclass(frozen=True)
class AlignmentResult:
    """Результат поиска постоянного сдвига по X."""

    shift: float
    correlation: float
    points: int
    resolution: float


def _sorted_finite(ppm: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(ppm, dtype=float).ravel()
    y = np.asarray(signal, dtype=float).ravel()
    if x.size != y.size:
        raise AlignmentError("ось ppm и сигнал имеют разную длину")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 20:
        raise AlignmentError("для выравнивания нужно не меньше 20 точек")
    order = np.argsort(x)
    return x[order], y[order]


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Сглаживание без SciPy; край отражается, чтобы не создавать ложный пик."""
    width = max(3, int(width) | 1)
    if width >= values.size:
        return np.full_like(values, float(np.median(values)))
    padded = np.pad(values, width // 2, mode="reflect")
    return np.convolve(padded, np.full(width, 1.0 / width), mode="valid")


def _feature_signal(values: np.ndarray) -> np.ndarray:
    """Оставить форму пиков, устранив уровень, наклон и единицы интенсивности."""
    short = _moving_average(values, min(9, max(3, values.size // 900 * 2 + 1)))
    background_width = min(401, max(31, values.size // 32 * 2 + 1))
    feature = short - _moving_average(short, background_width)
    low, high = np.percentile(feature, (1.0, 99.0))
    if not np.isfinite(low + high) or np.isclose(low, high):
        raise AlignmentError("на выбранном участке нет различимой структуры пиков")
    feature = np.clip(feature, low, high)
    feature -= float(np.mean(feature))
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise AlignmentError("на выбранном участке нет различимой структуры пиков")
    return feature / norm


def _correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    candidate = candidate - float(np.mean(candidate))
    norm = float(np.linalg.norm(candidate))
    if norm <= np.finfo(float).eps or not np.isfinite(norm):
        return float("-inf")
    return float(np.dot(reference, candidate / norm))


def find_x_shift(
    reference_ppm: np.ndarray,
    reference_signal: np.ndarray,
    target_ppm: np.ndarray,
    target_signal: np.ndarray,
    xmin: float,
    xmax: float,
    *,
    max_shift: float = 0.05,
    resolution: float = 0.0001,
) -> AlignmentResult:
    """Найти добавку ``offset`` для отображения ``target_ppm + offset``.

    Сопоставляются только видимые данные между ``xmin`` и ``xmax``. Перед
    корреляцией вычитается локальный фон и нормируется амплитуда, поэтому
    метод устойчивее к разной концентрации и масштабу двух спектров.
    """
    if not (np.isfinite(xmin) and np.isfinite(xmax) and max_shift > 0 and resolution > 0):
        raise AlignmentError("неверные границы или точность поиска")
    ref_x, ref_y = _sorted_finite(reference_ppm, reference_signal)
    target_x, target_y = _sorted_finite(target_ppm, target_signal)
    lower, upper = sorted((float(xmin), float(xmax)))

    # Оставляем пространство по краям, чтобы каждый пробный сдвиг использовал
    # одни и те же точки и не получал преимущество из-за границы интерполяции.
    lower = max(lower, float(ref_x[0]), float(target_x[0]) + max_shift)
    upper = min(upper, float(ref_x[-1]), float(target_x[-1]) - max_shift)
    if upper <= lower:
        raise AlignmentError("у спектров нет общего участка для выбранного диапазона")

    native_step = min(
        float(np.median(np.diff(ref_x))),
        float(np.median(np.diff(target_x))),
    )
    if not np.isfinite(native_step) or native_step <= 0:
        raise AlignmentError("не удалось определить шаг оси ppm")
    # 6000 точек достаточно для поиска с шагом 10⁻⁴ ppm, но не подвешивает
    # интерфейс при выравнивании целой серии спектров.
    count = int(np.clip(round((upper - lower) / native_step) + 1, 256, 6_000))
    grid = np.linspace(lower, upper, count)
    reference = _feature_signal(np.interp(grid, ref_x, ref_y))
    # Выделение формы пиков инвариантно к сдвигу. Поэтому преобразуем target
    # один раз на расширенной оси, а на каждом пробном шаге только сдвигаем
    # уже готовый массив. Для длинных 1r это быстрее на порядки.
    grid_step = (upper - lower) / max(count - 1, 1)
    extended = np.arange(lower - max_shift, upper + max_shift + grid_step * .5, grid_step)
    target_feature = _feature_signal(np.interp(extended, target_x, target_y))

    candidates = np.arange(-max_shift, max_shift + resolution * .51, resolution)
    scores = np.empty(candidates.size, dtype=float)
    for index, shift in enumerate(candidates):
        # После отображения p -> p + shift значение в координате p берётся
        # из исходной точки p - shift.
        target = np.interp(grid - shift, extended, target_feature)
        scores[index] = _correlation(reference, target)

    best = int(np.argmax(scores))
    score = float(scores[best])
    if not np.isfinite(score) or score < .20:
        raise AlignmentError("пики на выбранном участке недостаточно похожи для автовыравнивания")
    if best in (0, len(candidates) - 1):
        raise AlignmentError(
            f"нужен сдвиг больше ±{max_shift:.3f} ppm; увеличьте диапазон поиска вручную"
        )

    # Парабола по трём соседним точкам даёт более плавный результат, чем
    # жёсткая сетка 0.0001 ppm, но не позволяет уйти дальше одной ячейки.
    left, center, right = scores[best - 1:best + 2]
    denominator = left - 2.0 * center + right
    correction = 0.0 if abs(denominator) < 1e-12 else .5 * (left - right) / denominator
    correction = float(np.clip(correction, -1.0, 1.0))
    shift = float(candidates[best] + correction * resolution)
    return AlignmentResult(shift=shift, correlation=score, points=count, resolution=resolution)
