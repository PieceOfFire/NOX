"""Чтение спектров Bruker и алгоритмы коррекции базовой линии.

Интерфейс и построение графиков находятся в ``main.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_bruker_parameters(filename: Path) -> dict[str, str]:
    """Считать простые параметры JCAMP из файла Bruker ``procs``."""
    parameters: dict[str, str] = {}
    for line in filename.read_text(encoding="latin-1", errors="replace").splitlines():
        match = re.match(r"^##\$(\w+)=\s*(.*)$", line)
        if match:
            parameters[match.group(1)] = match.group(2).strip()
    return parameters


def find_processed_spectrum(folder: str | Path) -> tuple[Path, Path]:
    """Найти ``1r`` и ``procs`` в папке эксперимента либо обработки."""
    folder = Path(folder)
    candidates: list[Path] = [folder / "1r"]
    candidates += list((folder / "pdata").glob("*/1r"))
    candidates += list(folder.glob("*/1r"))
    candidates += list(folder.glob("*/pdata/*/1r"))
    valid = [(one_r, one_r.parent / "procs") for one_r in candidates
             if one_r.is_file() and (one_r.parent / "procs").is_file()]
    if not valid:
        raise FileNotFoundError("Не найден файл Bruker pdata/<номер>/1r вместе с procs.")
    return max(valid, key=lambda pair: int(pair[0].parent.name)
               if pair[0].parent.name.isdigit() else -1)


def load_bruker_spectrum(folder: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Path]:
    """Загрузить ppm, действительную/мнимую части, SW_p и путь к ``1r``."""
    one_r, procs_file = find_processed_spectrum(folder)
    p = read_bruker_parameters(procs_file)
    try:
        points = int(float(p["SI"]))
        offset_ppm = float(p["OFFSET"])
        spectral_width_hz = float(p["SW_p"])
        sf_mhz = float(p["SF"])
    except KeyError as error:
        raise ValueError(f"В {procs_file} нет параметра {error.args[0]}.") from error
    dtype_code = int(float(p.get("DTYPP", "0")))
    dtypes = {0: "i4", 1: "f4", 2: "f8"}
    if dtype_code not in dtypes:
        raise ValueError(f"Неподдерживаемый тип данных DTYPP={dtype_code}.")
    byte_order = "<" if int(float(p.get("BYTORDP", "0"))) == 0 else ">"
    dtype = np.dtype(byte_order + dtypes[dtype_code])
    real = np.fromfile(one_r, dtype=dtype, count=points)
    if real.size != points:
        raise ValueError(f"В 1r найдено {real.size} точек, ожидалось {points}.")
    imag_path = one_r.parent / "1i"
    imaginary = (np.fromfile(imag_path, dtype=dtype, count=points)
                 if imag_path.is_file() else np.zeros(points, dtype=dtype))
    if imaginary.size != points:
        imaginary = np.zeros(points, dtype=dtype)
    scale = 2.0 ** int(float(p.get("NC_proc", "0")))
    ppm = offset_ppm - np.arange(points) * spectral_width_hz / sf_mhz / points
    return ppm, real.astype(float) * scale, imaginary.astype(float) * scale, spectral_width_hz, one_r


def phase_correct(
    real: np.ndarray, imaginary: np.ndarray, phase0_deg: float, phase1_deg: float,
    pivot: float = .5,
) -> np.ndarray:
    """Применить нулевую и линейную фазовую коррекцию к 1r/1i.

    ``pivot`` — положение опорного пика для фазы первого порядка (0…1).
    """
    position = np.linspace(0., 1., real.size) - pivot
    phase = np.deg2rad(phase0_deg + phase1_deg * position)
    return np.real((real + 1j * imaginary) * np.exp(-1j * phase))


def lorentzian_component(
    x: np.ndarray, amplitude: float, center: float, half_width: float,
) -> np.ndarray:
    """Лоренциан с высотой ``amplitude`` и полушириной на полувысоте.

    Его полная аналитическая площадь равна ``pi * amplitude * half_width``.
    Отдельная функция нужна и для подгонки, и для прозрачного отображения
    результата в интерфейсе.
    """
    width = max(abs(float(half_width)), np.finfo(float).eps)
    return float(amplitude) * width ** 2 / ((np.asarray(x, dtype=float) - float(center)) ** 2 + width ** 2)


def fit_lorentzian_peak(ppm: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    """Аппроксимировать выделенный фрагмент одним лоренцианом и прямым фоном.

    Границы фрагмента используются только для *подгонки*. Возвращаемая
    ``area`` — полная площадь лоренциана от ``-inf`` до ``+inf``. Локальная
    прямая входит в модель исключительно для того, чтобы остаточный наклон
    базовой линии и хвост близкого пика не вошли в площадь выбранного пика.

    Функция допускает пики любого знака: знак хранится в ``amplitude``;
    вызывающий код решает, нужна ли площадь со знаком либо её модуль.
    """
    from scipy.optimize import least_squares

    x = np.asarray(ppm, dtype=float).ravel()
    y = np.asarray(signal, dtype=float).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 7:
        raise ValueError("Для аппроксимации Лоренца нужно не меньше 7 точек.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = float(x[-1] - x[0])
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("У выбранного диапазона должна быть ненулевая ширина.")

    midpoint = float((x[0] + x[-1]) / 2.0)
    coordinate = x - midpoint
    edge_count = max(2, min(x.size // 6, 31))
    left_x, right_x = float(np.mean(coordinate[:edge_count])), float(np.mean(coordinate[-edge_count:]))
    left_y, right_y = float(np.median(y[:edge_count])), float(np.median(y[-edge_count:]))
    slope0 = (right_y - left_y) / max(right_x - left_x, np.finfo(float).eps)
    intercept0 = (left_y + right_y) / 2.0 - slope0 * (left_x + right_x) / 2.0
    initial_background = intercept0 + slope0 * coordinate
    residual = y - initial_background
    peak_index = int(np.argmax(np.abs(residual)))
    amplitude0 = float(residual[peak_index])
    signal_scale = max(float(np.ptp(y)), float(np.max(np.abs(residual))), np.finfo(float).eps)
    if abs(amplitude0) <= signal_scale * 1e-8:
        return {
            "amplitude": 0.0, "center": float(x[peak_index]), "half_width": span / 8.0,
            "background_offset": intercept0, "background_slope": slope0,
            "background_center": midpoint, "area": 0.0,
        }

    spacing = float(np.median(np.diff(x)))
    min_width = max(spacing * .75, span / max(20.0 * x.size, 1.0))
    max_width = span * 2.0
    half_width0 = min(max(span / 8.0, min_width * 2.0), max_width * .8)
    amplitude_limit = max(abs(amplitude0), signal_scale) * 100.0
    # Точки в ядре линии получают больший вес. Обычный робастный fit считал
    # вершину сильного пика выбросом, подстраивался под многочисленные точки
    # в хвостах и поэтому заметно занижал высоту лоренциана. Вес ограничен,
    # чтобы узкий соседний пик всё же не перетянул модель целиком на себя.
    core_weight = 1.0 + 6.0 * np.clip(np.abs(residual) / signal_scale, 0.0, 1.0) ** 2

    def model(parameters: np.ndarray) -> np.ndarray:
        amplitude, center, half_width, background_offset, background_slope = parameters
        return (background_offset + background_slope * coordinate
                + lorentzian_component(x, amplitude, center, half_width))

    start = np.array([amplitude0, float(x[peak_index]), half_width0, intercept0, slope0], dtype=float)
    lower = np.array([-amplitude_limit, float(x[0]), min_width, -np.inf, -np.inf])
    upper = np.array([amplitude_limit, float(x[-1]), max_width, np.inf, np.inf])
    result = least_squares(
        lambda parameters: (model(parameters) - y) * core_weight, start, bounds=(lower, upper),
        loss="linear", x_scale="jac", max_nfev=1200,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError("Не удалось устойчиво аппроксимировать этот участок лоренцианом.")
    amplitude, center, half_width, background_offset, background_slope = map(float, result.x)
    return {
        "amplitude": amplitude,
        "center": center,
        "half_width": half_width,
        "background_offset": background_offset,
        "background_slope": background_slope,
        "background_center": midpoint,
        "area": float(np.pi * amplitude * half_width),
    }


def _phase_result(
    phase0: float, phase1: float, reliable: bool, return_reliable: bool,
) -> tuple[float, float] | tuple[float, float, bool]:
    """Form a public autophase result with an optional reliability flag."""
    if return_reliable:
        return float(phase0), float(phase1), bool(reliable)
    return float(phase0), float(phase1)


def auto_phase_parameters(
    real: np.ndarray,
    imaginary: np.ndarray,
    segment: slice | tuple[int, int] | None = None,
    pivot: float = .5,
    return_reliable: bool = False,
) -> tuple[float, float] | tuple[float, float, bool]:
    """Оценить нулевую и первую фазу для комплексного спектра 1r/1i.

    Возвращаемые ``(p0, p1)`` используются напрямую в :func:`phase_correct`:
    ``Re((1r + i*1i) * exp(-i * (p0 + p1 * position)))``.  Поэтому это
    *абсолютные*, а не добавочные, значения фазы.  ``pivot`` задаётся в той
    же нормированной шкале, что и в ``phase_correct``.

    Алгоритм намеренно не опирается на знак пиков. Сначала он находит
    выраженные максимумы модуля и объединяет близкие максимумы в области
    мультиплетов. Для каждой области берётся комплексная площадь после
    вычитания локальной линейной базовой линии: так дисперсионные хвосты
    взаимно компенсируются. Удвоение фазы площадей делает положительные и
    отрицательные абсорбционные пики эквивалентными, что важно для
    гиперполяризованных и двухполярных спектров.

    ``segment`` можно передать как обычный ``slice`` либо пару индексов
    ``(start, stop)``.  Расчёт тогда использует только видимый участок, но
    фаза первого порядка всё равно относится ко всей исходной оси спектра.
    Функция использует только NumPy и безопасно возвращает ``(0, 0)``, если
    квадратурный канал отсутствует, сигнал недостаточен или наклон нельзя
    оценить устойчиво. При ``return_reliable=True`` третьим элементом
    возвращается признак надёжности; в таком случае интерфейс сохраняет
    текущую ручную фазу при ``False``.
    If ``return_reliable`` is true, a third boolean is returned.  ``False``
    means that the safe ``(0, 0)`` fallback must not replace a phase already
    chosen by the user.
    """
    real = np.asarray(real, dtype=float).ravel()
    imaginary = np.asarray(imaginary, dtype=float).ravel()
    if real.size != imaginary.size:
        raise ValueError("Каналы 1r и 1i должны иметь одинаковую длину.")
    if real.size < 8:
        return _phase_result(0.0, 0.0, False, return_reliable)
    if not np.isfinite(pivot):
        raise ValueError("Опорная позиция фазы должна быть конечным числом.")

    n = real.size
    if segment is None:
        start, stop = 0, n
    elif isinstance(segment, slice):
        start, stop, step = segment.indices(n)
        if step != 1:
            # Фазировка по прореженной выборке искажает координату p1.
            indices = np.arange(start, stop, step, dtype=int)
            if indices.size < 8:
                return _phase_result(0.0, 0.0, False, return_reliable)
            return _auto_phase_from_samples(
                real[indices], imaginary[indices], indices, n, pivot,
                return_reliable=return_reliable,
            )
    else:
        if len(segment) != 2:
            raise ValueError("segment должен быть slice или парой (start, stop).")
        start, stop = int(segment[0]), int(segment[1])
        start, stop, _ = slice(start, stop).indices(n)
    if stop - start < 8:
        return _phase_result(0.0, 0.0, False, return_reliable)
    indices = np.arange(start, stop, dtype=int)
    return _auto_phase_from_samples(
        real[start:stop], imaginary[start:stop], indices, n, pivot,
        return_reliable=return_reliable,
    )


def _auto_phase_from_samples(
    real: np.ndarray,
    imaginary: np.ndarray,
    indices: np.ndarray,
    total_points: int,
    pivot: float,
    *,
    return_reliable: bool = False,
) -> tuple[float, float] | tuple[float, float, bool]:
    """Внутренняя NumPy-реализация :func:`auto_phase_parameters`."""
    z = np.asarray(real, dtype=float) + 1j * np.asarray(imaginary, dtype=float)
    finite = np.isfinite(z.real) & np.isfinite(z.imag)
    if finite.sum() < 8:
        return _phase_result(0.0, 0.0, False, return_reliable)
    z = z[finite]
    indices = np.asarray(indices, dtype=int)[finite]
    magnitude = np.abs(z)
    energy = float(np.sqrt(np.mean(magnitude ** 2)))
    if not np.isfinite(energy) or energy <= np.finfo(float).eps:
        return _phase_result(0.0, 0.0, False, return_reliable)

    # Без реального квадратурного сигнала фазу определить невозможно.  Это
    # нормальный случай для папок, где Bruker сохранил только 1r.
    imag_energy = float(np.sqrt(np.mean(np.asarray(imaginary, dtype=float)[finite] ** 2)))
    if imag_energy <= energy * 1e-8:
        return _phase_result(0.0, 0.0, False, return_reliable)

    length = magnitude.size
    # Небольшое сглаживание служит только для поиска центров линий. Саму фазу
    # читаем из исходного комплексного значения в центре максимума.
    window = min(17, max(3, (length // 2048) * 2 + 3))
    if window % 2 == 0:
        window += 1
    smooth = np.convolve(magnitude, np.full(window, 1.0 / window), mode="same")
    candidates = np.flatnonzero(
        (smooth[1:-1] >= smooth[:-2]) & (smooth[1:-1] > smooth[2:])
    ) + 1
    if candidates.size == 0:
        return _phase_result(0.0, 0.0, False, return_reliable)

    # Убираем шумовые максимумы. Разность соседних точек даёт устойчивую
    # оценку шума даже при медленном наклоне базовой линии.
    differences = np.diff(np.concatenate((real[finite], imaginary[finite])))
    noise = 1.4826 * float(np.median(np.abs(differences - np.median(differences))))
    noise = max(noise, np.finfo(float).eps * energy)
    median_smooth = float(np.median(smooth))
    # Относительный порог особенно важен: при разреженном спектре обычный
    # ``median + k*noise`` всё ещё оставляет сотни максимумов в хвостах
    # сильной линии, а их случайная фаза портит оценку p1.
    dynamic = max(float(np.quantile(smooth, .999) - median_smooth), 8.0 * noise)
    floor = median_smooth + max(8.0 * noise, .03 * dynamic)
    candidates = candidates[smooth[candidates] >= floor]
    # Локальный максимум на наклонном хвосте сильной линии — не отдельный
    # резонанс. Отсекаем такие шумовые «бугорки» по локальной prominence.
    # Радиус существенно шире окна сглаживания, но остаётся меньше типичной
    # дистанции между независимыми NMR-сигналами.
    prominence_radius = max(window * 4, length // 128)
    if candidates.size:
        prominences = np.empty(candidates.size)
        for number, candidate in enumerate(candidates):
            left = smooth[max(0, candidate - prominence_radius):candidate]
            right = smooth[candidate + 1:min(length, candidate + prominence_radius + 1)]
            shoulder = max(float(np.min(left)) if left.size else smooth[candidate],
                           float(np.min(right)) if right.size else smooth[candidate])
            prominences[number] = smooth[candidate] - shoulder
        candidates = candidates[prominences >= max(3.0 * noise, .01 * dynamic)]
    if candidates.size == 0:
        return _phase_result(0.0, 0.0, False, return_reliable)

    # Оставляем наиболее выразительные и пространственно разнесённые линии.
    # Сами вершины линий дальше *не* используются для чтения фазы: в
    # мультиплете максимум |z| часто лежит на дисперсионном плече соседней
    # линии. Именно это было причиной случайного φ1 у сильных мультиплетов.
    order = candidates[np.argsort(smooth[candidates])[::-1]]
    separation = max(3, length // 512)
    selected: list[int] = []
    for candidate in order:
        if all(abs(candidate - previous) >= separation for previous in selected):
            selected.append(int(candidate))
            if len(selected) >= 96:
                break
    if not selected:
        return _phase_result(0.0, 0.0, False, return_reliable)
    peak_indices = np.asarray(selected, dtype=int)

    # Сглаженный максимум иногда сдвинут на несколько отсчётов. Уточняем
    # координаты только для формирования областей сигнала, а не для оценки
    # фазы одной точки.
    radius = max(1, window // 2)
    centres: list[int] = []
    for candidate in peak_indices:
        left, right = max(0, candidate - radius), min(length, candidate + radius + 1)
        centres.append(left + int(np.argmax(magnitude[left:right])))
    peak_indices = np.unique(np.asarray(centres, dtype=int))
    if peak_indices.size == 0:
        return _phase_result(0.0, 0.0, False, return_reliable)

    # Близкие вершины принадлежат одному мультиплету. Вместо фазы в каждой
    # вершине берём комплексную площадь всей области. У абсорбционной линии
    # дисперсионная часть при таком интегрировании взаимно компенсируется,
    # поэтому фаза площади значительно устойчивее к перекрытию линий.
    cluster_gap = max(8, total_points // 128)
    clusters: list[list[int]] = []
    for centre in np.sort(peak_indices):
        if not clusters or centre - clusters[-1][-1] > cluster_gap:
            clusters.append([int(centre)])
        else:
            clusters[-1].append(int(centre))

    # Около 80 отсчётов для SI=16k: достаточно, чтобы захватить хвосты линии,
    # но соседние, разнесённые мультиплеты не смешиваются. Проверка ниже также
    # сравнивает результаты для нескольких таких окон.
    default_pad = max(24, int(round(total_points / 205)))

    def regional_areas(pad: int) -> tuple[np.ndarray, np.ndarray]:
        areas: list[complex] = []
        positions: list[float] = []
        for cluster in clusters:
            first, last = cluster[0], cluster[-1]
            begin = max(0, first - pad)
            end = min(length, last + pad + 1)
            if end - begin < 3:
                continue

            # Комплексная линейная базовая линия из соседних тихих участков.
            # Среднее по симметричным окнам сохраняет компенсацию
            # дисперсионных хвостов; ``np.median`` для complex сортирует
            # значения не по фазе и даёт скачки φ1 при смещении окна на 1
            # точку.
            left_samples = z[max(0, begin - pad):begin]
            right_samples = z[end:min(length, end + pad)]
            left_base = np.mean(left_samples) if left_samples.size else 0j
            right_base = np.mean(right_samples) if right_samples.size else left_base
            baseline = np.linspace(left_base, right_base, end - begin)
            area = complex(np.sum(z[begin:end] - baseline))
            areas.append(area)
            # Координата центра группы, а не максимум отдельной линии.
            positions.append(float(np.mean(cluster)) / max(total_points - 1, 1) - float(pivot))
        return np.asarray(areas, dtype=complex), np.asarray(positions, dtype=float)

    def phase_fit(areas: np.ndarray, positions: np.ndarray) -> tuple[float, float, float]:
        """Вернуть φ0, φ1 и согласованность круговой регрессии."""
        if areas.size == 0:
            return 0.0, 0.0, 0.0
        # Слабые области несут в основном шум. Корень из площади делает вес
        # сильного пика заметным, но не позволяет ему скрыть остальные.
        strongest = float(np.max(np.abs(areas)))
        keep = np.abs(areas) >= strongest * .03 if strongest > 0.0 else np.ones(areas.size, dtype=bool)
        areas = areas[keep]
        positions = positions[keep]
        if areas.size == 0:
            return 0.0, 0.0, 0.0
        weights = np.sqrt(np.abs(areas))
        median_weight = float(np.median(weights[weights > 0])) if np.any(weights > 0) else 1.0
        weights = np.clip(weights, 0.0, median_weight * 5.0)
        if not np.any(weights > 0):
            weights = np.ones(areas.size)
        def circular_fit(fit_areas: np.ndarray, fit_positions: np.ndarray,
                         fit_weights: np.ndarray) -> tuple[float, float, float]:
            """Одна взвешенная регрессия фаз площадей."""
            doubled_phase = np.exp(2j * np.angle(fit_areas))
            # Для φ1 перебираем только физически разумный диапазон. В отличие
            # от фазы нулевого порядка, φ1+360° не эквивалентно φ1 на всей оси.
            p1_grid = np.arange(-360.0, 360.001, .1)
            correlation = np.exp(-2j * np.outer(np.deg2rad(p1_grid), fit_positions)) @ (
                fit_weights * doubled_phase
            )
            best = int(np.argmax(np.abs(correlation)))
            total_weight = float(np.sum(fit_weights))
            coherence = float(np.abs(correlation[best]) / total_weight) if total_weight else 0.0
            return (float(np.rad2deg(.5 * np.angle(correlation[best]))),
                    float(p1_grid[best]), coherence)

        p0, p1, coherence = circular_fit(areas, positions, weights)
        # Небольшой паразитный максимум (обычно на дисперсионном плече очень
        # сильной линии) иногда даёт комплексную площадь почти под 90°. Раньше
        # один такой выброс полностью отменял автофазировку: именно так было
        # со спектром 11. Удаляем только явно чужие фазы и только когда первая
        # регрессия уже показала недостаточную согласованность. Хорошие
        # спектры, в том числе с законной слабой линией, не меняются.
        if areas.size >= 4 and coherence < .90:
            expected = np.deg2rad(p0 + p1 * positions)
            residual = .5 * np.rad2deg(np.angle(np.exp(2j * (np.angle(areas) - expected))))
            absolute_residual = np.abs(residual)
            median_residual = float(np.median(absolute_residual))
            robust_spread = 1.4826 * float(np.median(np.abs(absolute_residual - median_residual)))
            # 20° — намеренно консервативная нижняя граница: линию не следует
            # отбрасывать лишь из-за небольшой ошибки локальной базы.
            limit = max(20.0, median_residual + 5.0 * robust_spread)
            inliers = absolute_residual <= limit
            if 3 <= int(np.count_nonzero(inliers)) < areas.size:
                p0, p1, coherence = circular_fit(areas[inliers], positions[inliers], weights[inliers])
        return p0, p1, coherence

    areas, positions = regional_areas(default_pad)
    if areas.size == 0:
        return _phase_result(0.0, 0.0, False, return_reliable)
    p0_deg, p1_deg, coherence = phase_fit(areas, positions)

    # Результат φ1 применяется только при доказуемой устойчивости. Иначе
    # первая фаза оставляется нулевой: это безопаснее, чем заметно исказить
    # весь спектр произвольным наклоном. Проверяем число независимых областей,
    # их разнос, согласованность и чувствительность к ширине окна/одной
    # исключённой области.
    strongest = float(np.max(np.abs(areas)))
    reliable = np.abs(areas) >= strongest * .03 if strongest > 0.0 else np.zeros(areas.size, dtype=bool)
    fit_areas = areas[reliable]
    fit_positions = positions[reliable]
    stable_p1: list[float] = []
    for pad in sorted({max(24, default_pad - 16), default_pad, default_pad + 16}):
        test_areas, test_positions = regional_areas(pad)
        test_p0, test_p1, test_coherence = phase_fit(test_areas, test_positions)
        if test_areas.size:
            test_strongest = float(np.max(np.abs(test_areas)))
            test_keep = np.abs(test_areas) >= test_strongest * .03 if test_strongest > 0.0 else np.zeros(test_areas.size, dtype=bool)
            if int(np.count_nonzero(test_keep)) >= 3 and test_coherence >= .90:
                stable_p1.append(test_p1)
    for omitted in range(fit_areas.size):
        if fit_areas.size <= 3:
            break
        _, leave_one_out_p1, leave_one_out_coherence = phase_fit(
            np.delete(fit_areas, omitted), np.delete(fit_positions, omitted)
        )
        if leave_one_out_coherence >= .90:
            stable_p1.append(leave_one_out_p1)

    if (fit_areas.size < 3 or np.ptp(fit_positions) < .12 or coherence < .90
            or len(stable_p1) < 3 or np.ptp(stable_p1) > 30.0):
        # Если наклон не доказан, нельзя доверять и фазе, полученной из тех же
        # областей: на антифазных/гиперполяризованных спектрах она способна
        # повернуть уже хороший сигнал в дисперсию. Безопасный результат
        # автофазы в такой ситуации — ничего не менять; φ0 пользователь при
        # необходимости по-прежнему может выставить вручную по красному пику.
        return _phase_result(0.0, 0.0, False, return_reliable)

    # Удвоенная фаза не различает пики, отличающиеся знаком. Не принуждаем
    # все пики быть положительными: сохраняем знак реальной части в опорном
    # пике, выбранном в диалоге фазирования. Это важно для гиперполяризации.
    reference_index = int(round(float(pivot) * max(total_points - 1, 1)))
    nearest = int(np.argmin(np.abs(indices - reference_index)))
    if abs(real[finite][nearest]) <= noise:
        nearest = int(np.argmax(magnitude))
    reference_position = indices[nearest] / max(total_points - 1, 1) - float(pivot)
    corrected_reference = float(np.real(z[nearest] * np.exp(-1j * np.deg2rad(
        p0_deg + p1_deg * reference_position
    ))))
    if float(real[finite][nearest]) * corrected_reference < 0.0:
        p0_deg += 180.0

    # Удобный для интерфейса канонический диапазон p0. При сдвиге p0 на 360°
    # результат phase_correct не меняется.
    p0_deg = (p0_deg + 180.0) % 360.0 - 180.0
    return _phase_result(p0_deg, p1_deg, True, return_reliable)


def whittaker_lambda(filter_hz: float, smoothness: float, spectral_width_hz: float) -> float:
    """Перевести параметры интерфейса в λ с калибровкой шкалы TopSpin."""
    if not 0 < filter_hz < spectral_width_hz / 2:
        raise ValueError(f"Фильтр должен быть в пределах 0…{spectral_width_hz / 2:g} Гц.")
    if smoothness <= 0:
        raise ValueError("Коэффициент сглаживания должен быть положительным.")
    return smoothness * (10 * spectral_width_hz / 512 / filter_hz) ** 4


def auto_parameters(signal: np.ndarray, spectral_width_hz: float) -> tuple[float, float]:
    """Автонастройки, согласованные с фильтром Уиттакера Mestrenova.

    Частота фильтра вычисляется из ширины спектра, а не подставляется как
    константа: ``SW_p / 512``. Поэтому для 13 получается 11.74 Гц, а для
    10 с меньшей шириной — около 8.07 Гц. Коэффициент 16384 — стандартная
    автонастройка Mestrenova для этого фильтра.
    """
    if signal.size < 3:
        raise ValueError("Для коррекции нужны хотя бы 3 точки.")
    return spectral_width_hz / 512, 16384.


def peak_polarity(signal: np.ndarray) -> int:
    """Вернуть +1, -1 или 0 для положительных, отрицательных и двуполярных пиков."""
    finite = signal[np.isfinite(signal)]
    if finite.size < 3:
        return 1
    low, high = np.quantile(finite, (.01, .99))
    if high <= low:
        center = float(np.median(finite))
    else:
        counts, edges = np.histogram(finite, bins=128, range=(low, high))
        index = int(np.argmax(counts))
        center = float((edges[index] + edges[index + 1]) / 2)
    # Гиперполяризованный пик может занимать существенно меньше 0.5 % точек.
    # Поэтому берём почти экстремальные отклонения, а не прежние 99.5/0.5 %
    # квантили, которые пропускали узкий сильный отрицательный пик.
    positive = float(np.quantile(finite, .9998) - center)
    negative = float(center - np.quantile(finite, .0002))
    # Если обе полярности заметны, нельзя выбирать верхний/нижний конверт:
    # оба вида линий должны быть исключены из оценки базовой линии.
    if min(positive, negative) > .15 * max(positive, negative, np.finfo(float).eps):
        return 0
    return -1 if negative > positive else 1


def whittaker_baseline(
    signal: np.ndarray, lam: float, asymmetry: float = .001, iterations: int = 15,
    polarity: int | str = "auto",
) -> np.ndarray:
    """Асимметричный фильтр Уиттакера для пиков любого знака.

    При ``polarity='auto'`` выбирается нижний конверт для обычного спектра
    и верхний конверт для гиперполяризованного спектра с отрицательными пиками.
    Для двуполярного спектра пики подавляются с обеих сторон робастными весами.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    if signal.size < 3 or lam <= 0:
        raise ValueError("Недопустимый сигнал или параметр сглаживания.")
    direction = peak_polarity(signal) if polarity == "auto" else int(polarity)
    if direction not in (-1, 0, 1):
        raise ValueError("Полярность должна быть +1, 0, -1 или 'auto'.")

    def solve(data: np.ndarray, local_lam: float, p: float) -> np.ndarray:
        n = data.size
        d = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
                  [0, 1, 2], shape=(n - 2, n), format="csc")
        result = data.copy()
        penalty = local_lam * (d.T @ d)
        for _ in range(iterations):
            if direction == 0:
                # Двусторонняя робастная оценка: подавляет как положительные,
                # так и отрицательные пики/дисперсионные компоненты.
                residual = data - result
                scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
                scale = max(float(scale), np.finfo(float).eps)
                weights = 1 / (1 + (residual / (3 * scale)) ** 2)
            else:
                # Сильные пики получают малый вес: сверху для обычного
                # спектра и снизу для гиперполяризованного.
                is_peak = data > result if direction > 0 else data < result
                weights = np.where(is_peak, p, 1 - p)
            result = spsolve(diags(weights, format="csc") + penalty, weights * data)
        return result

    n = signal.size
    padding = min(2048, max(32, n // 16))
    baseline = solve(np.pad(signal, padding, mode="reflect"), lam, asymmetry)[padding:padding + n]
    edge = min(n, 1024, max(128, n // 32))
    blend = np.linspace(1., 0., edge)
    local_lam = max(lam / 1e4, 1.)
    baseline[:edge] = blend * solve(signal[:edge], local_lam, max(asymmetry, .05)) + (1 - blend) * baseline[:edge]
    baseline[-edge:] = blend[::-1] * solve(signal[-edge:], local_lam, max(asymmetry, .05)) + (1 - blend[::-1]) * baseline[-edge:]
    return baseline


def arpls_baseline(
    signal: np.ndarray,
    *,
    total_points: int | None = None,
    polarity: int | str = "auto",
    iterations: int = 40,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Автоматическая базовая линия методом adaptive reweighted PLS (arPLS).

    В отличие от обычного асимметричного Whittaker, arPLS на каждой итерации
    оценивает статистику отрицательных отклонений от текущей линии и по ней
    автоматически уменьшает вес пиков. Поэтому линия лучше следует широкому
    неровному фону, но остаётся устойчивой к узким интенсивным резонансам.
    ``total_points`` передаётся для сохранения одного физического масштаба
    сглаживания при коррекции только увеличенного участка спектра.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size < 16 or not np.isfinite(signal).all():
        raise ValueError("Для arPLS нужен одномерный спектр без пропусков (минимум 16 точек).")
    direction = peak_polarity(signal) if polarity == "auto" else int(polarity)
    if direction not in (-1, 0, 1):
        raise ValueError("Полярность должна быть +1, 0, -1 или 'auto'.")

    # Консервативная настройка для стандартного Bruker 1r с SI=16384.
    # Более жёсткий штраф не даёт arPLS принять широкие основания реальных
    # сигналов за фон и тем самым «съесть» их интегральную интенсивность.
    # Для более плотной цифровой сетки λ растёт как четвёртая степень шага.
    points = max(int(total_points or signal.size), 512)
    lam = max(1e5, 3e7 * (points / 16384.0) ** 4)

    def fit_lower(data: np.ndarray) -> np.ndarray:
        n = data.size
        difference = diags(
            [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
            [0, 1, 2], shape=(n - 2, n), format="csc",
        )
        penalty = lam * (difference.T @ difference)
        weights = np.ones(n)
        result = data.copy()
        for _ in range(iterations):
            result = spsolve(diags(weights, format="csc") + penalty, weights * data)
            residual = data - result
            negative = residual[residual < 0]
            if negative.size < max(8, n // 200):
                break
            mean = float(np.mean(negative))
            deviation = max(float(np.std(negative)), np.finfo(float).eps)
            argument = np.clip(2 * (residual - (2 * deviation - mean)) / deviation, -100, 100)
            next_weights = 1 / (1 + np.exp(argument))
            if np.linalg.norm(weights - next_weights) / max(np.linalg.norm(weights), np.finfo(float).eps) < tolerance:
                result = spsolve(diags(next_weights, format="csc") + penalty, next_weights * data)
                break
            weights = next_weights
        return np.asarray(result, dtype=float)

    if direction == 0:
        # У двуполярного гиперполяризованного сигнала нельзя однозначно
        # выбрать нижнюю сторону. Оставляем симметричную робастную оценку.
        return whittaker_baseline(signal, lam, polarity=0)

    padding = min(2048, max(32, signal.size // 16))
    if direction < 0:
        result = -fit_lower(np.pad(-signal, padding, mode="reflect"))
    else:
        result = fit_lower(np.pad(signal, padding, mode="reflect"))
    return result[padding:padding + signal.size]


def spline_baseline(signal: np.ndarray, polarity: int | str = "auto") -> np.ndarray:
    """Полностью автоматическая робастная базовая линия на spline.

    Сплайн получает редкие опорные точки из нижней части спектра и сильно
    сглаживается, поэтому он повторяет медленную базу, а не NMR-пики.
    """
    from scipy.interpolate import UnivariateSpline
    if signal.size < 16:
        raise ValueError("Для spline-коррекции нужны как минимум 16 точек.")
    # Около 64 опорных блоков на спектр; это намеренно гораздо реже пиков.
    width = max(64, signal.size // 64)
    starts = np.arange(0, signal.size, width)
    x = np.array([(start + min(start + width, signal.size) - 1) / 2 for start in starts])
    direction = peak_polarity(signal) if polarity == "auto" else int(polarity)
    if direction not in (-1, 0, 1):
        raise ValueError("Полярность должна быть +1, 0, -1 или 'auto'.")
    quantile = .10 if direction > 0 else .90 if direction < 0 else .50
    y = np.array([np.quantile(signal[start:min(start + width, signal.size)], quantile) for start in starts])
    if x.size < 4:
        raise ValueError("Для spline нужны 4 опорные точки.")
    noise = np.median(np.abs(np.diff(signal) - np.median(np.diff(signal)))) / .6745 / np.sqrt(2)
    budget = 100 * x.size * max(float(noise), np.finfo(float).eps) ** 2
    return UnivariateSpline(x, y, k=3, s=budget, ext="extrapolate")(np.arange(signal.size))


if __name__ == "__main__":
    # Совместимость с прежним способом запуска: интерфейс живёт в main.py,
    # но команда ``python base_line.py`` по-прежнему открывает окно.
    from main import SpectrumApp
    SpectrumApp().mainloop()
