"""Модальные окна интерфейса NOX.

Окна получают ссылку на SpectrumApp, но не зависят от его реализации при импорте.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from base_line import (
    arpls_baseline, auto_parameters, auto_phase_parameters, peak_polarity,
    phase_correct, spline_baseline, whittaker_baseline, whittaker_lambda,
)

if TYPE_CHECKING:
    from main import SpectrumApp

class ShiftSpectrumDialog(tk.Toplevel):
    """Точная интерактивная подстройка положения записи на графике сравнения."""

    FINE_STEP = .0001
    SLIDER_SPAN = .05

    def __init__(self, app: SpectrumApp, comparison: dict[str, object]) -> None:
        super().__init__(app)
        self.app = app
        self.comparison = comparison
        self.title("Сдвиг спектра по X")
        self.transient(app)
        self.grab_set()
        self.resizable(False, False)

        self.spectrum_ids = [int(value) for value in comparison.get("spectrum_ids", [])
                             if int(value) in app.spectra]
        self.original_offsets = {
            spectrum_id: float(comparison.get("offsets", {}).get(spectrum_id, 0.0))
            for spectrum_id in self.spectrum_ids
        }
        initial_id = (app.active_spectrum_id if app.active_spectrum_id in self.spectrum_ids
                      else self.spectrum_ids[0])
        self.selected_id = tk.IntVar(value=initial_id)
        self.offset = tk.DoubleVar()
        self.value_text = tk.StringVar()
        self._updating = False

        frame = tk.Frame(self, padx=14, pady=12)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="Спектр:").pack(anchor="w")
        self.spectrum_box = ttk.Combobox(
            frame, state="readonly", width=42,
            values=[str(app.spectra[spectrum_id]["name"]) for spectrum_id in self.spectrum_ids],
        )
        self.spectrum_box.pack(fill="x", pady=(2, 12))
        self.spectrum_box.current(self.spectrum_ids.index(initial_id))
        self.spectrum_box.bind("<<ComboboxSelected>>", self.select_spectrum)

        tk.Label(frame, text="Точная подстройка химического сдвига", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        tk.Label(frame, textvariable=self.value_text, fg="#185fa5", font=("TkDefaultFont", 12, "bold")).pack(pady=(3, 4))
        self.slider = tk.Scale(
            frame, orient="horizontal", showvalue=False, resolution=self.FINE_STEP,
            length=430, variable=self.offset, command=self.slide,
            highlightthickness=0,
        )
        self.slider.pack(fill="x")
        self.slider.bind("<Left>", lambda _event: self.nudge(-self.FINE_STEP))
        self.slider.bind("<Right>", lambda _event: self.nudge(self.FINE_STEP))
        self.slider.bind("<MouseWheel>", self.on_mousewheel)
        tk.Label(
            frame,
            text="Перетаскивайте бегунок для грубой настройки. Стрелки клавиатуры и колесо — шаг 0.0001 ppm.",
            justify="left", fg="#68707d", wraplength=430,
        ).pack(anchor="w", pady=(4, 0))

        fine = tk.Frame(frame)
        fine.pack(pady=(10, 0))
        for text, delta in (("−0.001", -.001), ("−0.0001", -self.FINE_STEP),
                            ("+0.0001", self.FINE_STEP), ("+0.001", .001)):
            tk.Button(fine, text=text, width=9, command=lambda value=delta: self.nudge(value)).pack(
                side="left", padx=2
            )
        tk.Button(frame, text="Центрировать бегунок", command=self.recenter).pack(pady=(8, 0))

        buttons = tk.Frame(frame)
        buttons.pack(fill="x", pady=(12, 0))
        tk.Button(buttons, text="Готово", command=self.accept).pack(side="right")
        tk.Button(buttons, text="Отмена", command=self.cancel).pack(side="right", padx=(0, 6))
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.configure_slider()
        self.slider.focus_set()

    def current_id(self) -> int:
        return self.spectrum_ids[self.spectrum_box.current()]

    def configure_slider(self) -> None:
        spectrum_id = self.current_id()
        self.selected_id.set(spectrum_id)
        current = float(self.comparison.get("offsets", {}).get(spectrum_id, 0.0))
        self._updating = True
        self.slider.configure(from_=current - self.SLIDER_SPAN, to=current + self.SLIDER_SPAN)
        self.offset.set(current)
        self._updating = False
        self.update_value_text(current)

    def select_spectrum(self, _event=None) -> None:
        self.configure_slider()
        self.slider.focus_set()

    def update_value_text(self, value: float) -> None:
        name = str(self.app.spectra[self.current_id()]["name"])
        self.value_text.set(f"{name}: {value:+.5f} ppm")

    def slide(self, value: str) -> None:
        if self._updating:
            return
        offset = round(float(value), 5)
        spectrum_id = self.current_id()
        offsets = self.comparison.setdefault("offsets", {})
        if np.isclose(float(offsets.get(spectrum_id, 0.0)), offset, atol=5e-8):
            self.update_value_text(offset)
            return
        offsets[spectrum_id] = offset
        self.update_value_text(offset)
        self.app.draw_comparison()

    def nudge(self, delta: float):
        value = round(float(self.offset.get()) + delta, 5)
        lower, upper = float(self.slider.cget("from")), float(self.slider.cget("to"))
        if value < lower or value > upper:
            self.recenter()
        self.offset.set(value)
        self.slide(str(value))
        return "break"

    def on_mousewheel(self, event):
        if event.delta:
            return self.nudge(self.FINE_STEP if event.delta > 0 else -self.FINE_STEP)
        return "break"

    def recenter(self) -> None:
        current = float(self.offset.get())
        self._updating = True
        self.slider.configure(from_=current - self.SLIDER_SPAN, to=current + self.SLIDER_SPAN)
        self._updating = False

    def accept(self) -> None:
        spectrum_id = self.current_id()
        offset = float(self.comparison.get("offsets", {}).get(spectrum_id, 0.0))
        self.app.status.set(f"{self.app.spectra[spectrum_id]['name']}: сдвиг {offset:+.5f} ppm.")
        self.destroy()

    def cancel(self) -> None:
        offsets = self.comparison.setdefault("offsets", {})
        changed = any(not np.isclose(float(offsets.get(spectrum_id, 0.0)), original)
                      for spectrum_id, original in self.original_offsets.items())
        offsets.update(self.original_offsets)
        if changed:
            self.app.draw_comparison()
        self.destroy()


class MultiSpectrumDialog(tk.Toplevel):
    """Выбор записей и вида для сравнительного графика."""
    def __init__(self, app: SpectrumApp) -> None:
        super().__init__(app)
        self.app = app
        self.title("Вывести несколько спектров")
        self.transient(app)
        self.grab_set()
        self.resizable(False, False)
        frame = tk.Frame(self, padx=12, pady=12)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="Выберите спектры для общего графика:").pack(anchor="w")
        self.variables: dict[int, tk.BooleanVar] = {}
        for spectrum_id, record in app.spectra.items():
            variable = tk.BooleanVar(value=True)
            self.variables[spectrum_id] = variable
            tk.Checkbutton(frame, text=str(record["name"]), variable=variable).pack(anchor="w", pady=1)
        mode_box = tk.LabelFrame(frame, text="Оси", padx=8, pady=6)
        mode_box.pack(fill="x", pady=(10, 0))
        self.mode = tk.StringVar(value="common")
        tk.Radiobutton(mode_box, text="Одна общая ось", variable=self.mode, value="common").pack(anchor="w")
        tk.Radiobutton(mode_box, text="У каждого своя ось", variable=self.mode, value="separate").pack(anchor="w")
        buttons = tk.Frame(frame)
        buttons.pack(fill="x", pady=(12, 0))
        tk.Button(buttons, text="Готово", command=self.accept).pack(side="right")
        tk.Button(buttons, text="Отмена", command=self.destroy).pack(side="right", padx=(0, 6))
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def accept(self) -> None:
        selected = [spectrum_id for spectrum_id, variable in self.variables.items() if variable.get()]
        if len(selected) < 2:
            messagebox.showinfo("Несколько спектров", "Отметьте хотя бы два спектра.", parent=self)
            return
        self.app.create_comparison(selected, self.mode.get())
        self.destroy()


class PhaseDialog(tk.Toplevel):
    def __init__(self, app: SpectrumApp) -> None:
        super().__init__(app)
        self.app = app
        self.previous_state = app.capture_spectrum_state()
        app._spectrum_dialog_open = True
        self.title("Фазовая коррекция")
        self.transient(app); self.grab_set(); self.resizable(False, False)
        frame = tk.Frame(self, padx=12, pady=12); frame.pack()
        self.p0, self.p1 = tk.DoubleVar(), tk.DoubleVar()
        self.drag_start_y: int | None = None
        self.drag_start_phase = 0.0
        self.drag_mode = "p0"

        segment = app.visible_slice()
        assert app.real is not None and app.imaginary is not None
        magnitude = np.abs(app.real[segment] + 1j * app.imaginary[segment])
        app.phase_reference = segment.start + int(np.argmax(magnitude))
        self.reference_pivot = app.phase_reference / max(app.real.size - 1, 1)
        # В полях φ₀ показывается фаза именно на красном пике. Внутри
        # приложения φ₀ всегда хранится относительно фиксированного центра
        # спектра, поэтому открытие/закрытие диалога не меняет сигнал.
        self.p1.set(app.phase1)
        self.p0.set(app.phase0 + app.phase1 * (self.reference_pivot - app.phase_pivot))
        app.draw()

        tk.Label(frame, text="Нулевая фаза настраивается по пику, отмеченному красным.").pack(anchor="w")
        self.value_label = tk.StringVar()
        tk.Label(frame, textvariable=self.value_label, font=("TkDefaultFont", 10, "bold")).pack(pady=(6, 4))
        self.pad = tk.Canvas(frame, width=280, height=180, bg="#f2f2f2", highlightthickness=1,
                             highlightbackground="#999")
        self.pad.pack()
        self.pad.create_text(140, 75, text="Наведите мышь и тяните\nвверх или вниз", justify="center",
                             fill="#444", font=("TkDefaultFont", 12))
        self.pad.create_text(140, 120, text="Shift + перетаскивание — фаза 1-го порядка",
                             fill="#666", font=("TkDefaultFont", 8))
        self.pad.bind("<ButtonPress-1>", self.start_drag)
        self.pad.bind("<B1-Motion>", self.drag)
        self.pad.bind("<ButtonRelease-1>", self.stop_drag)

        fields = tk.Frame(frame); fields.pack(fill="x", pady=(8, 0))
        for row, (label, value) in enumerate((("Фаза 0-го порядка, °", self.p0), ("Фаза 1-го порядка, °", self.p1))):
            tk.Label(fields, text=label).grid(row=row, column=0, sticky="w", pady=2)
            tk.Spinbox(fields, textvariable=value, from_=-720, to=720, increment=.01, width=11,
                       command=self.update).grid(row=row, column=1, padx=8)
            value.trace_add("write", lambda *_: self.update())
        buttons = tk.Frame(frame); buttons.pack(anchor="e", pady=(10, 0))
        tk.Button(buttons, text="Автофаза", command=self.auto_phase).pack(side="left", padx=4)
        tk.Button(buttons, text="Готово", command=self.done).pack(side="left", padx=4)
        tk.Button(buttons, text="Отмена", command=self.cancel).pack(side="left")
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.update_label()

    def start_drag(self, event) -> None:
        self.drag_start_y = event.y
        self.drag_mode = "p1" if event.state & 0x0001 else "p0"
        self.drag_start_phase = self.p1.get() if self.drag_mode == "p1" else self.p0.get()

    def drag(self, event) -> None:
        if self.drag_start_y is None:
            return
        # 0.03° на пиксель: движение получается заметно более плавным, чем
        # у прежних ползунков с шагом 0.5°.
        value = self.drag_start_phase + (self.drag_start_y - event.y) * .03
        (self.p1 if self.drag_mode == "p1" else self.p0).set(value)

    def stop_drag(self, _event) -> None:
        self.drag_start_y = None

    def auto_phase(self) -> None:
        """Подобрать фазу и сразу показать результат.

        φ₁ — это наклон фазы по *всему* спектру.  Узкий увеличенный фрагмент
        почти никогда не содержит достаточно разнесённых сигналов для его
        оценки, поэтому автофаза всегда анализирует полный спектр. Красная
        опорная линия по-прежнему задаёт точку, в которой выражается φ₀.
        """
        if self.app.real is None or self.app.imaginary is None:
            return
        try:
            phase0, phase1, reliable = auto_phase_parameters(
                self.app.real, self.app.imaginary, segment=None,
                pivot=self.reference_pivot,
                return_reliable=True,
            )
        except (ValueError, FloatingPointError) as error:
            messagebox.showerror("Автофаза", f"Не удалось подобрать фазу:\n{error}", parent=self)
            return
        if not reliable:
            # Недостаточно независимых областей спектра для честной оценки
            # наклона. Не сбрасываем уже вручную выставленную фазу в ноль.
            self.value_label.set(
                "Автофаза: φ₁ нельзя определить надёжно — текущая фаза сохранена"
            )
            return
        # trace_add у переменных применит параметры и перерисует спектр.
        # Точность 0.01° заметно тоньше ручного управления, а дальнейшие
        # знаки после запятой являются лишь численным шумом оптимизации.
        phase0, phase1 = round(phase0, 2), round(phase1, 2)
        self.p0.set(phase0)
        self.p1.set(phase1)
        self.value_label.set(f"φ₀ = {phase0:.2f}°     φ₁ = {phase1:.2f}°  (авто)")

    def update(self) -> None:
        try:
            phase0_at_reference, phase1 = self.p0.get(), self.p1.get()
        except tk.TclError:
            return
        self.app.phase1 = phase1
        self.app.phase0 = phase0_at_reference - phase1 * (
            self.reference_pivot - self.app.phase_pivot
        )
        self.app.corrected = None
        self.app.draw()
        self.update_label()

    def update_label(self) -> None:
        self.value_label.set(f"φ₀ = {self.p0.get():.2f}°     φ₁ = {self.p1.get():.2f}°")

    def done(self) -> None:
        self.app.remember_spectrum_state(self.previous_state, "фазирование")
        self.app.phase_reference = None
        self.app.draw()
        self.app._spectrum_dialog_open = False
        self.destroy()

    def cancel(self) -> None:
        self.app.restore_spectrum_state(self.previous_state)
        self.app.draw()
        self.app._spectrum_dialog_open = False
        self.destroy()


class CorrectionDialog(tk.Toplevel):
    def __init__(self, app: SpectrumApp) -> None:
        super().__init__(app)
        self.app = app
        self.previous_state = app.capture_spectrum_state()
        app._spectrum_dialog_open = True
        self.title("Коррекция базовой линии")
        self.transient(app); self.grab_set(); self.resizable(False, False)
        self.method = tk.StringVar(value="Адаптивный arPLS")
        self.polarity = tk.StringVar(value="Авто")
        self.filter_hz, self.smoothness = tk.StringVar(), tk.StringVar()
        self._job: str | None = None
        self.manual_widgets: list[tk.Widget] = []
        frame = tk.Frame(self, padx=12, pady=12); frame.pack()
        tk.Label(frame, text="Метод").grid(row=0, column=0, sticky="w")
        box = ttk.Combobox(frame, textvariable=self.method, state="readonly", width=23,
                           values=("Фильтр Уиттакера", "Сглаживающий spline", "Адаптивный arPLS"))
        box.grid(row=0, column=1, columnspan=2, pady=3); box.bind("<<ComboboxSelected>>", self.on_method_changed)
        self.auto_button = tk.Button(frame, text="Авто", command=self.set_auto)
        self.auto_button.grid(row=1, column=0, sticky="w", pady=5)
        self.manual_widgets.append(self.auto_button)
        tk.Label(frame, text="Полярность пиков").grid(row=1, column=1, sticky="e", padx=(8, 3))
        polarity_box = ttk.Combobox(frame, textvariable=self.polarity, state="readonly", width=15,
                                    values=("Авто", "Положительные", "Отрицательные", "Двуполярные"))
        polarity_box.grid(row=1, column=2, sticky="w")
        polarity_box.bind("<<ComboboxSelected>>", lambda _event: self.schedule())
        self.manual_widgets += self.make_control(frame, 2, "Фильтр, Гц", self.filter_hz, .01, app.spectral_width_hz / 2 * .999, .01)
        self.manual_widgets += self.make_control(frame, 4, "Сглаживание", self.smoothness, 1, 1e9, 1)
        self.note = tk.StringVar()
        self.note_label = tk.Label(frame, textvariable=self.note, wraplength=340, justify="left")
        self.note_label.grid(row=6, column=0, columnspan=3, sticky="w", pady=8)
        self.buttons = tk.Frame(frame)
        self.buttons.grid(row=7, column=1, columnspan=2, sticky="e")
        tk.Button(self.buttons, text="Готово", command=self.commit).pack(side="left", padx=4)
        tk.Button(self.buttons, text="Отмена", command=self.cancel).pack(side="left")
        self.filter_hz.trace_add("write", lambda *_: self.schedule())
        self.smoothness.trace_add("write", lambda *_: self.schedule())
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.set_auto()

    def make_control(self, frame, row, label, variable, low, high, increment) -> list[tk.Widget]:
        name = tk.Label(frame, text=label); name.grid(row=row, column=0, sticky="w")
        spinbox = tk.Spinbox(frame, textvariable=variable, from_=low, to=high, increment=increment, width=12,
                             command=self.schedule)
        spinbox.grid(row=row, column=1, sticky="w")
        log_low, log_high = np.log10(low), np.log10(high)
        scale = tk.Scale(frame, from_=log_low, to=log_high, resolution=.03, orient="horizontal",
                         showvalue=False, length=180, command=lambda v, var=variable: self.from_slider(var, v))
        scale.grid(row=row + 1, column=0, columnspan=3, sticky="ew")
        return [name, spinbox, scale]

    def from_slider(self, variable: tk.StringVar, value: str) -> None:
        variable.set(f"{10 ** float(value):.6g}")

    def set_auto(self) -> None:
        data = self.app.signal()[self.app.visible_slice()]
        f, s = auto_parameters(data, self.app.spectral_width_hz)
        self.filter_hz.set(f"{f:.4g}"); self.smoothness.set(f"{s:g}")

    def on_method_changed(self, _event: object) -> None:
        automatic = self.method.get() != "Фильтр Уиттакера"
        for widget in self.manual_widgets:
            if automatic:
                widget.grid_remove()
            else:
                widget.grid()
        self.note_label_row(automatic)
        self.schedule()

    def note_label_row(self, automatic: bool) -> None:
        """Сжать окно в полностью автоматических режимах."""
        self.note_label.grid_configure(row=2 if automatic else 6)
        self.buttons.grid_configure(row=3 if automatic else 7)

    def parameters(self) -> tuple[float, float]:
        return float(self.filter_hz.get().replace(",", ".")), float(self.smoothness.get().replace(",", "."))

    def baseline(self) -> tuple[np.ndarray, slice, float, float]:
        segment = self.app.visible_slice(); signal = self.app.signal()[segment]
        polarity = {"Авто": "auto", "Положительные": 1,
                    "Отрицательные": -1, "Двуполярные": 0}[self.polarity.get()]
        if self.method.get() == "Сглаживающий spline":
            base = spline_baseline(signal, polarity=polarity)
            f, s = 0.0, 0.0
        elif self.method.get() == "Адаптивный arPLS":
            assert self.app.real is not None
            base = arpls_baseline(signal, total_points=self.app.real.size, polarity=polarity)
            f, s = 0.0, 0.0
        else:
            f, s = self.parameters()
            base = whittaker_baseline(signal, whittaker_lambda(f, s, self.app.spectral_width_hz),
                                      polarity=polarity)
        return base, segment, f, s

    def schedule(self) -> None:
        if self._job is not None: self.after_cancel(self._job)
        self._job = self.after(140, self.preview)

    def preview(self) -> None:
        self._job = None
        try:
            base, segment, f, s = self.baseline()
        except (ValueError, RuntimeError) as error:
            self.note.set(str(error)); return
        self.app.draw(base, segment, preview=True)
        if self.method.get() == "Сглаживающий spline":
            text = "Spline: автоматическая коррекция без ручных параметров."
        elif self.method.get() == "Адаптивный arPLS":
            text = "arPLS: адаптивная робастная коррекция широкого фона."
        else:
            text = f"{self.method.get()}: {f:.4g} Гц; сглаживание {s:g}."
        self.note.set(text + "\nПостроение выполнено только по текущему диапазону X.")

    def commit(self) -> None:
        try: base, segment, f, s = self.baseline()
        except (ValueError, RuntimeError) as error:
            messagebox.showerror("Ошибка коррекции", str(error), parent=self); return
        result = self.app.signal().copy(); result[segment] -= base
        self.app.corrected = result
        self.app.remember_spectrum_state(self.previous_state, "коррекция базовой линии")
        self.app.draw()
        self.app.after_idle(self.app.restore_interaction_mode)
        self.app.status.set(f"Коррекция применена только к выбранному диапазону: {self.method.get()}.")
        self.app._spectrum_dialog_open = False
        self.destroy()

    def cancel(self) -> None:
        self.app.draw()
        self.app.after_idle(self.app.restore_interaction_mode)
        self.app._spectrum_dialog_open = False
        self.destroy()

