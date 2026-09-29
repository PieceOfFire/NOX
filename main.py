"""Оконный просмотрщик и корректор 1D NMR Bruker."""

from __future__ import annotations

import ctypes
import json
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from alignment import AlignmentError, find_x_shift
from base_line import (
    arpls_baseline, auto_parameters, auto_phase_parameters, fit_lorentzian_peak, load_bruker_spectrum,
    lorentzian_component, peak_polarity, phase_correct, spline_baseline,
    whittaker_baseline, whittaker_lambda,
)
from nmr_io import choose_windows_folders as _choose_windows_folders
from project_store import decode_signal, encode_signal, read_project, write_project as write_project_file

try:
    # Tk сам по себе не умеет принимать файлы из Проводника. tkinterdnd2
    # добавляет нативный Windows drag-and-drop и остаётся необязательным для
    # запуска уже сохранённых проектов.
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _RootWindow = TkinterDnD.Tk
    DND_ENABLED = True
except ImportError:
    DND_FILES = "DND_Files"
    _RootWindow = tk.Tk
    DND_ENABLED = False

def choose_windows_folders(parent_hwnd: int) -> tuple[str, ...] | None:
    """Открыть нативный Проводник Windows с Ctrl/Shift для папок.

    ``None`` означает, что системный COM-диалог недоступен; пустой кортеж —
    что пользователь просто нажал «Отмена».
    """
    return _choose_windows_folders(parent_hwnd)


class SpectrumApp(_RootWindow):
    DECONVOLUTION_METHOD = "Деконволюция"
    LEGACY_LORENTZIAN_METHOD = "Аппроксимация Лоренца"
    INTEGRATION_METHODS = (
        "Обычная площадь",
        "Сумма модулей относительно 0",
    )
    INTEGRATION_DISPLAYS = (
        "Площадь",
        "Сумма дискретных точек (MestReNova)",
    )

    @classmethod
    def normalize_integration_method(cls, method: str) -> str:
        """Безопасно открыть проект, сохранённый с экспериментальным методом."""
        if method in (cls.LEGACY_LORENTZIAN_METHOD, cls.DECONVOLUTION_METHOD):
            return "Обычная площадь"
        return method if method in cls.INTEGRATION_METHODS else "Обычная площадь"

    def __init__(self) -> None:
        super().__init__()
        self.title("NMR: фазировка и коррекция базовой линии")
        self.minsize(950, 620)
        self.project_path: Path | None = None
        self.source_folder: Path | None = None
        self.ppm_shift = 0.0
        # История едина для фазировки, базовой линии и всех операций с
        # интегралами. Состояния хранятся до действия, поэтому Ctrl+Z и
        # Ctrl+Shift+Z не зависят от того, какой именно инструмент был активен.
        self._undo_stack: list[tuple[str, str, dict[str, object]]] = []
        self._redo_stack: list[tuple[str, str, dict[str, object]]] = []
        self._spectrum_dialog_open = False
        self.integration_method = tk.StringVar(value="Обычная площадь")
        self.integration_display = tk.StringVar(value="Площадь")
        # Отдельная переменная нужна виджетам выбора: радиокнопка изменяет
        # свою переменную до callback, а для истории нам важно ещё знать
        # прежний режим расчёта.
        self.integration_choice = tk.StringVar(value=self.integration_method.get())
        self.integration_display_choice = tk.StringVar(value=self.integration_display.get())
        self.build_application_menu()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.spectrum_tab = tk.Frame(self.notebook)
        self.formula_tab = tk.Frame(self.notebook)
        self.notebook.add(self.spectrum_tab, text="Спектр и интегралы")
        self.notebook.add(self.formula_tab, text="Формулы")
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.ppm: np.ndarray | None = None
        self.real: np.ndarray | None = None
        self.imaginary: np.ndarray | None = None
        self.spectral_width_hz = 0.0
        self.phase0 = self.phase1 = 0.0
        # Внутренняя точка отсчёта фазы фиксирована. Красная линия в диалоге
        # служит только для удобной ручной настройки φ₀ и не меняет формулу.
        self.phase_pivot = .5
        self.phase_reference: int | None = None
        self.corrected: np.ndarray | None = None
        self._title = "NMR"
        self.integration_mode = False
        self.x_zoom_mode = False
        self.integrals: list[dict[str, float | int | str]] = []
        self.reference_integral_id: int | None = None
        self._next_integral_id = 1
        self.integral_colors = ("#e45756", "#4c78a8", "#59a14f", "#f28e2b", "#b279a2", "#76b7b2")
        # Библиотека хранит независимое состояние каждого добавленного
        # спектра. Поля self.ppm/self.corrected ниже всегда соответствуют
        # только выбранной строке библиотеки — так существующие диалоги
        # фазирования и коррекции не знают о многоспектральном режиме.
        self.spectra: dict[int, dict[str, object]] = {}
        self._next_spectrum_id = 1
        self.active_spectrum_id: int | None = None
        self.comparisons: dict[int, dict[str, object]] = {}
        self._next_comparison_id = 1
        self.current_comparison_id: int | None = None
        self._comparison_axes: list[object] = []
        self._library_items: list[tuple[str, int] | None] = []

        controls = tk.Frame(self.spectrum_tab, padx=10, pady=8)
        controls.pack(fill="x")
        tk.Button(controls, text="Добавить папки…", command=self.choose_folder).pack(side="left")
        tk.Label(controls, text="Инструменты:").pack(side="left", padx=(12, 3))
        phase_tool = tk.Button(controls, text="φ", width=3, command=self.open_phase_dialog)
        phase_tool.pack(side="left", padx=1)
        self.add_formula_tooltip(phase_tool, "Фазирование спектра")
        baseline_tool = tk.Button(controls, text="B", width=3, command=self.open_correction_dialog)
        baseline_tool.pack(side="left", padx=1)
        self.add_formula_tooltip(baseline_tool, "Коррекция базовой линии")
        self.integration_button = tk.Button(controls, text="∫", width=3, command=self.toggle_integration)
        self.integration_button.pack(side="left", padx=1)
        self.add_formula_tooltip(self.integration_button, "Режим интегрирования")
        self.zoom_button = tk.Button(controls, text="🔍 Выбрать X", command=self.toggle_x_zoom)
        self.zoom_button.pack(side="left", padx=5)
        tk.Button(controls, text="Сбросить ширину", command=self.reset_x_zoom).pack(side="left")
        self.status = tk.StringVar(value="Выберите папку с обработанным Bruker-спектром.")
        tk.Label(self.spectrum_tab, textvariable=self.status, anchor="w", padx=10).pack(fill="x")

        # Разделитель можно перетаскивать мышью: это важнее фиксированной
        # высоты на ноутбуках с разным разрешением экрана.
        body = tk.PanedWindow(self.spectrum_tab, orient=tk.HORIZONTAL, sashrelief="raised", sashwidth=6,
                              showhandle=True, bd=0, relief="flat")
        body.pack(fill="both", expand=True)
        library = tk.LabelFrame(body, text="  Спектры  ", padx=6, pady=6)
        analysis = tk.Frame(body)
        body.add(library, minsize=150)
        body.add(analysis, minsize=620)
        tk.Label(library, text="Спектры и сравнения", anchor="w").pack(fill="x")
        list_body = tk.Frame(library)
        list_body.pack(fill="both", expand=True, pady=(4, 0))
        list_scroll = tk.Scrollbar(list_body, orient="vertical")
        list_scroll.pack(side="right", fill="y")
        self.spectrum_list = tk.Listbox(list_body, exportselection=False, activestyle="none",
                                        yscrollcommand=list_scroll.set, width=20)
        self.spectrum_list.pack(side="left", fill="both", expand=True)
        list_scroll.configure(command=self.spectrum_list.yview)
        self.spectrum_list.bind("<<ListboxSelect>>", self.on_spectrum_selected)
        self.enable_spectrum_drop(library)
        tk.Label(library, text="Клик по спектру открывает\nобработку; по сравнению —\nобщий график. Можно\nперетащить 1r или папку.", justify="left",
                 fg="#68707d", wraplength=145).pack(fill="x", pady=(8, 0))

        self.spectrum_split = tk.PanedWindow(
            analysis, orient=tk.VERTICAL, sashrelief="raised", sashwidth=7,
            showhandle=True, bd=0, relief="flat",
        )
        self.spectrum_split.pack(fill="both", expand=True)
        self.plot_frame = tk.Frame(self.spectrum_split)
        # Размер Matplotlib-холста не задаёт минимальную высоту панели:
        # разделитель должен уметь сжать график на маленьком экране.
        self.plot_frame.pack_propagate(False)
        self.spectrum_split.add(self.plot_frame, minsize=180)
        self._integral_table_visible = False
        self.graph_height_percent = 68
        self.figure = Figure(figsize=(9, 5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("scroll_event", self.zoom_y)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.span: SpanSelector | None = None
        self.create_span_selector()
        self.bind("<FocusIn>", lambda _event: self.restore_interaction_mode())
        # У Frame в Tk для Windows padding должен быть одним числом;
        # асимметричные отступы допустимы только в pack/grid.
        self.table_frame = tk.Frame(self.spectrum_split, padx=10, pady=0)
        table_controls = tk.Frame(self.table_frame)
        table_controls.pack(fill="x", pady=(4, 0))
        tk.Label(table_controls, text="Интегралы").pack(side="left")
        tk.Label(table_controls, text="Вывод:").pack(side="right", padx=(8, 3))
        display_box = ttk.Combobox(
            table_controls, textvariable=self.integration_display_choice, state="readonly", width=31,
            values=self.INTEGRATION_DISPLAYS,
        )
        display_box.pack(side="right")
        display_box.bind("<<ComboboxSelected>>",
                         lambda _event: self.change_integration_display(self.integration_display_choice.get()))
        tk.Label(table_controls, text="Метод:").pack(side="right", padx=(12, 3))
        integration_box = ttk.Combobox(
            table_controls, textvariable=self.integration_choice, state="readonly", width=29,
            values=self.INTEGRATION_METHODS,
        )
        integration_box.pack(side="right")
        integration_box.bind("<<ComboboxSelected>>",
                             lambda _event: self.change_integration_method(self.integration_choice.get()))
        columns = ("number", "name", "limits", "area", "relative")
        self.integral_table = ttk.Treeview(self.table_frame, columns=columns, show="headings", height=5)
        headings = {"number": "№", "name": "Название", "limits": "Пределы, ppm", "area": "Площадь", "relative": "Отн. площадь"}
        widths = {"number": 45, "name": 140, "limits": 180, "area": 160, "relative": 130}
        for column in columns:
            self.integral_table.heading(column, text=headings[column])
            self.integral_table.column(column, width=widths[column], anchor="center")
        self.integral_table.pack(fill="both", expand=True)
        self.integral_table.bind("<Button-3>", self.integral_table_menu)
        self.integral_table.bind("<Control-KeyPress>", self.copy_integral_table_shortcut)
        self.build_formula_workspace()

    def build_application_menu(self) -> None:
        """Основное меню проекта и команды обработки сигналов."""
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(label="Добавить папки…", command=self.choose_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Открыть проект…", accelerator="Ctrl+O", command=self.open_project)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить", accelerator="Ctrl+S", command=self.save_project)
        file_menu.add_command(label="Сохранить как…", accelerator="Ctrl+Shift+S", command=self.save_project_as)
        menu_bar.add_cascade(label="Файл", menu=file_menu)
        edit_menu = tk.Menu(menu_bar, tearoff=False)
        edit_menu.add_command(label="Отменить", accelerator="Ctrl+Z", command=self.undo_last_action)
        edit_menu.add_command(label="Повторить", accelerator="Ctrl+Shift+Z", command=self.redo_last_action)
        menu_bar.add_cascade(label="Правка", menu=edit_menu)
        settings_menu = tk.Menu(menu_bar, tearoff=False)
        integration_menu = tk.Menu(settings_menu, tearoff=False)
        for method in self.INTEGRATION_METHODS:
            integration_menu.add_radiobutton(
                label=method, variable=self.integration_choice, value=method,
                command=lambda value=method: self.change_integration_method(value),
            )
        settings_menu.add_cascade(label="Метод интегрирования", menu=integration_menu)
        display_menu = tk.Menu(settings_menu, tearoff=False)
        for display in self.INTEGRATION_DISPLAYS:
            display_menu.add_radiobutton(
                label=display, variable=self.integration_display_choice, value=display,
                command=lambda value=display: self.change_integration_display(value),
            )
        settings_menu.add_cascade(label="Вывод значений интегралов", menu=display_menu)
        menu_bar.add_cascade(label="Настройки", menu=settings_menu)
        processing_menu = tk.Menu(menu_bar, tearoff=False)
        processing_menu.add_command(label="Фазировать", command=self.open_phase_dialog)
        processing_menu.add_command(label="Скорректировать базовую линию", command=self.open_correction_dialog)
        processing_menu.add_separator()
        processing_menu.add_command(label="Интегрирование", command=self.toggle_integration)
        processing_menu.add_command(label="Вывести несколько…", command=self.open_comparison_dialog)
        processing_menu.add_command(label="Автоматически выровнять по X", command=self.auto_align_comparison)
        processing_menu.add_command(label="Сдвинуть спектр по X…", command=self.shift_comparison_spectrum)
        menu_bar.add_cascade(label="Обработка сигналов", menu=processing_menu)
        self.configure(menu=menu_bar)
        self.bind_all("<Control-o>", lambda _event: self.open_project())
        self.bind_all("<Control-s>", lambda _event: self.save_project())
        self.bind_all("<Control-Shift-S>", lambda _event: self.save_project_as())
        # На русской раскладке Windows физическая клавиша Z приходит в Tk
        # как «я»/Cyrillic_ya и шаблон <Control-z> не совпадает. Общий
        # обработчик смотрит также на физический virtual key 90, поэтому
        # Ctrl+Z и Ctrl+Shift+Z работают независимо от раскладки и фокуса.
        self.bind_all("<Control-KeyPress>", self.handle_control_shortcut, add="+")

    def handle_control_shortcut(self, event) -> str | None:
        """Обработать Undo/Redo для латинской и русской раскладки клавиатуры."""
        keysym = str(getattr(event, "keysym", "")).lower()
        is_z_key = keysym in {"z", "cyrillic_ya", "я"} or int(getattr(event, "keycode", -1)) == 90
        if not is_z_key:
            return None
        if int(getattr(event, "state", 0)) & 0x0001:  # Shift
            return self.redo_last_action(event)
        return self.undo_last_action(event)

    def choose_folder(self) -> None:
        """Стандартный Windows-проводник с множественным выбором папок."""
        folders = choose_windows_folders(self.winfo_id())
        if folders is None:
            # Редкий fallback для не-Windows окружения: Tk умеет только одну
            # папку, но приложение всё равно остаётся запускаемым.
            folder = filedialog.askdirectory(parent=self, title="Папка Bruker-эксперимента")
            folders = (folder,) if folder else ()
        if not folders:
            return
        self.add_spectrum_paths(folders)

    @staticmethod
    def spectrum_path_from_drop(path: str | Path) -> Path:
        """Получить папку, которую понимает загрузчик, из файла или каталога."""
        source = Path(path)
        return source.parent if source.is_file() else source

    def add_spectrum_paths(self, paths: tuple[str, ...] | list[str]) -> None:
        """Добавить несколько файлов 1r/папок, не прерываясь на одном сбое."""
        added = 0
        failed: list[str] = []
        for path in paths:
            before = len(self.spectra)
            try:
                success = self.add_spectrum_folder(self.spectrum_path_from_drop(path), show_error=False)
            except (OSError, ValueError):
                success = False
            if success and len(self.spectra) > before:
                added += 1
            elif not success:
                failed.append(Path(path).name)
        if added:
            self.status.set(f"Добавлено спектров: {added}. Выберите нужный слева.")
        if failed:
            messagebox.showwarning("Добавление спектров",
                                   "Не удалось прочитать:\n" + "\n".join(failed[:8]), parent=self)

    def enable_spectrum_drop(self, library: tk.Widget) -> None:
        """Разрешить перетаскивание из Проводника на левую библиотеку."""
        if not DND_ENABLED:
            return
        for widget in (library, self.spectrum_list):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<DropEnter>>", lambda _event: self.spectrum_list.configure(bg="#eaf3ff"))
            widget.dnd_bind("<<DropLeave>>", lambda _event: self.spectrum_list.configure(bg="white"))
            widget.dnd_bind("<<Drop>>", self.on_spectrum_drop)

    def on_spectrum_drop(self, event):
        """Обработчик набора путей, переданного Windows через tkinterdnd2."""
        self.spectrum_list.configure(bg="white")
        try:
            paths = tuple(str(path) for path in self.tk.splitlist(event.data))
        except tk.TclError:
            paths = (str(event.data),)
        self.add_spectrum_paths(paths)
        return getattr(event, "action", "copy")

    def clear_formula_workspace(self) -> None:
        """Очистить формулы только при открытии нового проекта, не при добавлении спектра."""
        if not hasattr(self, "formula_blocks"):
            return
        self.formula_blocks.clear()
        self.formula_frames.clear()
        self._next_formula_frame_id = 1
        self.formula_comments.clear()
        self._next_formula_comment_id = 1
        self.selected_formula_blocks.clear()
        self.wire_routes.clear()
        self.refresh_formula_workspace()

    def save_active_spectrum(self) -> None:
        """Записать текущее рабочее состояние в выбранную строку библиотеки."""
        if self.active_spectrum_id is None or self.active_spectrum_id not in self.spectra:
            return
        record = self.spectra[self.active_spectrum_id]
        record.update({
            "source_folder": self.source_folder, "ppm": self.ppm, "real": self.real,
            "imaginary": self.imaginary, "spectral_width_hz": self.spectral_width_hz,
            "ppm_shift": self.ppm_shift, "phase0": self.phase0, "phase1": self.phase1,
            "phase_pivot": self.phase_pivot, "phase_reference": self.phase_reference,
            "corrected": self.corrected, "integrals": self.integrals,
            "reference_integral_id": self.reference_integral_id,
            "next_integral_id": self._next_integral_id,
            "integration_method": self.integration_method.get(),
            "integration_display": self.integration_display.get(), "title": self._title,
        })

    def restore_active_spectrum(self, spectrum_id: int) -> None:
        record = self.spectra[spectrum_id]
        self.active_spectrum_id = spectrum_id
        self.source_folder = record["source_folder"]  # type: ignore[assignment]
        self.ppm = record["ppm"]  # type: ignore[assignment]
        self.real = record["real"]  # type: ignore[assignment]
        self.imaginary = record["imaginary"]  # type: ignore[assignment]
        self.spectral_width_hz = float(record["spectral_width_hz"])
        self.ppm_shift = float(record["ppm_shift"])
        self.phase0, self.phase1 = float(record["phase0"]), float(record["phase1"])
        self.phase_pivot = float(record["phase_pivot"])
        self.phase_reference = record["phase_reference"]  # type: ignore[assignment]
        self.corrected = record["corrected"]  # type: ignore[assignment]
        self.integrals = record["integrals"]  # type: ignore[assignment]
        self.reference_integral_id = record["reference_integral_id"]  # type: ignore[assignment]
        self._next_integral_id = int(record["next_integral_id"])
        self.integration_method.set(self.normalize_integration_method(
            str(record.get("integration_method", "Обычная площадь"))
        ))
        display = str(record.get("integration_display", "Площадь"))
        self.integration_display.set(display if display in self.INTEGRATION_DISPLAYS else "Площадь")
        self.sync_integration_method_choice()
        self._title = str(record["title"])

    def refresh_spectrum_library(self) -> None:
        """Обновить левую библиотеку спектров и созданных сравнений."""
        if not hasattr(self, "spectrum_list"):
            return
        self.spectrum_list.delete(0, "end")
        self._library_items = []
        ids = list(self.spectra)
        for spectrum_id in ids:
            self.spectrum_list.insert("end", str(self.spectra[spectrum_id]["name"]))
            self._library_items.append(("spectrum", spectrum_id))
        for comparison_id, comparison in self.comparisons.items():
            if len(self.comparisons) and comparison_id == next(iter(self.comparisons)):
                self.spectrum_list.insert("end", "— Сравнения —")
                self._library_items.append(None)
            self.spectrum_list.insert("end", f"▦  {comparison['name']}")
            self._library_items.append(("comparison", comparison_id))
        selected_item: tuple[str, int] | None = (
            ("comparison", self.current_comparison_id) if self.current_comparison_id is not None
            else ("spectrum", self.active_spectrum_id) if self.active_spectrum_id is not None else None
        )
        if selected_item in self._library_items:
            position = self._library_items.index(selected_item)
            self.spectrum_list.selection_set(position)
            self.spectrum_list.activate(position)

    def on_spectrum_selected(self, _event=None) -> None:
        selected = self.spectrum_list.curselection()
        if not selected:
            return
        if selected[0] >= len(self._library_items):
            return
        item = self._library_items[selected[0]]
        if item is None:
            self.spectrum_list.selection_clear(0, "end")
            return
        kind, item_id = item
        if kind == "spectrum":
            self.activate_spectrum(item_id)
        elif item_id in self.comparisons:
            self.save_active_spectrum()
            self.current_comparison_id = item_id
            self.clear_history()
            self.refresh_spectrum_library()
            self.draw_comparison(reset_x=True, reset_y=True)
            self.status.set(f"Открыто {self.comparisons[item_id]['name']}.")

    def activate_spectrum(self, spectrum_id: int) -> None:
        """Открыть один спектр из библиотеки и выйти из обзорного графика."""
        if spectrum_id not in self.spectra:
            return
        self.save_active_spectrum()
        self.current_comparison_id = None
        self._comparison_axes = []
        self.restore_active_spectrum(spectrum_id)
        self.clear_history()
        self.refresh_spectrum_library()
        self.draw(reset_x=True, reset_y=True)
        self.status.set(f"Открыт спектр: {self.spectra[spectrum_id]['name']}.")

    def add_spectrum_folder(self, folder: str | Path, *, clear_formulas: bool = False,
                            reset_project: bool = True, show_error: bool = True) -> bool:
        """Добавить Bruker-спектр в библиотеку, не сбрасывая остальные."""
        try:
            ppm, real, imaginary, spectral_width_hz, source = load_bruker_spectrum(folder)
        except (OSError, ValueError) as error:
            if show_error:
                messagebox.showerror("Ошибка загрузки", str(error), parent=self)
            return False
        # Канонический адрес — папка именно выбранной обработки pdata/N.
        # Поэтому один и тот же спектр не добавится дважды, если сначала
        # выбрали папку эксперимента, а затем перетащили файл 1r.
        source_folder = source.parent.resolve()
        existing = next((item_id for item_id, item in self.spectra.items()
                         if item.get("source_folder") == source_folder), None)
        if existing is not None:
            self.activate_spectrum(existing)
            self.status.set("Этот спектр уже есть в списке — он открыт.")
            return True
        if clear_formulas:
            self.clear_formula_workspace()
        spectrum_id = self._next_spectrum_id
        self._next_spectrum_id += 1
        experiment_folder = source_folder.parent.parent if source_folder.parent.name == "pdata" else source_folder
        base_name = experiment_folder.name or source_folder.name or f"Спектр {spectrum_id}"
        used_names = {str(item["name"]) for item in self.spectra.values()}
        name, suffix = base_name, 2
        while name in used_names:
            name = f"{base_name} ({suffix})"
            suffix += 1
        self.spectra[spectrum_id] = {
            "id": spectrum_id, "name": name, "source_folder": source_folder,
            "ppm": ppm, "real": real, "imaginary": imaginary,
            "spectral_width_hz": float(spectral_width_hz), "ppm_shift": 0.0,
            "phase0": 0.0, "phase1": 0.0, "phase_pivot": .5,
            "phase_reference": None, "corrected": None, "integrals": [],
            "reference_integral_id": None, "next_integral_id": 1,
            "integration_method": "Обычная площадь", "integration_display": "Площадь",
            "title": f"NMR: {source.parent}",
        }
        if reset_project:
            self.project_path = None
        self.activate_spectrum(spectrum_id)
        self.status.set(f"Добавлен спектр «{name}». Выберите его слева для отдельной обработки.")
        return True

    def load_spectrum_folder(self, folder: str | Path) -> bool:
        """Открыть одиночный спектр: используется при загрузке старого проекта."""
        self.spectra.clear()
        self.comparisons.clear()
        self.active_spectrum_id = None
        self.current_comparison_id = None
        self._next_spectrum_id = 1
        self._next_comparison_id = 1
        self.clear_history()
        return self.add_spectrum_folder(folder, clear_formulas=True, reset_project=False)

    def capture_spectrum_state(self) -> dict[str, object]:
        """Полный снимок изменяемого состояния активного спектра.

        Интегралы входят в тот же снимок, что фаза и линия: пользователь может
        подряд фазировать, добавить область и удалить другую, а затем идти по
        этой единой последовательности Ctrl+Z / Ctrl+Shift+Z.
        """
        return {
            "phase0": float(self.phase0),
            "phase1": float(self.phase1),
            "phase_reference": self.phase_reference,
            "ppm": None if self.ppm is None else self.ppm.copy(),
            "ppm_shift": float(self.ppm_shift),
            "corrected": None if self.corrected is None else self.corrected.copy(),
            "integrals": [dict(item) for item in self.integrals],
            "reference_integral_id": self.reference_integral_id,
            "next_integral_id": int(self._next_integral_id),
            "integration_method": self.integration_method.get(),
            "integration_display": self.integration_display.get(),
        }

    def spectrum_state_changed(self, previous: dict[str, object]) -> bool:
        """Есть ли отличие текущего спектра от сохранённого снимка."""
        if (not np.isclose(float(previous["phase0"]), self.phase0)
                or not np.isclose(float(previous["phase1"]), self.phase1)
                or previous.get("phase_reference") != self.phase_reference
                or not np.isclose(float(previous.get("ppm_shift", 0.0)), self.ppm_shift)
                or previous.get("reference_integral_id") != self.reference_integral_id
                or int(previous.get("next_integral_id", 1)) != self._next_integral_id
                or previous.get("integration_method", "Обычная площадь") != self.integration_method.get()
                or previous.get("integration_display", "Площадь") != self.integration_display.get()
                or previous.get("integrals", []) != self.integrals):
            return True
        before_ppm = previous.get("ppm")
        if (before_ppm is None) != (self.ppm is None):
            return True
        if before_ppm is not None and not np.array_equal(np.asarray(before_ppm), self.ppm):
            return True
        before = previous["corrected"]
        if (before is None) != (self.corrected is None):
            return True
        return before is not None and not np.array_equal(np.asarray(before), self.corrected)

    def clear_history(self) -> None:
        """Смена спектра/проекта не должна смешивать независимые истории."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def trim_history(self) -> None:
        if len(self._undo_stack) > 20:
            del self._undo_stack[:-20]
        if len(self._redo_stack) > 20:
            del self._redo_stack[:-20]

    def remember_spectrum_state(self, previous: dict[str, object], action: str) -> None:
        """Добавить состояние до действия в историю и очистить ветку redo."""
        if not self.spectrum_state_changed(previous):
            return
        self._undo_stack.append(("spectrum", action, previous))
        self._redo_stack.clear()
        self.trim_history()

    def remember_comparison_integrals(self, comparison: dict[str, object], action: str) -> None:
        """Сохранить общие интегралы обзора для Ctrl+Z.

        В режиме отдельных осей один диапазон принадлежит самому сравнению,
        а не активному спектру. Поэтому снимок `self.integrals` здесь был бы
        пустым и отмена ранее ничего не делала.
        """
        comparison_id = comparison.get("id", self.current_comparison_id)
        if not isinstance(comparison_id, int):
            return
        previous = {
            "comparison_id": comparison_id,
            "integrals": [dict(item) for item in comparison.get("integrals", [])],
            "next_integral_id": int(comparison.get("next_integral_id", 1)),
        }
        self._undo_stack.append(("comparison_integrals", action, previous))
        self._redo_stack.clear()
        self.trim_history()

    def remember_comparison_offsets(self, comparison: dict[str, object], action: str) -> None:
        """Сохранить X-сдвиги сравнения, чтобы автовыравнивание отменялось через Ctrl+Z."""
        comparison_id = comparison.get("id", self.current_comparison_id)
        if not isinstance(comparison_id, int):
            return
        previous = {
            "comparison_id": comparison_id,
            "offsets": dict(comparison.get("offsets", {})),
        }
        self._undo_stack.append(("comparison_offsets", action, previous))
        self._redo_stack.clear()
        self.trim_history()

    @staticmethod
    def comparison_integral_state(comparison: dict[str, object]) -> dict[str, object]:
        return {
            "comparison_id": comparison.get("id"),
            "integrals": [dict(item) for item in comparison.get("integrals", [])],
            "next_integral_id": int(comparison.get("next_integral_id", 1)),
        }

    @staticmethod
    def comparison_offset_state(comparison: dict[str, object]) -> dict[str, object]:
        return {
            "comparison_id": comparison.get("id"),
            "offsets": dict(comparison.get("offsets", {})),
        }

    def restore_comparison_integral_state(self, state: dict[str, object]) -> bool:
        comparison_id = state.get("comparison_id")
        comparison = self.comparisons.get(comparison_id) if isinstance(comparison_id, int) else None
        if comparison is None:
            return False
        comparison["integrals"] = [dict(item) for item in state.get("integrals", [])]
        comparison["next_integral_id"] = int(state.get("next_integral_id", 1))
        return True

    def restore_comparison_offset_state(self, state: dict[str, object]) -> bool:
        comparison_id = state.get("comparison_id")
        comparison = self.comparisons.get(comparison_id) if isinstance(comparison_id, int) else None
        offsets = state.get("offsets")
        if comparison is None or not isinstance(offsets, dict):
            return False
        comparison["offsets"] = dict(offsets)
        return True

    def restore_spectrum_state(self, state: dict[str, object]) -> None:
        self.phase0 = float(state["phase0"])
        self.phase1 = float(state["phase1"])
        self.phase_reference = state.get("phase_reference")  # type: ignore[assignment]
        restored_ppm = state.get("ppm")
        self.ppm = None if restored_ppm is None else np.asarray(restored_ppm, dtype=float).copy()
        self.ppm_shift = float(state.get("ppm_shift", 0.0))
        corrected = state["corrected"]
        self.corrected = None if corrected is None else np.asarray(corrected, dtype=float).copy()
        self.integrals = [dict(item) for item in state.get("integrals", [])]
        self.reference_integral_id = state.get("reference_integral_id")  # type: ignore[assignment]
        self._next_integral_id = int(state.get("next_integral_id", 1))
        self.integration_method.set(self.normalize_integration_method(
            str(state.get("integration_method", "Обычная площадь"))
        ))
        display = str(state.get("integration_display", "Площадь"))
        self.integration_display.set(display if display in self.INTEGRATION_DISPLAYS else "Площадь")
        self.sync_integration_method_choice()

    def current_history_state(self, kind: str, target: dict[str, object]) -> dict[str, object] | None:
        if kind == "spectrum":
            return self.capture_spectrum_state()
        if kind == "comparison_integrals":
            comparison_id = target.get("comparison_id")
            comparison = self.comparisons.get(comparison_id) if isinstance(comparison_id, int) else None
            return self.comparison_integral_state(comparison) if comparison is not None else None
        if kind == "comparison_offsets":
            comparison_id = target.get("comparison_id")
            comparison = self.comparisons.get(comparison_id) if isinstance(comparison_id, int) else None
            return self.comparison_offset_state(comparison) if comparison is not None else None
        return None

    def apply_history_state(self, kind: str, state: dict[str, object]) -> bool:
        if kind == "spectrum":
            self.restore_spectrum_state(state)
            self.draw()
            return True
        if kind == "comparison_integrals":
            if not self.restore_comparison_integral_state(state):
                return False
            if self.current_comparison_id == state.get("comparison_id"):
                self.draw_comparison()
            return True
        if kind == "comparison_offsets":
            if not self.restore_comparison_offset_state(state):
                return False
            if self.current_comparison_id == state.get("comparison_id"):
                self.draw_comparison()
            return True
        return False

    def undo_last_action(self, _event: object | None = None) -> str:
        """Отменить последнее действие над спектром либо общим интегралом."""
        if self._spectrum_dialog_open:
            return "break"
        if not self._undo_stack:
            self.status.set("Нет действий для отмены.")
            return "break"
        kind, action, previous = self._undo_stack.pop()
        current = self.current_history_state(kind, previous)
        if current is None or not self.apply_history_state(kind, previous):
            self.status.set("Нельзя отменить: исходный объект истории уже недоступен.")
            return "break"
        self._redo_stack.append((kind, action, current))
        self.trim_history()
        self.after_idle(self.restore_interaction_mode)
        self.status.set(f"Отменено: {action}.")
        return "break"

    def redo_last_action(self, _event: object | None = None) -> str:
        """Вернуть действие, отменённое через Ctrl+Z."""
        if self._spectrum_dialog_open:
            return "break"
        if not self._redo_stack:
            self.status.set("Нет действий для повтора.")
            return "break"
        kind, action, following = self._redo_stack.pop()
        current = self.current_history_state(kind, following)
        if current is None or not self.apply_history_state(kind, following):
            self.status.set("Нельзя повторить: исходный объект истории уже недоступен.")
            return "break"
        self._undo_stack.append((kind, action, current))
        self.trim_history()
        self.after_idle(self.restore_interaction_mode)
        self.status.set(f"Повторено: {action}.")
        return "break"

    @staticmethod
    def encode_project_signal(signal: np.ndarray | None) -> dict[str, object] | None:
        return encode_signal(signal)

    @staticmethod
    def decode_project_signal(payload: object, expected_size: int) -> np.ndarray | None:
        return decode_signal(payload, expected_size)

    def project_spectrum_data(self, record: dict[str, object]) -> dict[str, object]:
        integrals = [
            {"id": int(item["id"]), "left": float(item["left"]), "right": float(item["right"]),
             "color": str(item["color"]), "name": str(item["name"]),
             "number": int(item.get("number", index))}
            for index, item in enumerate(sorted(record.get("integrals", []),
                                                key=lambda item: int(item.get("number", 10**9))), start=1)
        ]
        return {
            "name": str(record["name"]), "source_folder": str(record["source_folder"]),
            "phase": {"p0": float(record["phase0"]), "p1": float(record["phase1"]),
                      "reference": record["phase_reference"]},
            "ppm_shift": float(record["ppm_shift"]),
            "corrected_signal": self.encode_project_signal(record.get("corrected")),
            "integrals": integrals, "reference_integral_id": record.get("reference_integral_id"),
            "integration_method": str(record.get("integration_method", "Обычная площадь")),
            "integration_display": str(record.get("integration_display", "Площадь")),
        }

    def project_data(self) -> dict[str, object]:
        if self.ppm is None or self.source_folder is None:
            raise ValueError("сначала откройте спектр")
        self.save_active_spectrum()
        integrals = [
            {"id": int(item["id"]), "left": float(item["left"]), "right": float(item["right"]),
             "color": str(item["color"]), "name": str(item["name"]),
             "number": int(item.get("number", index))}
            for index, item in enumerate(self.ordered_integrals(), start=1)
        ]
        data = {
            "format": "nmr-analysis-project", "version": 2,
            "source_folder": str(self.source_folder),
            "phase": {"p0": self.phase0, "p1": self.phase1, "reference": self.phase_reference},
            "ppm_shift": self.ppm_shift,
            # Сохраняется уже вычтенный сигнал, поэтому результат коррекции
            # восстанавливается точно, а не пересчитывается с другими настройками.
            "corrected_signal": self.encode_project_signal(self.corrected),
            "integrals": integrals, "reference_integral_id": self.reference_integral_id,
            "integration_method": self.integration_method.get(),
            "integration_display": self.integration_display.get(),
            "formula_workspace": self.formula_preset_data(),
            "view": {"xlim": list(self.axes.get_xlim()), "ylim": list(self.axes.get_ylim())},
        }
        if self.spectra:
            spectrum_ids = list(self.spectra)
            data["spectra"] = [self.project_spectrum_data(record) for record in self.spectra.values()]
            data["active_spectrum_index"] = spectrum_ids.index(self.active_spectrum_id) \
                if self.active_spectrum_id in spectrum_ids else 0
            data["comparisons"] = [
                {"name": str(comparison["name"]),
                 "spectrum_indices": [spectrum_ids.index(int(value)) for value in comparison.get("spectrum_ids", [])
                                      if int(value) in spectrum_ids],
                 "mode": str(comparison.get("mode", "common")), "offsets": dict(comparison.get("offsets", {})),
                 "offsets_by_index": [float(comparison.get("offsets", {}).get(int(value), 0.0))
                                      for value in comparison.get("spectrum_ids", []) if int(value) in spectrum_ids],
                 "integrals": list(comparison.get("integrals", [])),
                 "next_integral_id": comparison.get("next_integral_id", 1)}
                for comparison in self.comparisons.values()
            ]
        return data

    def save_project(self) -> None:
        if self.project_path is None:
            self.save_project_as()
            return
        self.write_project(self.project_path)

    def save_project_as(self) -> None:
        filename = filedialog.asksaveasfilename(
            parent=self, title="Сохранить проект NMR", defaultextension=".nmrproj.json",
            filetypes=(("Проект NMR", "*.nmrproj.json"), ("JSON", "*.json"), ("Все файлы", "*.*")),
        )
        if filename:
            self.write_project(Path(filename))

    def write_project(self, filename: Path) -> None:
        try:
            write_project_file(filename, self.project_data())
        except (OSError, ValueError) as error:
            messagebox.showerror("Проект NMR", f"Не удалось сохранить проект:\n{error}", parent=self)
            return
        self.project_path = filename
        self.status.set(f"Проект сохранён: {filename.name}")

    def open_project(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self, title="Открыть проект NMR",
            filetypes=(("Проект NMR", "*.nmrproj.json *.json"), ("Все файлы", "*.*")),
        )
        if not filename:
            return
        try:
            project = read_project(filename)
            source_folder = Path(str(project["source_folder"]))
            if not source_folder.is_dir():
                raise ValueError(f"не найдена папка спектра: {source_folder}")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            messagebox.showerror("Проект NMR", f"Не удалось открыть проект:\n{error}", parent=self)
            return
        try:
            saved_spectra = project.get("spectra")
            payloads = [item for item in saved_spectra if isinstance(item, dict)] \
                if isinstance(saved_spectra, list) else [project]
            if not payloads:
                raise ValueError("в проекте нет спектров")
            first_source = Path(str(payloads[0].get("source_folder", source_folder)))
            if not first_source.is_dir() or not self.load_spectrum_folder(first_source):
                return
            for index, payload in enumerate(payloads):
                payload_source = Path(str(payload.get("source_folder", source_folder)))
                if index:
                    if not payload_source.is_dir():
                        raise ValueError(f"не найдена папка спектра: {payload_source}")
                    if not self.add_spectrum_folder(payload_source, reset_project=False):
                        return
                self.restore_project_spectrum_payload(payload)
                if payload.get("name"):
                    self.spectra[self.active_spectrum_id]["name"] = str(payload["name"])
                self.save_active_spectrum()
            active_index = project.get("active_spectrum_index", 0)
            ids = list(self.spectra)
            if isinstance(active_index, int) and 0 <= active_index < len(ids):
                self.restore_active_spectrum(ids[active_index])
            self.comparisons.clear()
            for raw in project.get("comparisons", []):
                if not isinstance(raw, dict):
                    continue
                selected = [ids[int(item)] for item in raw.get("spectrum_indices", [])
                            if isinstance(item, int) and 0 <= item < len(ids)]
                if len(selected) < 2:
                    continue
                comparison_id = self._next_comparison_id
                self._next_comparison_id += 1
                raw_offsets = raw.get("offsets_by_index", [])
                offsets = {spectrum_id: float(raw_offsets[index]) if isinstance(raw_offsets, list)
                           and index < len(raw_offsets) else 0.0
                           for index, spectrum_id in enumerate(selected)}
                self.comparisons[comparison_id] = {
                    "id": comparison_id, "name": str(raw.get("name") or f"Сравнение {comparison_id}"),
                    "spectrum_ids": selected, "mode": str(raw.get("mode") or "common"),
                    "offsets": offsets,
                    "integrals": list(raw.get("integrals", [])),
                    "next_integral_id": int(raw.get("next_integral_id", 1)),
                }
            workspace = project.get("formula_workspace")
            if isinstance(workspace, dict):
                self.load_formula_preset(workspace, show_error=False)
            self.refresh_spectrum_library()
            self.draw(reset_x=True, reset_y=True)
            view = project.get("view", {})
            if isinstance(view, dict):
                xlim, ylim = view.get("xlim"), view.get("ylim")
                if isinstance(xlim, list) and len(xlim) == 2:
                    self.axes.set_xlim(float(xlim[0]), float(xlim[1]))
                if isinstance(ylim, list) and len(ylim) == 2:
                    self.axes.set_ylim(float(ylim[0]), float(ylim[1]))
                self.canvas.draw_idle()
        except (TypeError, ValueError, KeyError) as error:
            messagebox.showerror("Проект NMR", f"Проект повреждён или несовместим:\n{error}", parent=self)
            return
        self.project_path = Path(filename)
        self.status.set(f"Проект открыт: {self.project_path.name}")

    def restore_project_spectrum_payload(self, payload: dict[str, object]) -> None:
        """Применить сохранённую обработку к уже загруженному активному 1r."""
        phase = payload.get("phase", {})
        if not isinstance(phase, dict):
            phase = {}
        self.phase0, self.phase1 = float(phase.get("p0", 0.0)), float(phase.get("p1", 0.0))
        reference = phase.get("reference")
        self.phase_reference = int(reference) if isinstance(reference, int) and self.real is not None and \
            0 <= reference < self.real.size else None
        self.ppm_shift = float(payload.get("ppm_shift", 0.0))
        assert self.ppm is not None and self.real is not None
        self.ppm = self.ppm + self.ppm_shift
        self.corrected = self.decode_project_signal(payload.get("corrected_signal"), self.real.size)
        self.restore_project_integrals(payload.get("integrals"), payload.get("reference_integral_id"))
        self.integration_method.set(self.normalize_integration_method(
            str(payload.get("integration_method", "Обычная площадь"))
        ))
        display = str(payload.get("integration_display", "Площадь"))
        self.integration_display.set(display if display in self.INTEGRATION_DISPLAYS else "Площадь")
        self.sync_integration_method_choice()

    def restore_project_integrals(self, raw_integrals: object, reference_id: object) -> None:
        self.integrals = []
        if not isinstance(raw_integrals, list):
            return
        for raw in raw_integrals:
            if not isinstance(raw, dict):
                continue
            try:
                integral_id = int(raw["id"])
                left, right = float(raw["left"]), float(raw["right"])
            except (KeyError, TypeError, ValueError):
                continue
            if integral_id <= 0 or not np.isfinite(left) or not np.isfinite(right) or left == right:
                continue
            color = str(raw.get("color") or self.integral_colors[(integral_id - 1) % len(self.integral_colors)])
            self.integrals.append({"id": integral_id, "left": min(left, right), "right": max(left, right),
                                   "color": color, "name": str(raw.get("name") or f"Интеграл {integral_id}"),
                                   "number": raw.get("number")})
        # Старые проекты не содержали номер: восстанавливаем привычный порядок
        # слева направо и одновременно устраняем повторы в повреждённых файлах.
        used_numbers: set[int] = set()
        for fallback, item in enumerate(sorted(self.integrals, key=lambda value: -float(value["left"]) - float(value["right"])), start=1):
            raw_number = item.get("number")
            number = int(raw_number) if isinstance(raw_number, int) and raw_number > 0 else fallback
            while number in used_numbers:
                number += 1
            item["number"] = number
            used_numbers.add(number)
        ids = {int(item["id"]) for item in self.integrals}
        self.reference_integral_id = int(reference_id) if isinstance(reference_id, int) and reference_id in ids else None
        self._next_integral_id = max(ids, default=0) + 1

    def phased_signal(self) -> np.ndarray:
        assert self.real is not None and self.imaginary is not None
        return phase_correct(self.real, self.imaginary, self.phase0, self.phase1, self.phase_pivot)

    def signal(self) -> np.ndarray:
        return self.corrected if self.corrected is not None else self.phased_signal()

    @staticmethod
    def record_signal(record: dict[str, object]) -> np.ndarray:
        """Сигнал записи библиотеки без переключения активного спектра."""
        corrected = record.get("corrected")
        if isinstance(corrected, np.ndarray):
            return corrected
        return phase_correct(
            np.asarray(record["real"]), np.asarray(record["imaginary"]),
            float(record.get("phase0", 0.0)), float(record.get("phase1", 0.0)),
            float(record.get("phase_pivot", .5)),
        )

    def comparison_record(self) -> dict[str, object] | None:
        if self.current_comparison_id is None:
            return None
        return self.comparisons.get(self.current_comparison_id)

    @staticmethod
    def lorentzian_fit_in_region(ppm: np.ndarray, signal: np.ndarray, left: float, right: float,
                                 *, x_offset: float = 0.0) -> dict[str, float]:
        """Подогнать один лоренциан внутри границ интеграла.

        В обзорном графике границы живут уже в сдвинутой системе координат,
        поэтому перед выборкой их нужно вернуть на ось отдельного спектра.
        """
        low, high = min(left - x_offset, right - x_offset), max(left - x_offset, right - x_offset)
        mask = (ppm >= low) & (ppm <= high)
        if np.count_nonzero(mask) < 7:
            raise ValueError("Для аппроксимации Лоренца выделите не меньше 7 точек спектра.")
        return fit_lorentzian_peak(ppm[mask], signal[mask])

    def lorentzian_integral_area(self, ppm: np.ndarray, signal: np.ndarray, left: float, right: float,
                                 *, x_offset: float = 0.0) -> float:
        """Вернуть модуль полной площади fitted-лоренциана.

        При неудачной подгонке таблица остаётся рабочей и показывает ноль;
        при добавлении нового интеграла ошибка сообщается отдельно до записи
        интеграла в проект.
        """
        try:
            fit = self.lorentzian_fit_in_region(ppm, signal, left, right, x_offset=x_offset)
        except (ValueError, RuntimeError, FloatingPointError):
            return 0.0
        return abs(float(fit["area"]))

    def record_integral_area(self, record: dict[str, object], left: float, right: float,
                             *, x_offset: float = 0.0) -> float:
        """Физическая площадь записи в единицах интенсивность × ppm.

        Формулы всегда используют именно это значение; способ вывода таблицы
        не должен незаметно менять математическую схему пользователя.
        """
        ppm = np.asarray(record["ppm"])
        low, high = min(left - x_offset, right - x_offset), max(left - x_offset, right - x_offset)
        mask = (ppm >= low) & (ppm <= high)
        if np.count_nonzero(mask) < 2:
            return 0.0
        signal = self.record_signal(record)
        method = self.integration_method.get()
        if method == self.DECONVOLUTION_METHOD:
            return self.lorentzian_integral_area(ppm, signal, left, right, x_offset=x_offset)
        values = signal[mask]
        if method == "Сумма модулей относительно 0":
            values = np.abs(values)
        return abs(float(np.trapezoid(values, ppm[mask])))

    def record_integral_display_value(self, record: dict[str, object], left: float, right: float,
                                      *, x_offset: float = 0.0) -> float:
        """Вернуть только отображаемое значение интеграла для таблицы."""
        if self.integration_method.get() == self.DECONVOLUTION_METHOD:
            # Сумма дискретных отсчётов относится только к обычному
            # интегрированию. Для fitted-пика всегда отображаем его полную
            # аналитическую площадь, иначе выбранный способ теряет смысл.
            return self.record_integral_area(record, left, right, x_offset=x_offset)
        if self.integration_display.get() != "Сумма дискретных точек (MestReNova)":
            return self.record_integral_area(record, left, right, x_offset=x_offset)
        ppm = np.asarray(record["ppm"])
        low, high = min(left - x_offset, right - x_offset), max(left - x_offset, right - x_offset)
        mask = (ppm >= low) & (ppm <= high)
        if np.count_nonzero(mask) < 2:
            return 0.0
        values = self.record_signal(record)[mask]
        if self.integration_method.get() == "Сумма модулей относительно 0":
            values = np.abs(values)
        return abs(float(np.sum(values)))

    def visible_slice(self) -> slice:
        assert self.ppm is not None
        left, right = sorted(self.axes.get_xlim())
        indices = np.flatnonzero((self.ppm >= left) & (self.ppm <= right))
        if indices.size < 3:
            return slice(0, self.ppm.size)
        return slice(int(indices[0]), int(indices[-1]) + 1)

    def draw(self, baseline: np.ndarray | None = None, segment: slice | None = None,
             preview: bool = False, reset_x: bool = False, reset_y: bool = False) -> None:
        if self.ppm is None:
            return
        if self.current_comparison_id is not None and not preview:
            self.draw_comparison(reset_x=reset_x, reset_y=reset_y)
            return
        self.save_active_spectrum()
        # После обзорного графика в Figure может быть несколько осей. Для
        # обычного спектра снова оставляем ровно одну, привычную всем диалогам.
        if self._comparison_axes:
            self.figure.clear()
            self.axes = self.figure.add_subplot(111)
            self._comparison_axes = []
        old_xlim = self.axes.get_xlim()
        old_ylim = self.axes.get_ylim()
        self.axes.clear()
        self.axes.plot(self.ppm, self.signal(), color="#1f77b4", linewidth=.75,
                       label="Спектр" if not preview else "Исходный спектр")
        if self.phase_reference is not None:
            self.axes.axvline(self.ppm[self.phase_reference], color="#d62728", linewidth=1.2)
            self.axes.text(self.ppm[self.phase_reference], .97, "Опорный пик φ₀",
                           color="#d62728", transform=self.axes.get_xaxis_transform(),
                           ha="center", va="top", fontsize=9)
        if baseline is not None and segment is not None:
            self.axes.plot(self.ppm[segment], baseline, "r--", linewidth=1.1,
                           label="Базовая линия для вычитания")
        self.axes.set_xlabel("Химический сдвиг, ppm")
        self.axes.set_ylabel("Интенсивность, отн. ед.")
        self.axes.set_title(self._title if not preview else "Предпросмотр базовой линии")
        self.axes.grid(alpha=.25)
        self.draw_integrals()
        if baseline is not None:
            self.axes.legend(loc="best")
        if reset_x:
            self.axes.set_xlim(float(self.ppm.max()), float(self.ppm.min()))
        elif old_xlim != (0.0, 1.0):
            self.axes.set_xlim(old_xlim)
        if not reset_y and old_ylim != (0.0, 1.0):
            self.axes.set_ylim(old_ylim)
        self.create_span_selector()
        self.canvas.draw_idle()
        self.save_active_spectrum()

    def draw_comparison(self, *, reset_x: bool = False, reset_y: bool = False) -> None:
        """Отрисовать наложение либо стек отдельных спектров на одной оси X."""
        comparison = self.comparison_record()
        if comparison is None:
            return
        ids = [int(value) for value in comparison.get("spectrum_ids", []) if int(value) in self.spectra]
        if not ids:
            self.current_comparison_id = None
            self.draw(reset_x=True, reset_y=True)
            return
        old_xlim = self.axes.get_xlim() if hasattr(self, "axes") else (0.0, 1.0)
        old_ylim = self.axes.get_ylim() if hasattr(self, "axes") else (0.0, 1.0)
        self.figure.clear()
        mode = str(comparison.get("mode", "common"))
        # «У каждого своя ось» здесь означает не набор маленьких subplot, а
        # вертикальный стек на одном поле — как в Mnova. Каждый ряд получает
        # собственный визуальный масштаб, исходные данные для интегрирования
        # остаются без нормировки.
        self.axes = self.figure.add_subplot(111)
        self._comparison_axes = [self.axes]
        axis = self.axes
        offsets = comparison.setdefault("offsets", {})
        colors = ("#1f77b4", "#e45756", "#59a14f", "#f28e2b", "#9467bd", "#17becf")
        maximum, minimum = -np.inf, np.inf
        stack_levels: list[float] = []
        for index, spectrum_id in enumerate(ids):
            record = self.spectra[spectrum_id]
            ppm = np.asarray(record["ppm"])
            offset = float(offsets.get(spectrum_id, offsets.get(str(spectrum_id), 0.0)))
            signal = self.record_signal(record)
            color = colors[index % len(colors)]
            if mode == "separate":
                # Медиана устойчива к узким линиям и задаёт положение местной
                # базовой линии. Положительная и отрицательная части имеют
                # разные пределы, чтобы обычный спектр выглядел естественно,
                # а двуполярный не налезал на соседний ряд.
                baseline = float(np.median(signal))
                positive = max(float(np.max(signal) - baseline), np.finfo(float).eps)
                negative = max(float(baseline - np.min(signal)), np.finfo(float).eps)
                scale = min(.78 / positive, .48 / negative)
                level = 1.0 + (len(ids) - 1 - index) * 1.42
                stack_levels.append(level)
                displayed = (signal - baseline) * scale + level
                axis.plot(ppm + offset, displayed, color=color, linewidth=.75)
                axis.axhline(level, color=color, linewidth=.65, alpha=.7)
                axis.text(.002, level + .88, str(record["name"]), color=color,
                          transform=axis.get_yaxis_transform(), ha="left", va="bottom", fontsize=9)
            else:
                axis.plot(ppm + offset, signal, color=color, linewidth=.75, label=str(record["name"]))
            maximum = max(maximum, float(np.max(ppm + offset)))
            minimum = min(minimum, float(np.min(ppm + offset)))
        axis.grid(alpha=.25)
        axis.set_xlabel("Химический сдвиг, ppm")
        if mode == "common":
            axis.set_ylabel("Интенсивность, отн. ед.")
            axis.set_title(str(comparison["name"]))
            axis.legend(loc="best")
        else:
            axis.set_title(str(comparison["name"]) + " — отдельные шкалы")
            axis.set_ylabel("")
            axis.set_yticks(stack_levels)
            axis.set_yticklabels([str(number) for number in range(len(ids), 0, -1)])
            axis.yaxis.tick_right()
            axis.tick_params(axis="y", length=0, pad=5)
            axis.set_ylim(.35, max(stack_levels) + 1.06)
        if reset_x or old_xlim == (0.0, 1.0):
            axis.set_xlim(maximum, minimum)
        else:
            axis.set_xlim(old_xlim)
        if not reset_y and old_ylim != (0.0, 1.0):
            axis.set_ylim(old_ylim)
        self.draw_comparison_integrals(axis, comparison)
        self.figure.tight_layout()
        self.create_span_selector()
        self.update_comparison_integral_table(comparison)
        self.canvas.draw_idle()

    def draw_comparison_integrals(self, axis, comparison: dict[str, object]) -> None:
        integrals = list(comparison.get("integrals", []))
        ymin, ymax = axis.get_ylim()
        for item in sorted(integrals, key=lambda value: int(value.get("number", 0))):
            left, right, color = float(item["left"]), float(item["right"]), str(item["color"])
            axis.axvspan(left, right, color=color, alpha=.11)
            axis.axvline(left, color=color, linewidth=1.1)
            axis.axvline(right, color=color, linewidth=1.1)
            axis.text((left + right) / 2, ymax - (ymax - ymin) * .035, str(item["number"]),
                      color=color, ha="center", va="top", fontsize=10, fontweight="bold")

    def open_comparison_dialog(self) -> None:
        if len(self.spectra) < 2:
            messagebox.showinfo("Несколько спектров", "Добавьте как минимум два спектра.", parent=self)
            return
        MultiSpectrumDialog(self)

    def create_comparison(self, spectrum_ids: list[int], mode: str) -> None:
        ids = [spectrum_id for spectrum_id in spectrum_ids if spectrum_id in self.spectra]
        if len(ids) < 2:
            messagebox.showinfo("Несколько спектров", "Выберите хотя бы два спектра.", parent=self)
            return
        self.save_active_spectrum()
        comparison_id = self._next_comparison_id
        self._next_comparison_id += 1
        self.comparisons[comparison_id] = {
            "id": comparison_id, "name": f"Сравнение {comparison_id}", "spectrum_ids": ids,
            "mode": mode, "offsets": {spectrum_id: 0.0 for spectrum_id in ids}, "integrals": [],
            "next_integral_id": 1,
        }
        self.current_comparison_id = comparison_id
        # Палитра формул показывает площадь первого выбранного спектра, а
        # таблица формул ниже всё равно пересчитывает все строки обзора.
        self.restore_active_spectrum(ids[0])
        self.integration_mode = False
        self.integration_button.configure(relief="raised")
        self.hide_integral_table()
        self.refresh_spectrum_library()
        self.draw_comparison(reset_x=True, reset_y=True)
        text = "общей оси" if mode == "common" else "отдельных осей"
        self.status.set(f"Создано {self.comparisons[comparison_id]['name']} в режиме {text}.")

    def shift_comparison_spectrum(self) -> None:
        comparison = self.comparison_record()
        if comparison is None:
            messagebox.showinfo("Сдвиг по X", "Сначала откройте график «Сравнение» справа.", parent=self)
            return
        ShiftSpectrumDialog(self, comparison)

    def auto_align_comparison(self) -> bool:
        """Совместить все линии обзора с первым спектром по текущему окну X.

        На полном спектре метод использует все общие пики. Если нужна привязка
        к одному конкретному сигналу, сначала увеличьте его колёсиком и затем
        запустите эту команду: границы текущего окна сохраняются.
        """
        comparison = self.comparison_record()
        if comparison is None:
            messagebox.showinfo(
                "Автовыравнивание по X", "Сначала откройте график «Сравнение» слева.", parent=self,
            )
            return False
        spectrum_ids = [int(value) for value in comparison.get("spectrum_ids", []) if int(value) in self.spectra]
        if len(spectrum_ids) < 2:
            messagebox.showinfo(
                "Автовыравнивание по X", "Для выравнивания нужны как минимум два спектра.", parent=self,
            )
            return False

        reference_id = spectrum_ids[0]
        reference = self.spectra[reference_id]
        offsets = comparison.setdefault("offsets", {})
        reference_offset = float(offsets.get(reference_id, offsets.get(str(reference_id), 0.0)))
        shown_left, shown_right = sorted(self.axes.get_xlim())
        # Окно на графике выражено уже в отображаемой системе p + offset.
        # Алгоритм получает исходную ось опорного спектра.
        raw_left, raw_right = shown_left - reference_offset, shown_right - reference_offset
        proposed = {reference_id: reference_offset}
        aligned: list[tuple[str, float, float]] = []
        failures: list[str] = []

        for spectrum_id in spectrum_ids[1:]:
            record = self.spectra[spectrum_id]
            try:
                result = find_x_shift(
                    np.asarray(reference["ppm"]), self.record_signal(reference),
                    np.asarray(record["ppm"]), self.record_signal(record),
                    raw_left, raw_right,
                )
            except AlignmentError as error:
                failures.append(f"{record['name']}: {error}")
                continue
            proposed[spectrum_id] = reference_offset + result.shift
            aligned.append((str(record["name"]), result.shift, result.correlation))

        if not aligned:
            messagebox.showwarning(
                "Автовыравнивание по X",
                "Ни один спектр не удалось совместить с опорным.\n\n" + "\n".join(failures[:4]),
                parent=self,
            )
            return False

        changed = any(
            not np.isclose(float(offsets.get(spectrum_id, offsets.get(str(spectrum_id), 0.0))), offset,
                          atol=5e-7)
            for spectrum_id, offset in proposed.items()
        )
        if changed:
            self.remember_comparison_offsets(comparison, "автовыравнивание спектров по X")
            offsets.update(proposed)
            self.draw_comparison()

        average = float(np.mean([score for _name, _shift, score in aligned]))
        self.status.set(
            f"Автовыравнивание: совмещено {len(aligned)} из {len(spectrum_ids) - 1}; "
            f"средняя корреляция {average:.3f}. Ctrl+Z отменит сдвиги."
        )
        if failures:
            messagebox.showwarning(
                "Автовыравнивание по X",
                "Часть спектров не удалось совместить автоматически:\n\n" + "\n".join(failures[:6])
                + "\n\nОстальные линии уже выровнены; проблемные можно довести вручную.",
                parent=self,
            )
        return changed

    def create_span_selector(self) -> None:
        """Создать выделение заново: ``axes.clear()`` удаляет его художники."""
        if self.span is not None:
            self.span.disconnect_events()
        self.span = SpanSelector(self.axes, self.on_x_selected, "horizontal", useblit=True,
                                 props={"facecolor": "#4c78a8", "alpha": .22})
        self.span.set_active(self.integration_mode or self.x_zoom_mode)

    def restore_interaction_mode(self) -> None:
        """Вернуть выделение после закрытия модального окна или смены метода."""
        if self.span is not None:
            self.span.set_active(self.integration_mode or self.x_zoom_mode)

    def zoom_y(self, event) -> None:
        active_axes = self._comparison_axes if self.current_comparison_id is not None else [self.axes]
        if event.inaxes not in active_axes:
            return
        axis = event.inaxes
        bottom, top = axis.get_ylim()
        factor = .8 if event.button == "up" else 1.25
        # Ноль — фиксированная опора масштаба. Для спектра с пиками обоих
        # знаков он стоит по центру; для обычного NMR — в нижней десятой
        # части экрана, оставляя место для положительных пиков.
        if self.current_comparison_id is not None:
            comparison = self.comparison_record()
            assert comparison is not None
            if str(comparison.get("mode")) == "separate":
                # В вертикальном стеке нет единого физического нуля: уровни
                # рядов являются только визуальным размещением. Масштабируем
                # весь стек вокруг его центра, не склеивая линии обратно.
                centre = (bottom + top) / 2
                axis.set_ylim(centre - new_range / 2, centre + new_range / 2)
                self.canvas.draw_idle()
                return
            # В наложении сохраняем прежнюю логику, привязанную к базовой
            # линии первого спектра, а не к положению курсора.
            ids = [int(value) for value in comparison["spectrum_ids"]]
            visible = self.record_signal(self.spectra[ids[0]])
        else:
            visible = self.signal()[self.visible_slice()]
        new_range = (top - bottom) * factor
        if peak_polarity(visible) == 0:
            axis.set_ylim(-new_range / 2, new_range / 2)
        else:
            axis.set_ylim(-new_range / 10, new_range * 9 / 10)
        self.canvas.draw_idle()

    def toggle_x_zoom(self) -> None:
        if self.integration_mode:
            self.integration_mode = False
            self.integration_button.configure(relief="raised")
            self.hide_integral_table()
            self.x_zoom_mode = True
        else:
            self.x_zoom_mode = not self.x_zoom_mode
        assert self.span is not None
        self.span.set_active(self.x_zoom_mode)
        self.zoom_button.configure(relief="sunken" if self.x_zoom_mode else "raised")
        self.status.set("Протяните мышью нужный диапазон ppm." if self.x_zoom_mode else "Выбор диапазона X выключен.")

    def on_x_selected(self, xmin: float, xmax: float) -> None:
        if abs(xmax - xmin) < 1e-9:
            return
        if self.integration_mode:
            if self.current_comparison_id is not None:
                self.add_comparison_integral(xmin, xmax)
            else:
                self.add_integral(xmin, xmax)
            return
        self.axes.set_xlim(max(xmin, xmax), min(xmin, xmax))
        self.x_zoom_mode = False
        assert self.span is not None
        self.span.set_active(False)
        self.zoom_button.configure(relief="raised")
        self.canvas.draw_idle()
        self.status.set("Выбранный диапазон будет использован для коррекции базовой линии.")

    def reset_x_zoom(self) -> None:
        if self.current_comparison_id is not None:
            self.draw_comparison(reset_x=True)
        elif self.ppm is not None:
            self.axes.set_xlim(float(self.ppm.max()), float(self.ppm.min()))
            self.canvas.draw_idle()

    def show_integral_table(self) -> None:
        """Показать таблицу вторым окном вертикального разделителя."""
        if not self._integral_table_visible:
            self.spectrum_split.add(self.table_frame, minsize=120)
            self._integral_table_visible = True
        self.after_idle(self.apply_graph_height)

    def hide_integral_table(self) -> None:
        if self._integral_table_visible:
            self.spectrum_split.forget(self.table_frame)
            self._integral_table_visible = False

    def apply_graph_height(self) -> None:
        """Установить положение разделителя в процентах от доступной высоты."""
        if not self._integral_table_visible:
            return
        self.update_idletasks()
        total_height = self.spectrum_split.winfo_height()
        if total_height < 250:
            return
        # Оставляем таблице как минимум 120 px даже при слишком маленьком окне.
        graph_height = min(int(total_height * self.graph_height_percent / 100), total_height - 120)
        graph_height = max(180, graph_height)
        try:
            self.spectrum_split.sash_place(0, 0, graph_height)
        except tk.TclError:
            # До первого размещения панелей sash ещё может не существовать.
            self.after(40, self.apply_graph_height)

    def configure_graph_size(self) -> None:
        """Точная настройка высоты графика; разделитель также можно тянуть."""
        if not self.integration_mode:
            messagebox.showinfo(
                "Размер графика",
                "Включите режим интегрирования: появится таблица и разделитель высоты.",
                parent=self,
            )
            return
        percent = simpledialog.askinteger(
            "Размер графика",
            "Высота графика, % от доступного пространства (30–85):\n"
            "Разделитель между графиком и таблицей также можно перетянуть мышью.",
            initialvalue=self.graph_height_percent, minvalue=30, maxvalue=85, parent=self,
        )
        if percent is not None:
            self.graph_height_percent = percent
            self.apply_graph_height()

    def toggle_integration(self) -> None:
        if self.real is None and self.current_comparison_id is None:
            messagebox.showinfo("Нет спектра", "Сначала загрузите спектр.", parent=self)
            return
        self.integration_mode = not self.integration_mode
        self.x_zoom_mode = False
        self.zoom_button.configure(relief="raised")
        assert self.span is not None
        self.span.set_active(self.integration_mode)
        self.zoom_button.configure(relief="raised")
        self.integration_button.configure(relief="sunken" if self.integration_mode else "raised")
        if self.integration_mode:
            self.show_integral_table()
            if self.current_comparison_id is not None:
                comparison = self.comparison_record()
                if comparison is not None:
                    self.update_comparison_integral_table(comparison)
            else:
                self.update_integral_table(self.ordered_integrals())
        else:
            self.hide_integral_table()
        self.status.set(
            "Режим интегрирования: протяните мышью границы пика."
            if self.integration_mode else "Режим интегрирования выключен."
        )

    def add_integral(self, xmin: float, xmax: float) -> None:
        left, right = min(xmin, xmax), max(xmin, xmax)
        if self.ppm is None or np.count_nonzero((self.ppm >= left) & (self.ppm <= right)) < 3:
            return
        if self.integration_method.get() == self.DECONVOLUTION_METHOD:
            try:
                self.lorentzian_fit_in_region(self.ppm, self.signal(), left, right)
            except (ValueError, RuntimeError, FloatingPointError) as error:
                messagebox.showwarning("Деконволюция", str(error), parent=self)
                return
        previous = self.capture_spectrum_state()
        integral_id = self._next_integral_id
        self._next_integral_id += 1
        self.integrals.append({"id": integral_id, "left": left, "right": right,
                               "color": self.integral_colors[(integral_id - 1) % len(self.integral_colors)],
                               "name": f"Интеграл {integral_id}",
                               "number": max((int(item.get("number", 0)) for item in self.integrals), default=0) + 1})
        self.remember_spectrum_state(previous, "добавление интеграла")
        self.draw()
        if self.integration_method.get() == self.DECONVOLUTION_METHOD:
            self.status.set("Интеграл добавлен: пунктиром показан лоренциан, в таблице — его полная площадь.")
        else:
            self.status.set("Интеграл добавлен. Правый клик по нему открывает действия.")

    def add_comparison_integral(self, xmin: float, xmax: float) -> None:
        """Один выделенный диапазон применяется ко всем линиям обзора."""
        comparison = self.comparison_record()
        if comparison is None:
            return
        left, right = min(xmin, xmax), max(xmin, xmax)
        if self.integration_method.get() == self.DECONVOLUTION_METHOD:
            offsets = comparison.get("offsets", {})
            failed: list[str] = []
            for spectrum_id in [int(value) for value in comparison.get("spectrum_ids", [])]:
                record = self.spectra.get(spectrum_id)
                if record is None:
                    continue
                offset = float(offsets.get(spectrum_id, offsets.get(str(spectrum_id), 0.0)))
                try:
                    self.lorentzian_fit_in_region(
                        np.asarray(record["ppm"]), self.record_signal(record), left, right, x_offset=offset,
                    )
                except (ValueError, RuntimeError, FloatingPointError):
                    failed.append(str(record["name"]))
            if failed:
                messagebox.showwarning(
                    "Деконволюция",
                    "Для некоторых спектров в выбранном диапазоне нельзя надёжно построить лоренциан:\n"
                    + ", ".join(failed), parent=self,
                )
                return
        self.remember_comparison_integrals(comparison, "добавление общего интеграла")
        integral_id = int(comparison.get("next_integral_id", 1))
        comparison["next_integral_id"] = integral_id + 1
        integrals = comparison.setdefault("integrals", [])
        next_number = max((int(item.get("number", 0)) for item in integrals), default=0) + 1
        integrals.append({
            "id": integral_id, "number": next_number, "left": left, "right": right,
            "color": self.integral_colors[(integral_id - 1) % len(self.integral_colors)],
        })
        self.draw_comparison()
        if self.integration_method.get() == self.DECONVOLUTION_METHOD:
            self.status.set("Общий интеграл добавлен: для каждого спектра рассчитана полная площадь его лоренциана.")
        else:
            self.status.set("Общий интеграл добавлен: площади рассчитаны отдельно для каждого спектра.")

    @staticmethod
    def comparison_integral_at(comparison: dict[str, object], ppm: float) -> dict[str, object] | None:
        """Вернуть верхний общий интеграл, в чьи пределы попала мышь."""
        return next((item for item in reversed(list(comparison.get("integrals", [])))
                     if float(item["left"]) <= ppm <= float(item["right"])), None)

    def remove_comparison_integral(self, item: dict[str, object]) -> None:
        comparison = self.comparison_record()
        if comparison is None:
            return
        integrals = comparison.get("integrals", [])
        if item not in integrals:
            return
        number = int(item.get("number", 0))
        self.remember_comparison_integrals(comparison, f"удаление общего интеграла I{number}")
        integrals.remove(item)
        self.draw_comparison()
        self.status.set(f"Общий интеграл I{number} удалён. Ctrl+Z вернёт его.")

    def ordered_integrals(self) -> list[dict[str, float | int | str]]:
        """Интегралы в заданном пользователем порядке номеров."""
        return sorted(self.integrals, key=lambda item: (int(item.get("number", 10**9)), -float(item["left"]) - float(item["right"])))

    def integral_area(self, item: dict[str, float | int | str]) -> float:
        assert self.ppm is not None
        if self.active_spectrum_id in self.spectra:
            return self.record_integral_area(self.spectra[self.active_spectrum_id],
                                             float(item["left"]), float(item["right"]))
        mask = (self.ppm >= float(item["left"])) & (self.ppm <= float(item["right"]))
        if np.count_nonzero(mask) < 2:
            return 0.0
        signal = self.signal()
        method = self.integration_method.get()
        if method == self.DECONVOLUTION_METHOD:
            return self.lorentzian_integral_area(self.ppm, signal, float(item["left"]), float(item["right"]))
        values = signal[mask]
        if method == "Сумма модулей относительно 0":
            values = np.abs(values)
        return abs(float(np.trapezoid(values, self.ppm[mask])))

    def integral_display_value(self, item: dict[str, float | int | str]) -> float:
        """Число для таблицы: физическая площадь либо сумма её отсчётов."""
        if self.integration_method.get() == self.DECONVOLUTION_METHOD:
            return self.integral_area(item)
        if self.integration_display.get() != "Сумма дискретных точек (MestReNova)":
            return self.integral_area(item)
        assert self.ppm is not None
        if self.active_spectrum_id in self.spectra:
            return self.record_integral_display_value(self.spectra[self.active_spectrum_id],
                                                      float(item["left"]), float(item["right"]))
        mask = (self.ppm >= float(item["left"])) & (self.ppm <= float(item["right"]))
        if np.count_nonzero(mask) < 2:
            return 0.0
        values = self.signal()[mask]
        if self.integration_method.get() == "Сумма модулей относительно 0":
            values = np.abs(values)
        return abs(float(np.sum(values)))

    def sync_integration_method_choice(self) -> None:
        """Синхронизировать combobox и радиокнопки с активным спектром."""
        self.integration_choice.set(self.integration_method.get())
        self.integration_display_choice.set(self.integration_display.get())

    def change_integration_method(self, method: str) -> None:
        """Сменить метод интегрирования как обычное отменяемое действие."""
        method = self.normalize_integration_method(method)
        previous = self.capture_spectrum_state()
        self.integration_method.set(method)
        self.sync_integration_method_choice()
        self.remember_spectrum_state(previous, "смена способа расчёта интегралов")
        self.refresh_integrals()

    def change_integration_display(self, display: str) -> None:
        """Изменить только единицы вывода таблицы, не затрагивая формулы."""
        if display not in self.INTEGRATION_DISPLAYS:
            display = "Площадь"
        previous = self.capture_spectrum_state()
        self.integration_display.set(display)
        self.sync_integration_method_choice()
        self.remember_spectrum_state(previous, "смена способа вывода интегралов")
        self.refresh_integrals()

    def refresh_integrals(self) -> None:
        """Пересчитать таблицу после смены способа интегрирования."""
        if self.integration_method.get() == self.DECONVOLUTION_METHOD:
            # При этом способе пунктирная fitted-кривая — часть результата,
            # поэтому требуется полноценная перерисовка, а не только таблица.
            self.draw()
            return
        comparison = self.comparison_record()
        if comparison is not None:
            self.update_comparison_integral_table(comparison)
        else:
            self.update_integral_table(self.ordered_integrals())
        self.canvas.draw_idle()

    def draw_integrals(self) -> None:
        if self.ppm is None:
            return
        ordered = self.ordered_integrals()
        ymin, ymax = self.axes.get_ylim()
        for item in ordered:
            number = self.integral_number(item["id"])
            left, right = float(item["left"]), float(item["right"])
            color = str(item["color"])
            self.axes.axvspan(left, right, color=color, alpha=.12)
            self.axes.axvline(left, color=color, linewidth=1.2)
            self.axes.axvline(right, color=color, linewidth=1.2)
            self.axes.text((left + right) / 2, ymax - (ymax - ymin) * .035, number, color=color,
                           ha="center", va="top", fontsize=11, fontweight="bold")
            if self.integration_method.get() == self.DECONVOLUTION_METHOD:
                try:
                    fit = self.lorentzian_fit_in_region(self.ppm, self.signal(), left, right)
                except (ValueError, RuntimeError, FloatingPointError):
                    continue
                half_width = float(fit["half_width"])
                center = float(fit["center"])
                extent = max(4.0 * half_width, (right - left) / 2.0)
                curve_x = np.linspace(
                    max(float(self.ppm.min()), center - extent),
                    min(float(self.ppm.max()), center + extent), 240,
                )
                curve_y = (
                    lorentzian_component(curve_x, fit["amplitude"], center, half_width)
                    + fit["background_offset"]
                    + fit["background_slope"] * (curve_x - fit["background_center"])
                )
                self.axes.plot(curve_x, curve_y, color=color, linestyle="--", linewidth=1.25, alpha=.95)
        self.update_integral_table(ordered)

    def update_integral_table(self, ordered: list[dict[str, float | int | str]]) -> None:
        self.configure_normal_integral_table()
        for row in self.integral_table.get_children():
            self.integral_table.delete(row)
        reference_area = next((self.integral_display_value(item) for item in self.integrals
                               if int(item["id"]) == self.reference_integral_id), None)
        for item in ordered:
            number = self.integral_number(item["id"])
            area = self.integral_display_value(item)
            relative = "—" if not reference_area else f"{area / reference_area:.5g}"
            name = str(item["name"]) + (" (1.000)" if int(item["id"]) == self.reference_integral_id else "")
            limits = f"{float(item['left']):.4f} … {float(item['right']):.4f}"
            self.integral_table.insert("", "end", iid=str(item["id"]),
                                       values=(number, name, limits, f"{area:.6g}", relative))
        if hasattr(self, "formula_canvas"):
            self.refresh_formula_workspace()

    def configure_normal_integral_table(self) -> None:
        columns = ("number", "name", "limits", "area", "relative")
        if self.integration_method.get() == self.DECONVOLUTION_METHOD:
            area_heading = "Площадь Лоренца"
        else:
            area_heading = ("Сумма точек" if self.integration_display.get() == "Сумма дискретных точек (MestReNova)"
                            else "Площадь")
        headings = {"number": "№", "name": "Название", "limits": "Пределы, ppm",
                    "area": area_heading, "relative": "Отн. площадь"}
        widths = {"number": 45, "name": 140, "limits": 180, "area": 160, "relative": 130}
        self.integral_table.configure(columns=columns, show="headings")
        for column in columns:
            self.integral_table.heading(column, text=headings[column])
            self.integral_table.column(column, width=widths[column], anchor="center", stretch=True)

    def update_comparison_integral_table(self, comparison: dict[str, object]) -> None:
        """Таблица обзора: одна строка на спектр, одна колонка на диапазон."""
        integrals = sorted(list(comparison.get("integrals", [])), key=lambda item: int(item["number"]))
        columns = ("spectrum",) + tuple(f"i_{int(item['id'])}" for item in integrals)
        self.integral_table.configure(columns=columns, show="headings")
        self.integral_table.heading("spectrum", text="Спектр")
        self.integral_table.column("spectrum", width=155, anchor="w", stretch=True)
        for item in integrals:
            column = f"i_{int(item['id'])}"
            self.integral_table.heading(column, text=f"I{int(item['number'])}")
            self.integral_table.column(column, width=120, anchor="center", stretch=True)
        for row in self.integral_table.get_children():
            self.integral_table.delete(row)
        offsets = comparison.get("offsets", {})
        for spectrum_id in [int(value) for value in comparison.get("spectrum_ids", [])]:
            record = self.spectra.get(spectrum_id)
            if record is None:
                continue
            offset = float(offsets.get(spectrum_id, offsets.get(str(spectrum_id), 0.0)))
            values = [str(record["name"])]
            values.extend(f"{self.record_integral_display_value(record, float(item['left']), float(item['right']), x_offset=offset):.6g}"
                          for item in integrals)
            self.integral_table.insert("", "end", iid=f"s{ spectrum_id }", values=values)
        if hasattr(self, "formula_canvas"):
            self.refresh_formula_workspace()

    def copy_treeview_rows(self, table: ttk.Treeview, *, context: str) -> bool:
        """Скопировать выделенные строки Treeview либо всю таблицу в TSV для Excel."""
        row_ids = list(table.selection()) or list(table.get_children())
        columns = tuple(str(column) for column in table.cget("columns"))
        if not row_ids or not columns:
            return False
        header = [str(table.heading(column, "text")) for column in columns]
        rows = ["\t".join(str(value) for value in table.item(row_id, "values")) for row_id in row_ids]
        self.clipboard_clear()
        self.clipboard_append("\r\n".join(("\t".join(header), *rows)))
        amount = f"выделенных строк: {len(row_ids)}" if table.selection() else f"строк: {len(row_ids)}"
        if context == "formula":
            self.formula_status.set(f"Таблица результатов скопирована в буфер ({amount}).")
        else:
            self.status.set(f"Таблица интегралов скопирована в буфер ({amount}).")
        return True

    def copy_integral_table(self, _event=None) -> str:
        """Кнопка и Ctrl+C для таблицы обычных либо общих интегралов."""
        if not self.copy_treeview_rows(self.integral_table, context="integrals"):
            self.status.set("В таблице интегралов пока нет данных для копирования.")
        return "break"

    def copy_integral_table_shortcut(self, event) -> str | None:
        keysym = str(getattr(event, "keysym", "")).lower()
        is_c_key = keysym in {"c", "cyrillic_es", "с"} or int(getattr(event, "keycode", -1)) == 67
        return self.copy_integral_table(event) if is_c_key else None

    def integral_table_menu(self, event) -> str:
        """Контекстное меню строки таблицы; номер не связан с внутренним ID."""
        row_id = self.integral_table.identify_row(event.y)
        comparison = self.comparison_record()
        if comparison is not None:
            column = self.integral_table.identify_column(event.x)
            try:
                column_index = int(column.removeprefix("#")) - 1
            except ValueError:
                return "break"
            integrals = sorted(list(comparison.get("integrals", [])), key=lambda item: int(item["number"]))
            # Первый столбец — название спектра; остальные соответствуют I1,
            # I2… и потому подходят для удаления общей области.
            if not row_id or not 1 <= column_index <= len(integrals):
                return "break"
            item = integrals[column_index - 1]
            self.integral_table.selection_set(row_id)
            menu = tk.Menu(self, tearoff=False)
            menu.add_command(label=f"Убрать общий интеграл I{int(item['number'])}",
                             command=lambda selected=item: self.remove_comparison_integral(selected))
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()
            return "break"
        if not row_id:
            return "break"
        try:
            item = self.integral_by_id(int(row_id))
        except ValueError:
            item = None
        if item is None:
            return "break"
        self.integral_table.selection_set(row_id)
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="Изменить номер…", command=lambda: self.rename_integral_number(item))
        menu.add_command(label="Переименовать…", command=lambda: self.rename_integral(item))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    # ---------- Первый прототип вкладки формул (не используется) ----------
    def build_formula_workspace_prototype(self) -> None:
        tk.Label(
            self.formula_tab,
            text="Перетащите цветной блок площади в слот операции. Правый клик по полю добавляет блоки.",
            anchor="w", padx=10, pady=8,
        ).pack(fill="x")
        self.formula_canvas = tk.Canvas(self.formula_tab, bg="#f7f7f7", highlightthickness=0)
        self.formula_canvas.pack(fill="both", expand=True)
        self.formula_blocks: dict[int, dict[str, object]] = {}
        self._next_formula_id = 1
        self._formula_drag: dict[str, float | int] | None = None
        self.formula_canvas.bind("<Button-3>", self.formula_menu)
        self.refresh_formula_workspace()

    def on_tab_changed(self, _event: object) -> None:
        if self.notebook.select() == str(self.formula_tab):
            self.refresh_formula_workspace()

    def refresh_formula_workspace_prototype(self) -> None:
        """Обновить палитру площадей и значения ранее созданных блоков."""
        self.formula_canvas.delete("palette")
        self.formula_canvas.create_text(12, 16, text="Площади интегралов — перетащите блок в рабочую область:",
                                        anchor="w", fill="#444", tags="palette")
        x = 14
        ordered = self.ordered_integrals()
        for number, item in enumerate(ordered, start=1):
            integral_id = int(item["id"])
            color = str(item["color"])
            tag = f"palette_{integral_id}"
            self.formula_canvas.create_rectangle(x, 30, x + 62, 70, fill=color, outline="", tags=("palette", tag))
            self.formula_canvas.create_text(x + 31, 50, text=f"{number}\nI{number}", fill="white",
                                            font=("TkDefaultFont", 9, "bold"), tags=("palette", tag))
            self.formula_canvas.tag_bind(tag, "<ButtonPress-1>",
                                         lambda event, iid=integral_id: self.start_palette_drag(event, iid))
            x += 72
        if not ordered:
            self.formula_canvas.create_text(14, 50, text="Сначала создайте хотя бы один интеграл на вкладке «Спектр и интегралы».",
                                            anchor="w", fill="#777", tags="palette")
        self.render_formula_blocks()

    def start_palette_drag_prototype(self, event, integral_id: int) -> None:
        block_id = self.add_formula_block("area", event.x, event.y, integral_id=integral_id)
        self.start_formula_drag(event, block_id)

    def add_formula_block_prototype(self, kind: str, x: float, y: float, *, integral_id: int | None = None,
                                    operation: str | None = None) -> int:
        block_id = self._next_formula_id
        self._next_formula_id += 1
        self.formula_blocks[block_id] = {
            "kind": kind, "x": x, "y": y, "integral_id": integral_id,
            "operation": operation, "inputs": [None] if operation == "abs" else [None, None],
            "input": None,
        }
        self.render_formula_blocks()
        return block_id

    def render_formula_blocks_prototype(self) -> None:
        self.formula_canvas.delete("formula_block")
        for block_id, block in self.formula_blocks.items():
            kind, x, y = str(block["kind"]), float(block["x"]), float(block["y"])
            tag = ("formula_block", f"block_{block_id}")
            if kind == "area":
                item = self.integral_by_id(block.get("integral_id"))
                color = str(item["color"]) if item else "#999"
                number = self.integral_number(block.get("integral_id"))
                value = self.formula_value(block_id)
                self.formula_canvas.create_rectangle(x, y, x + 70, y + 48, fill=color, outline="", tags=tag)
                self.formula_canvas.create_text(x + 35, y + 24, text=f"I{number}\n{self.format_formula_value(value)}",
                                                fill="white", font=("TkDefaultFont", 9, "bold"), tags=tag)
            elif kind == "operation":
                operation = str(block["operation"])
                self.formula_canvas.create_rectangle(x, y, x + 250, y + 82, fill="#ffbf69", outline="#b66c13", tags=tag)
                self.formula_canvas.create_text(x + 125, y + 15, text=f"Операция: {operation}", tags=tag)
                inputs = block["inputs"]
                slots = [(x + 14, x + 104)] if operation == "abs" else [(x + 14, x + 104), (x + 146, x + 236)]
                for index, (left, right) in enumerate(slots):
                    self.formula_canvas.create_rectangle(left, y + 29, right, y + 60, fill="#fff5df", outline="#b66c13", tags=tag)
                    label = self.formula_block_label(inputs[index])
                    self.formula_canvas.create_text((left + right) / 2, y + 44, text=label, tags=tag)
                self.formula_canvas.create_text(x + 125, y + 71, text=f"= {self.format_formula_value(self.formula_value(block_id))}", tags=tag)
            else:
                self.formula_canvas.create_rectangle(x, y, x + 210, y + 58, fill="#8ac926", outline="#4c7b13", tags=tag)
                self.formula_canvas.create_text(x + 105, y + 16, text="Вывод значения", font=("TkDefaultFont", 9, "bold"), tags=tag)
                self.formula_canvas.create_rectangle(x + 12, y + 27, x + 198, y + 49, fill="#efffdc", outline="#4c7b13", tags=tag)
                self.formula_canvas.create_text(x + 105, y + 38,
                                                text=f"{self.formula_block_label(block.get('input'))}  →  {self.format_formula_value(self.formula_value(block_id))}", tags=tag)
            self.formula_canvas.tag_bind(f"block_{block_id}", "<ButtonPress-1>",
                                         lambda event, bid=block_id: self.start_formula_drag(event, bid))
            self.formula_canvas.tag_bind(f"block_{block_id}", "<B1-Motion>", self.drag_formula_block)
            self.formula_canvas.tag_bind(f"block_{block_id}", "<ButtonRelease-1>", self.drop_formula_block)

    def integral_by_id(self, integral_id: object) -> dict[str, float | int | str] | None:
        return next((item for item in self.integrals if int(item["id"]) == integral_id), None)

    def integral_number(self, integral_id: object) -> str:
        for fallback, item in enumerate(self.ordered_integrals(), start=1):
            if int(item["id"]) == integral_id:
                number = item.get("number", fallback)
                return str(number)
        return "?"

    def formula_block_label_prototype(self, block_id: object) -> str:
        if not isinstance(block_id, int) or block_id not in self.formula_blocks:
            return "перетащите блок"
        block = self.formula_blocks[block_id]
        return f"I{self.integral_number(block.get('integral_id'))}" if block["kind"] == "area" else f"Блок {block_id}"

    def formula_value_prototype(self, block_id: object, visited: set[int] | None = None) -> float | None:
        if not isinstance(block_id, int) or block_id not in self.formula_blocks:
            return None
        visited = set() if visited is None else visited
        if block_id in visited:
            return None
        visited.add(block_id)
        block = self.formula_blocks[block_id]
        if block["kind"] == "area":
            item = self.integral_by_id(block.get("integral_id"))
            # Если в загруженном пресете нет соответствующего интеграла,
            # площадь является нулём, а не «неопределённым» значением всей
            # формулы. Это позволяет пересчитать остальные ветви схемы.
            return self.integral_area(item) if item else 0.0
        if block["kind"] == "output":
            return self.formula_value(block.get("input"), visited)
        values = [self.formula_value(value, visited.copy()) for value in block["inputs"]]
        if any(value is None for value in values):
            return None
        a, b = values[0], values[0] if len(values) == 1 else values[1]
        try:
            return {"+": a + b, "−": a - b, "×": a * b, "÷": a / b, "^": a ** b,
                    "abs": abs(a)}[str(block["operation"])]
        except ZeroDivisionError:
            return None

    @staticmethod
    def format_formula_value(value: float | None) -> str:
        return "—" if value is None or not np.isfinite(value) else f"{value:.6g}"

    def start_formula_drag_prototype(self, event, block_id: int) -> None:
        self._formula_drag = {"id": block_id, "x": event.x, "y": event.y}
        self.formula_canvas.tag_raise(f"block_{block_id}")

    def drag_formula_block_prototype(self, event) -> None:
        if self._formula_drag is None:
            return
        block_id = int(self._formula_drag["id"])
        dx, dy = event.x - float(self._formula_drag["x"]), event.y - float(self._formula_drag["y"])
        self.formula_blocks[block_id]["x"] = float(self.formula_blocks[block_id]["x"]) + dx
        self.formula_blocks[block_id]["y"] = float(self.formula_blocks[block_id]["y"]) + dy
        self._formula_drag["x"], self._formula_drag["y"] = event.x, event.y
        self.formula_canvas.move(f"block_{block_id}", dx, dy)

    def drop_formula_block_prototype(self, event) -> None:
        if self._formula_drag is None:
            return
        source_id = int(self._formula_drag["id"])
        for target_id, target in self.formula_blocks.items():
            if target_id == source_id:
                continue
            x, y = float(target["x"]), float(target["y"])
            if target["kind"] == "operation":
                slots = [(x + 14, x + 104)] if target["operation"] == "abs" else [(x + 14, x + 104), (x + 146, x + 236)]
                for index, (left, right) in enumerate(slots):
                    if left <= event.x <= right and y + 29 <= event.y <= y + 60:
                        target["inputs"][index] = source_id
            elif target["kind"] == "output" and x + 12 <= event.x <= x + 198 and y + 27 <= event.y <= y + 49:
                target["input"] = source_id
        self._formula_drag = None
        self.render_formula_blocks()

    def formula_menu_prototype(self, event) -> None:
        menu = tk.Menu(self, tearoff=False)
        area_menu = tk.Menu(menu, tearoff=False)
        for item in self.ordered_integrals():
            number = self.integral_number(item["id"])
            area_menu.add_command(label=f"Площадь I{number}", command=lambda i=int(item["id"]): self.add_formula_block("area", event.x, event.y, integral_id=i))
        menu.add_cascade(label="Площадь", menu=area_menu)
        operation_menu = tk.Menu(menu, tearoff=False)
        for operation in ("+", "−", "×", "÷", "^", "abs"):
            operation_menu.add_command(label=operation, command=lambda op=operation: self.add_formula_block("operation", event.x, event.y, operation=op))
        menu.add_cascade(label="Мат. операция", menu=operation_menu)
        menu.add_command(label="Блок вывода", command=lambda: self.add_formula_block("output", event.x, event.y))
        menu.tk_popup(event.x_root, event.y_root)

    # ---------- Улучшенный визуальный калькулятор (переопределяет прототип выше) ----------
    _UNARY_OPERATIONS = {"abs", "√", "ln", "log₁₀", "exp", "round", "−x"}
    _OPERATION_NAMES = ("+", "−", "×", "÷", "^", "%", "abs", "√", "ln", "log₁₀", "exp", "round", "−x", "min", "max")

    def build_formula_workspace(self) -> None:
        """Создать аккуратную Scratch-подобную вкладку для живых расчётов."""
        header = tk.Frame(self.formula_tab, bg="#ffffff", padx=14, pady=10)
        header.pack(fill="x")
        tk.Label(header, text="Визуальные формулы", font=("Segoe UI", 14, "bold"), bg="#ffffff").pack(side="left")
        tk.Button(header, text="Загрузить пресет…", command=self.load_formula_preset).pack(side="right")
        tk.Button(header, text="Сохранить пресет…", command=self.save_formula_preset).pack(side="right", padx=(0, 6))
        self.formula_status = tk.StringVar(
            value="Тяните провод от оранжевого выхода к синему входу; пустой фон перемещает поле."
        )
        tk.Label(header, textvariable=self.formula_status, fg="#68707d", bg="#ffffff").pack(side="left", padx=16)
        layout = tk.Frame(self.formula_tab, bg="#e9edf2", padx=10, pady=10)
        layout.pack(fill="both", expand=True)

        palette = tk.LabelFrame(layout, text="  Источники  ", bg="#ffffff", padx=8, pady=8, width=215)
        palette.pack(side="left", fill="y", padx=(0, 10))
        palette.pack_propagate(False)
        palette_body = tk.Frame(palette, bg="#ffffff")
        palette_body.pack(fill="both", expand=True)
        palette_scroll = tk.Scrollbar(palette_body, orient="vertical")
        palette_scroll.pack(side="right", fill="y")
        self.palette_canvas = tk.Canvas(
            palette_body, bg="#ffffff", highlightthickness=0, width=195,
            yscrollcommand=palette_scroll.set,
        )
        self.palette_canvas.pack(side="left", fill="both", expand=True)
        palette_scroll.configure(command=self.palette_canvas.yview)
        self.palette_canvas.bind("<MouseWheel>", self.scroll_formula_palette)

        workspace = tk.LabelFrame(layout, text="  Рабочая область  ", bg="#ffffff", padx=0, pady=0)
        workspace.pack(side="left", fill="both", expand=True)
        toolbar = tk.Frame(workspace, bg="#f5f7fa", padx=8, pady=5)
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="Инструменты", bg="#f5f7fa", fg="#536171").pack(side="left", padx=(2, 8))
        self.formula_tool_buttons: dict[str, tk.Button] = {}
        for tool, icon, label in (
            ("pan", "✥", "Перемещение поля"),
            ("select", "▣", "Выделение блоков"),
            ("comment", "✎", "Комментарий"),
        ):
            button = tk.Button(toolbar, text=icon, width=3, relief="flat", bd=1,
                               command=lambda value=tool: self.set_formula_tool(value))
            button.pack(side="left", padx=2)
            self.formula_tool_buttons[tool] = button
            self.add_formula_tooltip(button, label)
        tk.Button(toolbar, text="Удалить выделенное", command=self.delete_selected_formula_blocks).pack(side="left", padx=(10, 2))
        tk.Label(toolbar, text="Delete", bg="#f5f7fa", fg="#8b96a3", font=("Segoe UI", 8)).pack(side="left")
        tk.Button(toolbar, text="Создать фрейм", command=self.create_formula_frame).pack(side="left", padx=(12, 2))
        workspace_body = tk.Frame(workspace, bg="#ffffff")
        workspace_body.pack(fill="both", expand=True)
        workspace_body.rowconfigure(0, weight=1)
        workspace_body.columnconfigure(0, weight=1)
        formula_x_scroll = tk.Scrollbar(workspace_body, orient="horizontal")
        formula_y_scroll = tk.Scrollbar(workspace_body, orient="vertical")
        self.formula_canvas = tk.Canvas(
            workspace_body, bg="#fbfcfe", highlightthickness=0, scrollregion=(0, 0, 1800, 1200),
            xscrollcommand=formula_x_scroll.set, yscrollcommand=formula_y_scroll.set,
        )
        self.formula_canvas.grid(row=0, column=0, sticky="nsew")
        formula_y_scroll.grid(row=0, column=1, sticky="ns")
        formula_x_scroll.grid(row=1, column=0, sticky="ew")
        formula_x_scroll.configure(command=self.formula_canvas.xview)
        formula_y_scroll.configure(command=self.formula_canvas.yview)
        self.formula_canvas.bind("<Button-3>", self.formula_menu)
        self.formula_canvas.bind("<ButtonPress-1>", self.start_formula_pan, add="+")
        self.formula_canvas.bind("<B1-Motion>", self.drag_formula_wire, add="+")
        self.formula_canvas.bind("<B1-Motion>", self.drag_formula_pan, add="+")
        self.formula_canvas.bind("<ButtonRelease-1>", self.drop_formula_wire, add="+")
        self.formula_canvas.bind("<ButtonRelease-1>", self.stop_formula_pan, add="+")
        self.formula_canvas.bind("<Configure>", lambda _event: self.draw_formula_grid())
        self.formula_canvas.bind("<Delete>", self.delete_selected_formula_blocks)
        self.formula_canvas.bind("<Escape>", lambda _event: self.clear_formula_selection())
        self._formula_world = (1800.0, 1200.0)

        results = tk.LabelFrame(layout, text="  Результаты  ", bg="#ffffff", padx=8, pady=8, width=220)
        results.pack(side="right", fill="y", padx=(10, 0))
        results.pack_propagate(False)
        result_body = tk.Frame(results, bg="#ffffff")
        result_body.pack(fill="both", expand=True)
        result_scroll = tk.Scrollbar(result_body, orient="vertical")
        result_scroll.pack(side="right", fill="y")
        self.result_canvas = tk.Canvas(result_body, bg="#ffffff", highlightthickness=0,
                                       yscrollcommand=result_scroll.set)
        self.result_canvas.pack(side="left", fill="both", expand=True)
        result_scroll.configure(command=self.result_canvas.yview)
        self.result_panel = tk.Frame(self.result_canvas, bg="#ffffff")
        self._result_panel_window = self.result_canvas.create_window((0, 0), window=self.result_panel, anchor="nw")
        self.formula_results_table: ttk.Treeview | None = None
        self.result_panel.bind("<Configure>", self.update_result_scrollregion)
        self.result_canvas.bind("<Configure>", self.resize_result_panel)
        self.result_canvas.bind("<MouseWheel>", self.scroll_result_panel)
        tk.Label(results, text="Правый клик по рабочей области\nдобавляет блоки.", bg="#ffffff",
                 fg="#68707d", justify="left").pack(anchor="w", pady=(12, 0))

        self.formula_blocks = {}
        self._next_formula_id = 1
        self._formula_drag = None
        self._formula_drag_origin = "workspace"
        self._wire_drag: dict[str, object] | None = None
        self._wire_handle_drag: dict[str, object] | None = None
        self._formula_pan = False
        self.formula_tool = "pan"
        self.selected_formula_blocks: set[int] = set()
        self._selection_drag: dict[str, float | int] | None = None
        self._selection_rect: int | None = None
        self.formula_comments: dict[int, dict[str, float | str]] = {}
        self._next_formula_comment_id = 1
        self.formula_frames: dict[int, dict[str, object]] = {}
        self._next_formula_frame_id = 1
        self._frame_drag: dict[str, float | int] | None = None
        self._comment_drag: dict[str, float | int] | None = None
        self.wire_routes: dict[str, tuple[float, float]] = {}
        self.set_formula_tool("pan")
        self.refresh_formula_workspace()

    def scroll_formula_palette(self, event) -> str:
        steps = max(1, abs(event.delta) // 120)
        self.palette_canvas.yview_scroll(-steps if event.delta > 0 else steps, "units")
        return "break"

    def update_result_scrollregion(self, _event=None) -> None:
        if hasattr(self, "result_canvas"):
            self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all") or (0, 0, 1, 1))

    def resize_result_panel(self, event) -> None:
        self.result_canvas.itemconfigure(self._result_panel_window, width=event.width)
        self.update_result_scrollregion()

    def scroll_result_panel(self, event) -> str:
        steps = max(1, abs(event.delta) // 120)
        self.result_canvas.yview_scroll(-steps if event.delta > 0 else steps, "units")
        return "break"

    def add_formula_tooltip(self, widget: tk.Widget, text: str) -> None:
        """Небольшая подсказка для кнопок, чтобы иконки не занимали место текстом."""
        tip: tk.Toplevel | None = None

        def show(_event=None) -> None:
            nonlocal tip
            if tip is not None:
                return
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{widget.winfo_rootx()}+{widget.winfo_rooty() + widget.winfo_height() + 3}")
            tk.Label(tip, text=text, bg="#293746", fg="white", padx=6, pady=3).pack()

        def hide(_event=None) -> None:
            nonlocal tip
            if tip is not None:
                tip.destroy()
                tip = None

        widget.bind("<Enter>", show, add="+")
        widget.bind("<Leave>", hide, add="+")

    def set_formula_tool(self, tool: str) -> None:
        if tool != "select" and hasattr(self, "selected_formula_blocks"):
            self.clear_formula_selection(render=False)
        self.formula_tool = tool
        labels = {
            "pan": "Режим перемещения поля.",
            "select": "Выделяйте рамкой; выбранные блоки можно переносить вместе или удалить клавишей Delete.",
            "comment": "Щёлкните по доске, чтобы добавить комментарий.",
        }
        for name, button in getattr(self, "formula_tool_buttons", {}).items():
            button.configure(relief="sunken" if name == tool else "flat",
                             bg="#dcecff" if name == tool else "#f5f7fa")
        if hasattr(self, "formula_status"):
            self.formula_status.set(labels[tool])
        if hasattr(self, "formula_canvas"):
            self.formula_canvas.configure(cursor={"pan": "fleur", "select": "crosshair", "comment": "pencil"}[tool])

    def clear_formula_selection(self, *, render: bool = True) -> None:
        """Снять выделение одной командой; новое рамочное выделение его заменяет."""
        self.selected_formula_blocks.clear()
        # Контуры выбранных блоков и временная рамка — это разные объекты
        # Canvas. Удаляем оба типа, иначе после удаления блока оставался
        # «висящий» пунктир.
        self.formula_canvas.delete("selection_outline")
        self.formula_canvas.delete("selection_marquee")
        self._selection_rect = None
        self._selection_drag = None
        if render and hasattr(self, "formula_canvas"):
            self.render_formula_blocks()

    def start_formula_selection(self, event) -> None:
        canvas = self.formula_canvas
        canvas.focus_set()
        current = canvas.find_withtag("current")
        if current and "formula_item" in canvas.gettags(current[0]):
            return
        x, y = float(canvas.canvasx(event.x)), float(canvas.canvasy(event.y))
        if not (event.state & 0x0001):
            # Новая область всегда заменяет старую, даже если она пуста.
            self.clear_formula_selection(render=False)
        if self._selection_rect is not None:
            canvas.delete(self._selection_rect)
        self._selection_rect = canvas.create_rectangle(x, y, x, y, outline="#3179c7", width=1,
                                                       dash=(4, 3), tags="selection_marquee")
        self._selection_drag = {"x": x, "y": y}

    def drag_formula_selection(self, event) -> None:
        if self._selection_drag is None or self._selection_rect is None:
            return
        x, y = float(self.formula_canvas.canvasx(event.x)), float(self.formula_canvas.canvasy(event.y))
        self.formula_canvas.coords(self._selection_rect, self._selection_drag["x"], self._selection_drag["y"], x, y)

    def finish_formula_selection(self, event) -> None:
        if self._selection_drag is None:
            return
        start_x, start_y = float(self._selection_drag["x"]), float(self._selection_drag["y"])
        end_x, end_y = float(self.formula_canvas.canvasx(event.x)), float(self.formula_canvas.canvasy(event.y))
        left, right = sorted((start_x, end_x))
        top, bottom = sorted((start_y, end_y))
        if abs(right - left) > 3 or abs(bottom - top) > 3:
            for block_id, block in self.formula_blocks.items():
                width, height = self.formula_block_size(block)
                x, y = float(block["x"]), float(block["y"])
                if x < right and x + width > left and y < bottom and y + height > top:
                    self.selected_formula_blocks.add(block_id)
        if self._selection_rect is not None:
            self.formula_canvas.delete(self._selection_rect)
        self._selection_rect = None
        self._selection_drag = None
        self.render_formula_blocks()

    def add_formula_comment_at(self, x: float, y: float) -> None:
        text = simpledialog.askstring("Комментарий", "Текст комментария:", parent=self)
        if not text or not text.strip():
            return
        comment_id = self._next_formula_comment_id
        self._next_formula_comment_id += 1
        self.formula_comments[comment_id] = {"x": x, "y": y, "text": text.strip()}
        self.render_formula_blocks()

    def start_formula_comment(self, event) -> None:
        self.add_formula_comment_at(float(self.formula_canvas.canvasx(event.x)), float(self.formula_canvas.canvasy(event.y)))

    def delete_selected_formula_blocks(self, _event=None) -> str | None:
        selected = set(self.selected_formula_blocks) & set(self.formula_blocks)
        if not selected:
            self.formula_status.set("Сначала выделите один или несколько блоков рамкой.")
            return "break" if _event is not None else None
        for block_id in selected:
            self.formula_blocks.pop(block_id, None)
        for block in self.formula_blocks.values():
            if block.get("input") in selected:
                block["input"] = None
            if block.get("inputs"):
                block["inputs"] = [None if value in selected else value for value in block["inputs"]]
        self.selected_formula_blocks.clear()
        self.cleanup_formula_frames()
        self.renumber_formula_outputs()
        self.cleanup_wire_routes()
        self.formula_status.set(f"Удалено блоков: {len(selected)}.")
        self.refresh_formula_workspace()
        # Не отдаём событие дальше Tk: это исключает появление меню рабочего
        # поля после нажатия клавиши Delete.
        return "break" if _event is not None else None

    def formula_frame_members(self, frame: dict[str, object]) -> list[int]:
        return [int(block_id) for block_id in frame.get("members", [])
                if isinstance(block_id, int) and block_id in self.formula_blocks]

    def formula_frame_terminals(self, frame: dict[str, object]) -> list[int]:
        members = self.formula_frame_members(frame)
        member_set = set(members)
        consumed = {
            int(source_id) for target_id in member_set for source_id in
            (self.formula_blocks[target_id].get("inputs", []) if self.formula_blocks[target_id]["kind"] == "operation"
             else []) if isinstance(source_id, int) and source_id in member_set
        }
        terminals = [block_id for block_id in members
                     if block_id not in consumed and self.formula_blocks[block_id]["kind"] != "output"]
        return terminals or [block_id for block_id in members if self.formula_blocks[block_id]["kind"] != "output"]

    def formula_frame_input_targets(self, frame: dict[str, object]) -> list[tuple[int, int]]:
        members = set(self.formula_frame_members(frame))
        targets: list[tuple[int, int]] = []
        for block_id in self.formula_frame_members(frame):
            block = self.formula_blocks[block_id]
            if block["kind"] == "operation":
                targets.extend((block_id, index) for index, source_id in enumerate(block.get("inputs", []))
                               if source_id not in members)
            elif block["kind"] == "output":
                if block.get("input") not in members:
                    targets.append((block_id, 0))
        return targets

    def formula_frame_bounds(self, frame: dict[str, object]) -> tuple[float, float, float, float]:
        members = self.formula_frame_members(frame)
        if not members:
            return float(frame.get("x", 40)), float(frame.get("y", 40)), 170.0, 72.0
        # Оставляем достаточно места у стенок для подписей и проводов портов.
        world_width, _world_height = self._formula_world
        left = max(10.0, min(float(self.formula_blocks[block_id]["x"]) for block_id in members) - 96)
        top = min(float(self.formula_blocks[block_id]["y"]) for block_id in members) - 34
        right = min(world_width - 10.0, max(float(self.formula_blocks[block_id]["x"]) +
                                              self.formula_block_size(self.formula_blocks[block_id])[0]
                                              for block_id in members) + 96)
        bottom = max(float(self.formula_blocks[block_id]["y"]) + self.formula_block_size(self.formula_blocks[block_id])[1]
                     for block_id in members) + 26
        port_rows = max(1, len(frame.get("inputs", [])), len(frame.get("outputs", [])))
        bottom = max(bottom, top + 54.0 + 21.0 * port_rows)
        return left, top, right - left, bottom - top

    def formula_frame_compact_size(self, frame: dict[str, object]) -> tuple[float, float]:
        ports = max(1, len(frame.get("inputs", [])), len(frame.get("outputs", [])))
        return 164.0, max(52.0, 28.0 + ports * 21.0)

    def create_formula_frame(self) -> None:
        members = sorted(set(self.selected_formula_blocks) & set(self.formula_blocks))
        if not members:
            self.formula_status.set("Выделите блоки рамкой, затем создайте из них фрейм.")
            return
        name = simpledialog.askstring("Новый фрейм", "Название фрейма:", initialvalue="Расчёт", parent=self)
        if not name or not name.strip():
            return
        frame_id = self._next_formula_frame_id
        self._next_formula_frame_id += 1
        temporary = {"members": members}
        x, y, _width, _height = self.formula_frame_bounds(temporary)
        terminals = self.formula_frame_terminals(temporary)
        input_targets = self.formula_frame_input_targets(temporary)
        self.formula_frames[frame_id] = {
            "name": name.strip(), "members": members, "collapsed": False, "x": x, "y": y,
            "inputs": [{"name": f"Вход {index + 1}", "target_id": target_id, "index": slot}
                       for index, (target_id, slot) in enumerate(input_targets)],
            "outputs": [{"name": f"Выход {index + 1}", "source_id": source_id}
                        for index, source_id in enumerate(terminals)] or
                       [{"name": "Выход 1", "source_id": None}],
        }
        self.selected_formula_blocks.clear()
        self.formula_status.set(f"Создан фрейм «{name.strip()}». ПКМ по его заголовку — настройки и сворачивание.")
        self.render_formula_blocks()

    def configure_formula_frame(self, frame_id: int) -> None:
        frame = self.formula_frames.get(frame_id)
        if frame is None:
            return
        name = simpledialog.askstring("Фрейм", "Название фрейма:", initialvalue=str(frame.get("name", "Фрейм")), parent=self)
        if not name or not name.strip():
            return
        input_count = simpledialog.askinteger("Фрейм", "Количество входов:",
                                              initialvalue=len(frame.get("inputs", [])), minvalue=0, maxvalue=8, parent=self)
        if input_count is None:
            return
        output_count = simpledialog.askinteger("Фрейм", "Количество выходов:",
                                               initialvalue=len(frame.get("outputs", [])), minvalue=1, maxvalue=8, parent=self)
        if output_count is None:
            return
        frame["name"] = name.strip()
        targets = self.formula_frame_input_targets(frame)
        if input_count > len(targets):
            messagebox.showinfo(
                "Фрейм",
                f"Внутри этой схемы доступны только {len(targets)} внешних входов. "
                "Количество входов будет ограничено этим числом.",
                parent=self,
            )
            input_count = len(targets)
        old_inputs = list(frame.get("inputs", []))
        frame["inputs"] = []
        for index in range(input_count):
            old = old_inputs[index] if index < len(old_inputs) and isinstance(old_inputs[index], dict) else {}
            target_id, slot = targets[index % len(targets)] if targets else (None, None)
            port_name = simpledialog.askstring("Вход фрейма", f"Название входа {index + 1}:",
                                               initialvalue=str(old.get("name", f"Вход {index + 1}")), parent=self)
            frame["inputs"].append({"name": (port_name or f"Вход {index + 1}").strip(),
                                    "target_id": target_id, "index": slot})
        terminals = self.formula_frame_terminals(frame)
        members = set(self.formula_frame_members(frame))
        old_outputs = list(frame.get("outputs", []))
        frame["outputs"] = []
        for index in range(output_count):
            old = old_outputs[index] if index < len(old_outputs) and isinstance(old_outputs[index], dict) else {}
            source_id = old.get("source_id") if isinstance(old.get("source_id"), int) else None
            if source_id not in members:
                source_id = terminals[index % len(terminals)] if terminals else None
            port_name = simpledialog.askstring("Выход фрейма", f"Название выхода {index + 1}:",
                                               initialvalue=str(old.get("name", f"Выход {index + 1}")), parent=self)
            frame["outputs"].append({"name": (port_name or f"Выход {index + 1}").strip(), "source_id": source_id})
        self.cleanup_formula_frames()
        self.render_formula_blocks()

    def toggle_formula_frame(self, frame_id: int) -> None:
        frame = self.formula_frames.get(frame_id)
        if frame is None:
            return
        if not bool(frame.get("collapsed")):
            x, y, _width, _height = self.formula_frame_bounds(frame)
            frame["x"], frame["y"] = x, y
        frame["collapsed"] = not bool(frame.get("collapsed"))
        self.render_formula_blocks()

    def ungroup_formula_frame(self, frame_id: int) -> None:
        if frame_id in self.formula_frames:
            self.formula_frames.pop(frame_id)
            self.render_formula_blocks()

    def cleanup_formula_frames(self) -> None:
        for frame_id in list(self.formula_frames):
            frame = self.formula_frames[frame_id]
            frame["members"] = self.formula_frame_members(frame)
            if not frame["members"]:
                self.formula_frames.pop(frame_id)
                continue
            members = set(frame["members"])
            frame["inputs"] = [port for port in frame.get("inputs", []) if isinstance(port, dict)
                               and port.get("target_id") in members]
            frame["outputs"] = [port for port in frame.get("outputs", []) if isinstance(port, dict)
                                and port.get("source_id") in members]
            if not frame["outputs"]:
                terminals = self.formula_frame_terminals(frame)
                frame["outputs"] = [{"name": "Выход 1", "source_id": terminals[-1] if terminals else None}]

    def collapsed_frame_output(self, source_id: int) -> tuple[int, int] | None:
        for frame_id, frame in self.formula_frames.items():
            if bool(frame.get("collapsed")):
                for index, port in enumerate(frame.get("outputs", [])):
                    if isinstance(port, dict) and port.get("source_id") == source_id:
                        return frame_id, index
        return None

    def collapsed_frame_input(self, target_id: int, index: int) -> tuple[int, int] | None:
        for frame_id, frame in self.formula_frames.items():
            if bool(frame.get("collapsed")):
                for port_index, port in enumerate(frame.get("inputs", [])):
                    if isinstance(port, dict) and port.get("target_id") == target_id and port.get("index") == index:
                        return frame_id, port_index
        return None

    def frame_output_for_connection(self, source_id: int, target_id: int) -> tuple[int, int] | None:
        """Публичный выход, если провод покидает фрейм."""
        for frame_id, frame in self.formula_frames.items():
            members = set(self.formula_frame_members(frame))
            if source_id not in members or target_id in members:
                continue
            for index, port in enumerate(frame.get("outputs", [])):
                if isinstance(port, dict) and port.get("source_id") == source_id:
                    return frame_id, index
        return None

    def frame_input_for_connection(self, source_id: int, target_id: int, index: int) -> tuple[int, int] | None:
        """Публичный вход, если провод приходит в фрейм снаружи."""
        for frame_id, frame in self.formula_frames.items():
            members = set(self.formula_frame_members(frame))
            if target_id not in members or source_id in members:
                continue
            for port_index, port in enumerate(frame.get("inputs", [])):
                if isinstance(port, dict) and port.get("target_id") == target_id and port.get("index") == index:
                    return frame_id, port_index
        return None

    def hidden_formula_block(self, block_id: int) -> bool:
        return any(bool(frame.get("collapsed")) and block_id in self.formula_frame_members(frame)
                   for frame in self.formula_frames.values())

    def formula_frame_at(self, x: float, y: float) -> int | None:
        """Найти фрейм по координате холста, даже если ПКМ попало в его фон."""
        for frame_id, frame in reversed(list(self.formula_frames.items())):
            if bool(frame.get("collapsed")):
                left, top = float(frame.get("x", 40.0)), float(frame.get("y", 40.0))
                width, height = self.formula_frame_compact_size(frame)
            else:
                left, top, width, height = self.formula_frame_bounds(frame)
            if left <= x <= left + width and top <= y <= top + height:
                return frame_id
        return None

    def draw_expanded_formula_frames(self) -> None:
        """Открытый фрейм: стенки, порты и видимые связи с содержимым."""
        canvas = self.formula_canvas
        for frame_id, frame in self.formula_frames.items():
            if bool(frame.get("collapsed")):
                continue
            x, y, width, height = self.formula_frame_bounds(frame)
            frame["x"], frame["y"] = x, y
            tags = ("formula_item", f"frame_{frame_id}", f"frame_body_{frame_id}")
            self.rounded_rect(canvas, x, y, x + width, y + height, 12,
                              fill="#f3f7fd", outline="#8aa8c9", width=1, tags=tags)
            canvas.create_rectangle(x, y, x + width, y + 27, fill="#dfeaf7", outline="",
                                    tags=tags)
            canvas.create_text(x + 11, y + 13.5, text=str(frame.get("name") or "Фрейм"), anchor="w",
                               fill="#294d70", font=("Segoe UI", 9, "bold"), tags=tags)
            canvas.create_text(x + width - 9, y + 13.5, text="ПКМ", anchor="e",
                               fill="#64809c", font=("Segoe UI", 7), tags=tags)
            self.draw_frame_internal_connections(frame_id, frame)
            for index, port in enumerate(frame.get("inputs", [])):
                if not isinstance(port, dict):
                    continue
                port_x, port_y = self.formula_frame_port_position(frame_id, "in", index)
                canvas.create_text(port_x + 12, port_y, text=str(port.get("name") or f"Вход {index + 1}"),
                                   anchor="w", fill="#3b6487", font=("Segoe UI", 8), width=68, tags=tags)
                target_id, target_index = port.get("target_id"), port.get("index")
                self.draw_formula_frame_port(
                    frame_id, "in", index, int(target_id) if isinstance(target_id, int) else None,
                    int(target_index) if isinstance(target_index, int) else None,
                )
            for index, port in enumerate(frame.get("outputs", [])):
                if not isinstance(port, dict):
                    continue
                port_x, port_y = self.formula_frame_port_position(frame_id, "out", index)
                canvas.create_text(port_x - 12, port_y, text=str(port.get("name") or f"Выход {index + 1}"),
                                   anchor="e", fill="#8a5a18", font=("Segoe UI", 8), width=68, tags=tags)
                source_id = port.get("source_id")
                self.draw_formula_frame_port(frame_id, "out", index,
                                             int(source_id) if isinstance(source_id, int) else None)
            for sequence, callback in (
                ("<ButtonPress-1>", lambda event, fid=frame_id: self.start_formula_frame_drag(event, fid)),
                ("<B1-Motion>", self.drag_formula_frame),
                ("<ButtonRelease-1>", self.drop_formula_frame),
            ):
                canvas.tag_bind(f"frame_body_{frame_id}", sequence, callback)

    def draw_frame_internal_connections(self, frame_id: int, frame: dict[str, object]) -> None:
        """Показать, к каким внутренним блокам привязаны порты стенок."""
        canvas = self.formula_canvas
        tags = ("formula_item", f"frame_{frame_id}", "frame_internal")
        for index, port in enumerate(frame.get("inputs", [])):
            if not isinstance(port, dict):
                continue
            target_id, target_index = port.get("target_id"), port.get("index")
            if not isinstance(target_id, int) or not isinstance(target_index, int) or target_id not in self.formula_blocks:
                continue
            start_x, start_y = self.formula_frame_port_position(frame_id, "in", index)
            end_x, end_y = self.formula_port_position(target_id, "in", target_index)
            middle_x = (start_x + end_x) / 2
            canvas.create_line(start_x, start_y, middle_x, start_y, middle_x, end_y, end_x, end_y,
                               fill="#5d94c3", width=1.5, dash=(3, 2), joinstyle="round", tags=tags)
        for index, port in enumerate(frame.get("outputs", [])):
            if not isinstance(port, dict):
                continue
            source_id = port.get("source_id")
            if not isinstance(source_id, int) or source_id not in self.formula_blocks:
                continue
            start_x, start_y = self.formula_port_position(source_id, "out")
            end_x, end_y = self.formula_frame_port_position(frame_id, "out", index)
            middle_x = (start_x + end_x) / 2
            canvas.create_line(start_x, start_y, middle_x, start_y, middle_x, end_y, end_x, end_y,
                               fill="#d49336", width=1.5, dash=(3, 2), joinstyle="round", tags=tags)

    def draw_collapsed_formula_frames(self) -> None:
        """Компактные операторы: только имя и настраиваемые входы/выходы."""
        canvas = self.formula_canvas
        for frame_id, frame in self.formula_frames.items():
            if not bool(frame.get("collapsed")):
                continue
            x, y = float(frame.get("x", 40.0)), float(frame.get("y", 40.0))
            width, height = self.formula_frame_compact_size(frame)
            tags = ("formula_item", f"frame_{frame_id}", f"frame_body_{frame_id}")
            self.rounded_rect(canvas, x, y, x + width, y + height, 12, fill="#386f9c", outline="#245271", width=1,
                              tags=tags)
            canvas.create_rectangle(x + 1, y + 1, x + width - 1, y + 26, fill="#4d86b3", outline="", tags=tags)
            canvas.create_text(x + 10, y + 13.5, text=str(frame.get("name") or "Фрейм"), anchor="w",
                               fill="white", font=("Segoe UI", 9, "bold"), width=132, tags=tags)
            inputs = [port for port in frame.get("inputs", []) if isinstance(port, dict)]
            outputs = [port for port in frame.get("outputs", []) if isinstance(port, dict)]
            rows = max(len(inputs), len(outputs), 1)
            for index in range(rows):
                port_y = y + 34.0 + index * 21.0
                if index < len(inputs):
                    canvas.create_text(x + 11, port_y, text=str(inputs[index].get("name") or f"Вход {index + 1}"),
                                       anchor="w", fill="#eaf4ff", font=("Segoe UI", 8), width=60, tags=tags)
                    target_id = inputs[index].get("target_id")
                    target_slot = inputs[index].get("index")
                    self.draw_formula_frame_port(frame_id, "in", index,
                                                 int(target_id) if isinstance(target_id, int) else None,
                                                 int(target_slot) if isinstance(target_slot, int) else None)
                if index < len(outputs):
                    canvas.create_text(x + width - 11, port_y,
                                       text=str(outputs[index].get("name") or f"Выход {index + 1}"), anchor="e",
                                       fill="#fff2d5", font=("Segoe UI", 8), width=65, tags=tags)
                    source_id = outputs[index].get("source_id")
                    self.draw_formula_frame_port(frame_id, "out", index,
                                                 int(source_id) if isinstance(source_id, int) else None)
            for sequence, callback in (
                ("<ButtonPress-1>", lambda event, fid=frame_id: self.start_formula_frame_drag(event, fid)),
                ("<B1-Motion>", self.drag_formula_frame),
                ("<ButtonRelease-1>", self.drop_formula_frame),
            ):
                canvas.tag_bind(f"frame_body_{frame_id}", sequence, callback)

    def start_formula_frame_drag(self, event, frame_id: int) -> str:
        if self.formula_tool == "comment":
            self.add_formula_comment_at(float(self.formula_canvas.canvasx(event.x)),
                                        float(self.formula_canvas.canvasy(event.y)))
            return "break"
        if frame_id not in self.formula_frames:
            return "break"
        x, y = self.formula_canvas.canvasx(event.x), self.formula_canvas.canvasy(event.y)
        self._formula_drag = None
        self._frame_drag = {"id": frame_id, "x": float(x), "y": float(y)}
        return "break"

    def drag_formula_frame(self, event) -> str | None:
        if self._frame_drag is None:
            return None
        frame_id = int(self._frame_drag["id"])
        frame = self.formula_frames.get(frame_id)
        if frame is None:
            return "break"
        x, y = float(self.formula_canvas.canvasx(event.x)), float(self.formula_canvas.canvasy(event.y))
        previous_x, previous_y = float(self._frame_drag["x"]), float(self._frame_drag["y"])
        dx, dy = x - previous_x, y - previous_y
        left, top, width, height = self.formula_frame_bounds(frame)
        world_width, world_height = self._formula_world
        dx = min(max(dx, 10.0 - left), world_width - 10.0 - (left + width))
        dy = min(max(dy, 10.0 - top), world_height - 10.0 - (top + height))
        for block_id in self.formula_frame_members(frame):
            block = self.formula_blocks[block_id]
            block["x"] = float(block["x"]) + dx
            block["y"] = float(block["y"]) + dy
            if not bool(frame.get("collapsed")):
                self.formula_canvas.move(f"block_{block_id}", dx, dy)
        frame["x"], frame["y"] = float(frame.get("x", left)) + dx, float(frame.get("y", top)) + dy
        self.formula_canvas.move(f"frame_{frame_id}", dx, dy)
        self._frame_drag["x"], self._frame_drag["y"] = x, y
        return "break"

    def drop_formula_frame(self, _event) -> str | None:
        if self._frame_drag is None:
            return None
        self._frame_drag = None
        self.render_formula_blocks()
        return "break"

    def formula_frame_menu(self, event, frame_id: int) -> str:
        frame = self.formula_frames.get(frame_id)
        if frame is None:
            return "break"
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="Открыть" if bool(frame.get("collapsed")) else "Сжать",
                         command=lambda: self.toggle_formula_frame(frame_id))
        menu.add_command(label="Настроить фрейм…", command=lambda: self.configure_formula_frame(frame_id))
        menu.add_separator()
        menu.add_command(label="Удалить фрейм (оставить содержимое)", command=lambda: self.ungroup_formula_frame(frame_id))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def start_formula_pan(self, event) -> None:
        """ЛКМ по пустому фону перемещает вид рабочей области."""
        if self.formula_tool == "select":
            self.start_formula_selection(event)
            return
        if self.formula_tool == "comment":
            self.start_formula_comment(event)
            return
        current = self.formula_canvas.find_withtag("current")
        if current and "formula_item" in self.formula_canvas.gettags(current[0]):
            self._formula_pan = False
            return
        self._formula_pan = True
        self.formula_canvas.scan_mark(event.x, event.y)

    def drag_formula_pan(self, event) -> None:
        if self.formula_tool == "select":
            self.drag_formula_selection(event)
            return
        if self._formula_pan:
            self.formula_canvas.scan_dragto(event.x, event.y, gain=1)

    def stop_formula_pan(self, event) -> None:
        if self.formula_tool == "select":
            self.finish_formula_selection(event)
        self._formula_pan = False

    def draw_formula_grid(self) -> None:
        if not hasattr(self, "formula_canvas"):
            return
        canvas = self.formula_canvas
        canvas.delete("formula_grid")
        width, height = (int(value) for value in getattr(self, "_formula_world", (1800, 1200)))
        for x in range(0, width + 1, 24):
            canvas.create_line(x, 0, x, height, fill="#f0f3f7", tags="formula_grid")
        for y in range(0, height + 1, 24):
            canvas.create_line(0, y, width, y, fill="#f0f3f7", tags="formula_grid")
        canvas.tag_lower("formula_grid")

    def refresh_formula_workspace(self) -> None:
        if not hasattr(self, "formula_canvas"):
            return
        self.render_formula_palette()
        self.render_formula_blocks()

    def render_formula_palette(self) -> None:
        canvas = self.palette_canvas
        canvas.delete("all")
        y = 12
        canvas.create_text(8, y, text="Площади интегралов", anchor="w", fill="#536171",
                           font=("Segoe UI", 10, "bold"))
        y += 18
        comparison = self.comparison_record()
        source_integrals = (sorted(list(comparison.get("integrals", [])), key=lambda item: int(item["number"]))
                            if comparison is not None else self.ordered_integrals())
        for item in source_integrals:
            number = str(item["number"]) if comparison is not None else self.integral_number(item["id"])
            integral_id, color = int(item["id"]), str(item["color"])
            tag = f"palette_area_{integral_id}"
            self.rounded_rect(canvas, 8, y, 187, y + 34, 13, fill=color, outline="", tags=(tag,))
            canvas.create_text(20, y + 17, text=f"I{number}", anchor="w", fill="white",
                               font=("Segoe UI", 10, "bold"), tags=(tag,))
            if comparison is not None and self.active_spectrum_id in self.spectra:
                record = self.spectra[self.active_spectrum_id]
                offsets = comparison.get("offsets", {})
                offset = float(offsets.get(record["id"], offsets.get(str(record["id"]), 0.0)))
                value = self.record_integral_area(record, float(item["left"]), float(item["right"]),
                                                  x_offset=offset)
            else:
                value = self.integral_area(item)
            canvas.create_text(174, y + 17, text=self.format_formula_value(value), anchor="e",
                               fill="white", font=("Consolas", 9, "bold"), tags=(tag,))
            canvas.tag_bind(tag, "<ButtonPress-1>",
                            lambda event, iid=integral_id, num=int(number): self.start_palette_drag(event, iid, num))
            y += 41
        if not source_integrals:
            canvas.create_text(8, y + 8, text="Сначала выделите интеграл", anchor="w", fill="#98a1aa")
            y += 30
        canvas.create_line(8, y, 187, y, fill="#e5e9ee")
        y += 14
        canvas.create_text(8, y, text="Константы", anchor="w", fill="#536171",
                           font=("Segoe UI", 10, "bold"))
        y += 18
        action_y = y + 5
        canvas.create_text(10, action_y + 12, text="＋ Константа", anchor="w", fill="#7b61ff",
                                          font=("Segoe UI", 10, "bold"), tags=("palette_action", "add_constant"))
        canvas.tag_bind("add_constant", "<ButtonPress-1>", lambda _event: self.add_formula_constant())
        y = action_y + 35
        canvas.create_line(8, y, 187, y, fill="#e5e9ee")
        y += 15
        canvas.create_text(8, y, text="Операторы", anchor="w", fill="#536171", font=("Segoe UI", 10, "bold"))
        y += 18
        for row in (("+", "−"), ("×", "÷"), ("^", "%"), ("abs", "√"),
                    ("ln", "log₁₀"), ("exp", "round"), ("min", "max")):
            for column, operation in enumerate(row):
                left = 8 + column * 91
                tag = f"palette_operation_{operation}_{y}"
                self.draw_palette_operation(canvas, left, y, left + 84, y + 32, operation, tag)
                canvas.tag_bind(tag, "<ButtonPress-1>",
                                lambda _event, op=operation: self.add_formula_operation_from_palette(op))
            y += 39
        output_tag = "palette_output"
        self.rounded_rect(canvas, 8, y, 187, y + 32, 13, fill="#3d7ea6", outline="", tags=(output_tag,))
        canvas.create_text(20, y + 16, text="＋ Карточка результата", anchor="w", fill="white",
                           font=("Segoe UI", 9, "bold"), tags=(output_tag,))
        canvas.tag_bind(output_tag, "<ButtonPress-1>", lambda _event: self.add_formula_output_from_palette())
        y += 40
        canvas.create_text(8, y + 2, text="ПКМ в поле — все команды", anchor="w", fill="#98a1aa",
                           font=("Segoe UI", 8))
        canvas.configure(scrollregion=(0, 0, 195, y + 25))

    def draw_palette_operation(self, canvas, x1: float, y1: float, x2: float, y2: float,
                               operation: str, tag: str) -> None:
        """Небольшая зелёная Scratch-капсула в палитре операций."""
        self.rounded_rect(canvas, x1, y1, x2, y2, 14, fill="#49a942", outline="#2f8738", tags=(tag,))
        if operation in self._UNARY_OPERATIONS:
            self.rounded_rect(canvas, x1 + 7, y1 + 7, x1 + 31, y2 - 7, 10,
                              fill="#ffffff", outline="#2f8738", tags=(tag,))
            canvas.create_text(x1 + 56, (y1 + y2) / 2, text=operation, fill="white",
                               font=("Segoe UI", 10, "bold"), tags=(tag,))
        else:
            self.rounded_rect(canvas, x1 + 6, y1 + 7, x1 + 28, y2 - 7, 9,
                              fill="#ffffff", outline="#2f8738", tags=(tag,))
            self.rounded_rect(canvas, x2 - 28, y1 + 7, x2 - 6, y2 - 7, 9,
                              fill="#ffffff", outline="#2f8738", tags=(tag,))
            canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=operation, fill="white",
                               font=("Segoe UI", 11, "bold"), tags=(tag,))

    @staticmethod
    def rounded_rect(canvas, x1, y1, x2, y2, radius, **kwargs):
        points = [x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
                  x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
                  x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1]
        return canvas.create_polygon(points, smooth=True, splinesteps=12, **kwargs)

    def add_formula_block(self, kind: str, x: float, y: float, *, integral_id: int | None = None,
                          operation: str | None = None, value: float | None = None,
                          name: str | None = None, area_number: int | None = None) -> int:
        block_id = self._next_formula_id
        self._next_formula_id += 1
        unary = operation in self._UNARY_OPERATIONS
        width, height = self.formula_block_size({"kind": kind})
        world_width, world_height = getattr(self, "_formula_world", (1800.0, 1200.0))
        x = min(max(10.0, float(x)), world_width - width - 10.0)
        y = min(max(10.0, float(y)), world_height - height - 10.0)
        output_number = 1 + sum(1 for item in self.formula_blocks.values() if item["kind"] == "output")
        auto_name = kind == "output" and name is None
        output_name = "Вывод" if output_number == 1 else f"Вывод {output_number}"
        self.formula_blocks[block_id] = {
            "kind": kind, "x": x, "y": y, "integral_id": integral_id,
            "area_number": area_number if area_number is not None else
            (int(self.integral_number(integral_id)) if kind == "area" and self.integral_number(integral_id).isdigit() else None),
            "comparison_id": self.current_comparison_id if kind == "area" else None,
            "operation": operation,
            "inputs": ([None] if unary else [None, None]) if kind == "operation" else [],
            "input": None, "value": value,
            "name": output_name if auto_name else (name or ""), "auto_name": auto_name,
        }
        self.render_formula_blocks()
        return block_id

    def formula_spawn_point(self) -> tuple[float, float]:
        """Свободное место в текущем видимом участке рабочей области."""
        canvas = self.formula_canvas
        index = len(self.formula_blocks)
        left = float(canvas.canvasx(45)) + (index % 4) * 34
        top = float(canvas.canvasy(55)) + (index % 5) * 30
        return left, top

    def start_palette_drag(self, _event, integral_id: int, area_number: int | None = None) -> None:
        x, y = self.formula_spawn_point()
        self.add_formula_block("area", x, y, integral_id=integral_id, area_number=area_number)

    def add_formula_operation_from_palette(self, operation: str) -> None:
        x, y = self.formula_spawn_point()
        self.add_formula_block("operation", x, y, operation=operation)

    def add_formula_output_from_palette(self) -> None:
        x, y = self.formula_spawn_point()
        self.add_formula_block("output", x, y)

    def ask_formula_number(self, title: str, prompt: str, initial: float | None = None) -> float | None:
        """Числовой ввод, принимающий и запятую, и точку."""
        initial_text = None if initial is None else f"{initial:.12g}"
        entered = simpledialog.askstring(title, prompt, initialvalue=initial_text, parent=self)
        if entered is None:
            return None
        try:
            return float(entered.strip().replace(",", "."))
        except ValueError:
            messagebox.showerror(title, "Введите корректное число.", parent=self)
            return None

    def add_formula_constant(self, x: float | None = None, y: float | None = None) -> None:
        value = self.ask_formula_number("Константа", "Введите численное значение:")
        if value is not None:
            if x is None or y is None:
                x, y = self.formula_spawn_point()
            self.add_formula_block("constant", x, y, value=value)


    def render_formula_blocks(self) -> None:
        canvas = self.formula_canvas
        canvas.delete("formula_item")
        canvas.delete("wire_preview")
        canvas.delete("selection_outline")
        # Если перерисовка произошла во время смены инструмента, временная
        # рамка также не должна превращаться в постоянный пунктир.
        if self._selection_drag is None:
            canvas.delete("selection_marquee")
        # Развёрнутый фрейм рисуется первым: его рамка остаётся под блоками и
        # не перекрывает ни сами блоки, ни соединения.
        self.draw_expanded_formula_frames()
        self.draw_formula_connections()
        for block_id, block in self.formula_blocks.items():
            if self.hidden_formula_block(block_id):
                continue
            kind, x, y = str(block["kind"]), float(block["x"]), float(block["y"])
            body_tag = ("formula_item", f"block_{block_id}", f"node_body_{block_id}")
            if kind in {"area", "constant"}:
                item = self.integral_by_id(block.get("integral_id"))
                stored_comparison_id = block.get("comparison_id")
                comparison = self.comparisons.get(stored_comparison_id) \
                    if isinstance(stored_comparison_id, int) else self.comparison_record()
                if kind == "area" and comparison is not None:
                    item = next((value for value in comparison.get("integrals", [])
                                 if str(value.get("number")) == self.formula_area_number(block)), item)
                color = str(item["color"]) if kind == "area" and item else \
                    "#7b61ff" if kind == "constant" else "#9aa5b1"
                if kind == "area":
                    width, height = self.formula_block_size(block)
                    self.rounded_rect(canvas, x, y, x + width, y + height, 9, fill=color, outline="", tags=body_tag)
                    canvas.create_text(x + width / 2, y + height / 2, text=self.formula_area_number(block),
                                       fill="white", font=("Segoe UI", 14, "bold"), tags=body_tag)
                else:
                    width, height = self.formula_block_size(block)
                    self.rounded_rect(canvas, x, y, x + width, y + height, 10, fill=color, outline="", tags=body_tag)
                    canvas.create_text(x + width / 2, y + height / 2,
                                       text=self.format_formula_value(self.formula_value(block_id)), fill="white",
                                       font=("Consolas", 10, "bold"), tags=body_tag)
                self.draw_formula_port(block_id, "out", None)
            elif kind == "operation":
                operation = str(block["operation"])
                self.rounded_rect(canvas, x, y, x + 88, y + 64, 16, fill="#48a942", outline="#2d8734", tags=body_tag)
                canvas.create_text(x + 44, y + 32, text=operation, fill="white",
                                   font=("Segoe UI", 20, "bold"), tags=body_tag)
                for index in range(len(block["inputs"])):
                    self.draw_formula_port(block_id, "in", index)
                self.draw_formula_port(block_id, "out", None)
            else:
                width, height = self.formula_block_size(block)
                self.rounded_rect(canvas, x, y, x + width, y + height, 15, fill="#3d7ea6", outline="#275a7a", tags=body_tag)
                canvas.create_text(x + width / 2, y + height / 2, text=str(block.get("name") or "Вывод"),
                                   fill="white", font=("Segoe UI", 9, "bold"), tags=body_tag)
                self.draw_formula_port(block_id, "in", 0)
            for sequence, callback in (("<ButtonPress-1>", lambda event, bid=block_id: self.start_formula_drag(event, bid)),
                                       ("<B1-Motion>", self.drag_formula_block), ("<ButtonRelease-1>", self.drop_formula_block)):
                canvas.tag_bind(f"node_body_{block_id}", sequence, callback)
            if kind == "constant":
                canvas.tag_bind(f"node_body_{block_id}", "<Double-Button-1>",
                                lambda _event, bid=block_id: self.edit_formula_constant(bid))
            if block_id in self.selected_formula_blocks:
                width, height = self.formula_block_size(block)
                canvas.create_rectangle(x - 4, y - 4, x + width + 4, y + height + 4,
                                        outline="#3179c7", width=2, dash=(4, 2), tags="selection_outline")
        # Свёрнутый фрейм показывается поверх проводов, чтобы его порты были
        # легко различимы и за них можно было тянуть соединения.
        self.draw_collapsed_formula_frames()
        self.render_formula_comments()
        self.refresh_result_panel()

    def render_formula_comments(self) -> None:
        """Лёгкие заметки на доске; они не участвуют в вычислениях."""
        canvas = self.formula_canvas
        for comment_id, comment in self.formula_comments.items():
            x, y = float(comment["x"]), float(comment["y"])
            tag = ("formula_item", "formula_comment", f"comment_{comment_id}")
            self.rounded_rect(canvas, x, y, x + 170, y + 50, 9, fill="#fff4b8", outline="#d8bd56", tags=tag)
            canvas.create_text(x + 10, y + 9, text=str(comment["text"]), anchor="nw", width=150,
                               fill="#5c4a0d", font=("Segoe UI", 9), tags=tag)
            canvas.tag_bind(f"comment_{comment_id}", "<ButtonPress-1>",
                            lambda event, cid=comment_id: self.start_formula_comment_drag(event, cid))
            canvas.tag_bind(f"comment_{comment_id}", "<B1-Motion>", self.drag_formula_comment)
            canvas.tag_bind(f"comment_{comment_id}", "<ButtonRelease-1>", self.drop_formula_comment)
            canvas.tag_bind(f"comment_{comment_id}", "<Button-3>",
                            lambda event, cid=comment_id: self.formula_comment_menu(event, cid))

    def start_formula_comment_drag(self, event, comment_id: int) -> str | None:
        if self.formula_tool == "comment":
            return "break"
        x, y = self.formula_canvas.canvasx(event.x), self.formula_canvas.canvasy(event.y)
        self._comment_drag = {"id": comment_id, "x": float(x), "y": float(y)}
        return "break"

    def drag_formula_comment(self, event) -> str | None:
        if self._comment_drag is None:
            return None
        comment_id = int(self._comment_drag["id"])
        x, y = float(self.formula_canvas.canvasx(event.x)), float(self.formula_canvas.canvasy(event.y))
        comment = self.formula_comments[comment_id]
        dx, dy = x - float(self._comment_drag["x"]), y - float(self._comment_drag["y"])
        comment["x"], comment["y"] = float(comment["x"]) + dx, float(comment["y"]) + dy
        self._comment_drag["x"], self._comment_drag["y"] = x, y
        self.formula_canvas.move(f"comment_{comment_id}", dx, dy)
        return "break"

    def drop_formula_comment(self, _event) -> str | None:
        if self._comment_drag is None:
            return None
        self._comment_drag = None
        return "break"

    def formula_comment_menu(self, event, comment_id: int) -> str:
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="Изменить комментарий…", command=lambda: self.edit_formula_comment(comment_id))
        menu.add_command(label="Удалить комментарий", command=lambda: self.delete_formula_comment(comment_id))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def edit_formula_comment(self, comment_id: int) -> None:
        comment = self.formula_comments.get(comment_id)
        if comment is None:
            return
        text = simpledialog.askstring("Комментарий", "Текст комментария:", initialvalue=str(comment["text"]), parent=self)
        if text and text.strip():
            comment["text"] = text.strip()
            self.render_formula_blocks()

    def delete_formula_comment(self, comment_id: int) -> None:
        self.formula_comments.pop(comment_id, None)
        self.render_formula_blocks()

    @staticmethod
    def formula_block_size(block: dict[str, object]) -> tuple[float, float]:
        if block["kind"] == "area":
            return 42.0, 42.0
        if block["kind"] == "constant":
            return 66.0, 36.0
        if block["kind"] == "operation":
            return 88.0, 64.0
        if block["kind"] == "output":
            return 104.0, 44.0
        return 100.0, 44.0

    def formula_slot_rects(self, block: dict[str, object]) -> list[tuple[float, float, float, float]]:
        x, y = float(block["x"]), float(block["y"])
        if block["kind"] == "operation":
            if block["operation"] in self._UNARY_OPERATIONS:
                return [(x, y + 26, x + 14, y + 38)]
            return [(x, y + 14, x + 14, y + 26), (x, y + 38, x + 14, y + 50)]
        if block["kind"] == "output":
            return [(x, y + 16, x + 12, y + 28)]
        return []

    def formula_port_position(self, block_id: int, direction: str, index: int | None = None) -> tuple[float, float]:
        block = self.formula_blocks[block_id]
        x, y = float(block["x"]), float(block["y"])
        if direction == "out":
            if block["kind"] == "operation":
                return x + 88, y + 32
            width, height = self.formula_block_size(block)
            return x + width, y + height / 2
        if block["kind"] == "output":
            return x, y + 22
        if block["operation"] in self._UNARY_OPERATIONS:
            return x, y + 32
        return x, y + (20 if index == 0 else 44)

    def formula_frame_port_position(self, frame_id: int, direction: str, index: int) -> tuple[float, float]:
        """Координата публичного порта на стенке или компактном фрейме."""
        frame = self.formula_frames[frame_id]
        x, y = float(frame.get("x", 40.0)), float(frame.get("y", 40.0))
        if bool(frame.get("collapsed")):
            width, _height = self.formula_frame_compact_size(frame)
            port_y = y + 34.0 + index * 21.0
            return (x, port_y) if direction == "in" else (x + width, port_y)
        x, y, width, height = self.formula_frame_bounds(frame)
        inputs = len(frame.get("inputs", []))
        outputs = len(frame.get("outputs", []))
        rows = max(1, inputs, outputs)
        top, bottom = y + 42.0, y + height - 16.0
        port_y = (top + bottom) / 2 if rows == 1 else top + (bottom - top) * index / (rows - 1)
        return (x, port_y) if direction == "in" else (x + width, port_y)

    def formula_source_port_position(self, source_id: int) -> tuple[float, float]:
        return self.formula_port_position(source_id, "out")

    def formula_target_port_position(self, target_id: int, index: int) -> tuple[float, float]:
        return self.formula_port_position(target_id, "in", index)

    def formula_connection_port_positions(self, source_id: int, target_id: int, index: int) -> tuple[float, float, float, float]:
        """Концы провода с учётом перехода через стенки фреймов."""
        output_proxy = self.frame_output_for_connection(source_id, target_id)
        input_proxy = self.frame_input_for_connection(source_id, target_id, index)
        source_x, source_y = (self.formula_frame_port_position(output_proxy[0], "out", output_proxy[1])
                              if output_proxy is not None else self.formula_port_position(source_id, "out"))
        target_x, target_y = (self.formula_frame_port_position(input_proxy[0], "in", input_proxy[1])
                              if input_proxy is not None else self.formula_port_position(target_id, "in", index))
        return source_x, source_y, target_x, target_y

    def draw_formula_port(self, block_id: int, direction: str, index: int | None) -> None:
        canvas = self.formula_canvas
        x, y = self.formula_port_position(block_id, direction, index)
        suffix = "out" if direction == "out" else f"in_{index}"
        tag = f"port_{block_id}_{suffix}"
        color = "#f3a530" if direction == "out" else "#4285c5"
        canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ffffff", outline=color, width=2,
                           tags=("formula_item", f"block_{block_id}", "formula_port", tag))
        if direction == "out":
            canvas.tag_bind(tag, "<ButtonPress-1>",
                            lambda event, bid=block_id: self.start_formula_wire(event, bid, "out", None))
        else:
            canvas.tag_bind(tag, "<ButtonPress-1>",
                            lambda event, bid=block_id, idx=index: self.start_formula_wire(event, bid, "in", idx))
        canvas.tag_bind(tag, "<Enter>", lambda _event, item=tag: canvas.itemconfigure(item, fill="#fff4cf"))
        canvas.tag_bind(tag, "<Leave>", lambda _event, item=tag: canvas.itemconfigure(item, fill="#ffffff"))

    def draw_formula_frame_port(self, frame_id: int, direction: str, index: int,
                                endpoint_id: int | None, endpoint_index: int | None = None) -> None:
        """Порт на компактном представлении фрейма.

        Снаружи это обычный порт. Внутри он перенаправляется на выбранный
        блок, поэтому и цепочки из нескольких фреймов пересчитываются сразу.
        """
        canvas = self.formula_canvas
        x, y = self.formula_frame_port_position(frame_id, direction, index)
        tag = f"frame_port_{frame_id}_{direction}_{index}"
        color = "#f3a530" if direction == "out" else "#4285c5"
        canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ffffff", outline=color, width=2,
                           tags=("formula_item", f"frame_{frame_id}", "formula_port", tag))
        if isinstance(endpoint_id, int) and endpoint_id in self.formula_blocks:
            if direction == "out":
                canvas.tag_bind(tag, "<ButtonPress-1>",
                                lambda event, bid=endpoint_id, fid=frame_id, slot=index:
                                self.start_formula_wire(event, bid, "out", None, (fid, "out", slot)))
            elif endpoint_index is not None:
                canvas.tag_bind(tag, "<ButtonPress-1>",
                                lambda event, bid=endpoint_id, slot=endpoint_index:
                                self.start_formula_wire(event, bid, "in", slot, (frame_id, "in", index)))
        canvas.tag_bind(tag, "<Enter>", lambda _event, item=tag: canvas.itemconfigure(item, fill="#fff4cf"))
        canvas.tag_bind(tag, "<Leave>", lambda _event, item=tag: canvas.itemconfigure(item, fill="#ffffff"))

    @staticmethod
    def formula_wire_key(target_id: int, index: int) -> str:
        return f"{target_id}_{index}"

    def formula_wire_points(self, source_id: int, target_id: int, index: int) -> list[float]:
        source_x, source_y, target_x, target_y = self.formula_connection_port_positions(source_id, target_id, index)
        key = self.formula_wire_key(target_id, index)
        route = self.wire_routes.get(key)
        if route is None:
            # Обычный провод — короткая «скоба» из трёх ортогональных отрезков.
            # Более длинный маршрут появляется только после ручного переноса узла.
            middle_x = (source_x + target_x) / 2
            return [source_x, source_y, middle_x, source_y, middle_x, target_y, target_x, target_y]
        middle_x, middle_y = route
        return [source_x, source_y, middle_x, source_y, middle_x, middle_y,
                target_x, middle_y, target_x, target_y]

    def formula_wire_handle_position(self, source_id: int, target_id: int, index: int) -> tuple[float, float]:
        key = self.formula_wire_key(target_id, index)
        route = self.wire_routes.get(key)
        if route is not None:
            return route
        source_x, source_y, target_x, target_y = self.formula_connection_port_positions(source_id, target_id, index)
        return (source_x + target_x) / 2, (source_y + target_y) / 2

    def formula_wire_color(self, source_id: int) -> str:
        source = self.formula_blocks[source_id]
        if source["kind"] == "area":
            integral = self.integral_by_id(source.get("integral_id"))
            return str(integral["color"]) if integral else "#6f8ca6"
        if source["kind"] == "constant":
            return "#7b61ff"
        return "#48a942"

    def draw_formula_connections(self) -> None:
        """Ломаные LabVIEW-проводки, всегда состоящие только из H/V-сегментов."""
        canvas = self.formula_canvas
        for target_id, target in self.formula_blocks.items():
            sources = list(target.get("inputs", [])) if target["kind"] == "operation" else [target.get("input")]
            for index, source_id in enumerate(sources):
                if not isinstance(source_id, int) or source_id not in self.formula_blocks:
                    continue
                # Внутренние проводки свёрнутого фрейма не должны торчать
                # наружу. Видны только связи, выведенные через его порты.
                if self.hidden_formula_block(source_id) and self.frame_output_for_connection(source_id, target_id) is None:
                    continue
                if self.hidden_formula_block(target_id) and self.frame_input_for_connection(source_id, target_id, index) is None:
                    continue
                key = self.formula_wire_key(target_id, index)
                points = self.formula_wire_points(source_id, target_id, index)
                canvas.create_line(*points, fill=self.formula_wire_color(source_id), width=2, joinstyle="round",
                                   tags=("formula_item", "formula_wire", f"wire_line_{key}"))
                middle_x, middle_y = self.formula_wire_handle_position(source_id, target_id, index)
                handle_tag = f"wire_handle_{key}"
                canvas.create_rectangle(middle_x - 4, middle_y - 4, middle_x + 4, middle_y + 4,
                                        fill="#ffffff", outline=self.formula_wire_color(source_id), width=1,
                                        tags=("formula_item", "formula_wire", "wire_handle", handle_tag))
                canvas.tag_bind(handle_tag, "<ButtonPress-1>",
                                lambda event, tid=target_id, idx=index: self.start_wire_handle_drag(event, tid, idx))
                canvas.tag_bind(handle_tag, "<B1-Motion>", self.drag_wire_handle)
                canvas.tag_bind(handle_tag, "<ButtonRelease-1>", self.drop_wire_handle)
                canvas.tag_bind(handle_tag, "<Button-3>",
                                lambda event, tid=target_id, idx=index: self.formula_wire_menu(event, tid, idx))
                canvas.tag_bind(f"wire_line_{key}", "<Button-3>",
                                lambda event, tid=target_id, idx=index: self.formula_wire_menu(event, tid, idx))

    def formula_wire_menu(self, event, target_id: int, index: int) -> str:
        """Контекстное меню отдельного провода, не затрагивающее остальные связи."""
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="Отсоединить провод", command=lambda: self.disconnect_formula_wire(target_id, index))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def disconnect_formula_wire(self, target_id: int, index: int) -> None:
        target = self.formula_blocks.get(target_id)
        if target is None:
            return
        if target["kind"] == "operation" and 0 <= index < len(target.get("inputs", [])):
            target["inputs"][index] = None
        elif target["kind"] == "output" and index == 0:
            target["input"] = None
        else:
            return
        self.wire_routes.pop(self.formula_wire_key(target_id, index), None)
        self.formula_status.set("Провод отсоединён.")
        self.render_formula_blocks()

    def formula_block_label(self, block_id: object) -> str:
        if not isinstance(block_id, int) or block_id not in self.formula_blocks:
            return "вставьте блок"
        block = self.formula_blocks[block_id]
        kind = block["kind"]
        if kind == "area":
            return f"I{self.formula_area_number(block)}"
        if kind == "constant":
            return self.format_formula_value(block.get("value"))
        if kind == "output":
            return str(block.get("name") or "Вывод")
        return f"результат {block_id}"

    def formula_area_number(self, block: dict[str, object]) -> str:
        saved = block.get("area_number")
        if isinstance(saved, int) and saved > 0:
            return str(saved)
        return self.integral_number(block.get("integral_id"))

    def formula_value_for_record(self, block_id: object, record: dict[str, object] | None,
                                 visited: set[int] | None = None) -> float | None:
        """Посчитать схему для конкретного спектра, не меняя активную строку."""
        if not isinstance(block_id, int) or block_id not in self.formula_blocks:
            return None
        visited = set() if visited is None else visited
        if block_id in visited:
            return None
        visited.add(block_id)
        block = self.formula_blocks[block_id]
        kind = block["kind"]
        if kind == "area":
            if record is None:
                return 0.0
            wanted_number = self.formula_area_number(block)
            stored_comparison_id = block.get("comparison_id")
            comparison = self.comparisons.get(stored_comparison_id) \
                if isinstance(stored_comparison_id, int) else self.comparison_record()
            # В обзоре диапазон I1 общий, но собственная площадь вычисляется
            # по исходной линии конкретной строки с её X-сдвигом.
            if comparison is not None and int(record.get("id", -1)) in comparison.get("spectrum_ids", []):
                shared = next((item for item in comparison.get("integrals", [])
                               if str(item.get("number")) == wanted_number), None)
                if shared is not None:
                    offsets = comparison.get("offsets", {})
                    offset = float(offsets.get(record["id"], offsets.get(str(record["id"]), 0.0)))
                    return self.record_integral_area(record, float(shared["left"]), float(shared["right"]),
                                                     x_offset=offset)
            item = next((value for value in record.get("integrals", [])
                         if str(value.get("number")) == wanted_number), None)
            if item is None:
                item = next((value for value in record.get("integrals", [])
                             if value.get("id") == block.get("integral_id")), None)
            return self.record_integral_area(record, float(item["left"]), float(item["right"])) if item else 0.0
        if kind == "constant":
            return float(block["value"])
        if kind == "output":
            return self.formula_value_for_record(block.get("input"), record, visited)
        values = [self.formula_value_for_record(value, record, visited.copy()) for value in block["inputs"]]
        if any(value is None for value in values):
            return None
        a, b = float(values[0]), float(values[0] if len(values) == 1 else values[1])
        operation = str(block["operation"])
        # Нельзя собирать здесь словарь с готовыми выражениями: тогда Python
        # вычисляет *все* ветки (включая a / b) даже для операции «+».
        try:
            if operation == "+":
                value = a + b
            elif operation == "−":
                value = a - b
            elif operation == "×":
                value = a * b
            elif operation == "÷":
                value = 0.0 if b == 0.0 else a / b
            elif operation == "^":
                value = a ** b
            elif operation == "%":
                value = 0.0 if b == 0.0 else a % b
            elif operation == "abs":
                value = abs(a)
            elif operation == "√":
                value = np.sqrt(a) if a >= 0 else np.nan
            elif operation == "ln":
                value = np.log(a) if a > 0 else np.nan
            elif operation == "log₁₀":
                value = np.log10(a) if a > 0 else np.nan
            elif operation == "exp":
                value = np.exp(a)
            elif operation == "round":
                value = round(a)
            elif operation == "−x":
                value = -a
            elif operation == "min":
                value = min(a, b)
            elif operation == "max":
                value = max(a, b)
            else:
                return None
            value = float(value)
            return value if np.isfinite(value) else None
        except (ZeroDivisionError, ValueError, OverflowError, TypeError):
            return None

    def formula_value(self, block_id: object, visited: set[int] | None = None) -> float | None:
        record = self.spectra.get(self.active_spectrum_id) if self.active_spectrum_id is not None else None
        return self.formula_value_for_record(block_id, record, visited)

    def start_formula_drag(self, event, block_id: int) -> str | None:
        x, y = self.formula_canvas.canvasx(event.x), self.formula_canvas.canvasy(event.y)
        self.formula_canvas.focus_set()
        if self.formula_tool == "comment":
            self.add_formula_comment_at(float(x), float(y))
            return "break"
        if self.formula_tool == "select":
            if event.state & 0x0001:
                if block_id in self.selected_formula_blocks:
                    self.selected_formula_blocks.remove(block_id)
                else:
                    self.selected_formula_blocks.add(block_id)
                self.render_formula_blocks()
                return "break"
            if block_id not in self.selected_formula_blocks:
                self.selected_formula_blocks = {block_id}
            self._formula_drag = {"id": block_id, "x": float(x), "y": float(y), "group": True}
            self.render_formula_blocks()
            return "break"
        self._formula_drag = {"id": block_id, "x": float(x), "y": float(y)}
        self.formula_canvas.tag_raise(f"block_{block_id}")
        return "break"

    def drag_formula_block(self, event) -> str | None:
        if self._formula_drag is None:
            return None
        x, y = self.formula_canvas.canvasx(event.x), self.formula_canvas.canvasy(event.y)
        if self._formula_drag.get("group"):
            self.move_selected_formula_blocks(float(x), float(y))
            return "break"
        self.move_formula_block_to(float(x), float(y))
        return "break"

    def move_selected_formula_blocks(self, x: float, y: float) -> None:
        if self._formula_drag is None:
            return
        previous_x, previous_y = float(self._formula_drag["x"]), float(self._formula_drag["y"])
        dx, dy = x - previous_x, y - previous_y
        world_width, world_height = self._formula_world
        for block_id in self.selected_formula_blocks:
            block = self.formula_blocks.get(block_id)
            if block is None:
                continue
            width, height = self.formula_block_size(block)
            old_x, old_y = float(block["x"]), float(block["y"])
            next_x = min(max(10.0, old_x + dx), world_width - width - 10.0)
            next_y = min(max(10.0, old_y + dy), world_height - height - 10.0)
            block["x"], block["y"] = next_x, next_y
            self.formula_canvas.move(f"block_{block_id}", next_x - old_x, next_y - old_y)
        self._formula_drag["x"], self._formula_drag["y"] = x, y

    def move_formula_block_to(self, x: float, y: float) -> None:
        """Передвинуть блок в координатах рабочего холста, не давая ему потеряться."""
        if self._formula_drag is None:
            return
        block_id = int(self._formula_drag["id"])
        block = self.formula_blocks[block_id]
        width, height = self.formula_block_size(block)
        dx, dy = x - float(self._formula_drag["x"]), y - float(self._formula_drag["y"])
        world_width, world_height = self._formula_world
        next_x = min(max(10.0, float(block["x"]) + dx), world_width - width - 10.0)
        next_y = min(max(10.0, float(block["y"]) + dy), world_height - height - 10.0)
        dx, dy = next_x - float(block["x"]), next_y - float(block["y"])
        block["x"], block["y"] = next_x, next_y
        self._formula_drag["x"], self._formula_drag["y"] = x, y
        self.formula_canvas.move(f"block_{block_id}", dx, dy)

    def drop_formula_block(self, event) -> str | None:
        if self._formula_drag is None:
            return None
        self._formula_drag = None
        # После перемещения блока проводки и порты перестраиваются по новой
        # геометрии; соединение создаётся только через сами порты.
        self.render_formula_blocks()
        return "break"

    def formula_port_under_cursor(self, x: float, y: float, direction: str) -> tuple[int, int | None] | None:
        """Найти ближайший совместимый порт в радиусе его кружка."""
        # Сначала проверяем порты стенок и компактные фреймы: они
        # перенаправляют провод на реальный внутренний блок.
        for frame_id, frame in reversed(list(self.formula_frames.items())):
            ports = frame.get("outputs", []) if direction == "out" else frame.get("inputs", [])
            for port_index, port in enumerate(ports):
                if not isinstance(port, dict):
                    continue
                port_x, port_y = self.formula_frame_port_position(frame_id, direction, port_index)
                if (port_x - x) ** 2 + (port_y - y) ** 2 > 12 ** 2:
                    continue
                if direction == "out":
                    source_id = port.get("source_id")
                    if isinstance(source_id, int) and source_id in self.formula_blocks:
                        return source_id, None
                else:
                    target_id, index = port.get("target_id"), port.get("index")
                    if isinstance(target_id, int) and isinstance(index, int) and target_id in self.formula_blocks:
                        return target_id, index
        for block_id, block in reversed(list(self.formula_blocks.items())):
            if self.hidden_formula_block(block_id):
                continue
            if direction == "out":
                if block["kind"] == "output":
                    continue
                port_x, port_y = self.formula_port_position(block_id, "out")
                if (port_x - x) ** 2 + (port_y - y) ** 2 <= 12 ** 2:
                    return block_id, None
            elif block["kind"] in {"operation", "output"}:
                count = len(block.get("inputs", [])) if block["kind"] == "operation" else 1
                for index in range(count):
                    port_x, port_y = self.formula_port_position(block_id, "in", index)
                    if (port_x - x) ** 2 + (port_y - y) ** 2 <= 12 ** 2:
                        return block_id, index
        return None

    def start_formula_wire(self, event, block_id: int, direction: str, index: int | None,
                           frame_port: tuple[int, str, int] | None = None) -> None:
        if direction == "out" and self.formula_blocks[block_id]["kind"] == "output":
            return
        self._formula_drag = None
        self._wire_drag = {"direction": direction, "block_id": block_id, "index": index,
                           "frame_port": frame_port}
        x, y = self.formula_canvas.canvasx(event.x), self.formula_canvas.canvasy(event.y)
        self.draw_wire_preview(float(x), float(y))

    def wire_preview_points(self, x: float, y: float) -> list[float]:
        assert self._wire_drag is not None
        block_id = int(self._wire_drag["block_id"])
        index = self._wire_drag["index"]
        frame_port = self._wire_drag.get("frame_port")
        if self._wire_drag["direction"] == "out":
            start_x, start_y = (self.formula_frame_port_position(*frame_port)
                                if isinstance(frame_port, tuple) else self.formula_source_port_position(block_id))
            end_x, end_y = x, y
        else:
            start_x, start_y = x, y
            end_x, end_y = (self.formula_frame_port_position(*frame_port)
                            if isinstance(frame_port, tuple) else self.formula_target_port_position(block_id, int(index)))
        middle_x = (start_x + end_x) / 2
        return [start_x, start_y, middle_x, start_y, middle_x, end_y, end_x, end_y]

    def draw_wire_preview(self, x: float, y: float) -> None:
        canvas = self.formula_canvas
        canvas.delete("wire_preview")
        if self._wire_drag is not None:
            canvas.create_line(*self.wire_preview_points(x, y), fill="#f6b73c", width=2, dash=(4, 3),
                               tags="wire_preview")

    def drag_formula_wire(self, event) -> None:
        if self._wire_drag is None:
            return
        x, y = self.formula_canvas.canvasx(event.x), self.formula_canvas.canvasy(event.y)
        self.draw_wire_preview(float(x), float(y))

    def connect_formula_wire(self, source_id: int, target_id: int, index: int) -> None:
        target = self.formula_blocks[target_id]
        if target["kind"] == "operation":
            target["inputs"][index] = source_id
        else:
            target["input"] = source_id
        self.wire_routes.pop(self.formula_wire_key(target_id, index), None)
        self.formula_status.set("Провод подключён — результат пересчитан.")
        self.render_formula_blocks()

    def drop_formula_wire(self, event) -> None:
        if self._wire_drag is None:
            return
        x, y = self.formula_canvas.canvasx(event.x), self.formula_canvas.canvasy(event.y)
        drag = self._wire_drag
        self._wire_drag = None
        self.formula_canvas.delete("wire_preview")
        if drag["direction"] == "out":
            source_id = int(drag["block_id"])
            target_port = self.formula_port_under_cursor(float(x), float(y), "in")
            if target_port is None:
                return
            target_id, index = target_port
        else:
            target_id, index = int(drag["block_id"]), int(drag["index"])
            source_port = self.formula_port_under_cursor(float(x), float(y), "out")
            if source_port is None:
                return
            source_id, _ = source_port
        assert index is not None
        if source_id == target_id or self.formula_depends_on(source_id, target_id):
            self.formula_status.set("Нельзя создать циклическую формулу.")
            return
        self.connect_formula_wire(source_id, target_id, index)

    def start_wire_handle_drag(self, event, target_id: int, index: int) -> None:
        self._wire_handle_drag = {"target_id": target_id, "index": index}

    def drag_wire_handle(self, event) -> None:
        if self._wire_handle_drag is None:
            return
        target_id, index = int(self._wire_handle_drag["target_id"]), int(self._wire_handle_drag["index"])
        x, y = self.formula_canvas.canvasx(event.x), self.formula_canvas.canvasy(event.y)
        world_width, world_height = self._formula_world
        route = (min(max(8.0, float(x)), world_width - 8.0), min(max(8.0, float(y)), world_height - 8.0))
        key = self.formula_wire_key(target_id, index)
        self.wire_routes[key] = route
        target = self.formula_blocks.get(target_id)
        source_id = (target.get("inputs", [])[index] if target and target["kind"] == "operation" else
                     target.get("input") if target else None)
        if not isinstance(source_id, int):
            return
        points = self.formula_wire_points(source_id, target_id, index)
        self.formula_canvas.coords(f"wire_line_{key}", *points)
        self.formula_canvas.coords(f"wire_handle_{key}", route[0] - 4, route[1] - 4, route[0] + 4, route[1] + 4)

    def drop_wire_handle(self, _event) -> None:
        self._wire_handle_drag = None

    def formula_depends_on(self, block_id: int, searched_id: int, seen: set[int] | None = None) -> bool:
        """Проверка циклов: источник не может зависеть от будущего получателя."""
        if block_id == searched_id:
            return True
        if block_id not in self.formula_blocks:
            return False
        seen = set() if seen is None else seen
        if block_id in seen:
            return False
        seen.add(block_id)
        block = self.formula_blocks[block_id]
        linked = block.get("inputs", []) if block["kind"] == "operation" else [block.get("input")]
        return any(isinstance(child, int) and self.formula_depends_on(child, searched_id, seen.copy())
                   for child in linked)

    def candidate_formula_slot(self, x: float, y: float, source_id: int) -> tuple[int, int] | None:
        """Вернуть слот под указателем, если его можно безопасно соединить."""
        for target_id, target in reversed(list(self.formula_blocks.items())):
            if target_id == source_id or target["kind"] not in {"operation", "output"}:
                continue
            for index, (left, top, right, bottom) in enumerate(self.formula_slot_rects(target)):
                if left <= x <= right and top <= y <= bottom:
                    if self.formula_depends_on(source_id, target_id):
                        return None
                    return target_id, index
        return None

    def highlight_formula_slot(self, x: float, y: float, source_id: int) -> None:
        canvas = self.formula_canvas
        canvas.itemconfigure("input_slot_shape", outline="#2d8734", width=1)
        target = self.candidate_formula_slot(x, y, source_id)
        if target is not None:
            target_id, index = target
            canvas.itemconfigure(f"slot_{target_id}_{index}_shape", outline="#ffd24a", width=3)

    def dock_formula_source(self, source_id: int, target_id: int) -> None:
        """После соединения источник становится рядом со слотом, а не закрывает его."""
        source, target = self.formula_blocks[source_id], self.formula_blocks[target_id]
        source_width, source_height = self.formula_block_size(source)
        target_width, _target_height = self.formula_block_size(target)
        desired_x = float(target["x"]) - source_width - 28.0
        if desired_x < 12.0:
            desired_x = float(target["x"]) + target_width + 28.0
        world_width, world_height = self._formula_world
        source["x"] = min(max(12.0, desired_x), world_width - source_width - 12.0)
        source["y"] = min(max(12.0, float(target["y"]) + 18.0), world_height - source_height - 12.0)

    def drop_formula_block_at(self, x: float, y: float) -> None:
        if self._formula_drag is None:
            return
        source_id = int(self._formula_drag["id"])
        source = self.formula_blocks[source_id]
        candidate = None if source["kind"] == "output" else self.candidate_formula_slot(x, y, source_id)
        if candidate is not None:
            target_id, index = candidate
            target = self.formula_blocks[target_id]
            if target["kind"] == "operation":
                target["inputs"][index] = source_id
            else:
                target["input"] = source_id
            self.dock_formula_source(source_id, target_id)
            self.formula_status.set("Связь создана — результат пересчитан.")
        elif source["kind"] != "output":
            # Указатель на слоте, но соединение отклонено из-за цикла.
            for target_id, target in self.formula_blocks.items():
                if target_id == source_id:
                    continue
                if any(left <= x <= right and top <= y <= bottom
                       for left, top, right, bottom in self.formula_slot_rects(target)):
                    self.formula_status.set("Нельзя создать циклическую формулу.")
                    break
        self._formula_drag = None
        self.formula_canvas.itemconfigure("input_slot_shape", outline="#2d8734", width=1)
        self.render_formula_blocks()

    def formula_menu(self, event) -> str | None:
        # Меню рабочего поля — строго реакция на правую кнопку мыши. Это
        # дополнительно страхует обработчик от событий клавиатуры и кнопки
        # «Удалить выделенное».
        if getattr(event, "num", None) != 3:
            return
        x, y = float(self.formula_canvas.canvasx(event.x)), float(self.formula_canvas.canvasy(event.y))
        # Проверка геометрии нужна, потому что Canvas может считать текущим
        # элементом пунктирный внутренний провод, а не саму рамку.
        frame_at_pointer = self.formula_frame_at(x, y)
        if frame_at_pointer is not None:
            self.formula_frame_menu(event, frame_at_pointer)
            return "break"
        current = self.formula_canvas.find_withtag("current")
        block_id = None
        frame_id = None
        if current:
            for tag in self.formula_canvas.gettags(current[0]):
                if tag.startswith("block_"):
                    block_id = int(tag.split("_")[1])
                elif tag.startswith("frame_") and tag[6:].isdigit():
                    frame_id = int(tag[6:])
        if frame_id is not None:
            self.formula_frame_menu(event, frame_id)
            return "break"
        menu = tk.Menu(self, tearoff=False)
        if block_id is not None:
            block = self.formula_blocks[block_id]
            if block["kind"] == "constant":
                menu.add_command(label="Изменить константу…", command=lambda: self.edit_formula_constant(block_id))
            elif block["kind"] == "output":
                menu.add_command(label="Переименовать вывод…", command=lambda: self.rename_formula_output(block_id))
            if block["kind"] in {"operation", "output"}:
                menu.add_command(label="Отсоединить входы", command=lambda: self.disconnect_formula_block(block_id))
            menu.add_command(label="Дублировать", command=lambda: self.duplicate_formula_block(block_id))
            menu.add_separator()
            menu.add_command(label="Удалить блок", command=lambda: self.delete_formula_block(block_id))
        else:
            area_menu = tk.Menu(menu, tearoff=False)
            for item in self.ordered_integrals():
                number = self.integral_number(item["id"])
                area_menu.add_command(label=f"Площадь I{number}", command=lambda i=int(item["id"]): self.add_formula_block("area", x, y, integral_id=i))
            menu.add_cascade(label="Площадь", menu=area_menu)
            menu.add_command(label="Константа…", command=lambda: self.add_formula_constant(x, y))
            operation_menu = tk.Menu(menu, tearoff=False)
            for operation in self._OPERATION_NAMES:
                operation_menu.add_command(label=operation, command=lambda op=operation: self.add_formula_block("operation", x, y, operation=op))
            menu.add_cascade(label="Мат. операция", menu=operation_menu)
            menu.add_command(label="Блок вывода", command=lambda: self.add_formula_block("output", x, y))
            if self.selected_formula_blocks:
                menu.add_separator()
                menu.add_command(label="Создать фрейм из выделенных", command=self.create_formula_frame)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def edit_formula_constant(self, block_id: int) -> None:
        current = float(self.formula_blocks[block_id]["value"])
        value = self.ask_formula_number("Константа", "Новое значение:", current)
        if value is not None:
            self.formula_blocks[block_id]["value"] = value
            self.refresh_formula_workspace()

    def rename_formula_output(self, block_id: int) -> None:
        current = str(self.formula_blocks[block_id].get("name") or "Вывод")
        name = simpledialog.askstring("Вывод", "Название карточки результата:", initialvalue=current, parent=self)
        if name and name.strip():
            self.formula_blocks[block_id]["name"] = name.strip()
            self.formula_blocks[block_id]["auto_name"] = False
            self.render_formula_blocks()

    def renumber_formula_outputs(self) -> None:
        """Автоматические имена нумеруются только среди блоков вывода."""
        outputs = [block for block in self.formula_blocks.values() if block["kind"] == "output"]
        for number, block in enumerate(outputs, start=1):
            if block.get("auto_name", True):
                block["name"] = "Вывод" if number == 1 else f"Вывод {number}"

    def disconnect_formula_block(self, block_id: int) -> None:
        block = self.formula_blocks.get(block_id)
        if block is None:
            return
        if block["kind"] == "operation":
            block["inputs"] = [None for _ in block["inputs"]]
            for index in range(len(block["inputs"])):
                self.wire_routes.pop(self.formula_wire_key(block_id, index), None)
        elif block["kind"] == "output":
            block["input"] = None
            self.wire_routes.pop(self.formula_wire_key(block_id, 0), None)
        self.render_formula_blocks()

    def duplicate_formula_block(self, block_id: int) -> None:
        block = self.formula_blocks.get(block_id)
        if block is None:
            return
        x, y = float(block["x"]) + 36, float(block["y"]) + 36
        clone = self.add_formula_block(
            str(block["kind"]), x, y, integral_id=block.get("integral_id"),
            operation=block.get("operation"), value=block.get("value"),
            name=str(block.get("name") or "") + " копия" if block["kind"] == "output" else None,
        )
        if block["kind"] == "operation":
            self.formula_blocks[clone]["inputs"] = list(block.get("inputs", []))
        elif block["kind"] == "output":
            self.formula_blocks[clone]["input"] = block.get("input")
        self.render_formula_blocks()

    def delete_formula_block(self, block_id: int) -> None:
        self.formula_blocks.pop(block_id, None)
        self.renumber_formula_outputs()
        for block in self.formula_blocks.values():
            if block.get("input") == block_id:
                block["input"] = None
            if "inputs" in block:
                block["inputs"] = [None if value == block_id else value for value in block["inputs"]]
        self.cleanup_formula_frames()
        self.cleanup_wire_routes()
        self.refresh_formula_workspace()

    def cleanup_wire_routes(self) -> None:
        """Убрать маршруты, для которых после удаления больше нет связи."""
        for key in list(self.wire_routes):
            try:
                target_text, index_text = key.rsplit("_", 1)
                target_id, index = int(target_text), int(index_text)
            except ValueError:
                self.wire_routes.pop(key, None)
                continue
            target = self.formula_blocks.get(target_id)
            if target is None:
                self.wire_routes.pop(key, None)
                continue
            inputs = target.get("inputs", [])
            source_id = inputs[index] if target["kind"] == "operation" and index < len(inputs) else \
                target.get("input") if target["kind"] == "output" and index == 0 else None
            if not isinstance(source_id, int) or source_id not in self.formula_blocks:
                self.wire_routes.pop(key, None)

    def formula_preset_data(self) -> dict[str, object]:
        """Схема формул без численных площадей конкретного спектра.

        Площадь хранится не по внутреннему ID, а по номеру интеграла слева
        направо. Поэтому один и тот же пресет можно применить к другому
        спектру после ручного задания тех же участков интегрирования.
        """
        comparison = self.comparison_record()
        source_integrals = (sorted(list(comparison.get("integrals", [])), key=lambda item: int(item["number"]))
                            if comparison is not None else self.ordered_integrals())
        integral_positions = {int(item["id"]): number
                              for number, item in enumerate(source_integrals)}
        blocks: list[dict[str, object]] = []
        for block_id, block in self.formula_blocks.items():
            kind = str(block["kind"])
            if kind not in {"area", "constant", "operation", "output"}:
                continue
            saved: dict[str, object] = {
                "id": block_id, "kind": kind, "x": float(block["x"]), "y": float(block["y"]),
                "operation": block.get("operation"), "value": block.get("value"),
                "inputs": list(block.get("inputs", [])), "input": block.get("input"),
                "name": block.get("name", ""), "auto_name": bool(block.get("auto_name", True)),
                # ID обзора действует только в пределах открытого проекта. В
                # переносимом пресете его не сохраняем: при загрузке формула
                # должна взять активный обзор, а не случайный старый ID.
                "comparison_id": None if kind == "area" else block.get("comparison_id"),
            }
            if kind == "area":
                saved["area_index"] = integral_positions.get(block.get("integral_id"))
                number = self.formula_area_number(block)
                saved["area_number"] = int(number) if number.isdigit() else None
            blocks.append(saved)
        routes: list[dict[str, object]] = []
        for key, route in self.wire_routes.items():
            try:
                target_text, index_text = key.rsplit("_", 1)
                routes.append({"target_id": int(target_text), "index": int(index_text),
                               "x": float(route[0]), "y": float(route[1])})
            except (ValueError, TypeError, IndexError):
                continue
        comments = [
            {"x": float(comment["x"]), "y": float(comment["y"]), "text": str(comment["text"])}
            for comment in self.formula_comments.values()
        ]
        frames: list[dict[str, object]] = []
        for frame_id, frame in self.formula_frames.items():
            members = self.formula_frame_members(frame)
            if not members:
                continue
            inputs = []
            for port in frame.get("inputs", []):
                if isinstance(port, dict):
                    inputs.append({"name": str(port.get("name") or "Вход"),
                                   "target_id": port.get("target_id"), "index": port.get("index")})
            outputs = []
            for port in frame.get("outputs", []):
                if isinstance(port, dict):
                    outputs.append({"name": str(port.get("name") or "Выход"),
                                    "source_id": port.get("source_id")})
            frames.append({
                "id": frame_id, "name": str(frame.get("name") or "Фрейм"), "members": members,
                "collapsed": bool(frame.get("collapsed")), "x": float(frame.get("x", 40)),
                "y": float(frame.get("y", 40)), "inputs": inputs, "outputs": outputs,
            })
        return {"format": "nmr-formula-preset", "version": 1, "blocks": blocks,
                "wire_routes": routes, "comments": comments, "frames": frames}

    def save_formula_preset(self) -> None:
        if not self.formula_blocks:
            messagebox.showinfo("Пресет формул", "Сначала соберите хотя бы один блок формулы.", parent=self)
            return
        filename = filedialog.asksaveasfilename(
            parent=self, title="Сохранить пресет формул", defaultextension=".json",
            filetypes=(("Пресет формул", "*.json"), ("Все файлы", "*.*")),
        )
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(self.formula_preset_data(), file, ensure_ascii=False, indent=2)
        except OSError as error:
            messagebox.showerror("Пресет формул", f"Не удалось сохранить файл:\n{error}", parent=self)
            return
        self.formula_status.set("Пресет формул сохранён.")

    def load_formula_preset(self, preset: dict[str, object] | None = None, *, show_error: bool = True) -> bool:
        if preset is None:
            filename = filedialog.askopenfilename(
                parent=self, title="Загрузить пресет формул",
                filetypes=(("Пресет формул", "*.json"), ("Все файлы", "*.*")),
            )
            if not filename:
                return False
            try:
                with open(filename, encoding="utf-8") as file:
                    preset = json.load(file)
            except (OSError, json.JSONDecodeError) as error:
                if show_error:
                    messagebox.showerror("Пресет формул", f"Не удалось открыть пресет:\n{error}", parent=self)
                return False
        try:
            if not isinstance(preset, dict) or preset.get("format") != "nmr-formula-preset":
                raise ValueError("это не пресет формул NMR")
            raw_blocks = preset.get("blocks")
            if not isinstance(raw_blocks, list):
                raise ValueError("в файле нет списка блоков")
        except ValueError as error:
            if show_error:
                messagebox.showerror("Пресет формул", f"Не удалось открыть пресет:\n{error}", parent=self)
            return False

        allowed_kinds = {"area", "constant", "operation", "output"}
        current_comparison = self.comparison_record()
        # В обзоре I1, I2… принадлежат самому сравнению. Активный отдельный
        # спектр может вообще не содержать интегралов, хотя таблица и палитра
        # показывают общие площади — из-за этого и появлялось ложное сообщение
        # «Не найдены 10 площадей».
        available_integrals = (
            sorted(list(current_comparison.get("integrals", [])), key=lambda item: int(item["number"]))
            if current_comparison is not None else self.ordered_integrals()
        )
        new_blocks: dict[int, dict[str, object]] = {}
        id_map: dict[int, int] = {}
        pending: list[tuple[int, dict[str, object], dict[str, object]]] = []
        skipped = 0
        for raw in raw_blocks:
            if not isinstance(raw, dict) or raw.get("kind") not in allowed_kinds:
                skipped += 1
                continue
            try:
                old_id = int(raw["id"])
                if old_id in id_map:
                    raise ValueError
                kind = str(raw["kind"])
                x, y = float(raw.get("x", 40)), float(raw.get("y", 40))
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            operation = raw.get("operation") if kind == "operation" else None
            if kind == "operation" and operation not in self._OPERATION_NAMES:
                skipped += 1
                continue
            block_id = len(new_blocks) + 1
            id_map[old_id] = block_id
            unary = operation in self._UNARY_OPERATIONS
            area_id = None
            if kind == "area":
                # Новые пресеты привязаны к видимому номеру интеграла. Это
                # переживает ручную перестановку номеров; старые JSON всё ещё
                # загружаются по прежнему порядковому индексу.
                area_number = raw.get("area_number")
                if isinstance(area_number, int):
                    selected = next((item for item in available_integrals
                                     if int(item.get("number", -1)) == area_number), None)
                    if selected is not None:
                        area_id = int(selected["id"])
                if area_id is None:
                    area_index = raw.get("area_index")
                    if isinstance(area_index, int) and 0 <= area_index < len(available_integrals):
                        area_id = int(available_integrals[area_index]["id"])
            new_block: dict[str, object] = {
                "kind": kind, "x": x, "y": y, "integral_id": area_id, "operation": operation,
                "area_number": raw.get("area_number") if isinstance(raw.get("area_number"), int) else None,
                # При загрузке из файла используем именно открытый обзор. ID
                # из другого проекта не является стабильной ссылкой.
                "comparison_id": current_comparison.get("id") if kind == "area" and current_comparison is not None else None,
                "inputs": [None] if unary else [None, None] if kind == "operation" else [],
                "input": None, "value": raw.get("value"),
                "name": str(raw.get("name") or ""), "auto_name": bool(raw.get("auto_name", True)),
            }
            if kind == "constant":
                try:
                    new_block["value"] = float(raw.get("value"))
                except (TypeError, ValueError):
                    new_block["value"] = 0.0
            new_blocks[block_id] = new_block
            pending.append((block_id, raw, new_block))
        if not new_blocks:
            if show_error:
                messagebox.showerror("Пресет формул", "В пресете нет поддерживаемых блоков.", parent=self)
            return False
        for _block_id, raw, block in pending:
            if block["kind"] == "operation":
                raw_inputs = raw.get("inputs")
                if isinstance(raw_inputs, list):
                    expected_inputs = len(block["inputs"])
                    block["inputs"] = [id_map.get(value) if isinstance(value, int) else None
                                       for value in raw_inputs[:expected_inputs]]
                    block["inputs"] += [None] * (expected_inputs - len(block["inputs"]))
            elif block["kind"] == "output":
                raw_input = raw.get("input")
                block["input"] = id_map.get(raw_input) if isinstance(raw_input, int) else None
        self.formula_blocks = new_blocks
        self._next_formula_id = max(new_blocks) + 1
        self.selected_formula_blocks.clear()
        self.formula_frames = {}
        self._next_formula_frame_id = 1
        self.formula_comments = {}
        self._next_formula_comment_id = 1
        for raw_comment in preset.get("comments", []):
            if not isinstance(raw_comment, dict):
                continue
            try:
                text = str(raw_comment["text"]).strip()
                x, y = float(raw_comment["x"]), float(raw_comment["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if text:
                self.formula_comments[self._next_formula_comment_id] = {"x": x, "y": y, "text": text}
                self._next_formula_comment_id += 1
        self.wire_routes = {}
        for raw_route in preset.get("wire_routes", []):
            if not isinstance(raw_route, dict):
                continue
            try:
                target_id = id_map[int(raw_route["target_id"])]
                index, x, y = int(raw_route["index"]), float(raw_route["x"]), float(raw_route["y"])
            except (KeyError, TypeError, ValueError):
                continue
            self.wire_routes[self.formula_wire_key(target_id, index)] = (x, y)
        for raw_frame in preset.get("frames", []):
            if not isinstance(raw_frame, dict):
                continue
            raw_members = raw_frame.get("members")
            if not isinstance(raw_members, list):
                continue
            members = [id_map[old_id] for old_id in raw_members if isinstance(old_id, int) and old_id in id_map]
            if not members:
                continue
            inputs: list[dict[str, object]] = []
            for raw_port in raw_frame.get("inputs", []):
                if not isinstance(raw_port, dict):
                    continue
                old_target, slot = raw_port.get("target_id"), raw_port.get("index")
                target_id = id_map.get(old_target) if isinstance(old_target, int) else None
                if not isinstance(target_id, int) or not isinstance(slot, int):
                    continue
                target = new_blocks[target_id]
                max_slots = len(target.get("inputs", [])) if target["kind"] == "operation" else 1 if target["kind"] == "output" else 0
                if not 0 <= slot < max_slots:
                    continue
                inputs.append({"name": str(raw_port.get("name") or f"Вход {len(inputs) + 1}"),
                               "target_id": target_id, "index": slot})
            outputs: list[dict[str, object]] = []
            for raw_port in raw_frame.get("outputs", []):
                if not isinstance(raw_port, dict):
                    continue
                old_source = raw_port.get("source_id")
                source_id = id_map.get(old_source) if isinstance(old_source, int) else None
                if not isinstance(source_id, int) or new_blocks[source_id]["kind"] == "output":
                    continue
                outputs.append({"name": str(raw_port.get("name") or f"Выход {len(outputs) + 1}"),
                                "source_id": source_id})
            try:
                x, y = float(raw_frame.get("x", 40)), float(raw_frame.get("y", 40))
            except (TypeError, ValueError):
                x, y = 40.0, 40.0
            frame_id = self._next_formula_frame_id
            self._next_formula_frame_id += 1
            self.formula_frames[frame_id] = {
                "name": str(raw_frame.get("name") or "Фрейм"), "members": members,
                "collapsed": bool(raw_frame.get("collapsed")), "x": x, "y": y,
                "inputs": inputs, "outputs": outputs,
            }
        self.cleanup_wire_routes()
        self.cleanup_formula_frames()
        self.renumber_formula_outputs()
        self.refresh_formula_workspace()
        missing = sum(1 for block in new_blocks.values()
                      if block["kind"] == "area" and block.get("integral_id") is None)
        note = f"Пресет загружен: {len(new_blocks)} блоков."
        if missing:
            note += f" Не найдены {missing} площадей: в формулах они приняты за 0."
        if skipped:
            note += f" Пропущено блоков: {skipped}."
        self.formula_status.set(note)
        return True

    def refresh_result_panel(self) -> None:
        if not hasattr(self, "result_panel"):
            return
        for child in self.result_panel.winfo_children():
            child.destroy()
        self.formula_results_table = None
        outputs = [(block_id, block) for block_id, block in self.formula_blocks.items() if block["kind"] == "output"]
        if not outputs:
            tk.Label(self.result_panel, text="Добавьте блок\n«Карточка результата»", bg="#ffffff",
                     fg="#98a1aa", justify="left").pack(anchor="w", padx=4, pady=4)
            return
        # Для схемы, собранной на обзорном графике, отдельное число «Вывод»
        # неоднозначно: оно относится лишь к активной строке. Показываем
        # только таблицу для всех спектров этого обзора.
        comparison = self.comparison_record()
        if comparison is None:
            stored_ids = {
                block.get("comparison_id") for block in self.formula_blocks.values()
                if block.get("kind") == "area" and isinstance(block.get("comparison_id"), int)
            }
            if len(stored_ids) == 1:
                comparison = self.comparisons.get(next(iter(stored_ids)))
        comparison_spectrum_ids = [int(value) for value in comparison.get("spectrum_ids", [])
                                   if int(value) in self.spectra] if comparison is not None else []
        multi_result = len(comparison_spectrum_ids) > 1

        if not multi_result:
            for number, (block_id, block) in enumerate(outputs, start=1):
                name = str(block.get("name") or ("Вывод" if number == 1 else f"Вывод {number}"))
                value = self.formula_value(block_id)
                card = tk.Frame(self.result_panel, bg="#eff8ff", highlightbackground="#b7d5e8", highlightthickness=1,
                                padx=8, pady=7)
                card.pack(fill="x", pady=(0, 7))
                tk.Label(card, text=name, bg="#eff8ff", fg="#35617d", anchor="w",
                         font=("Segoe UI", 9, "bold")).pack(fill="x")
                value_row = tk.Frame(card, bg="#eff8ff")
                value_row.pack(fill="x", pady=(4, 0))
                tk.Label(value_row, text=self.format_formula_value(value), bg="#eff8ff", fg="#153f59",
                         font=("Consolas", 13, "bold"), anchor="w").pack(side="left", fill="x", expand=True)
                tk.Button(value_row, text="Копировать", command=lambda v=value: self.copy_formula_value(v),
                          padx=4, pady=1).pack(side="right")

        if multi_result:
            result_header = tk.Frame(self.result_panel, bg="#ffffff")
            result_header.pack(fill="x", pady=(0, 4))
            tk.Label(result_header, text="Результаты по спектрам", bg="#ffffff", fg="#536171",
                     font=("Segoe UI", 9, "bold")).pack(side="left")
            columns = tuple(f"out_{block_id}" for block_id, _block in outputs)
            table = ttk.Treeview(self.result_panel, columns=("spectrum",) + columns, show="headings",
                                 height=min(6, len(comparison_spectrum_ids)), selectmode="extended")
            table.heading("spectrum", text="Спектр")
            table.column("spectrum", width=105, anchor="w", stretch=True)
            for number, (block_id, block) in enumerate(outputs, start=1):
                name = str(block.get("name") or ("Вывод" if number == 1 else f"Вывод {number}"))
                column = f"out_{block_id}"
                table.heading(column, text=name)
                table.column(column, width=78, anchor="center", stretch=True)
            for spectrum_id in comparison_spectrum_ids:
                record = self.spectra[spectrum_id]
                values = [str(record["name"])]
                values.extend(self.format_formula_value(self.formula_value_for_record(block_id, record))
                              for block_id, _block in outputs)
                table.insert("", "end", values=values)
            self.formula_results_table = table
            table.bind("<Control-KeyPress>", self.copy_formula_results_table_shortcut)
            table.pack(fill="x", pady=(0, 4))

    def copy_formula_value(self, value: float | None) -> None:
        if value is None:
            return
        self.clipboard_clear()
        self.clipboard_append(self.format_formula_value(value))

    def copy_formula_results_table(self, _event=None) -> str:
        """Скопировать результаты формулы по спектрам в Excel-совместимый TSV."""
        table = self.formula_results_table
        if table is None or not self.copy_treeview_rows(table, context="formula"):
            self.formula_status.set("Таблица результатов по спектрам пока не построена.")
        return "break"

    def copy_formula_results_table_shortcut(self, event) -> str | None:
        keysym = str(getattr(event, "keysym", "")).lower()
        is_c_key = keysym in {"c", "cyrillic_es", "с"} or int(getattr(event, "keycode", -1)) == 67
        return self.copy_formula_results_table(event) if is_c_key else None

    def on_plot_click(self, event) -> None:
        if event.button != 3 or event.xdata is None:
            return
        if self.current_comparison_id is not None:
            comparison = self.comparison_record()
            if comparison is None or event.inaxes not in self._comparison_axes or event.guiEvent is None:
                return
            item = self.comparison_integral_at(comparison, float(event.xdata))
            if item is None:
                return
            number = int(item.get("number", 0))
            menu = tk.Menu(self, tearoff=False)
            menu.add_command(
                label=f"Убрать общий интеграл I{number}",
                command=lambda selected=item: self.remove_comparison_integral(selected),
            )
            try:
                menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
            finally:
                menu.grab_release()
            return
        if event.inaxes is not self.axes:
            return
        item = next((candidate for candidate in reversed(self.integrals)
                     if float(candidate["left"]) <= event.xdata <= float(candidate["right"])), None)
        if event.guiEvent is None:
            return
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="Задать химический сдвиг…", command=lambda: self.calibrate_shift(event.xdata))
        if item is not None:
            menu.add_separator()
            menu.add_command(label="Калибровать как 1.000", command=lambda: self.calibrate_integral(item))
            menu.add_command(label="Изменить номер…", command=lambda: self.rename_integral_number(item))
            menu.add_command(label="Переименовать…", command=lambda: self.rename_integral(item))
            menu.add_separator()
            menu.add_command(label="Убрать интеграл", command=lambda: self.remove_integral(item))
        menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)

    def calibrate_shift(self, clicked_ppm: float) -> None:
        """Сдвинуть всю ось так, чтобы выбранный пик получил известное ppm."""
        reference = simpledialog.askfloat(
            "Калибровка химического сдвига",
            f"Текущая координата выбранного пика: {clicked_ppm:.5f} ppm\n"
            "Введите его известный химический сдвиг, ppm:",
            initialvalue=round(clicked_ppm, 5), parent=self,
        )
        if reference is None or self.ppm is None:
            return
        previous = self.capture_spectrum_state()
        shift = reference - clicked_ppm
        self.ppm = self.ppm + shift
        self.ppm_shift += shift
        for item in self.integrals:
            item["left"] = float(item["left"]) + shift
            item["right"] = float(item["right"]) + shift
        self.remember_spectrum_state(previous, "калибровка химического сдвига")
        self.draw()
        self.status.set(f"Ось химических сдвигов сдвинута на {shift:+.5f} ppm.")

    def calibrate_integral(self, item: dict[str, float | int | str]) -> None:
        previous = self.capture_spectrum_state()
        self.reference_integral_id = int(item["id"])
        self.remember_spectrum_state(previous, "калибровка интеграла")
        self.draw()
        self.status.set("Выбранный интеграл принят за единицу площади.")

    def rename_integral(self, item: dict[str, float | int | str]) -> None:
        name = simpledialog.askstring("Название интеграла", "Название:", initialvalue=str(item["name"]), parent=self)
        if name:
            previous = self.capture_spectrum_state()
            item["name"] = name
            self.remember_spectrum_state(previous, "переименование интеграла")
            self.draw()

    def rename_integral_number(self, item: dict[str, float | int | str]) -> None:
        """Изменить отображаемый номер, сохранив ссылки формул на стабильный ID."""
        current = int(item.get("number", 1))
        number = simpledialog.askinteger(
            "Номер интеграла", "Новый номер интеграла:", initialvalue=current,
            minvalue=1, parent=self,
        )
        if number is None or number == current:
            return
        previous = self.capture_spectrum_state()
        # Номера остаются уникальными: если занятый номер выбран намеренно,
        # меняем два номера местами, а не ломаем порядок палитры и пресетов.
        occupied = next((other for other in self.integrals
                         if other is not item and int(other.get("number", -1)) == number), None)
        if occupied is not None:
            occupied["number"] = current
        item["number"] = number
        self.remember_spectrum_state(previous, "изменение номера интеграла")
        self.draw()
        self.status.set(f"Интегралу присвоен номер {number}.")

    def remove_integral(self, item: dict[str, float | int | str]) -> None:
        previous = self.capture_spectrum_state()
        if self.reference_integral_id == int(item["id"]):
            self.reference_integral_id = None
        self.integrals.remove(item)
        self.remember_spectrum_state(previous, "удаление интеграла")
        self.draw()

    def open_phase_dialog(self) -> None:
        if self.current_comparison_id is not None:
            messagebox.showinfo("Фазирование", "Для фазирования выберите отдельный спектр в списке слева.", parent=self)
            return
        if self.real is None:
            messagebox.showinfo("Нет спектра", "Сначала загрузите спектр.", parent=self)
            return
        PhaseDialog(self)

    def open_correction_dialog(self) -> None:
        if self.current_comparison_id is not None:
            messagebox.showinfo("Коррекция базовой линии",
                                "Для коррекции выберите отдельный спектр в списке слева.", parent=self)
            return
        if self.real is None:
            messagebox.showinfo("Нет спектра", "Сначала загрузите спектр.", parent=self)
            return
        CorrectionDialog(self)


from nmr_dialogs import CorrectionDialog, MultiSpectrumDialog, PhaseDialog, ShiftSpectrumDialog

if __name__ == "__main__":
    SpectrumApp().mainloop()
