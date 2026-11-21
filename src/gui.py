"""Small desktop GUI: run the pipeline steps and label the reference points.

    python src/gui.py            open the window
    python src/gui.py --selftest build the window, render one crop, close (used by CI)

All logic lives in ``gui_logic`` (forms, runner) and ``labeltool`` (labelling session), which are unit
tested.  This file only places widgets.  Labelling is blind: the tool never shows model output.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gui_logic as gl  # noqa: E402
import labeltool as lt  # noqa: E402

SETTINGS = os.path.join(os.path.expanduser("~"), ".benthic_cover_gui.json")
HOTKEYS = "1234567890abcdefghijklmnopqrstuvwxyz"
DISPLAY_PX = 420
CONTEXT_CM, CLOSE_CM = 40.0, 8.0


def load_settings() -> dict:
    try:
        with open(SETTINGS, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    try:
        with open(SETTINGS, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except OSError:
        pass


class RunTab(ttk.Frame):
    """Pick a step, fill its form, run it, watch the log."""

    def __init__(self, master, shared: dict, settings: dict):
        super().__init__(master, padding=8)
        self.shared, self.settings = shared, settings
        self.vars: dict = {}
        self.q: queue.Queue = queue.Queue()
        self.running = False

        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=(0, 10))
        ttk.Label(left, text="Steps", font=("", 10, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(left, width=34, height=len(gl.STEPS), exportselection=False, activestyle="none")
        for s in gl.STEPS:
            self.listbox.insert("end", s.title)
        self.listbox.pack(fill="y")
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.show_selected())

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)
        self.help = ttk.Label(right, wraplength=760, justify="left")
        self.help.pack(anchor="w", pady=(0, 6))
        self.form = ttk.Frame(right)
        self.form.pack(fill="x")
        bar = ttk.Frame(right)
        bar.pack(fill="x", pady=6)
        self.run_btn = ttk.Button(bar, text="Run", command=self.run)
        self.run_btn.pack(side="left")
        self.status = ttk.Label(bar, text="")
        self.status.pack(side="left", padx=10)
        box = ttk.Frame(right)
        box.pack(fill="both", expand=True)
        self.log = tk.Text(box, height=14, wrap="none", font=("Consolas", 9))
        sb = ttk.Scrollbar(box, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

        self.listbox.selection_set(0)
        self.show_selected()
        self.after(100, self._poll)

    # ------------------------------------------------------------- variables
    def var(self, step: gl.Step, f: gl.Field):
        key = f.key if f.key in gl.SHARED else f"{step.name}.{f.key}"
        if key not in self.vars:
            v = tk.BooleanVar() if f.kind == "flag" else tk.StringVar()
            saved = self.settings.get("values", {}).get(key)
            if f.kind == "flag":
                v.set(bool(saved) if saved is not None else False)
            else:
                v.set(saved if saved is not None else f.default)
            self.vars[key] = v
        return self.vars[key]

    def values(self, step: gl.Step) -> dict:
        return {f.key: self.var(step, f).get() for f in step.fields}

    def snapshot(self) -> dict:
        return {k: v.get() for k, v in self.vars.items()}

    # ------------------------------------------------------------- form
    def current_step(self) -> gl.Step:
        return gl.STEPS[self.listbox.curselection()[0]]

    def show_selected(self) -> None:
        step = self.current_step()
        self.help.configure(text=step.help)
        for w in self.form.winfo_children():
            w.destroy()
        self.form.columnconfigure(1, weight=1)
        for r, f in enumerate(step.fields):
            label = f.label + (" *" if f.required else "")
            ttk.Label(self.form, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=1)
            v = self.var(step, f)
            if f.kind == "flag":
                ttk.Checkbutton(self.form, variable=v).grid(row=r, column=1, sticky="w")
            elif f.kind == "choice":
                ttk.Combobox(self.form, textvariable=v, values=f.choices, state="readonly", width=22).grid(
                    row=r, column=1, sticky="w")
            else:
                ttk.Entry(self.form, textvariable=v).grid(row=r, column=1, sticky="ew")
            if f.kind in ("file", "save", "dir", "multi"):
                ttk.Button(self.form, text="Browse...", width=10,
                           command=lambda v=v, k=f.kind: self.browse(v, k)).grid(row=r, column=2, padx=(6, 0))

    @staticmethod
    def browse(var: tk.StringVar, kind: str) -> None:
        if kind == "dir":
            p = filedialog.askdirectory()
        elif kind == "save":
            p = filedialog.asksaveasfilename()
        elif kind == "multi":
            ps = filedialog.askopenfilenames()
            p = " ".join(f'"{x}"' if " " in x else x for x in ps)
        else:
            p = filedialog.askopenfilename()
        if p:
            var.set(p)

    # ------------------------------------------------------------- running
    def run(self) -> None:
        if self.running:
            return
        step = self.current_step()
        values = self.values(step)
        missing = gl.missing_fields(step, values)
        if missing:
            messagebox.showwarning("Missing values", "Please fill in:\n- " + "\n- ".join(missing))
            return
        self.running = True
        self.run_btn.state(["disabled"])
        self.status.configure(text=f"Running {step.title} ...", foreground="")
        self.log.insert("end", f"\n$ {step.name} {' '.join(gl.build_argv(step, values))}\n")

        def work():
            ok, msg = gl.run_step(step, values, self.q.put)
            self.q.put(("done", step, values, ok, msg))

        threading.Thread(target=work, daemon=True).start()

    def _append(self, s: str) -> None:
        parts = s.split("\r")
        self.log.insert("end", parts[0])
        for p in parts[1:]:                                   # progress lines overwrite themselves
            self.log.delete("end-1c linestart", "end-1c")
            self.log.insert("end", p)
        self.log.see("end")

    def _poll(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "done":
                    self._finished(*item[1:])
                else:
                    self._append(item)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finished(self, step, values, ok, msg) -> None:
        self.running = False
        self.run_btn.state(["!disabled"])
        self.status.configure(text=("OK: " if ok else "FAILED: ") + msg, foreground="#1a7f37" if ok else "#c62828")
        self._append("\n" + ("Done.\n" if ok else f"ERROR: {msg}\n"))
        if ok:
            for key, val in gl.propagate(step, values).items():
                if key in self.vars:
                    self.vars[key].set(val)
                self.shared[key] = val
        save_settings({**self.settings, "values": self.snapshot()})


class LabelTab(ttk.Frame):
    """Blind labelling of the reference points: two crops around the point, one key per class."""

    def __init__(self, master, shared: dict, settings: dict):
        super().__init__(master, padding=8)
        self.shared, self.settings = shared, settings
        self.session = None
        self.reader = None
        self.classes: list = list(settings.get("classes", []))
        self.mm_per_px = 0.4
        self._photos: list = []
        self.bright = tk.BooleanVar(value=True)

        top = ttk.Frame(self)
        top.pack(fill="x")
        self.points = tk.StringVar(value=settings.get("label_points", ""))
        self.raster = tk.StringVar(value=settings.get("label_raster", ""))
        self.mm = tk.StringVar(value=settings.get("label_mm", "0.412284"))
        self.schema = tk.StringVar(value=settings.get("label_schema", ""))
        for r, (text, var, kind) in enumerate((("Points CSV", self.points, "file"), ("Mosaic (GeoTIFF)", self.raster, "file"),
                                               ("Pixel size (mm/px)", self.mm, None), ("Class schema CSV (optional)", self.schema, "file"))):
            ttk.Label(top, text=text).grid(row=r, column=0, sticky="w", padx=(0, 8))
            ttk.Entry(top, textvariable=var, width=70 if kind else 14).grid(row=r, column=1, sticky="w" if not kind else "ew")
            if kind:
                ttk.Button(top, text="Browse...", width=10, command=lambda v=var: self._browse(v)).grid(row=r, column=2, padx=6)
        top.columnconfigure(1, weight=1)
        ttk.Button(top, text="Start / resume", command=self.start).grid(row=0, column=3, rowspan=2, padx=6)
        self.info = ttk.Label(top, text="Blind mode: model predictions are never shown here.", foreground="#555")
        self.info.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=8)
        views = ttk.Frame(body)
        views.pack(side="left")
        self.ctx = tk.Canvas(views, width=DISPLAY_PX, height=DISPLAY_PX, bg="#222", highlightthickness=0)
        self.close = tk.Canvas(views, width=DISPLAY_PX, height=DISPLAY_PX, bg="#222", highlightthickness=0)
        self.ctx.grid(row=0, column=0, padx=(0, 6))
        self.close.grid(row=0, column=1)
        self.ctx_caption = ttk.Label(views, text=f"Context ({CONTEXT_CM:g} cm)")
        self.close_caption = ttk.Label(views, text=f"Close-up ({CLOSE_CM:g} cm)")
        self.ctx_caption.grid(row=1, column=0)
        self.close_caption.grid(row=1, column=1)
        ttk.Checkbutton(views, text="Brighten", variable=self.bright, command=self.render).grid(row=2, column=0, sticky="w")

        side = ttk.Frame(body)
        side.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.point_label = ttk.Label(side, text="No session", font=("", 11, "bold"))
        self.point_label.pack(anchor="w")
        self.progress = ttk.Progressbar(side, maximum=1, value=0)
        self.progress.pack(fill="x", pady=4)
        row = ttk.Frame(side)
        row.pack(fill="x")
        ttk.Button(row, text="Undo (Backspace)", command=self.undo).pack(side="left")
        ttk.Button(row, text="Skip (Space)", command=self.skip).pack(side="left", padx=4)
        ttk.Button(row, text="Unscorable (Del)", command=self.unscorable).pack(side="left")
        add = ttk.Frame(side)
        add.pack(fill="x", pady=4)
        self.new_label = tk.StringVar()
        ttk.Entry(add, textvariable=self.new_label).pack(side="left", fill="x", expand=True)
        ttk.Button(add, text="Add label", command=self.add_label).pack(side="left", padx=4)
        holder = ttk.Frame(side)
        holder.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(holder, highlightthickness=0, height=300)
        sb = ttk.Scrollbar(holder, command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.rebuild_buttons()
        self.winfo_toplevel().bind_all("<Key>", self._on_key, add="+")

    # ------------------------------------------------------------- setup
    @staticmethod
    def _browse(var: tk.StringVar) -> None:
        p = filedialog.askopenfilename()
        if p:
            var.set(p)

    def prefill(self) -> None:
        """Take paths produced by earlier steps (reference sample, mosaic, pixel size)."""
        if self.shared.get("points") and not self.points.get():
            self.points.set(self.shared["points"])
        if self.shared.get("raster") and not self.raster.get():
            self.raster.set(self.shared["raster"])
        if self.shared.get("mm_per_px"):
            self.mm.set(self.shared["mm_per_px"])

    def start(self) -> None:
        try:
            import pipeline
            out = os.path.splitext(self.points.get())[0] + "_labelled.csv"
            session = lt.LabelSession(self.points.get(), out)
            reader = pipeline.open_reader(self.raster.get())
            if self.schema.get().strip():
                for c in lt.classes_from_schema(self.schema.get()):
                    if c not in self.classes:
                        self.classes.append(c)
            self.load_session(session, reader, float(self.mm.get()))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Cannot start", f"{type(e).__name__}: {e}")
            return
        self.shared["reference"] = session.out_csv
        self.persist()

    def load_session(self, session, reader, mm_per_px: float) -> None:
        self.session, self.reader, self.mm_per_px = session, reader, mm_per_px
        self.rebuild_buttons()
        self.render()

    def persist(self) -> None:
        save_settings({**self.settings, "classes": self.classes, "label_points": self.points.get(),
                       "label_raster": self.raster.get(), "label_mm": self.mm.get(), "label_schema": self.schema.get()})

    # ------------------------------------------------------------- classes
    def rebuild_buttons(self) -> None:
        for w in self.inner.winfo_children():
            w.destroy()
        for i, c in enumerate(self.classes):
            key = HOTKEYS[i] if i < len(HOTKEYS) else "-"
            ttk.Button(self.inner, text=f"[{key}]  {c}", width=44, command=lambda c=c: self.answer(c)).grid(
                row=i, column=0, sticky="w", pady=1)

    def add_label(self) -> None:
        c = self.new_label.get().strip()
        if c and c not in self.classes:
            self.classes.append(c)
            self.new_label.set("")
            self.rebuild_buttons()
            self.persist()

    # ------------------------------------------------------------- actions
    def answer(self, label: str) -> None:
        if self.session:
            self.session.set_label(label)
            self.render()

    def undo(self) -> None:
        if self.session:
            self.session.undo()
            self.render()

    def skip(self) -> None:
        if self.session:
            self.session.skip()
            self.render()

    def unscorable(self) -> None:
        if self.session:
            self.session.unscorable()
            self.render()

    def _on_key(self, e) -> None:
        if self.session is None or not self.winfo_ismapped():
            return
        try:
            if isinstance(self.focus_get(), (tk.Entry, ttk.Entry, ttk.Combobox)):
                return
        except KeyError:
            return
        if e.keysym == "BackSpace":
            self.undo()
        elif e.keysym == "space":
            self.skip()
        elif e.keysym == "Delete":
            self.unscorable()
        elif e.char and e.char.lower() in HOTKEYS:
            i = HOTKEYS.index(e.char.lower())
            if i < len(self.classes):
                self.answer(self.classes[i])

    # ------------------------------------------------------------- drawing
    @staticmethod
    def _ticks(im: Image.Image) -> None:
        d = ImageDraw.Draw(im)
        c, gap, ln = im.width // 2, 9, 16
        for width, col in ((3, "white"), (1, "red")):
            for a, b in (((c - gap - ln, c), (c - gap, c)), ((c + gap, c), (c + gap + ln, c)),
                         ((c, c - gap - ln), (c, c - gap)), ((c, c + gap), (c, c + gap + ln))):
                d.line([a, b], fill=col, width=width)

    def _view(self, canvas: tk.Canvas, cur: dict, cm: float) -> None:
        size = max(8, int(round(cm * 10.0 / self.mm_per_px)))
        arr = lt.crop_around(self.reader, cur["x_px"], cur["y_px"], size)
        if self.bright.get() and arr.any():
            lo, hi = np.percentile(arr[arr.any(axis=2)], (1, 99))
            if hi > lo:
                arr = np.clip((arr.astype(float) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        im = Image.fromarray(arr).resize((DISPLAY_PX, DISPLAY_PX), Image.BICUBIC)
        self._ticks(im)
        photo = ImageTk.PhotoImage(im)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        self._photos.append(photo)                              # keep a reference or Tk drops the image

    def render(self) -> None:
        if self.session is None:
            return
        done, total = self.session.progress()
        self.progress.configure(maximum=max(total, 1), value=done)
        cur = self.session.current()
        if cur is None:
            self.point_label.configure(text=f"All {total} points labelled. File: {self.session.out_csv}")
            self.ctx.delete("all")
            self.close.delete("all")
            return
        self.point_label.configure(text=f"{cur['id']}   ({done}/{total} done)")
        self._photos.clear()
        self._view(self.ctx, cur, CONTEXT_CM)
        self._view(self.close, cur, CLOSE_CM)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Benthic Cover Estimation")
        root.geometry("1280x800")
        self.settings = load_settings()
        self.shared: dict = {}
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)
        self.run_tab = RunTab(nb, self.shared, self.settings)
        self.label_tab = LabelTab(nb, self.shared, self.settings)
        nb.add(self.run_tab, text="Run steps")
        nb.add(self.label_tab, text="Label reference points")
        nb.bind("<<NotebookTabChanged>>", lambda e: self.label_tab.prefill() if nb.index("current") == 1 else None)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def close(self) -> None:
        save_settings({**self.settings, "values": self.run_tab.snapshot()})
        self.root.destroy()


def selftest() -> int:
    """Build the whole window, render one labelling view from synthetic data, close it."""
    import csv
    import tempfile

    import tiling

    with tempfile.TemporaryDirectory() as d:
        pts = os.path.join(d, "points.csv")
        with open(pts, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["id", "x_px", "y_px"])
            wr.writerow(["P1", 120, 90])
            wr.writerow(["P2", 10, 10])
        root = tk.Tk()
        app = App(root)
        arr = (np.random.default_rng(0).random((300, 400, 3)) * 255).astype(np.uint8)
        session = lt.LabelSession(pts, os.path.join(d, "out.csv"))
        app.label_tab.classes = ["sand", "rock"]
        app.label_tab.load_session(session, tiling.ArrayReader(arr), 1.0)
        app.label_tab.answer("sand")
        root.update()
        ok = session.progress() == (1, 2) and len(app.label_tab._photos) == 2
        app.run_tab.show_selected()
        root.update()
        root.destroy()
    print("GUI selftest OK" if ok else "GUI selftest FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
