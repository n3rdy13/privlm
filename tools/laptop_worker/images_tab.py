"""Images tab: Stable Diffusion image models (.safetensors, .ckpt, .gguf), the
kind PrivateLM's image generation uses, run on this computer with
stable-diffusion.cpp's sd-server (the same stable-diffusion.cpp the app
pins). Image models run on one device: the phone cluster can't split them.
"""

import base64
import json
import os
import queue
import random
import subprocess
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import privatelm_worker as core

PORT = 7861
SIZES = ["512 × 512", "512 × 768", "768 × 512", "384 × 384", "768 × 768"]
SAMPLERS = ["Euler a", "Euler", "DPM++ 2M", "DDIM", "LCM", "Heun"]
DEVICE_AUTO = "Automatic (GPU if it has one)"
DEVICE_CPU = "CPU only"
PREVIEW = 384  # px


def pictures_dir() -> Path:
    base = Path.home() / "Pictures"
    path = (base if base.exists() else Path.home()) / "PrivateLM"
    path.mkdir(parents=True, exist_ok=True)
    return path


class ImagesTab:
    POLL_MS = 1000

    def __init__(self, root: tk.Tk, container, logs: queue.Queue,
                 find_sd_server: Callable[[str, Callable[[str], None]], Path],
                 has_gpu: Callable[[], bool], keep_awake: Callable[[bool], None], no_window: int):
        self.root = root
        self.container = container
        self.logs = logs
        self.find_sd_server = find_sd_server
        self.has_gpu = has_gpu
        self.keep_awake = keep_awake
        self.no_window = no_window
        self.server = None
        self.ready = False
        self.generating = False
        self.photo = None
        self.last_png: Optional[bytes] = None
        self.state_path = core.config_dir() / "images.json"
        self.build_ui()
        self.load_state()

    # State ------------------------------------------------------------------

    def load_state(self):
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return
        if state.get("model") and Path(state["model"]).exists():
            self.model_var.set(state["model"])
            self.describe_model()
        for key, var in (("size", self.size_var), ("steps", self.steps_var), ("cfg", self.cfg_var),
                         ("sampler", self.sampler_var), ("negative", self.neg_var), ("device", self.device_var)):
            if key in state:
                var.set(state[key])
        if state.get("prompt"):
            self.prompt.insert("1.0", state["prompt"])

    def save_state(self):
        state = {"model": self.model_var.get(), "size": self.size_var.get(), "steps": self.steps_var.get(),
                 "cfg": self.cfg_var.get(), "sampler": self.sampler_var.get(), "negative": self.neg_var.get(),
                 "device": self.device_var.get(), "prompt": self.prompt.get("1.0", "end").strip()}
        try:
            self.state_path.write_text(json.dumps(state, indent=1))
        except OSError:
            pass

    # UI ---------------------------------------------------------------------

    def build_ui(self):
        outer = ttk.Frame(self.container, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)

        ttk.Label(outer, text="Make images with a Stable Diffusion model (.safetensors), like the "
                              "app's image generation.", style="Muted.TLabel", wraplength=460,
                  justify="left").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        status = ttk.Frame(outer)
        status.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        self.dot = tk.Canvas(status, width=14, height=14, highlightthickness=0, bd=0)
        self.dot.pack(side="left", padx=(0, 8))
        self.status_var = tk.StringVar(value="No image model loaded")
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").pack(side="left")
        self.set_dot("#9e9e9e")

        ttk.Label(outer, text="Model").grid(row=2, column=0, sticky="w", pady=4)
        self.model_var = tk.StringVar()
        ttk.Entry(outer, textvariable=self.model_var, state="readonly").grid(row=2, column=1, sticky="ew", pady=4)
        self.browse_btn = ttk.Button(outer, text="Browse…", command=self.browse)
        self.browse_btn.grid(row=2, column=2, sticky="e", padx=(6, 0), pady=4)
        self.model_desc = tk.StringVar(value="Choose a Stable Diffusion model: .safetensors, .ckpt or .gguf")
        ttk.Label(outer, textvariable=self.model_desc, style="Muted.TLabel")\
            .grid(row=3, column=1, columnspan=2, sticky="w")

        ttk.Label(outer, text="Use").grid(row=4, column=0, sticky="w", pady=4)
        self.device_var = tk.StringVar(value=DEVICE_AUTO)
        self.device_box = ttk.Combobox(outer, textvariable=self.device_var, state="readonly",
                                       values=[DEVICE_AUTO, DEVICE_CPU])
        self.device_box.grid(row=4, column=1, columnspan=2, sticky="ew", pady=4)

        self.load_btn = ttk.Button(outer, text="Load model", command=self.toggle)
        self.load_btn.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 8), ipady=3)

        # Generate
        gen = ttk.LabelFrame(outer, text="Image", padding=10)
        gen.grid(row=6, column=0, columnspan=3, sticky="nsew")
        gen.columnconfigure(1, weight=1)
        ttk.Label(gen, text="Prompt").grid(row=0, column=0, sticky="nw", pady=2)
        self.prompt = tk.Text(gen, height=3, width=40, wrap="word",
                              font=("Segoe UI", 10) if os.name == "nt" else ("", 10))
        self.prompt.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)
        ttk.Label(gen, text="Avoid").grid(row=1, column=0, sticky="w", pady=2)
        self.neg_var = tk.StringVar(value="blurry, low quality, deformed")
        ttk.Entry(gen, textvariable=self.neg_var).grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)

        opts = ttk.Frame(gen)
        opts.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 2))
        self.size_var = tk.StringVar(value=SIZES[0])
        self.steps_var = tk.StringVar(value="20")
        self.cfg_var = tk.StringVar(value="7")
        self.sampler_var = tk.StringVar(value=SAMPLERS[0])
        self.seed_var = tk.StringVar(value="-1")
        for label, widget in (
                ("Size", ttk.Combobox(opts, textvariable=self.size_var, values=SIZES, width=10, state="readonly")),
                ("Steps", ttk.Spinbox(opts, textvariable=self.steps_var, from_=1, to=80, width=4)),
                ("CFG", ttk.Spinbox(opts, textvariable=self.cfg_var, from_=1, to=20, increment=0.5, width=4)),
                ("Sampler", ttk.Combobox(opts, textvariable=self.sampler_var, values=SAMPLERS, width=9,
                                         state="readonly")),
                ("Seed", ttk.Entry(opts, textvariable=self.seed_var, width=11))):
            ttk.Label(opts, text=label).pack(side="left", padx=(0, 4))
            widget.pack(side="left", padx=(0, 10))

        buttons = ttk.Frame(gen)
        buttons.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        self.gen_btn = ttk.Button(buttons, text="Generate", command=self.generate)
        self.gen_btn.pack(side="left")
        self.gen_btn.state(["disabled"])
        self.save_btn = ttk.Button(buttons, text="Save as…", command=self.save_as)
        self.save_btn.pack(side="left", padx=(8, 0))
        self.save_btn.state(["disabled"])
        ttk.Button(buttons, text="Open folder", command=self.open_folder).pack(side="left", padx=(8, 0))
        self.gen_status = tk.StringVar(value="")
        ttk.Label(gen, textvariable=self.gen_status, style="Muted.TLabel", wraplength=460, justify="left")\
            .grid(row=4, column=0, columnspan=4, sticky="w")

        self.canvas = tk.Canvas(gen, width=PREVIEW, height=PREVIEW, bg="#eeeeee", highlightthickness=0)
        self.canvas.grid(row=5, column=0, columnspan=4, pady=(8, 0))
        self.canvas.create_text(PREVIEW // 2, PREVIEW // 2, text="Your image appears here", fill="#888888")

    def set_dot(self, color: str):
        self.dot.delete("all")
        self.dot.create_oval(2, 2, 12, 12, fill=color, outline=color)

    # Model ------------------------------------------------------------------

    def browse(self):
        path = filedialog.askopenfilename(
            title="Choose an image model",
            filetypes=[("Stable Diffusion models", "*.safetensors *.ckpt *.gguf"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)
            self.describe_model()
            self.save_state()

    def describe_model(self):
        path = Path(self.model_var.get())
        try:
            size = path.stat().st_size / (1024 ** 3)
        except OSError:
            return
        self.model_desc.set(f"{path.name} · {size:.1f} GB")

    def toggle(self):
        if self.server:
            self.unload()
        else:
            self.load()

    def load(self):
        model = self.model_var.get()
        if not model:
            messagebox.showinfo("PrivateLM", "Choose an image model first.")
            return
        self.save_state()
        self.load_btn.state(["disabled"])
        for w in (self.browse_btn, self.device_box):
            w.state(["disabled"])
        self.set_dot("#f9a825")
        self.status_var.set("Starting…")
        threading.Thread(target=self.start_server, args=(model, self.device_var.get()), daemon=True).start()

    def start_server(self, model: str, device: str):
        def status(msg):
            self.root.after(0, lambda: self.status_var.set(msg))
        try:
            variant = "cpu"
            if device != DEVICE_CPU:
                status("Checking for a GPU…")
                if self.has_gpu():
                    variant = "vulkan"
            binary = self.find_sd_server(variant, status)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            return self.root.after(0, lambda: self.on_failed(msg))
        threads = max(2, (os.cpu_count() or 4) // 2)
        args = [str(binary), "-m", model, "--listen-ip", "127.0.0.1", "--listen-port", str(PORT),
                "-t", str(threads)]
        self.logs.put("Starting: " + " ".join(args))
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, creationflags=self.no_window, encoding="utf-8",
                                    errors="replace")
        except OSError as e:
            msg = f"sd-server couldn't start:\n\n{e}"
            return self.root.after(0, lambda: self.on_failed(msg))
        self.root.after(0, lambda: self.on_started(proc, variant))

    def on_started(self, proc, variant: str):
        self.server = proc
        self.ready = False
        self.variant = variant
        self.tail = []
        threading.Thread(target=self.read_output, args=(proc,), daemon=True).start()
        self.load_btn.configure(text="Unload model")
        self.load_btn.state(["!disabled"])
        self.status_var.set("Loading the model…")
        self.root.after(self.POLL_MS, self.poll)

    def read_output(self, proc):
        for line in proc.stdout:
            line = line.rstrip()
            self.tail = (self.tail + [line])[-8:]
            self.logs.put(line)
            if "sampling" in line.lower() or "step" in line.lower():
                self.root.after(0, lambda l=line: self.progress(l))

    def progress(self, line: str):
        if self.generating:
            self.gen_status.set(line.split("] ")[-1][:120])

    def poll(self):
        proc = self.server
        if not proc:
            return
        code = proc.poll()
        if code is not None:
            detail = "\n".join(l for l in self.tail if l.strip())[-900:]
            self.unload(error=f"The image model stopped (code {code}).\n\n{detail}")
            return
        if not self.ready:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=2) as r:
                    if r.status == 200:
                        self.ready = True
                        self.set_dot("#2e7d32")
                        where = "GPU" if self.variant == "vulkan" else "CPU"
                        self.status_var.set(f"Ready on the {where}")
                        self.gen_btn.state(["!disabled"])
            except (urllib.error.URLError, OSError):
                pass
        self.root.after(self.POLL_MS, self.poll)

    def on_failed(self, message: str):
        self.load_btn.state(["!disabled"])
        for w in (self.browse_btn, self.device_box):
            w.state(["!disabled"])
        self.set_dot("#c62828")
        self.status_var.set("Couldn't load the model")
        self.logs.put(message)
        messagebox.showerror("PrivateLM", message)

    def unload(self, error: Optional[str] = None):
        proc, self.server = self.server, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.ready = False
        if self.generating:
            self.generating = False
            self.keep_awake(False)
        self.load_btn.configure(text="Load model")
        self.load_btn.state(["!disabled"])
        self.gen_btn.state(["disabled"])
        for w in (self.browse_btn, self.device_box):
            w.state(["!disabled"])
        if error:
            self.set_dot("#c62828")
            self.status_var.set("The image model stopped")
            self.logs.put(error)
            messagebox.showerror("PrivateLM", error)
        else:
            self.set_dot("#9e9e9e")
            self.status_var.set("No image model loaded")

    def shutdown(self):
        if self.server:
            self.unload()

    # Generating -------------------------------------------------------------

    def generate(self):
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            messagebox.showinfo("PrivateLM", "Write a prompt first.")
            return
        try:
            w, h = (int(x) for x in self.size_var.get().split("×"))
            steps = int(self.steps_var.get())
            cfg = float(self.cfg_var.get())
            seed = int(self.seed_var.get() or -1)
        except ValueError:
            messagebox.showerror("PrivateLM", "Check the steps, CFG and seed.")
            return
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        self.save_state()
        body = {"prompt": prompt, "negative_prompt": self.neg_var.get(), "width": w, "height": h,
                "steps": steps, "cfg_scale": cfg, "seed": seed, "sampler_name": self.sampler_var.get(),
                "batch_size": 1}
        self.generating = True
        self.keep_awake(True)
        self.gen_btn.state(["disabled"])
        self.started = time.time()
        self.gen_status.set(f"Generating {w}×{h}, {steps} steps, seed {seed}…")
        self.tick()
        threading.Thread(target=self.request, args=(body,), daemon=True).start()

    def tick(self):
        if self.generating:
            self.status_var.set(f"Generating… {int(time.time() - self.started)} s")
            self.root.after(1000, self.tick)

    def request(self, body: dict):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/sdapi/v1/txt2img",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                data = json.load(r)
            png = base64.b64decode(data["images"][0])
            self.root.after(0, lambda: self.on_image(png, body))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if isinstance(e, urllib.error.HTTPError):
                try:
                    msg = e.read().decode("utf-8", "replace")[:500]
                except Exception:  # noqa: BLE001
                    pass
            self.root.after(0, lambda: self.on_gen_failed(msg))

    def on_image(self, png: bytes, body: dict):
        self.generating = False
        self.keep_awake(False)
        seconds = int(time.time() - self.started)
        self.last_png = png
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{body['seed']}.png"
        path = pictures_dir() / name
        try:
            path.write_bytes(png)
            saved = f"Saved to {path}"
        except OSError as e:
            saved = f"Couldn't save: {e}"
        self.show(png)
        self.save_btn.state(["!disabled"])
        if self.ready:
            self.gen_btn.state(["!disabled"])
            self.status_var.set(f"Ready · last image took {seconds} s")
        self.gen_status.set(f"Seed {body['seed']} · {saved}")

    def on_gen_failed(self, message: str):
        self.generating = False
        self.keep_awake(False)
        if self.ready:
            self.gen_btn.state(["!disabled"])
            self.status_var.set("Ready")
        self.gen_status.set("Generation failed: see the log")
        self.logs.put(f"Generation failed: {message}")
        if self.server:
            messagebox.showerror("PrivateLM", f"The image couldn't be made:\n\n{message}")

    def show(self, png: bytes):
        photo = tk.PhotoImage(data=base64.b64encode(png).decode())
        factor = max(1, -(-max(photo.width(), photo.height()) // PREVIEW))  # ceil
        if factor > 1:
            photo = photo.subsample(factor, factor)
        self.photo = photo  # keep a reference
        self.canvas.delete("all")
        self.canvas.configure(width=max(PREVIEW, photo.width()), height=max(PREVIEW, photo.height()))
        self.canvas.create_image(int(self.canvas["width"]) // 2, int(self.canvas["height"]) // 2,
                                 image=photo)

    def save_as(self):
        if not self.last_png:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")],
                                            initialdir=str(pictures_dir()))
        if path:
            Path(path).write_bytes(self.last_png)

    def open_folder(self):
        folder = str(pictures_dir())
        if os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif os.uname().sysname == "Darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    @property
    def running(self) -> bool:
        return self.server is not None
