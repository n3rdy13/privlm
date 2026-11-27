"""Images tab: Stable Diffusion image models (.safetensors, .ckpt, .gguf), the
kind PrivateLM's image generation uses.

This computer runs them with stable-diffusion.cpp's sd-server (the same
stable-diffusion.cpp the app pins), and phones with "Images for a laptop"
turned on make images too: each device takes the next job as soon as it's
free, so several images are made at once. Two ways to fill the queue:
variations (one prompt, many seeds) and a prompt queue (one prompt per line).

Phones and sd-server speak the same A1111-style POST /sdapi/v1/txt2img.
"""

import base64
import json
import os
import queue
import random
import socket
import subprocess
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional

import privatelm_worker as core
import prompt_helper as ph

PORT = 7861                        # this computer's sd-server (localhost only)
PHONE_PORT = 50061                 # ImageWorkerService in the phone app
SERVICE_TYPE = "_privatelm-img._tcp.local."
PROTOCOL = "img1"
SIZES = ["512 × 512", "512 × 768", "768 × 512", "384 × 384", "768 × 768"]
SAMPLERS = ["Euler a", "Euler", "DPM++ 2M", "DDIM", "LCM", "Heun"]
DEVICE_AUTO = "Automatic (GPU if it has one)"
DEVICE_CPU = "CPU only"
MODE_VARIATIONS = "variations"
MODE_QUEUE = "queue"
PREVIEW = 360   # px, the big preview's longest side
THUMB = 76      # px, gallery thumbnails
THUMB_COLS = 4
LOCAL_ID = "local"


def pictures_dir() -> Path:
    base = Path.home() / "Pictures"
    path = (base if base.exists() else Path.home()) / "PrivateLM"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fit_photo(photo: tk.PhotoImage, box: int) -> tk.PhotoImage:
    """Scales a PhotoImage to fit `box` (Tk scales by whole-number ratios)."""
    longest = max(photo.width(), photo.height())
    if longest <= box:
        return photo
    best = (1, -(-longest // box))
    for zoom in (2, 3):
        shrink = -(-longest * zoom // box)
        if zoom / shrink > best[0] / best[1]:
            best = (zoom, shrink)
    zoom, shrink = best
    if zoom > 1:
        photo = photo.zoom(zoom, zoom)
    return photo.subsample(shrink, shrink)


def phone_status(host: str, port: int, timeout: float = 3.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/status", timeout=timeout) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def find_phones(known_hosts=(), log=None) -> List[dict]:
    """Phones with "Images for a laptop" on: a Wi-Fi search, plus each known
    address asked directly, then each one's live status."""
    import discovery

    found: Dict[tuple, dict] = {}
    try:
        for f in discovery.find_all(SERVICE_TYPE, PHONE_PORT, known_hosts, log=log):
            found[(f["host"], f["port"])] = f
    except ImportError:
        if log:
            log("Find nearby needs the zeroconf package; scanning the network instead")
        for host in set(known_hosts) | set(discovery.scan_subnet(PHONE_PORT, log=log)):
            found.setdefault((host, PHONE_PORT), {"host": host, "port": PHONE_PORT, "props": {},
                                                  "instance": host})
    phones = []
    for (host, port), f in found.items():
        status = phone_status(host, port)
        props = f.get("props", {})
        phones.append({
            "id": props.get("id") or f"{host}:{port}", "host": host, "port": port,
            "name": (status or {}).get("name") or props.get("model") or host,
            "model": (status or {}).get("model") or props.get("img") or "",
            "online": status is not None, "loaded": bool((status or {}).get("loaded")),
            "img2img": bool((status or {}).get("img2img")),
        })
    return phones


@dataclass
class Job:
    index: int
    prompt: str
    seed: int
    negative: str = ""


@dataclass
class Device:
    id: str
    name: str
    url: str           # base URL, e.g. http://192.168.1.5:50061
    model: str = ""
    use: bool = True
    img2img: bool = True
    status: str = ""
    made: int = 0
    host: str = ""
    port: int = 0


@dataclass
class Result:
    png: bytes
    job: Job
    device: str
    seconds: float
    width: int
    height: int
    seed: int
    path: Optional[Path] = None
    photo: Optional[tk.PhotoImage] = None
    thumb: Optional[tk.PhotoImage] = None


class ImagesTab:
    POLL_MS = 1000

    def __init__(self, root: tk.Tk, container, logs: queue.Queue,
                 find_sd_server: Callable[[str, Callable[[str], None]], Path],
                 has_gpu: Callable[[], bool], keep_awake: Callable[[bool], None], no_window: int,
                 find_server=None, master_endpoint: Callable[[], Optional[tuple]] = lambda: None):
        self.root = root
        self.container = container
        self.logs = logs
        self.find_sd_server = find_sd_server
        self.has_gpu = has_gpu
        self.keep_awake = keep_awake
        self.no_window = no_window
        self.server = None
        self.ready = False
        self.variant = "cpu"
        self.tail: List[str] = []
        self.phones: List[dict] = []          # saved phones (id, name, host, port, use)
        self.local_use = True
        self.results: List[Result] = []
        self.selected: Optional[Result] = None
        self.run: Optional[dict] = None       # the running batch
        self.state_path = core.config_dir() / "images.json"
        self.ai = ph.PromptAI(logs, find_server, master_endpoint, no_window)
        self.helper: Optional[ph.HelperWindow] = None
        self.build_ui()
        self.load_state()
        self.refresh_devices()

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
                         ("sampler", self.sampler_var), ("negative", self.neg_var), ("device", self.device_var),
                         ("mode", self.mode_var), ("count", self.count_var), ("seed", self.seed_var),
                         ("preset", self.preset_var)):
            if key in state:
                var.set(state[key])
        self.phones = state.get("phones", [])
        self.local_use = state.get("local_use", True)
        if state.get("prompt"):
            self.prompt.insert("1.0", state["prompt"])
        self.on_mode()

    def save_state(self):
        state = {"model": self.model_var.get(), "size": self.size_var.get(), "steps": self.steps_var.get(),
                 "cfg": self.cfg_var.get(), "sampler": self.sampler_var.get(), "negative": self.neg_var.get(),
                 "device": self.device_var.get(), "mode": self.mode_var.get(), "count": self.count_var.get(),
                 "seed": self.seed_var.get(), "preset": self.preset_var.get(),
                 "prompt": self.prompt.get("1.0", "end").strip(),
                 "phones": self.phones, "local_use": self.local_use}
        try:
            self.state_path.write_text(json.dumps(state, indent=1))
        except OSError:
            pass

    # UI ---------------------------------------------------------------------

    def build_ui(self):
        outer = ttk.Frame(self.container, padding=14)
        outer.pack(fill="both", expand=True)
        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(1, weight=1)
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="n", padx=(14, 0))

        status = ttk.Frame(left)
        status.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.dot = tk.Canvas(status, width=14, height=14, highlightthickness=0, bd=0)
        self.dot.pack(side="left", padx=(0, 8))
        self.status_var = tk.StringVar(value="No image model loaded here")
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").pack(side="left")
        self.set_dot("#9e9e9e")

        # This computer's model
        ttk.Label(left, text="Model").grid(row=1, column=0, sticky="w", pady=3)
        self.model_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.model_var, state="readonly").grid(row=1, column=1, sticky="ew", pady=3)
        self.browse_btn = ttk.Button(left, text="Browse…", command=self.browse)
        self.browse_btn.grid(row=1, column=2, sticky="e", padx=(6, 0), pady=3)
        self.model_desc = tk.StringVar(value="A Stable Diffusion model: .safetensors, .ckpt or .gguf")
        ttk.Label(left, textvariable=self.model_desc, style="Muted.TLabel")\
            .grid(row=2, column=1, columnspan=2, sticky="w")
        row = ttk.Frame(left)
        row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 6))
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text="Use").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.device_var = tk.StringVar(value=DEVICE_AUTO)
        self.device_box = ttk.Combobox(row, textvariable=self.device_var, state="readonly",
                                       values=[DEVICE_AUTO, DEVICE_CPU], width=26)
        self.device_box.grid(row=0, column=1, sticky="w")
        self.load_btn = ttk.Button(row, text="Load model", command=self.toggle)
        self.load_btn.grid(row=0, column=2, sticky="e")

        # Devices
        df = ttk.LabelFrame(left, text="Devices making images", padding=8)
        df.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 6))
        df.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(df, columns=("use", "name", "model", "status"), show="headings",
                                 height=4, selectmode="browse")
        for col, text, width, stretch in (("use", "Use", 38, False), ("name", "Device", 110, False),
                                          ("model", "Model", 120, True), ("status", "Status", 120, True)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="center" if col == "use" else "w", stretch=stretch)
        self.tree.grid(row=0, column=0, columnspan=4, sticky="ew")
        self.tree.bind("<ButtonRelease-1>", self.on_tree_click)
        self.find_btn = ttk.Button(df, text="Find nearby", command=self.find_nearby)
        self.find_btn.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ip_var = tk.StringVar()
        self.ip_entry = ttk.Entry(df, textvariable=self.ip_var, width=16)
        self.ip_entry.grid(row=1, column=1, sticky="e", padx=(6, 0), pady=(6, 0))
        self.ip_entry.bind("<Return>", lambda _: self.add_by_ip())
        self.add_btn = ttk.Button(df, text="Add by IP", command=self.add_by_ip)
        self.add_btn.grid(row=1, column=2, sticky="e", padx=(6, 0), pady=(6, 0))
        self.remove_btn = ttk.Button(df, text="Remove", command=self.remove_selected)
        self.remove_btn.grid(row=1, column=3, sticky="e", padx=(6, 0), pady=(6, 0))
        self.devices_hint = tk.StringVar(
            value="On each phone: Phone cluster → Images for a laptop, with an image model loaded.")
        ttk.Label(df, textvariable=self.devices_hint, style="Muted.TLabel", wraplength=440,
                  justify="left").grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # What to make
        gen = ttk.LabelFrame(left, text="What to make", padding=8)
        gen.grid(row=5, column=0, columnspan=3, sticky="ew")
        gen.columnconfigure(1, weight=1)
        tools = ttk.Frame(gen)
        tools.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        ttk.Button(tools, text="✨ Prompt helper…", command=self.open_helper).pack(side="left")
        self.preset_var = tk.StringVar(value="")
        self.preset_box = ttk.Combobox(tools, textvariable=self.preset_var, values=list(ph.PRESETS),
                                       state="readonly", width=34)
        self.preset_box.pack(side="right")
        self.preset_box.bind("<<ComboboxSelected>>", lambda _: self.apply_preset())
        ttk.Label(tools, text="Settings").pack(side="right", padx=(0, 6))
        modes = ttk.Frame(gen)
        modes.grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 4))
        self.mode_var = tk.StringVar(value=MODE_VARIATIONS)
        ttk.Radiobutton(modes, text="Variations of one prompt", value=MODE_VARIATIONS,
                        variable=self.mode_var, command=self.on_mode).pack(side="left", padx=(0, 14))
        ttk.Radiobutton(modes, text="Queue: one prompt per line", value=MODE_QUEUE,
                        variable=self.mode_var, command=self.on_mode).pack(side="left")
        self.prompt_label = tk.StringVar(value="Prompt")
        ttk.Label(gen, textvariable=self.prompt_label).grid(row=2, column=0, sticky="nw", pady=2)
        self.prompt = tk.Text(gen, height=3, width=40, wrap="word",
                              font=("Segoe UI", 10) if os.name == "nt" else ("", 10))
        self.prompt.grid(row=2, column=1, columnspan=3, sticky="ew", pady=2)
        ttk.Label(gen, text="Avoid").grid(row=3, column=0, sticky="w", pady=2)
        self.neg_var = tk.StringVar(value="blurry, low quality, deformed")
        ttk.Entry(gen, textvariable=self.neg_var).grid(row=3, column=1, columnspan=3, sticky="ew", pady=2)

        # Image-to-image
        init = ttk.Frame(gen)
        init.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(4, 2))
        self.init_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(init, text="Start from an image", variable=self.init_var,
                        command=self.on_init_toggle).grid(row=0, column=0, sticky="w")
        self.init_thumb = tk.Label(init, width=4, height=2, bg="#e0e0e0", text="", relief="flat")
        self.init_thumb.grid(row=0, column=1, rowspan=2, padx=(8, 6))
        self.init_choose = ttk.Button(init, text="Choose…", command=self.choose_init)
        self.init_choose.grid(row=0, column=2, sticky="w")
        self.init_use = ttk.Button(init, text="Use selected", command=self.use_selected_init)
        self.init_use.grid(row=0, column=3, sticky="w", padx=(6, 0))
        strength = ttk.Frame(init)
        strength.grid(row=1, column=2, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(strength, text="Change: a little").pack(side="left")
        self.strength_var = tk.DoubleVar(value=0.55)
        self.strength_scale = ttk.Scale(strength, from_=0.1, to=1.0, variable=self.strength_var, length=120,
                                        command=lambda _: self.strength_label.set(f"{self.strength_var.get():.2f}"))
        self.strength_scale.pack(side="left", padx=4)
        ttk.Label(strength, text="a lot").pack(side="left")
        self.strength_label = tk.StringVar(value="0.55")
        self.strength_var.trace_add("write", lambda *_: self.strength_label.set(f"{self.strength_var.get():.2f}"))
        ttk.Label(strength, textvariable=self.strength_label, width=5, style="Muted.TLabel").pack(side="left", padx=(6, 0))
        self.init_image = None      # PIL image
        self.init_photo = None
        self.on_init_toggle()

        self.size_var = tk.StringVar(value=SIZES[0])
        self.steps_var = tk.StringVar(value="20")
        self.cfg_var = tk.StringVar(value="7")
        self.sampler_var = tk.StringVar(value=SAMPLERS[0])
        self.seed_var = tk.StringVar(value="-1")
        self.count_var = tk.StringVar(value="4")
        self.count_label = tk.StringVar(value="Images")
        grid = (
            (("Size", ttk.Combobox(gen, textvariable=self.size_var, values=SIZES, width=10, state="readonly")),
             ("Steps", ttk.Spinbox(gen, textvariable=self.steps_var, from_=1, to=80, width=5))),
            (("Sampler", ttk.Combobox(gen, textvariable=self.sampler_var, values=SAMPLERS, width=10,
                                      state="readonly")),
             ("CFG", ttk.Spinbox(gen, textvariable=self.cfg_var, from_=1, to=20, increment=0.5, width=5))),
            (("Seed", ttk.Entry(gen, textvariable=self.seed_var, width=12)),
             (self.count_label, ttk.Spinbox(gen, textvariable=self.count_var, from_=1, to=100, width=5))),
        )
        for r, pair in enumerate(grid, 5):
            for c, (label, widget) in enumerate(pair):
                if isinstance(label, tk.StringVar):
                    ttk.Label(gen, textvariable=label).grid(row=r, column=c * 2, sticky="w", pady=2, padx=(0, 6))
                else:
                    ttk.Label(gen, text=label).grid(row=r, column=c * 2, sticky="w", pady=2, padx=(0, 6))
                widget.grid(row=r, column=c * 2 + 1, sticky="w", pady=2, padx=(0, 12))
        self.seed_hint = tk.StringVar(value="−1: a random seed for each image")
        ttk.Label(gen, textvariable=self.seed_hint, style="Muted.TLabel")\
            .grid(row=8, column=0, columnspan=4, sticky="w")

        buttons = ttk.Frame(gen)
        buttons.grid(row=9, column=0, columnspan=4, sticky="ew", pady=(8, 2))
        self.gen_btn = ttk.Button(buttons, text="Generate", command=self.generate)
        self.gen_btn.pack(side="left")
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop_run)
        self.stop_btn.pack(side="left", padx=(8, 0))
        self.stop_btn.state(["disabled"])
        ttk.Button(buttons, text="Open folder", command=self.open_folder).pack(side="left", padx=(8, 0))
        self.gen_status = tk.StringVar(value="")
        ttk.Label(gen, textvariable=self.gen_status, style="Muted.TLabel", wraplength=440, justify="left")\
            .grid(row=10, column=0, columnspan=4, sticky="w")

        # Preview + gallery
        preview = ttk.LabelFrame(right, text="Preview", padding=8)
        preview.pack(fill="x")
        self.canvas = tk.Canvas(preview, width=PREVIEW, height=PREVIEW, bg="#eeeeee", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_text(PREVIEW // 2, PREVIEW // 2, text="Your images appear here", fill="#888888")
        self.caption = tk.StringVar(value="")
        ttk.Label(preview, textvariable=self.caption, style="Muted.TLabel", wraplength=PREVIEW)\
            .pack(anchor="w", pady=(4, 0))
        self.save_btn = ttk.Button(preview, text="Save as…", command=self.save_as)
        self.save_btn.pack(anchor="w", pady=(4, 0))
        self.save_btn.state(["disabled"])

        gal = ttk.LabelFrame(right, text="Gallery", padding=6)
        gal.pack(fill="both", expand=True, pady=(8, 0))
        width = THUMB_COLS * (THUMB + 6)
        self.gallery = tk.Canvas(gal, width=width, height=2 * (THUMB + 6), highlightthickness=0)
        scroll = ttk.Scrollbar(gal, orient="vertical", command=self.gallery.yview)
        self.gallery.configure(yscrollcommand=scroll.set)
        self.gallery.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.gallery.bind("<Button-1>", self.on_gallery_click)

    def set_dot(self, color: str):
        self.dot.delete("all")
        self.dot.create_oval(2, 2, 12, 12, fill=color, outline=color)

    def on_mode(self):
        if self.mode_var.get() == MODE_QUEUE:
            self.prompt_label.set("Prompts")
            self.count_label.set("Each")
            self.seed_hint.set("−1: a random seed for each image. One prompt per line.")
        else:
            self.prompt_label.set("Prompt")
            self.count_label.set("Images")
            self.seed_hint.set("−1: a random seed for each image; a number N gives seeds N, N+1, …")

    # Image-to-image ----------------------------------------------------------

    def on_init_toggle(self):
        flag = ["!disabled"] if self.init_var.get() else ["disabled"]
        for w in (self.init_choose, self.init_use, self.strength_scale):
            w.state(flag)

    def _pil(self):
        try:
            from PIL import Image  # type: ignore
            return Image
        except ImportError:
            messagebox.showerror("PrivateLM", "Starting from an image needs Pillow: pip install pillow")
            return None

    def set_init_image(self, image, label: str):
        Image = self._pil()
        if not Image:
            return
        self.init_image = image.convert("RGB")
        thumb = self.init_image.copy()
        thumb.thumbnail((40, 40))
        import io
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        self.init_photo = tk.PhotoImage(data=base64.b64encode(buf.getvalue()).decode())
        self.init_thumb.configure(image=self.init_photo, width=40, height=40)
        self.init_var.set(True)
        self.on_init_toggle()
        self.gen_status.set(f"Starting image: {label}")

    def choose_init(self):
        Image = self._pil()
        if not Image:
            return
        path = filedialog.askopenfilename(title="Choose a starting image", filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            with Image.open(path) as im:
                self.set_init_image(im.copy(), Path(path).name)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("PrivateLM", f"Can't open this image:\n\n{e}")

    def use_selected_init(self):
        Image = self._pil()
        if not Image:
            return
        if not self.selected:
            messagebox.showinfo("PrivateLM", "Click an image in the gallery first.")
            return
        import io
        self.set_init_image(Image.open(io.BytesIO(self.selected.png)), f"gallery image, seed {self.selected.seed}")

    def init_png_b64(self, w: int, h: int) -> Optional[str]:
        """The starting image cropped to the output's shape and resized to it."""
        Image = self._pil()
        if not Image or self.init_image is None:
            return None
        from PIL import ImageOps  # type: ignore
        fitted = ImageOps.fit(self.init_image, (w, h), method=Image.LANCZOS)
        import io
        buf = io.BytesIO()
        fitted.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # Prompt help -------------------------------------------------------------

    def open_helper(self):
        if self.helper and self.helper.win.winfo_exists():
            self.helper.win.lift()
            return
        self.helper = ph.HelperWindow(self.root, self.ai, self.apply_prompts, self.selected_prompt,
                                      lambda: Path(self.model_var.get()).name)

    def apply_prompts(self, prompts: List[str], negative: str):
        """From the prompt helper: one prompt for Variations, several for the Queue."""
        self.mode_var.set(MODE_QUEUE if len(prompts) > 1 else MODE_VARIATIONS)
        if len(prompts) > 1:
            self.count_var.set("1")
        self.on_mode()
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", "\n".join(prompts))
        if negative:
            self.neg_var.set(negative)
        self.save_state()

    def selected_prompt(self) -> Optional[tuple]:
        if not self.selected:
            return None
        return self.selected.job.prompt, self.selected.job.negative

    def apply_preset(self, name: Optional[str] = None):
        name = name or self.preset_var.get()
        if name not in ph.PRESETS:
            return
        sampler, steps, cfg, size, _ = ph.PRESETS[name]
        self.preset_var.set(name)
        self.sampler_var.set(sampler)
        self.steps_var.set(str(steps))
        self.cfg_var.set(f"{cfg:g}")
        self.size_var.set(size)
        self.save_state()

    # Devices ----------------------------------------------------------------

    def devices(self) -> List[Device]:
        """This computer (if its model is loaded) and the saved phones."""
        out = []
        local_name = Path(self.model_var.get()).name if self.model_var.get() else ""
        out.append(Device(LOCAL_ID, "This computer", f"http://127.0.0.1:{PORT}", local_name,
                          self.local_use))
        for p in self.phones:
            out.append(Device(p["id"], p["name"], f"http://{p['host']}:{p['port']}", p.get("model", ""),
                              p.get("use", True), img2img=bool(p.get("img2img")),
                              host=p["host"], port=p["port"]))
        return out

    def refresh_devices(self, statuses: Optional[Dict[str, str]] = None):
        statuses = statuses or {}
        self.tree.delete(*self.tree.get_children())
        for d in self.devices():
            if d.id == LOCAL_ID:
                default = ("Ready" if self.ready else "Model not loaded")
            else:
                p = next((x for x in self.phones if x["id"] == d.id), {})
                default = ("Ready" if p.get("loaded") else "No model loaded") if p.get("online") \
                    else "Not reachable" if "online" in p else ""
            self.tree.insert("", "end", iid=d.id, values=("☑" if d.use else "☐", d.name, d.model,
                                                         statuses.get(d.id, default)))

    def on_tree_click(self, event):
        if self.run or self.tree.identify_column(event.x) != "#1":
            return
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        if iid == LOCAL_ID:
            self.local_use = not self.local_use
        else:
            for p in self.phones:
                if p["id"] == iid:
                    p["use"] = not p.get("use", True)
        self.refresh_devices()
        self.save_state()

    def remove_selected(self):
        sel = self.tree.selection()
        if sel and sel[0] != LOCAL_ID and not self.run:
            self.phones = [p for p in self.phones if p["id"] != sel[0]]
            self.refresh_devices()
            self.save_state()

    def merge_phone(self, phone: dict):
        for p in self.phones:
            if p["id"] == phone["id"] or (p["host"] == phone["host"] and p["port"] == phone["port"]):
                use = p.get("use", True)
                p.update(phone)
                p["use"] = use
                return
        self.phones.append({**phone, "use": True})

    def find_nearby(self):
        self.find_btn.state(["disabled"])
        self.devices_hint.set("Looking for phones with Images for a laptop on…")
        known = [p["host"] for p in self.phones]

        def work():
            try:
                found, error = find_phones(known, log=self.logs.put), None
            except Exception as e:  # noqa: BLE001
                found, error = [], f"Find nearby failed: {e}"
            self.root.after(0, lambda: self.on_found(found, error))

        threading.Thread(target=work, daemon=True).start()

    def on_found(self, found: List[dict], error: Optional[str]):
        self.find_btn.state(["!disabled"])
        for phone in found:
            self.merge_phone(phone)
        self.refresh_devices()
        self.save_state()
        online = [p for p in found if p["online"]]
        if error:
            self.devices_hint.set(error)
        elif online:
            self.devices_hint.set(f"Found {len(online)}: " + ", ".join(p["name"] for p in online)
                                  + ". Click Use to include or leave out a device.")
        else:
            self.devices_hint.set("No phones found. On each phone: Phone cluster → Images for a "
                                  "laptop, keep PrivateLM open, same Wi-Fi. Or add one by IP.")

    def add_by_ip(self):
        text = self.ip_var.get().strip()
        if not text:
            return
        host, _, port = text.partition(":")
        try:
            port_n = int(port) if port else PHONE_PORT
            socket.inet_aton(host)
        except (ValueError, OSError):
            messagebox.showerror("PrivateLM", "Use the form 192.168.1.20 or 192.168.1.20:50061")
            return
        self.ip_var.set("")
        self.devices_hint.set(f"Asking {host}…")

        def work():
            status = phone_status(host, port_n)
            phone = {"id": f"{host}:{port_n}", "host": host, "port": port_n,
                     "name": (status or {}).get("name") or host, "model": (status or {}).get("model", ""),
                     "online": status is not None, "loaded": bool((status or {}).get("loaded")),
                     "img2img": bool((status or {}).get("img2img"))}
            self.root.after(0, lambda: self.on_added(phone))

        threading.Thread(target=work, daemon=True).start()

    def on_added(self, phone: dict):
        self.merge_phone(phone)
        self.refresh_devices()
        self.save_state()
        if phone["online"]:
            self.devices_hint.set(f"Added {phone['name']}.")
        else:
            self.devices_hint.set(f"Added {phone['host']}, but it didn't answer. Turn on Images for a "
                                  "laptop on it and keep PrivateLM open.")

    # This computer's model --------------------------------------------------

    def browse(self):
        path = filedialog.askopenfilename(
            title="Choose an image model",
            filetypes=[("Stable Diffusion models", "*.safetensors *.ckpt *.gguf"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)
            self.describe_model()
            self.apply_preset(ph.preset_for(Path(path).name))
            self.refresh_devices()
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
                        self.status_var.set(f"Model loaded here, on the {where}")
                        self.refresh_devices()
            except (urllib.error.URLError, OSError):
                pass
        self.root.after(self.POLL_MS, self.poll)

    def on_failed(self, message: str):
        self.load_btn.state(["!disabled"])
        for w in (self.browse_btn, self.device_box):
            w.state(["!disabled"])
        self.set_dot("#c62828")
        self.status_var.set("Couldn't load the model here")
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
        self.load_btn.configure(text="Load model")
        self.load_btn.state(["!disabled"])
        for w in (self.browse_btn, self.device_box):
            w.state(["!disabled"])
        self.refresh_devices()
        if error:
            self.set_dot("#c62828")
            self.status_var.set("The image model stopped")
            self.logs.put(error)
            messagebox.showerror("PrivateLM", error)
        else:
            self.set_dot("#9e9e9e")
            self.status_var.set("No image model loaded here")

    def shutdown(self):
        self.stop_run()
        self.ai.shutdown()
        if self.server:
            self.unload()

    # Generating -------------------------------------------------------------

    def build_jobs(self) -> Optional[List[Job]]:
        text = self.prompt.get("1.0", "end").strip()
        try:
            count = max(1, int(self.count_var.get()))
            seed = int(self.seed_var.get() or -1)
        except ValueError:
            messagebox.showerror("PrivateLM", "Check the seed and the number of images.")
            return None
        if self.mode_var.get() == MODE_QUEUE:
            prompts = [line.strip() for line in text.splitlines() if line.strip()]
        else:
            prompts = [" ".join(text.split())] if text else []
        if not prompts:
            messagebox.showinfo("PrivateLM", "Write a prompt first.")
            return None
        jobs = []
        for prompt in prompts:
            for i in range(count):
                s = seed + i if seed >= 0 else random.randint(0, 2**31 - 1)
                jobs.append(Job(len(jobs), prompt, s, self.neg_var.get()))
        return jobs

    def generate(self):
        jobs = self.build_jobs()
        if not jobs:
            return
        try:
            w, h = (int(x) for x in self.size_var.get().split("×"))
            steps = int(self.steps_var.get())
            cfg = float(self.cfg_var.get())
        except ValueError:
            messagebox.showerror("PrivateLM", "Check the steps and CFG.")
            return
        devices = [d for d in self.devices() if d.use]
        if self.local_use and not self.ready:
            devices = [d for d in devices if d.id != LOCAL_ID]
        init_b64 = None
        skipped_devices = []
        if self.init_var.get():
            if self.init_image is None:
                messagebox.showinfo("PrivateLM", "Choose a starting image, or untick Start from an image.")
                return
            skipped_devices = [d for d in devices if not d.img2img]
            skipped = [d.name for d in skipped_devices]
            devices = [d for d in devices if d.img2img]
            if skipped:
                self.logs.put("Can't start from an image (update PrivateLM on it): " + ", ".join(skipped))
            init_b64 = self.init_png_b64(w, h)
            if not init_b64:
                return
        if not devices:
            messagebox.showinfo("PrivateLM", "Load the model here, or add a phone with Images for a "
                                             "laptop on (updated for starting from an image), and tick Use.")
            return
        self.save_state()
        params = {"negative_prompt": self.neg_var.get(), "width": w, "height": h, "steps": steps,
                  "cfg_scale": cfg, "sampler_name": self.sampler_var.get()}
        if init_b64:
            params["init_images"] = [init_b64]
            params["denoising_strength"] = round(float(self.strength_var.get()), 2)
        pending: "queue.Queue[Job]" = queue.Queue()
        for job in jobs:
            pending.put(job)
        run = {"pending": pending, "total": len(jobs), "done": 0, "failed": 0, "stop": False,
               "statuses": {**{d.id: "Can't start from an image: update the app" for d in skipped_devices},
                            **{d.id: "Starting…" for d in devices}},
               "active": len(devices), "inflight": 0,
               "started": time.time(), "lock": threading.Lock(), "devices": devices}
        self.run = run
        self.keep_awake(True)
        self.gen_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])
        for w_ in (self.find_btn, self.add_btn, self.remove_btn):
            w_.state(["disabled"])
        names = ", ".join(d.name for d in devices)
        self.gen_status.set(f"Making {len(jobs)} image{'s' * (len(jobs) != 1)} on {names}…")
        for d in devices:
            threading.Thread(target=self.device_loop, args=(run, d, params), daemon=True).start()
        self.root.after(500, self.tick)

    def device_loop(self, run: dict, device: Device, params: dict):
        """One device: takes the next job whenever it's free, until none are left."""
        def set_status(text):
            with run["lock"]:
                run["statuses"][device.id] = text

        failures = 0
        while not run["stop"]:
            try:
                job = run["pending"].get_nowait()
            except queue.Empty:
                with run["lock"]:
                    others_busy = run["inflight"] > 0
                if not others_busy:
                    break
                time.sleep(1)  # a job another device fails comes back here
                continue
            with run["lock"]:
                run["inflight"] += 1
            set_status(f"Making #{job.index + 1}")
            started = time.time()
            body = {**params, "prompt": job.prompt, "seed": job.seed, "batch_size": 1}
            endpoint = "img2img" if "init_images" in params else "txt2img"
            req = urllib.request.Request(f"{device.url}/sdapi/v1/{endpoint}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    data = json.load(r)
                png = base64.b64decode(data["images"][0])
                info = data.get("info")
                if isinstance(info, str):
                    try:
                        info = json.loads(info)
                    except ValueError:
                        info = {}
                info = info or {}
                result = Result(png, job, device.name, time.time() - started,
                                int(info.get("width") or params["width"]),
                                int(info.get("height") or params["height"]),
                                int(info.get("seed", job.seed)))
                failures = 0
                with run["lock"]:
                    run["done"] += 1
                    run["inflight"] -= 1
                device.made += 1
                set_status(f"{device.made} made")
                self.root.after(0, lambda res=result: self.add_result(res))
            except Exception as e:  # noqa: BLE001 - per-device failure
                msg = str(e)
                if isinstance(e, (socket.timeout, TimeoutError)):
                    msg = "Timed out"
                elif isinstance(e, urllib.error.URLError) and not isinstance(e, urllib.error.HTTPError):
                    msg = "Not reachable"
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        msg = json.loads(e.read().decode("utf-8", "replace")).get("error", msg)
                    except Exception:  # noqa: BLE001
                        pass
                self.logs.put(f"{device.name}: image #{job.index + 1} failed: {msg}")
                run["pending"].put(job)  # someone else takes it
                with run["lock"]:
                    run["inflight"] -= 1
                failures += 1
                if failures >= 2 or isinstance(e, (urllib.error.URLError, OSError)) and \
                        not isinstance(e, urllib.error.HTTPError):
                    set_status(f"Stopped: {msg[:60]}")
                    break
                time.sleep(2)
        with run["lock"]:
            run["active"] -= 1
            if run["statuses"].get(device.id, "").startswith("Making"):
                run["statuses"][device.id] = f"{device.made} made"

    def tick(self):
        run = self.run
        if not run:
            return
        with run["lock"]:
            statuses = dict(run["statuses"])
            done, total, active = run["done"], run["total"], run["active"]
        self.refresh_devices(statuses)
        elapsed = int(time.time() - run["started"])
        self.status_var.set(f"{done} of {total} made · {elapsed} s")
        if active == 0:
            self.finish_run()
            return
        self.root.after(1000, self.tick)

    def finish_run(self):
        run, self.run = self.run, None
        if not run:
            return
        self.keep_awake(False)
        self.gen_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])
        for w_ in (self.find_btn, self.add_btn, self.remove_btn):
            w_.state(["!disabled"])
        left = run["pending"].qsize()
        elapsed = int(time.time() - run["started"])
        per = ", ".join(f"{d.name} {d.made}" for d in run["devices"] if d.made)
        if run["stop"]:
            self.gen_status.set(f"Stopped: {run['done']} made in {elapsed} s ({per or 'none'}).")
        elif left:
            self.gen_status.set(f"{run['done']} made, {left} not made: every device stopped. See the log.")
        else:
            self.gen_status.set(f"Done: {run['done']} images in {elapsed} s ({per}). Saved to {pictures_dir()}")
        self.status_var.set("Model loaded here" if self.ready else "No image model loaded here")
        with run["lock"]:
            statuses = dict(run["statuses"])
        self.refresh_devices(statuses)

    def stop_run(self):
        run = self.run
        if run:
            run["stop"] = True
            self.gen_status.set("Stopping: images already started will finish…")
            self.stop_btn.state(["disabled"])

    # Results ----------------------------------------------------------------

    def add_result(self, res: Result):
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{res.seed}-{len(self.results) + 1}.png"
        try:
            res.path = pictures_dir() / name
            res.path.write_bytes(res.png)
        except OSError as e:
            self.logs.put(f"Couldn't save an image: {e}")
            res.path = None
        try:
            photo = tk.PhotoImage(data=base64.b64encode(res.png).decode())
        except tk.TclError as e:
            self.logs.put(f"Couldn't show an image: {e}")
            return
        res.photo = fit_photo(photo, PREVIEW)
        res.thumb = fit_photo(photo, THUMB)
        self.results.append(res)
        self.draw_gallery()
        self.select(res)

    def draw_gallery(self):
        self.gallery.delete("all")
        cell = THUMB + 6
        for i, res in enumerate(self.results):
            x = (i % THUMB_COLS) * cell + cell // 2
            y = (i // THUMB_COLS) * cell + cell // 2
            if res is self.selected:
                self.gallery.create_rectangle(x - cell // 2 + 1, y - cell // 2 + 1, x + cell // 2 - 1,
                                              y + cell // 2 - 1, outline="#1565c0", width=2)
            self.gallery.create_image(x, y, image=res.thumb, tags=(f"r{i}",))
        rows = -(-len(self.results) // THUMB_COLS)
        self.gallery.configure(scrollregion=(0, 0, THUMB_COLS * cell, max(rows, 1) * cell))
        if self.results and self.results[-1] is self.selected:
            self.gallery.yview_moveto(1.0)

    def on_gallery_click(self, event):
        cell = THUMB + 6
        x, y = self.gallery.canvasx(event.x), self.gallery.canvasy(event.y)
        i = int(y // cell) * THUMB_COLS + int(x // cell)
        if 0 <= x < THUMB_COLS * cell and 0 <= i < len(self.results):
            self.select(self.results[i])

    def select(self, res: Result):
        self.selected = res
        self.canvas.delete("all")
        self.canvas.create_image(PREVIEW // 2, PREVIEW // 2, image=res.photo)
        prompt = res.job.prompt if len(res.job.prompt) <= 70 else res.job.prompt[:67] + "…"
        self.caption.set(f"{res.device} · seed {res.seed} · {res.width}×{res.height} · {res.seconds:.0f} s\n{prompt}")
        self.save_btn.state(["!disabled"])
        self.draw_gallery()

    def save_as(self):
        if not self.selected:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")],
                                            initialdir=str(pictures_dir()))
        if path:
            Path(path).write_bytes(self.selected.png)

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
