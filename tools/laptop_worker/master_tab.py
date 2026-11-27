"""Master tab: this computer runs the model, with part of its layers on phones
set to Worker, and serves it as an OpenAI-compatible API that the PrivateLM
app (Custom API) or a browser can chat with.

Uses llama.cpp's own llama-server, built from the release the app's llama.cpp
tree comes from, so it speaks the phones' RPC protocol.
"""

import json
import math
import os
import queue
import secrets
import socket
import struct
import subprocess
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional

import privatelm_worker as core

DEFAULT_PORT = 8080
UNKNOWN_RAM_MB = 2000      # a worker added by IP doesn't say (same as the app)
UNKNOWN_SPEED = 10.0       # GB/s, when a worker doesn't say
LAPTOP_SHARE = 0.7         # of its free RAM: it also holds the KV cache and runs Windows
WORKER_SHARE = 0.75        # of a worker's free RAM (same as the app)


# ── GGUF ────────────────────────────────────────────────────────────────────

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf_info(path: str) -> dict:
    """Name, architecture and layer count from a GGUF file's header."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("Not a GGUF model file")
        version, = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError("This GGUF version is too old")
        _tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def read_str():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def read_value(t):
            if t == 8:
                return read_str()
            if t == 9:
                item_t, count = struct.unpack("<IQ", f.read(12))
                if item_t in _SCALAR:
                    f.seek(struct.calcsize(_SCALAR[item_t]) * count, os.SEEK_CUR)
                else:
                    for _ in range(count):
                        read_value(item_t)
                return None
            fmt = _SCALAR.get(t)
            if fmt is None:
                raise ValueError(f"Unknown GGUF value type {t}")
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        kv = {}
        for _ in range(n_kv):
            key = read_str()
            t, = struct.unpack("<I", f.read(4))
            value = read_value(t)
            if key in ("general.architecture", "general.name") or key.endswith(".block_count"):
                kv[key] = value
            arch = kv.get("general.architecture")
            if arch and f"{arch}.block_count" in kv and "general.name" in kv:
                break
    arch = kv.get("general.architecture", "")
    layers = int(kv.get(f"{arch}.block_count") or 0)
    if layers <= 0:
        raise ValueError("Couldn't read the model's layer count")
    return {"name": kv.get("general.name") or Path(path).stem, "arch": arch, "layers": layers,
            "size_mb": os.path.getsize(path) / (1024 * 1024)}


# ── Planning ────────────────────────────────────────────────────────────────

@dataclass
class Device:
    name: str
    capacity_mb: float
    speed: float
    endpoint: Optional[str] = None  # None: this computer
    assigned_mb: float = 0.0


@dataclass
class Plan:
    local_layers: int
    n_layers: int
    workers: List[Device] = field(default_factory=list)  # used workers, in --rpc order
    laptop: Optional[Device] = None
    need_mb: float = 0.0

    @property
    def n_gpu_layers(self) -> int:
        # llama.cpp offloads the *last* n_gpu_layers blocks (+1 for the
        # output layer) to the RPC devices, so keeping `local_layers` here
        # means offloading the rest.
        return self.n_layers + 1 - self.local_layers

    def summary(self) -> str:
        if not self.workers:
            return "All layers on this computer"
        parts = [f"This computer {self.local_layers} layers"]
        remote = self.n_layers - self.local_layers
        total = sum(w.assigned_mb for w in self.workers) or 1
        given = 0
        for i, w in enumerate(self.workers):
            n = remote - given if i == len(self.workers) - 1 else round(remote * w.assigned_mb / total)
            given += n
            parts.append(f"{w.name} {n}")
        return " · ".join(parts)


def _split_by_speed(devices: List[Device], need_mb: float) -> None:
    """Shares in proportion to speed, capped at each device's capacity (what
    a capped device can't take goes to the others). Same as the app."""
    open_ = list(devices)
    left = need_mb
    while open_ and left > 0.5:
        speed_sum = sum(d.speed for d in open_)
        capped = [d for d in open_ if d.assigned_mb + left * d.speed / speed_sum >= d.capacity_mb]
        if not capped:
            for d in open_:
                d.assigned_mb += left * d.speed / speed_sum
            left = 0
        else:
            for d in capped:
                left -= d.capacity_mb - d.assigned_mb
                d.assigned_mb = d.capacity_mb
            open_ = [d for d in open_ if d not in capped]


def plan_split(model_mb: float, n_layers: int, laptop_free_mb: float, laptop_speed: float,
               workers: List[Device], use_phones_anyway: bool) -> Plan:
    """Decides which layers go where. Raises ValueError if the model doesn't
    fit on this computer plus the chosen workers."""
    need = model_mb * 1.1  # + KV cache and working buffers
    laptop = Device("This computer", laptop_free_mb * LAPTOP_SHARE, laptop_speed)
    if not workers or (need <= laptop.capacity_mb and not use_phones_anyway):
        if need > laptop.capacity_mb:
            raise ValueError(f"This model needs about {need / 1024:.1f} GB, and this computer has "
                             f"about {laptop.capacity_mb / 1024:.1f} GB to spare. Add phone workers, "
                             "close other apps, or pick a smaller model.")
        laptop.assigned_mb = need
        return Plan(n_layers, n_layers, [], laptop, need)

    total = laptop.capacity_mb + sum(w.capacity_mb for w in workers)
    if need > total:
        raise ValueError(f"This model needs about {need / 1024:.1f} GB, and this computer and the "
                         f"chosen phones have about {total / 1024:.1f} GB together. Add another "
                         "worker, close apps on the phones, or pick a smaller model.")
    for w in workers:
        w.assigned_mb = 0.0
    if need <= laptop.capacity_mb:
        # Fits here, but the user wants the phones in: share by speed.
        _split_by_speed([laptop, *workers], need)
    else:
        # This computer first (no Wi-Fi hop for its layers), the rest on the
        # phones by speed, each up to its free memory.
        laptop.assigned_mb = laptop.capacity_mb
        _split_by_speed(workers, need - laptop.assigned_mb)

    used = [w for w in workers if w.assigned_mb >= 1]
    local = round(n_layers * laptop.assigned_mb / need)
    local = max(0, min(local, n_layers - 1 if used else n_layers))
    return Plan(local, n_layers, used, laptop, need)


# ── Network ─────────────────────────────────────────────────────────────────

def _worker_from(found: dict) -> dict:
    props = found["props"]
    return {
        "name": props.get("model") or found["instance"].split(".")[0].replace("PrivateLM ", ""),
        "host": found["host"], "port": found["port"],
        "ram": int(props["ram"]) if props.get("ram", "").isdigit() else None,
        "bw": float(props["bw"]) if props.get("bw") else None,
        "id": props.get("id"), "v": props.get("v"),
    }


def discover_workers(known_hosts=(), seconds: float = 4.0, log=None) -> List[dict]:
    """Phones (and laptops) set to Worker on this Wi-Fi: a multicast search,
    plus a direct question to each known address, which works on networks
    that drop multicast between some devices."""
    import discovery

    found = {}
    for f in discovery.browse(core.SERVICE_TYPE, seconds, log=log):
        found[(f["host"], f["port"])] = f
    for host in sorted(set(known_hosts)):
        for f in discovery.query_host(host, core.SERVICE_TYPE, log=log):
            found.setdefault((f["host"], f["port"]), f)
    own = core.worker_id()
    workers = [_worker_from(f) for f in found.values()]
    return [w for w in workers if w.get("id") != own]


def stats_for(host: str, port: int, log=None) -> Optional[dict]:
    """A worker's name, free memory and speed, asked directly at its IP."""
    import discovery

    found = [_worker_from(f) for f in discovery.query_host(host, core.SERVICE_TYPE, log=log)]
    return next((w for w in found if w["port"] == port), found[0] if found else None)


def reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def api_key() -> str:
    """Lasting key the PrivateLM app needs (it requires one), so others on the
    Wi-Fi can't use the model."""
    path = core.config_dir() / "master_api_key"
    if path.exists():
        return path.read_text().strip()
    key = "plm-" + secrets.token_urlsafe(18)
    path.write_text(key)
    return key


# ── Tab ─────────────────────────────────────────────────────────────────────

class MasterTab:
    POLL_MS = 1000

    def __init__(self, root: tk.Tk, container, logs: queue.Queue,
                 measure_cpu: Callable[[], dict], find_server: Callable[[Callable[[str], None]], Path],
                 keep_awake: Callable[[bool], None], no_window: int):
        self.root = root
        self.container = container
        self.logs = logs
        self.measure_cpu = measure_cpu
        self.find_server = find_server
        self.keep_awake = keep_awake
        self.no_window = no_window
        self.server = None
        self.ready = False
        self.model_info = None
        self.workers: List[dict] = []
        self.is_busy = None  # set by the window: True while the Worker tab is sharing
        self.state_path = core.config_dir() / "master.json"
        self.build_ui()
        self.load_state()

    # State ------------------------------------------------------------------

    def load_state(self):
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return
        self.workers = state.get("workers", [])
        self.ctx_var.set(str(state.get("ctx", 4096)))
        self.port_var.set(str(state.get("port", DEFAULT_PORT)))
        self.anyway_var.set(bool(state.get("use_phones_anyway", False)))
        if state.get("model") and Path(state["model"]).exists():
            self.set_model(state["model"])
        self.refresh_workers()

    def save_state(self):
        state = {"model": self.model_var.get(), "workers": self.workers, "ctx": self.ctx_var.get(),
                 "port": self.port_var.get(), "use_phones_anyway": self.anyway_var.get()}
        try:
            self.state_path.write_text(json.dumps(state, indent=1))
        except OSError:
            pass

    # UI ---------------------------------------------------------------------

    def build_ui(self):
        outer = ttk.Frame(self.container, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        self.outer = outer

        ttk.Label(outer, text="Run a model here, with phones set to Worker holding part of it.",
                  style="Muted.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        status = ttk.Frame(outer)
        status.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        self.dot = tk.Canvas(status, width=14, height=14, highlightthickness=0, bd=0)
        self.dot.pack(side="left", padx=(0, 8))
        self.status_var = tk.StringVar(value="No model running")
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel", wraplength=420,
                  justify="left").pack(side="left")
        self.set_dot("#9e9e9e")

        # Model
        ttk.Label(outer, text="Model").grid(row=2, column=0, sticky="w", pady=4)
        self.model_var = tk.StringVar()
        self.model_entry = ttk.Entry(outer, textvariable=self.model_var, state="readonly")
        self.model_entry.grid(row=2, column=1, sticky="ew", pady=4)
        self.browse_btn = ttk.Button(outer, text="Browse…", command=self.browse)
        self.browse_btn.grid(row=2, column=2, sticky="e", padx=(6, 0), pady=4)
        self.model_desc = tk.StringVar(value="Choose a .gguf model file")
        ttk.Label(outer, textvariable=self.model_desc, style="Muted.TLabel")\
            .grid(row=3, column=1, columnspan=2, sticky="w")

        # Workers
        wf = ttk.LabelFrame(outer, text="Phone workers", padding=10)
        wf.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 4))
        wf.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(wf, columns=("use", "name", "addr", "ram", "bw"), show="headings",
                                 height=4, selectmode="browse")
        for col, text, width, anchor in (("use", "Use", 40, "center"), ("name", "Device", 130, "w"),
                                         ("addr", "Address", 140, "w"), ("ram", "Free", 70, "e"),
                                         ("bw", "Speed", 70, "e")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor, stretch=col in ("name", "addr"))
        self.tree.grid(row=0, column=0, columnspan=4, sticky="ew")
        self.tree.bind("<ButtonRelease-1>", self.on_tree_click)
        self.tree.bind("<space>", lambda _: self.toggle_selected())

        self.find_btn = ttk.Button(wf, text="Find nearby", command=self.find_nearby)
        self.find_btn.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.ip_var = tk.StringVar()
        self.ip_entry = ttk.Entry(wf, textvariable=self.ip_var, width=18)
        self.ip_entry.grid(row=1, column=1, sticky="e", padx=(6, 0), pady=(8, 0))
        self.ip_entry.bind("<Return>", lambda _: self.add_by_ip())
        self.add_btn = ttk.Button(wf, text="Add by IP", command=self.add_by_ip)
        self.add_btn.grid(row=1, column=2, sticky="e", padx=(6, 0), pady=(8, 0))
        self.remove_btn = ttk.Button(wf, text="Remove", command=self.remove_selected)
        self.remove_btn.grid(row=1, column=3, sticky="e", padx=(6, 0), pady=(8, 0))
        self.workers_hint = tk.StringVar(value="On each phone, set Phone cluster to Worker, then tap "
                                               "Find nearby. Click Use to include or leave out a phone.")
        ttk.Label(wf, textvariable=self.workers_hint, style="Muted.TLabel", wraplength=420,
                  justify="left").grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.anyway_var = tk.BooleanVar(value=False)
        self.anyway_check = ttk.Checkbutton(
            outer, variable=self.anyway_var,
            text="Use the phones even if the model fits here (usually slower)")
        self.anyway_check.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 2))

        opts = ttk.Frame(outer)
        opts.grid(row=6, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(opts, text="Context").pack(side="left")
        self.ctx_var = tk.StringVar(value="4096")
        self.ctx_box = ttk.Combobox(opts, textvariable=self.ctx_var, width=7,
                                    values=["2048", "4096", "8192", "16384", "32768"])
        self.ctx_box.pack(side="left", padx=(6, 18))
        ttk.Label(opts, text="Port").pack(side="left")
        self.port_var = tk.StringVar(value=str(DEFAULT_PORT))
        self.port_entry = ttk.Entry(opts, textvariable=self.port_var, width=7)
        self.port_entry.pack(side="left", padx=(6, 0))

        self.start_btn = ttk.Button(outer, text="Start model", command=self.toggle)
        self.start_btn.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(12, 6), ipady=4)

        # Connect panel
        self.connect = ttk.LabelFrame(outer, text="Chat with it", padding=12)
        self.connect.columnconfigure(1, weight=1)
        ttk.Label(self.connect, wraplength=420, justify="left",
                  text="In PrivateLM on your phone: Models → Cloud → Custom API, then enter:")\
            .grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self.conn_vars = {}
        mono = ("Consolas", 10) if os.name == "nt" else ("", 10)
        for row, (label, key) in enumerate((("Base URL", "url"), ("API key", "key"), ("Model", "model")), 1):
            ttk.Label(self.connect, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar()
            self.conn_vars[key] = var
            entry = ttk.Entry(self.connect, textvariable=var, state="readonly", font=mono)
            entry.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
            ttk.Button(self.connect, text="Copy", width=6,
                       command=lambda v=var: self.copy(v.get())).grid(row=row, column=2, pady=2)
        ttk.Button(self.connect, text="Open chat in this computer's browser", command=self.open_browser)\
            .grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def set_dot(self, color: str):
        self.dot.delete("all")
        self.dot.create_oval(2, 2, 12, 12, fill=color, outline=color)

    def copy(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def controls(self, enabled: bool):
        flag = ["!disabled"] if enabled else ["disabled"]
        for w in (self.browse_btn, self.find_btn, self.ip_entry, self.add_btn, self.remove_btn,
                  self.anyway_check, self.ctx_box, self.port_entry):
            w.state(flag)
        self.tree.state(flag)

    # Model ------------------------------------------------------------------

    def browse(self):
        path = filedialog.askopenfilename(title="Choose a model", filetypes=[("GGUF models", "*.gguf"),
                                                                             ("All files", "*.*")])
        if path:
            self.set_model(path)
            self.save_state()

    def set_model(self, path: str):
        try:
            info = read_gguf_info(path)
        except (OSError, ValueError, struct.error) as e:
            messagebox.showerror("PrivateLM", f"Can't use this file:\n\n{e}")
            return
        self.model_info = info
        self.model_var.set(path)
        size = info["size_mb"]
        size_text = f"{size / 1024:.1f} GB" if size >= 1024 else f"{size:.0f} MB"
        self.model_desc.set(f"{info['name']} · {size_text} · {info['layers']} layers")

    # Workers ----------------------------------------------------------------

    def refresh_workers(self):
        self.tree.delete(*self.tree.get_children())
        for i, w in enumerate(self.workers):
            ram = f"{w['ram'] / 1024:.1f} GB" if w.get("ram") else "?"
            bw = f"{w['bw']:.0f} GB/s" if w.get("bw") else "?"
            self.tree.insert("", "end", iid=str(i), values=("☑" if w.get("use", True) else "☐",
                                                            w["name"], f"{w['host']}:{w['port']}", ram, bw))

    def on_tree_click(self, event):
        if str(self.tree.cget("selectmode")) == "none" or "disabled" in self.tree.state():
            return
        if self.tree.identify_column(event.x) == "#1" and self.tree.identify_row(event.y):
            self.tree.selection_set(self.tree.identify_row(event.y))
            self.toggle_selected()

    def toggle_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        w = self.workers[int(sel[0])]
        w["use"] = not w.get("use", True)
        self.refresh_workers()
        self.tree.selection_set(sel[0])
        self.save_state()

    def remove_selected(self):
        sel = self.tree.selection()
        if sel:
            del self.workers[int(sel[0])]
            self.refresh_workers()
            self.save_state()

    def merge_worker(self, w: dict):
        for old in self.workers:
            same_id = w.get("id") and old.get("id") == w.get("id")
            if same_id or (old["host"] == w["host"] and old["port"] == w["port"]):
                use = old.get("use", True)
                old.update({k: v for k, v in w.items() if v is not None})
                old["use"] = use
                return
        self.workers.append({**w, "use": True})

    def find_nearby(self):
        self.find_btn.state(["disabled"])
        self.workers_hint.set("Looking for phones set to Worker…")

        known = [w["host"] for w in self.workers]

        def work():
            try:
                found = discover_workers(known, log=self.logs.put)
                error = None
            except ImportError:
                found, error = [], "Find nearby needs the zeroconf package. Add phones by IP instead."
            except Exception as e:  # noqa: BLE001 - shown to the user
                found, error = [], f"Find nearby failed: {e}"
            self.root.after(0, lambda: self.on_found(found, error))

        threading.Thread(target=work, daemon=True).start()

    def on_found(self, found: List[dict], error: Optional[str]):
        self.find_btn.state(["!disabled"])
        for w in found:
            if w.get("v") and w["v"] != core.PROTOCOL:
                self.logs.put(f"Skipping {w['name']}: its PrivateLM speaks {w['v']}, this needs {core.PROTOCOL}. "
                              "Update the app on that phone.")
                continue
            self.merge_worker(w)
        self.refresh_workers()
        self.save_state()
        if error:
            self.workers_hint.set(error)
        elif found:
            self.workers_hint.set(f"Found {len(found)}. Click Use to include or leave out a phone.")
        else:
            self.workers_hint.set("No workers found. On each phone, set Phone cluster to Worker, keep "
                                  "PrivateLM open, and check it's on the same Wi-Fi. Or add it by IP.")

    def add_by_ip(self):
        text = self.ip_var.get().strip()
        if not text:
            return
        host, _, port = text.partition(":")
        try:
            port_n = int(port) if port else core.DEFAULT_PORT
            socket.inet_aton(host)
        except (ValueError, OSError):
            messagebox.showerror("PrivateLM", "Use the form 192.168.1.20 or 192.168.1.20:50052")
            return
        self.merge_worker({"name": host, "host": host, "port": port_n, "ram": None, "bw": None, "id": None})
        self.ip_var.set("")
        self.refresh_workers()
        self.save_state()
        self.workers_hint.set(f"Asking {host} for its details…")

        def work():
            try:
                stats = stats_for(host, port_n, log=self.logs.put)
            except Exception as e:  # noqa: BLE001
                self.logs.put(f"Couldn't ask {host}: {e}")
                stats = None
            self.root.after(0, lambda: self.on_stats(host, port_n, stats))

        threading.Thread(target=work, daemon=True).start()

    def on_stats(self, host: str, port: int, stats: Optional[dict]):
        if stats:
            for w in self.workers:
                if w["host"] == host and w["port"] == port:
                    use = w.get("use", True)
                    w.update({k: v for k, v in stats.items() if v is not None})
                    w["port"], w["use"] = port, use
            self.workers_hint.set(f"Added {stats['name']}.")
        else:
            self.workers_hint.set(f"Added {host}, but it didn't say how much memory it has, so 2 GB is "
                                  "assumed. Check that Worker is on and PrivateLM is open on it, then "
                                  "tap Find nearby to try again.")
        self.refresh_workers()
        self.save_state()

    # Running ----------------------------------------------------------------

    def toggle(self):
        if self.server:
            self.stop()
        else:
            self.start()

    def start(self):
        if self.is_busy and self.is_busy():
            messagebox.showinfo("PrivateLM", "Stop sharing on the Worker tab first: this computer can "
                                             "be a master or a worker, not both.")
            return
        if not self.model_info:
            messagebox.showinfo("PrivateLM", "Choose a .gguf model first.")
            return
        try:
            port = int(self.port_var.get())
            ctx = int(self.ctx_var.get())
            if not (1 <= port <= 65535 and ctx >= 256):
                raise ValueError
        except ValueError:
            messagebox.showerror("PrivateLM", "Check the context size and port.")
            return
        self.save_state()
        self.controls(False)
        self.start_btn.state(["disabled"])
        self.set_dot("#f9a825")
        self.status_var.set("Checking this computer and the phones…")
        chosen = [w for w in self.workers if w.get("use", True)]
        threading.Thread(target=self.prepare, args=(chosen, port, ctx), daemon=True).start()

    def prepare(self, chosen: List[dict], port: int, ctx: int):
        """Background: measure, plan, probe workers, start llama-server."""
        def fail(msg):
            self.root.after(0, lambda: self.on_failed(msg))

        try:
            server_bin = self.find_server(lambda m: self.root.after(0, lambda: self.status_var.set(m)))
            laptop = self.measure_cpu()
        except Exception as e:  # noqa: BLE001
            return fail(str(e))

        offline = [w for w in chosen if not reachable(w["host"], w["port"])]
        online = [w for w in chosen if w not in offline]
        if offline:
            self.logs.put("Not reachable, starting without: " + ", ".join(w["name"] for w in offline))
        devices = [Device(w["name"], (w.get("ram") or UNKNOWN_RAM_MB) * WORKER_SHARE,
                          w.get("bw") or UNKNOWN_SPEED, f"{w['host']}:{w['port']}") for w in online]
        info = self.model_info
        try:
            plan = plan_split(info["size_mb"], info["layers"], laptop["usableMb"], laptop["bandwidth"],
                              devices, self.anyway_var.get())
        except ValueError as e:
            extra = f"\n\nNot reachable: {', '.join(w['name'] for w in offline)}." if offline else ""
            return fail(str(e) + extra)

        alias = Path(self.model_var.get()).stem.lower().replace(" ", "-")
        key = api_key()
        args = [str(server_bin), "-m", self.model_var.get(), "--host", "0.0.0.0", "--port", str(port),
                "-c", str(ctx), "--api-key", key, "-a", alias]
        if plan.workers:
            args += ["--rpc", ",".join(w.endpoint for w in plan.workers),
                     "-ngl", str(plan.n_gpu_layers),
                     "-ts", ",".join(str(max(1, int(w.assigned_mb))) for w in plan.workers),
                     # The output layer is the biggest matmul per token, and its
                     # result would cross the Wi-Fi every token: keep it here
                     # (the phone master does the same).
                     "-ot", r"^(output|token_embd)\.weight$=CPU"]
        self.logs.put("Starting: " + " ".join(a if a != key else "<key>" for a in args))
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, creationflags=self.no_window,
                                    encoding="utf-8", errors="replace")
        except OSError as e:
            return fail(f"llama-server couldn't start:\n\n{e}")
        self.root.after(0, lambda: self.on_started(proc, plan, port, key, alias))

    def on_started(self, proc, plan: Plan, port: int, key: str, alias: str):
        self.server = proc
        self.ready = False
        self.plan = plan
        self.port = port
        self.tail: List[str] = []
        threading.Thread(target=self.read_output, args=(proc,), daemon=True).start()
        self.keep_awake(True)
        host = core.lan_address()
        self.conn_vars["url"].set(f"http://{host}:{port}/v1")
        self.conn_vars["key"].set(key)
        self.conn_vars["model"].set(alias)
        self.start_btn.configure(text="Stop model")
        self.start_btn.state(["!disabled"])
        where = "across the phones" if plan.workers else "on this computer"
        self.status_var.set(f"Loading the model {where}…  ({plan.summary()})")
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
            self.stop(error=f"The model stopped (code {code}).\n\n{detail}")
            return
        if not self.ready:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        self.ready = True
                        self.set_dot("#2e7d32")
                        self.status_var.set(f"Ready · {self.plan.summary()}")
                        self.connect.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(8, 0))
            except (urllib.error.URLError, OSError):
                pass  # still loading
        self.root.after(self.POLL_MS, self.poll)

    def open_browser(self):
        webbrowser.open(f"http://127.0.0.1:{self.port}/")
        self.copy(self.conn_vars["key"].get())
        messagebox.showinfo("PrivateLM", "The API key is copied: if the chat page asks for it, paste it "
                                         "in its settings.")

    def on_failed(self, message: str):
        self.controls(True)
        self.start_btn.state(["!disabled"])
        self.set_dot("#c62828")
        self.status_var.set("Couldn't start the model")
        self.logs.put(message)
        messagebox.showerror("PrivateLM", message)

    def stop(self, error: Optional[str] = None):
        proc, self.server = self.server, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.keep_awake(False)
        self.ready = False
        self.connect.grid_remove()
        self.controls(True)
        self.start_btn.configure(text="Start model")
        self.start_btn.state(["!disabled"])
        if error:
            self.set_dot("#c62828")
            self.status_var.set("The model stopped")
            self.logs.put(error)
            messagebox.showerror("PrivateLM", error)
        else:
            self.set_dot("#9e9e9e")
            self.status_var.set("No model running")

    def shutdown(self):
        if self.server:
            self.stop()

    @property
    def running(self) -> bool:
        return self.server is not None

    def endpoint(self):
        """(base URL, API key, model name) of the running model, for the
        Images tab's prompt helper; None when no model is ready."""
        if not (self.server and self.ready):
            return None
        return (f"http://127.0.0.1:{self.port}", self.conn_vars["key"].get(), self.conn_vars["model"].get())
