#!/usr/bin/env python3
"""PrivateLM Worker: a window for sharing this computer with a phone cluster.

Made for Windows (built into PrivateLM-Worker.exe by the "Laptop worker"
GitHub Actions workflow), and it also runs from source on macOS and Linux:

    pip install zeroconf qrcode psutil
    python tools/laptop_worker/worker_gui.py
"""

import hashlib
import json
import os
import platform
import queue
import shutil
import socket
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from urllib.parse import urlencode

import privatelm_worker as core

IS_WINDOWS = platform.system() == "Windows"
FROZEN = getattr(sys, "frozen", False)
NO_WINDOW = 0x08000000 if IS_WINDOWS else 0  # CREATE_NO_WINDOW

DEVICE_AUTO = "Automatic (GPU if it has one)"
DEVICE_GPU = "GPU"
DEVICE_CPU = "CPU only"


# ── Worker binaries ─────────────────────────────────────────────────────────

def bundled_binary(variant: str):
    """The worker built into the .exe ("vulkan" or "cpu"), copied to a fixed
    folder first: the .exe unpacks to a new temp folder on every launch, and
    Windows Firewall would ask again for each new path."""
    if not FROZEN:
        return None
    name = "privatelm-worker.exe" if IS_WINDOWS else "privatelm-worker"
    src = Path(getattr(sys, "_MEIPASS", "")) / "bin" / variant / name
    if not src.exists():
        return None
    dst = core.config_dir() / "bin" / variant / name
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    stamp = dst.with_suffix(".sha256")
    if not dst.exists() or not stamp.exists() or stamp.read_text() != digest:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        stamp.write_text(digest)
    return dst


def candidate_binaries(choice: str):
    """Worker binaries to try, best first, as (binary, extra args)."""
    if FROZEN:
        gpu, cpu = bundled_binary("vulkan"), bundled_binary("cpu")
        out = []
        if choice != DEVICE_CPU and gpu:
            out.append((gpu, []))
        if cpu:
            out.append((cpu, ["--cpu"]))
        return out
    # From source: one build, Metal on a Mac, CPU elsewhere (or --gpu builds
    # made with privatelm_worker.py).
    binary = core.build("", rebuild=False)
    return [(binary, ["--cpu"] if choice == DEVICE_CPU else [])]


def measure(choice: str):
    """Runs the worker's --info. Returns (binary, args, info) for the first
    binary that works, keeping a GPU only if one was found (or the user
    asked for one)."""
    errors = []
    candidates = candidate_binaries(choice)
    for i, (binary, args) in enumerate(candidates):
        try:
            run = subprocess.run([str(binary), "--info", *args], capture_output=True, text=True,
                                 timeout=120, creationflags=NO_WINDOW)
            if run.returncode != 0:
                errors.append(run.stderr.strip() or f"exit code {run.returncode}")
                continue
            info = json.loads(run.stdout.strip().splitlines()[-1])
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            errors.append(str(e))
            continue
        if (choice == DEVICE_AUTO and info["type"] != "gpu" and "--cpu" not in args
                and i + 1 < len(candidates)):
            continue  # the GPU build found no GPU: use the CPU build
        return binary, args, info
    raise RuntimeError("The worker couldn't start on this computer."
                       + (f"\n\n{errors[-1]}" if errors else ""))


# ── Keeping the computer awake ──────────────────────────────────────────────

def keep_awake(on: bool) -> None:
    """Stops Windows from sleeping while sharing (the screen may still turn
    off). Closing the lid follows the power settings."""
    if not IS_WINDOWS:
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def connected_peers(pid: int, port: int):
    """IP addresses of masters connected to the worker right now."""
    try:
        import psutil  # type: ignore
        proc = psutil.Process(pid)
        conns = proc.net_connections("tcp") if hasattr(proc, "net_connections") else proc.connections("tcp")
    except Exception:
        return None  # unknown (psutil missing, or not allowed)
    return sorted({c.raddr.ip for c in conns
                   if c.status == "ESTABLISHED" and c.laddr and c.laddr.port == port and c.raddr})


# ── Window ──────────────────────────────────────────────────────────────────

class WorkerApp:
    POLL_MS = 1500

    def __init__(self, root: tk.Tk):
        self.root = root
        self.server = None
        self.published = None
        self.binary = None
        self.args = []
        self.info = None
        self.logs = queue.Queue()
        self.measuring = False

        root.title("PrivateLM Worker")
        root.minsize(460, 0)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.remeasure()
        self.root.after(200, self.drain_logs)

    # UI ---------------------------------------------------------------------

    def build_ui(self):
        style = ttk.Style()
        if IS_WINDOWS and "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold") if IS_WINDOWS else ("", 16, "bold"))
        style.configure("Muted.TLabel", foreground="#666666")
        style.configure("Status.TLabel", font=("Segoe UI", 11, "bold") if IS_WINDOWS else ("", 11, "bold"))

        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)

        ttk.Label(outer, text="PrivateLM Worker", style="Title.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(outer, text="Lend this computer's memory and processor to your phone cluster.",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 14))

        # Status
        status = ttk.Frame(outer)
        status.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        self.dot = tk.Canvas(status, width=14, height=14, highlightthickness=0, bd=0)
        self.dot.pack(side="left", padx=(0, 8))
        self.status_var = tk.StringVar(value="Not sharing")
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").pack(side="left")
        self.set_dot("#9e9e9e")

        # Settings
        ttk.Label(outer, text="Use").grid(row=3, column=0, sticky="w", pady=4)
        self.device_var = tk.StringVar(value=DEVICE_AUTO)
        self.device_box = ttk.Combobox(outer, textvariable=self.device_var, state="readonly",
                                       values=[DEVICE_AUTO, DEVICE_CPU])
        self.device_box.grid(row=3, column=1, columnspan=2, sticky="ew", pady=4)
        self.device_box.bind("<<ComboboxSelected>>", lambda _: self.remeasure())

        self.device_info_var = tk.StringVar(value="Checking this computer…")
        ttk.Label(outer, textvariable=self.device_info_var, style="Muted.TLabel", wraplength=400,
                  justify="left").grid(row=4, column=1, columnspan=2, sticky="w")

        ttk.Label(outer, text="Memory to offer").grid(row=5, column=0, sticky="w", pady=(10, 4))
        self.mem_var = tk.DoubleVar(value=0)
        self.mem_scale = ttk.Scale(outer, from_=1, to=2, variable=self.mem_var,
                                   command=lambda _: self.update_mem_label())
        self.mem_scale.grid(row=5, column=1, sticky="ew", pady=(10, 4))
        self.mem_label = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.mem_label, width=9, anchor="e").grid(row=5, column=2, sticky="e",
                                                                                 pady=(10, 4))

        ttk.Label(outer, text="Port").grid(row=6, column=0, sticky="w", pady=4)
        self.port_var = tk.StringVar(value=str(core.DEFAULT_PORT))
        self.port_entry = ttk.Entry(outer, textvariable=self.port_var, width=8)
        self.port_entry.grid(row=6, column=1, sticky="w", pady=4)

        self.start_btn = ttk.Button(outer, text="Start sharing", command=self.toggle, state="disabled")
        self.start_btn.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(14, 6), ipady=4)

        # Connect panel (shown while sharing)
        self.connect = ttk.LabelFrame(outer, text="Add it on the master phone", padding=12)
        self.connect.columnconfigure(1, weight=1)
        self.qr = tk.Canvas(self.connect, width=180, height=180, bg="white", highlightthickness=0)
        self.qr.grid(row=0, column=0, rowspan=4, sticky="nw", padx=(0, 12))
        ttk.Label(self.connect, text="In PrivateLM, open Phone cluster and tap Find nearby, or scan "
                                     "this code.", wraplength=220, justify="left").grid(row=0, column=1, sticky="nw")
        self.addr_var = tk.StringVar()
        ttk.Label(self.connect, textvariable=self.addr_var, font=("Consolas", 11) if IS_WINDOWS else ("", 11))\
            .grid(row=1, column=1, sticky="w", pady=(8, 0))
        self.discover_var = tk.StringVar()
        ttk.Label(self.connect, textvariable=self.discover_var, style="Muted.TLabel", wraplength=220,
                  justify="left").grid(row=2, column=1, sticky="w", pady=(4, 0))
        ttk.Button(self.connect, text="Copy address", command=self.copy_address)\
            .grid(row=3, column=1, sticky="sw", pady=(8, 0))
        self.connect_row = 8

        self.tips = ttk.Label(outer, style="Muted.TLabel", wraplength=420, justify="left",
                              text="Keep this computer plugged in, on the same Wi-Fi as the phones, and "
                                   "with its lid open. If Windows Firewall asks, allow private networks.")
        self.tips.grid(row=9, column=0, columnspan=3, sticky="w", pady=(10, 4))

        # Log
        self.log_shown = tk.BooleanVar(value=False)
        ttk.Checkbutton(outer, text="Show log", variable=self.log_shown, command=self.toggle_log)\
            .grid(row=10, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.log_frame = ttk.Frame(outer)
        self.log = tk.Text(self.log_frame, height=10, width=60, wrap="word", state="disabled",
                           font=("Consolas", 9) if IS_WINDOWS else ("", 9))
        scroll = ttk.Scrollbar(self.log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def set_dot(self, color: str):
        self.dot.delete("all")
        self.dot.create_oval(2, 2, 12, 12, fill=color, outline=color)

    def toggle_log(self):
        if self.log_shown.get():
            self.log_frame.grid(row=11, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        else:
            self.log_frame.grid_remove()

    def append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text if text.endswith("\n") else text + "\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 2000:
            self.log.delete("1.0", f"{lines - 2000}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def drain_logs(self):
        try:
            while True:
                self.append_log(self.logs.get_nowait())
        except queue.Empty:
            pass
        self.root.after(200, self.drain_logs)

    def update_mem_label(self):
        self.mem_label.set(f"{self.mem_var.get():.1f} GB")

    def draw_qr(self, data: str):
        self.qr.delete("all")
        try:
            import qrcode  # type: ignore
        except ImportError:
            self.qr.create_text(90, 90, text="Install qrcode\nto show a code", justify="center")
            return
        qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
        qr.add_data(data)
        qr.make(fit=True)
        matrix = qr.get_matrix()
        size = int(self.qr["width"])
        cell = size / len(matrix)
        for y, row in enumerate(matrix):
            for x, on in enumerate(row):
                if on:
                    self.qr.create_rectangle(x * cell, y * cell, (x + 1) * cell, (y + 1) * cell,
                                             fill="black", outline="black")

    def copy_address(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.addr_var.get())

    # Measuring --------------------------------------------------------------

    def remeasure(self):
        if self.measuring or self.server:
            return
        self.measuring = True
        self.start_btn.configure(state="disabled")
        self.device_info_var.set("Checking this computer's memory and speed…")
        choice = self.device_var.get()

        def work():
            try:
                result = measure(choice)
                self.root.after(0, lambda: self.on_measured(result))
            except Exception as e:  # noqa: BLE001 - shown to the user
                msg = str(e)
                self.root.after(0, lambda: self.on_measure_failed(msg))

        threading.Thread(target=work, daemon=True).start()

    def on_measured(self, result):
        self.measuring = False
        self.binary, self.args, self.info = result
        info = self.info
        free_gb = info["usableMb"] / 1024
        kind = "GPU" if info["type"] == "gpu" else "CPU"
        self.device_info_var.set(f"{info['description'] or info['device']} ({kind}) · "
                                 f"{free_gb:.1f} GB free · {info['bandwidth']:.1f} GB/s")
        if info["type"] == "gpu" and DEVICE_GPU not in self.device_box["values"]:
            self.device_box.configure(values=[DEVICE_AUTO, DEVICE_GPU, DEVICE_CPU])
        top = max(1.0, round(free_gb, 1))
        self.mem_scale.configure(from_=min(1.0, top), to=top)
        self.mem_var.set(top)
        self.update_mem_label()
        self.start_btn.configure(state="normal")

    def on_measure_failed(self, message: str):
        self.measuring = False
        self.device_info_var.set("This computer couldn't be checked. See the log.")
        self.append_log(message)
        if self.device_var.get() != DEVICE_CPU:
            self.device_var.set(DEVICE_CPU)
            self.remeasure()

    # Sharing ----------------------------------------------------------------

    def toggle(self):
        if self.server:
            self.stop()
        else:
            self.start()

    def start(self):
        try:
            port = int(self.port_var.get())
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            messagebox.showerror("PrivateLM Worker", "The port must be a number from 1 to 65535.")
            return
        ram_mb = int(self.mem_var.get() * 1024)
        host = core.lan_address()
        wid = core.worker_id()
        label = socket.gethostname().split(".")[0] or "Laptop"
        cache = core.config_dir() / "cluster_cache"
        cache.mkdir(exist_ok=True)

        try:
            self.server = subprocess.Popen(
                [str(self.binary), "--port", str(port), "--cache", str(cache), *self.args],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                creationflags=NO_WINDOW)
        except OSError as e:
            messagebox.showerror("PrivateLM Worker", f"The worker couldn't start:\n\n{e}")
            self.server = None
            return
        self.port = port
        threading.Thread(target=self.read_output, args=(self.server,), daemon=True).start()

        txt = {"ram": str(ram_mb), "bw": f"{self.info['bandwidth']:.1f}", "model": label,
               "v": core.PROTOCOL, "id": wid}
        try:
            self.published = core.advertise(f"PrivateLM {label}", host, port, txt)
        except Exception as e:  # noqa: BLE001 - discovery is optional
            self.published = None
            self.logs.put(f"Couldn't advertise on the network: {e}")
        keep_awake(True)

        self.draw_qr("privatelm://worker?" + urlencode({
            "host": host, "port": port, "name": label, "ram": ram_mb, "bw": txt["bw"], "id": wid}))
        self.addr_var.set(f"{host}:{port}")
        if host == "127.0.0.1":
            self.discover_var.set("This computer isn't on a network. Connect it to the phones' Wi-Fi.")
        elif self.published is None:
            self.discover_var.set("Find nearby won't see this computer (zeroconf is missing): "
                                  "scan the code instead.")
        else:
            self.discover_var.set("")
        self.connect.grid(row=self.connect_row, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        for widget in (self.device_box, self.mem_scale, self.port_entry):
            widget.state(["disabled"])
        self.start_btn.configure(text="Stop sharing")
        self.set_dot("#2e7d32")
        self.status_var.set(f"Sharing {ram_mb / 1024:.1f} GB · waiting for a master")
        self.logs.put(f"Sharing {self.info['device']} with {ram_mb} MB at {host}:{port}")
        self.root.after(self.POLL_MS, self.poll)

    def read_output(self, proc):
        for line in proc.stdout:
            self.logs.put(line.rstrip())

    def poll(self):
        proc = self.server
        if not proc:
            return
        code = proc.poll()
        if code is not None:
            self.stop(error=f"The worker stopped (code {code}). Port {self.port} may be in use: "
                            "see the log.")
            return
        peers = connected_peers(proc.pid, self.port)
        ram = self.mem_var.get()
        if peers:
            self.set_dot("#1565c0")
            self.status_var.set(f"Sharing {ram:.1f} GB · master connected ({', '.join(peers)})")
        else:
            self.set_dot("#2e7d32")
            self.status_var.set(f"Sharing {ram:.1f} GB · waiting for a master")
        self.root.after(self.POLL_MS, self.poll)

    def stop(self, error: str = None):
        proc, self.server = self.server, None
        if self.published:
            zc, info = self.published
            self.published = None
            try:
                zc.unregister_service(info)
                zc.close()
            except Exception:  # noqa: BLE001
                pass
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        keep_awake(False)
        self.connect.grid_remove()
        for widget in (self.device_box, self.mem_scale, self.port_entry):
            widget.state(["!disabled"])
        self.start_btn.configure(text="Start sharing")
        if error:
            self.set_dot("#c62828")
            self.status_var.set("Stopped")
            self.logs.put(error)
            messagebox.showerror("PrivateLM Worker", error)
        else:
            self.set_dot("#9e9e9e")
            self.status_var.set("Not sharing")

    def on_close(self):
        if self.server:
            self.stop()
        self.root.destroy()


def main():
    if IS_WINDOWS:
        try:  # sharp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass
    root = tk.Tk()
    WorkerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
