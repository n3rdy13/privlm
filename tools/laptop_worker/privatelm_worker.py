#!/usr/bin/env python3
"""Share this laptop with a PrivateLM phone cluster, like a phone set to Worker.

Builds the worker from the app's own llama.cpp tree (first run only), then
serves this computer's GPU or CPU on the Wi-Fi. On the master phone, open
Phone cluster and tap "Find nearby", or scan the QR code this prints.

    python3 tools/laptop_worker/privatelm_worker.py          # GPU if it has one
    python3 tools/laptop_worker/privatelm_worker.py --cpu    # CPU only
    python3 tools/laptop_worker/privatelm_worker.py --gpu cuda     # NVIDIA (Linux/Windows)
    python3 tools/laptop_worker/privatelm_worker.py --gpu vulkan   # other GPUs

Optional: `pip install zeroconf` so "Find nearby" finds the laptop, and
`pip install qrcode` to print the QR code in the terminal.
"""

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
SERVICE_TYPE = "_privatelm._tcp.local."
PROTOCOL = "rpc7"  # must match the phones (ClusterWorkerService.kt)
DEFAULT_PORT = 50052


def config_dir() -> Path:
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    elif platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    path = base / "PrivateLM"
    path.mkdir(parents=True, exist_ok=True)
    return path


def worker_id() -> str:
    """Lasting ID, so the master finds this laptop again on a new IP."""
    path = config_dir() / "worker_id"
    if path.exists():
        return path.read_text().strip()
    wid = str(uuid.uuid4())
    path.write_text(wid)
    return wid


def binary_path(build_dir: Path) -> Path:
    name = "privatelm-worker.exe" if platform.system() == "Windows" else "privatelm-worker"
    for candidate in (build_dir / name, build_dir / "Release" / name):
        if candidate.exists():
            return candidate
    return build_dir / name


def build(gpu: str, rebuild: bool) -> Path:
    build_dir = HERE / "build" / (gpu or "default")
    binary = binary_path(build_dir)
    if binary.exists() and not rebuild:
        return binary
    if not shutil.which("cmake"):
        sys.exit("cmake is needed to build the worker: install it (e.g. `brew install cmake`, "
                 "`sudo apt install cmake build-essential`, or Visual Studio with C++ on Windows).")
    flags = ["-DCMAKE_BUILD_TYPE=Release"]
    if gpu == "cuda":
        flags.append("-DGGML_CUDA=ON")
    elif gpu == "vulkan":
        flags.append("-DGGML_VULKAN=ON")
    print(f"Building the worker ({gpu or 'Metal' if platform.system() == 'Darwin' else gpu or 'CPU'}). "
          "This takes a few minutes the first time…", flush=True)
    subprocess.run(["cmake", "-S", str(HERE), "-B", str(build_dir), *flags], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release", "-j",
                    str(os.cpu_count() or 4)], check=True)
    return binary_path(build_dir)


def lan_address() -> str:
    """This computer's address on the Wi-Fi (no packets are sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.168.1.1", 9))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def print_qr(uri: str) -> None:
    try:
        import qrcode  # type: ignore
        qr = qrcode.QRCode(border=1)
        qr.add_data(uri)
        qr.print_ascii(invert=True)
    except ImportError:
        print("(pip install qrcode to show this as a QR code)")
    print(uri)


def advertise(name: str, host: str, port: int, txt: dict):
    """Bonjour/mDNS, the same service the phones advertise. None if zeroconf
    isn't installed."""
    try:
        from zeroconf import ServiceInfo, Zeroconf  # type: ignore
    except ImportError:
        print("zeroconf isn't installed, so \"Find nearby\" won't see this laptop: "
              "`pip install zeroconf`, or scan the QR code instead.")
        return None
    info = ServiceInfo(
        SERVICE_TYPE,
        f"{name}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(host)],
        port=port,
        properties=txt,
        server=f"privatelm-{txt['id'][:8]}.local.",
    )
    zc = Zeroconf()
    zc.register_service(info, allow_name_change=True)
    return zc, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--cpu", action="store_true", help="serve the CPU even if there's a GPU")
    parser.add_argument("--gpu", choices=["cuda", "vulkan"], default="",
                        help="build with CUDA or Vulkan (Metal is automatic on a Mac)")
    parser.add_argument("--threads", type=int, help="CPU threads (default: performance cores)")
    parser.add_argument("--max-ram-gb", type=float,
                        help="offer at most this much memory to the master")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    binary = build(args.gpu, args.rebuild)
    common = ["--cpu"] if args.cpu else []
    if args.threads:
        common += ["--threads", str(args.threads)]

    print("Measuring free memory and speed…", flush=True)
    info = json.loads(subprocess.run([str(binary), "--info", *common], check=True,
                                     capture_output=True, text=True).stdout)
    ram_mb = int(info["usableMb"])
    if args.max_ram_gb:
        ram_mb = min(ram_mb, int(args.max_ram_gb * 1024))

    host = lan_address()
    wid = worker_id()
    label = socket.gethostname().split(".")[0] or "Laptop"
    txt = {
        "ram": str(ram_mb),
        "bw": f"{info['bandwidth']:.1f}",
        "model": label,
        "v": PROTOCOL,
        "id": wid,
    }

    cache = config_dir() / "cluster_cache"
    cache.mkdir(exist_ok=True)
    server = subprocess.Popen([str(binary), "--port", str(args.port), "--cache", str(cache), *common])

    print(f"\nSharing {info['description'] or info['device']} ({info['type'].upper()}): "
          f"{ram_mb / 1024:.1f} GB free, {info['bandwidth']:.1f} GB/s, at {host}:{args.port}")
    if host == "127.0.0.1":
        print("This computer doesn't seem to be on a network: connect it to the phones' Wi-Fi.")
    print("On the master phone: Phone cluster → Find nearby, or scan this code:\n")
    print_qr("privatelm://worker?" + urlencode({
        "host": host, "port": args.port, "name": label, "ram": ram_mb,
        "bw": txt["bw"], "id": wid,
    }))
    published = advertise(f"PrivateLM {label}", host, args.port, txt)
    print("\nPress Ctrl+C to stop sharing. Keep the laptop awake and plugged in.\n", flush=True)

    def stop(*_):
        if published:
            zc, svc = published
            zc.unregister_service(svc)
            zc.close()
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    code = server.wait()
    if published:
        published[0].unregister_service(published[1])
        published[0].close()
    sys.exit(code)


if __name__ == "__main__":
    main()
