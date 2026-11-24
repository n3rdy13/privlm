# Laptop worker

Lets a Mac, Linux or Windows laptop join a PrivateLM phone cluster as a
**worker**, like a phone set to Worker. The master phone (iPhone or Android)
puts part of the model's layers on the laptop's GPU (or CPU) over Wi-Fi.

The worker is built from the app's own llama.cpp tree, so it speaks the same
RPC protocol (`rpc7`) as the phones. Rebuild it (`--rebuild`) whenever that
tree is updated, or the master will refuse to use the laptop.

## Requirements

- Python 3.8+, CMake 3.14+ and a C++ compiler (Xcode Command Line Tools on a
  Mac, `build-essential` on Linux, Visual Studio with C++ on Windows).
- Optional: `pip install zeroconf qrcode`. `zeroconf` lets the master's
  "Find nearby" see the laptop; `qrcode` prints the QR code in the terminal.

## Run

From the repository root:

```sh
python3 tools/laptop_worker/privatelm_worker.py              # Mac: Metal GPU; elsewhere: CPU
python3 tools/laptop_worker/privatelm_worker.py --gpu cuda   # NVIDIA GPU
python3 tools/laptop_worker/privatelm_worker.py --gpu vulkan # AMD / Intel GPU
python3 tools/laptop_worker/privatelm_worker.py --cpu        # CPU only
```

The first run builds the worker (a few minutes). Then it measures the free
memory and memory speed, starts serving on port 50052, and prints a QR code.
On the master phone, open **Phone cluster**, then tap **Find nearby** or scan
the code. You can also add the laptop by IP, but then the master doesn't know
how much memory it has and assumes 2 GB.

Other options: `--max-ram-gb 8` caps the memory offered to the master,
`--port` changes the port, `--threads` sets the CPU threads.

## Notes

- Allow incoming connections if the firewall asks (macOS, Windows).
- Keep the laptop awake and plugged in while it's sharing. If it sleeps, the
  master loses its layers mid-answer.
- The free memory is measured when the worker starts. Restart it after
  opening or closing big apps.
- Phones and laptop take turns on every token, and each hop crosses the
  Wi-Fi. Use 5 GHz Wi-Fi with the devices close to the router.
