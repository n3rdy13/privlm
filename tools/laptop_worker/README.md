# Laptop app

New to image models? Start with the step-by-step guide:
[docs/first-image-guide.html](../../docs/first-image-guide.html) (open it in a browser).

Lets a laptop join a PrivateLM phone cluster in either role:

- **Master:** the laptop runs the model, with part of its layers on phones
  set to Worker, and serves it as an OpenAI-compatible API. Chat with it from
  the PrivateLM app (Custom API) or the laptop's browser.
- **Images:** the laptop makes images with a Stable Diffusion model
  (`.safetensors`, `.ckpt` or `.gguf`, such as CyberRealistic), like the
  app's image generation, with phones making images at the same time.
- **Worker:** the laptop lends its GPU (or CPU) to a master phone, like a
  phone set to Worker.

Both use llama.cpp at the release the app's own tree comes from (b11157), so
they speak the phones' RPC protocol (`rpc7`).

## Windows app

No Python or compiler needed:

1. On GitHub, open **Actions → Laptop app (Windows)**, pick the latest green
   run (or **Run workflow** to start one), and download
   **PrivateLM-Laptop-windows** under Artifacts. Unzip it.
2. Run `PrivateLM-Laptop.exe`. It isn't signed, so Windows SmartScreen may
   warn: click **More info → Run anyway**. When Windows Firewall asks, allow
   **private networks**.

### Laptop as master

1. On each phone, set **Phone cluster** to **Worker** and keep PrivateLM open.
2. On the **Master** tab, click **Browse…** and pick a `.gguf` model, then
   **Find nearby** to list the phones (or **Add by IP**). Click **Use** to
   include or leave out a phone.
3. Click **Start model**. When it says **Ready**, it shows a Base URL, API key
   and Model name.
4. In PrivateLM on your phone: **Models → Cloud → Custom API**, enter those
   three, and chat. Or click **Open chat in this computer's browser**.

The laptop keeps as many layers as fit in its free memory (it's faster and
has no Wi-Fi hop) and puts the rest on the phones, sized by each phone's free
memory and speed. A model that fits on the laptop runs there alone, because
every word would otherwise cross the Wi-Fi to each phone; tick **Use the
phones even if the model fits here** to share it anyway. The master runs on
the laptop's CPU.

### Images

On the **Images** tab, click **Browse…** and pick the model, then **Load
model**. Write a prompt and click **Generate**. Each image is saved to
`Pictures\PrivateLM` and added to the gallery; click a thumbnail to see it
big. Size, steps, CFG, sampler and seed work as in other Stable Diffusion
apps; SD 1.5 models like CyberRealistic look best at 512 × 512 or 512 × 768
with 20–30 steps. **Automatic** uses the GPU (Vulkan) if the laptop has one,
else the CPU; integrated graphics can be faster or slower than the CPU, so
try both.

**Help with prompts.** Click **✨ Prompt helper…**:

- Type your idea in plain words ("an old fisherman mending nets on a pier"),
  then pick a **Style** (photo, portrait, cinematic, film, painting…),
  **Lighting**, **Framing** and **Mood**. The helper builds a well-ordered
  prompt and a matching **Avoid** list as you click.
- **✨ Improve with AI** rewrites it into a stronger prompt with concrete
  details. **Give me ideas** writes several different versions (setting,
  angle, light, mood) and sends them to the Queue, so the laptop and phones
  make one of each and you keep the best.
- Didn't like an image? Click it in the gallery, write what's wrong ("too
  dark", "face looks strange") and click **Fix it**.
- **Use this prompt** puts the result in the Images tab.

The AI helper uses the chat model running on the Master tab if there is
one, otherwise a small model of its own: click **Download recommended**
(Qwen3 4B Instruct, 2.5 GB, about 10–20 seconds per answer on a laptop CPU)
or **Choose a .gguf…**. The builder works without any model.

**Settings** (next to the helper button) sets sampler, steps, CFG and size
for the kind of model: choosing a model picks one automatically, e.g.
*Photo-realistic* for CyberRealistic (DPM++ 2M, 28 steps, CFG 6,
512 × 768).

**Start from an image (img2img).** Tick **Start from an image**, then
**Choose…** a picture (PNG, JPEG, WebP) or click an image in the gallery and
**Use selected**. The prompt says what you want it to become, and **Change**
sets how much: a little (around 0.3) keeps the picture and adjusts details
and style, a lot (0.7 and up) keeps only its rough layout and colors. The
picture is cropped to the chosen size's shape. Variations and the Queue work
the same way, so you can make several versions of one picture at once, or
refine a gallery favorite. Phones need the updated PrivateLM app for this
(older ones are skipped), and an iPhone's image Backend must be CPU (Settings
→ Image generation), not GPU (Apple NPU).

**LoRAs.** Click **LoRAs…**, choose the folder where you keep LoRA files
(`.safetensors`), tick the ones to use and set each one's weight (how
strongly it applies; 0.6–1.0 is typical, negative values push away from
it). Many LoRAs need a trigger word in the prompt: check the page you
downloaded it from. If you change the folder while the model is loaded,
Unload and Load the model again. Phones get the LoRAs a run needs sent to
them over the Wi-Fi when you click Generate, once per LoRA (they keep them
for next time), and need the updated PrivateLM app. LoRAs must match the
model family (SD 1.5 LoRAs for SD 1.5 models like CyberRealistic).

**Phones make images too.** On each phone, load the same image model in
PrivateLM, then open **Phone cluster** and turn on **Images for a laptop**.
On the laptop, click **Find nearby** under **Devices making images** (or
**Add by IP**). Every ticked device takes the next image as soon as it's
free, so the laptop and phones work at the same time and the fastest make
the most. Two ways to fill the queue:

- **Variations of one prompt:** *Images* copies of one prompt, each with its
  own seed. Seed −1 picks random seeds; a number N gives N, N+1, … so you can
  make the same set again.
- **Queue: one prompt per line:** each line is a prompt, made *Each* times.

The laptop doesn't need its own model loaded to use the phones. Phones
render at most 512 × 768. Each phone uses the model loaded in its own app,
shown in the Model column, so load the same one everywhere for matching
results. Anyone on the Wi-Fi can send a phone jobs while Images for a laptop
is on.

### Laptop as worker

On the **Worker** tab, pick the device and how much memory to offer, then
click **Start sharing**. On the master phone, open **Phone cluster** and tap
**Find nearby**, or scan the code in the window. The window shows when the
master connects. It uses the GPU through Vulkan (NVIDIA, AMD and Intel, with
an up-to-date driver) and falls back to the CPU.

In both roles the computer is kept from sleeping while it runs; also set
closing the lid to "Do nothing" in the power settings if you want to close
it. To run the window from source instead: `pip install zeroconf qrcode
psutil`, then `python tools/laptop_worker/worker_gui.py` (the first master
start downloads and builds llama-server, which needs git and CMake).

## Command line (worker only)

### Requirements

- Python 3.8+, CMake 3.14+ and a C++ compiler (Xcode Command Line Tools on a
  Mac, `build-essential` on Linux, Visual Studio with C++ on Windows).
- Optional: `pip install zeroconf qrcode`. `zeroconf` lets the master's
  "Find nearby" see the laptop; `qrcode` prints the QR code in the terminal.

### Run

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
