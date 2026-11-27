"""Help writing image prompts, for people who'd rather not learn prompting.

- A builder: type the subject, click a style, lighting, framing and mood, and
  get a well-ordered Stable Diffusion prompt with a matching "Avoid" list.
- Settings presets for the model in use (picked from its file name).
- An AI helper: a small chat model (the Master tab's, or its own llama-server)
  turns a rough idea into a prompt, writes several different ideas for the
  Queue, or fixes a prompt from "what's wrong with this image".
"""

import json
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional, Tuple

import privatelm_worker as core

# ── Builder ─────────────────────────────────────────────────────────────────

PHOTO_NEGATIVE = ("cartoon, painting, illustration, drawing, anime, 3d render, cgi, deformed, "
                  "disfigured, bad anatomy, extra fingers, mutated hands, poorly drawn face, blurry, "
                  "lowres, jpeg artifacts, watermark, text, signature, oversaturated")
ART_NEGATIVE = ("photo, deformed, disfigured, bad anatomy, extra fingers, mutated hands, blurry, "
                "lowres, jpeg artifacts, watermark, text, signature")
PEOPLE_NEGATIVE = "asymmetrical eyes, crossed eyes, extra limbs"

# name: (lead words before the subject, quality words at the end, photographic?)
STYLES = {
    "Photo": ("photo of", "realistic, highly detailed, sharp focus", True),
    "Portrait photo": ("portrait photo of", "detailed skin texture, sharp focus on the eyes, "
                                            "shallow depth of field", True),
    "Cinematic film still": ("cinematic film still of", "dramatic composition, film grain, "
                                                        "color graded, anamorphic", True),
    "Analog film photo": ("analog film photo of", "Kodak Portra 400, film grain, soft faded colors", True),
    "Fashion editorial": ("fashion editorial photo of", "magazine quality, professional styling, "
                                                        "sharp focus", True),
    "Street photography": ("candid street photo of", "35mm, natural light, documentary style", True),
    "Landscape photo": ("landscape photo of", "wide angle, vivid colors, highly detailed, "
                                              "sharp focus", True),
    "Studio product photo": ("studio product photo of", "clean background, softbox lighting, "
                                                        "commercial photography", True),
    "Digital painting": ("digital painting of", "highly detailed, artstation, concept art", False),
    "Oil painting": ("oil painting of", "visible brush strokes, rich colors, classical", False),
    "Watercolor": ("watercolor painting of", "soft washes, paper texture, delicate", False),
    "Anime illustration": ("anime illustration of", "clean line art, vibrant colors, detailed", False),
    "3D render": ("3d render of", "octane render, soft global illumination, highly detailed", False),
    "Fantasy concept art": ("fantasy concept art of", "epic, highly detailed, dramatic", False),
}
LIGHTING = ["", "golden hour sunlight", "soft window light", "overcast soft light", "dramatic rim light",
            "neon lights at night", "studio softbox lighting", "candlelight", "blue hour",
            "harsh midday sun", "moonlight", "backlit, glowing edges"]
SHOTS = ["", "close-up", "headshot", "half-body shot", "full-body shot", "wide shot", "aerial view",
         "low angle", "macro close-up", "85mm lens, f/1.8, bokeh", "35mm lens", "symmetrical, centered"]
MOODS = ["", "warm and cozy", "moody and dark", "bright and cheerful", "mysterious", "serene and calm",
         "epic", "melancholic", "vintage", "dreamy"]
PEOPLE_WORDS = ("man", "woman", "girl", "boy", "person", "people", "portrait", "face", "child", "lady",
                "guy", "fisherman", "old", "couple", "family", "warrior", "king", "queen")


def build_prompt(subject: str, extra: str, style: str, lighting: str, shot: str, mood: str) -> Tuple[str, str]:
    """A prompt in the order SD 1.5 follows best (medium and subject, details,
    light, framing, mood, quality) and a matching negative prompt."""
    subject = " ".join(subject.split()).strip(" ,")
    lead, quality, photographic = STYLES.get(style, STYLES["Photo"])
    parts = [f"{lead} {subject}" if subject else lead.replace(" of", "")]
    for piece in (extra, lighting, shot, mood, quality):
        piece = " ".join(piece.split()).strip(" ,")
        if piece:
            parts.append(piece)
    negative = PHOTO_NEGATIVE if photographic else ART_NEGATIVE
    if any(w in subject.lower().split() for w in PEOPLE_WORDS) or style == "Portrait photo":
        negative += ", " + PEOPLE_NEGATIVE
    return ", ".join(parts), negative


def merge_negative(*lists: str, limit: int = 45) -> str:
    """Joins negative prompts, keeping each term once, in order."""
    seen, out = set(), []
    for text in lists:
        for term in text.split(","):
            term = " ".join(term.split()).strip(" .")
            if term and term.lower() not in seen:
                seen.add(term.lower())
                out.append(term)
    return ", ".join(out[:limit])


# ── Presets ─────────────────────────────────────────────────────────────────

# name: (sampler, steps, cfg, size, words in a model file name that pick it)
PRESETS = {
    "Photo-realistic (CyberRealistic, Realistic Vision…)":
        ("DPM++ 2M", 28, 6.0, "512 × 768", ("cyber", "realis", "photon", "epicphoto", "photo", "real")),
    "General SD 1.5": ("Euler a", 25, 7.0, "512 × 512", ()),
    "LCM / Turbo / Hyper models": ("LCM", 6, 1.5, "512 × 512", ("lcm", "turbo", "hyper", "lightning", "sdxs")),
    "Fast draft (check ideas quickly)": ("Euler a", 12, 7.0, "384 × 384", ()),
}


def preset_for(model_name: str) -> str:
    name = model_name.lower()
    for preset, (*_, words) in PRESETS.items():
        if any(w in name for w in words) and words:
            if preset.startswith("Photo") and any(w in name for w in PRESETS["LCM / Turbo / Hyper models"][4]):
                return "LCM / Turbo / Hyper models"
            return preset
    return "General SD 1.5"


# ── AI helper ───────────────────────────────────────────────────────────────

RECOMMENDED_URL = ("https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/"
                   "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
RECOMMENDED_NAME = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
RECOMMENDED_SIZE = "2.5 GB"
HELPER_PORT = 8091

SYSTEM_PROMPT = """You write prompts for Stable Diffusion 1.5 image models.
A good prompt is a comma-separated list of short phrases, at most 60 words: first the medium and \
the main subject with a few concrete visual details, then the setting, lighting, camera or framing, \
colors and mood, then a few quality words. Describe only what can be seen. No full sentences, no \
explanations, no names of living artists, no quotation marks.
The negative prompt is a comma-separated list of things to keep out of the image, at most 30 words.
{model_note}"""
PHOTO_NOTE = ("The model is photo-realistic, so use photographic words (camera, lens, light) unless "
              "the person asks for art.")


def _schema_single():
    return {"type": "object", "properties": {"prompt": {"type": "string"}, "negative": {"type": "string"}},
            "required": ["prompt", "negative"]}


def _schema_many(n: int):
    return {"type": "object", "properties": {
        "prompts": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n},
        "negative": {"type": "string"}}, "required": ["prompts", "negative"]}


class PromptAI:
    """A chat model for the helper: the Master tab's model if it's running,
    otherwise a small one this computer runs just for prompts."""

    def __init__(self, logs: queue.Queue, find_server, master_endpoint: Callable[[], Optional[tuple]],
                 no_window: int):
        self.logs = logs
        self.find_server = find_server
        self.master_endpoint = master_endpoint
        self.no_window = no_window
        self.server = None
        self.ready = False
        self.lock = threading.Lock()
        self.config_path = core.config_dir() / "prompt_helper.json"

    @property
    def model_path(self) -> str:
        try:
            return json.loads(self.config_path.read_text()).get("model", "")
        except (OSError, ValueError):
            return ""

    @model_path.setter
    def model_path(self, path: str):
        try:
            self.config_path.write_text(json.dumps({"model": path}))
        except OSError:
            pass

    def describe(self) -> str:
        if self.master_endpoint():
            return "Uses the model running on the Master tab."
        path = self.model_path
        if path and Path(path).exists():
            return f"Uses {Path(path).name}" + (" (loaded)" if self.ready else " (loads on first use)")
        return "Needs a small chat model: download the recommended one, or choose a .gguf."

    def _endpoint(self, status: Callable[[str], None]) -> Tuple[str, Optional[str], str]:
        master = self.master_endpoint()
        if master:
            return master
        with self.lock:
            if self.server and self.server.poll() is None and self.ready:
                return f"http://127.0.0.1:{HELPER_PORT}", None, "helper"
            path = self.model_path
            if not path or not Path(path).exists():
                raise RuntimeError("Choose or download a chat model for the prompt helper first.")
            status("Loading the prompt helper model (first time only)…")
            binary = self.find_server(status)
            threads = max(2, (os.cpu_count() or 4) // 2)
            args = [str(binary), "-m", path, "--host", "127.0.0.1", "--port", str(HELPER_PORT),
                    "-c", "4096", "-t", str(threads), "-a", "helper", "--no-webui"]
            self.logs.put("Prompt helper: " + " ".join(args))
            self.server = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           bufsize=1, creationflags=self.no_window, encoding="utf-8",
                                           errors="replace")
            threading.Thread(target=self._read, args=(self.server,), daemon=True).start()
            deadline = time.time() + 300
            while time.time() < deadline:
                if self.server.poll() is not None:
                    raise RuntimeError("The prompt helper model stopped while loading. See the log.")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{HELPER_PORT}/health", timeout=2) as r:
                        if r.status == 200:
                            self.ready = True
                            return f"http://127.0.0.1:{HELPER_PORT}", None, "helper"
                except (urllib.error.URLError, OSError):
                    pass
                time.sleep(1)
            raise RuntimeError("The prompt helper model took too long to load.")

    def _read(self, proc):
        for line in proc.stdout:
            self.logs.put("[helper] " + line.rstrip())

    def _chat(self, user: str, schema: dict, photo: bool, temperature: float,
              status: Callable[[str], None]) -> dict:
        url, key, model = self._endpoint(status)
        status("Writing…")
        system = SYSTEM_PROMPT.format(model_note=PHOTO_NOTE if photo else "")
        body = {"model": model, "temperature": temperature, "max_tokens": 900,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "prompts", "schema": schema}}}
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
        text = data["choices"][0]["message"].get("content") or ""
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise RuntimeError("The model didn't answer with a prompt. Try again.")
        return json.loads(text[start:end + 1])

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(str(text).replace('"', "").split()).strip(" ,.")

    def improve(self, idea: str, builder_prompt: str, photo: bool, status) -> Tuple[str, str]:
        user = (f"My idea: {idea or '(none, use the draft)'}\n"
                f"Draft prompt from the options I picked: {builder_prompt}\n"
                "Write one strong prompt that keeps my idea and the style I picked, adding concrete "
                "visual details.")
        out = self._chat(user, _schema_single(), photo, 0.7, status)
        return self._clean(out["prompt"]), self._clean(out["negative"])

    def ideas(self, idea: str, builder_prompt: str, n: int, photo: bool, status) -> Tuple[List[str], str]:
        user = (f"My idea: {idea or builder_prompt}\nDraft prompt: {builder_prompt}\n"
                f"Write {n} clearly different prompts for this idea: vary the setting, angle, lighting, "
                "time of day, colors and mood, but keep the subject and style.")
        out = self._chat(user, _schema_many(n), photo, 1.0, status)
        prompts = [self._clean(p) for p in out["prompts"] if self._clean(p)]
        return prompts[:n], self._clean(out["negative"])

    def fix(self, prompt: str, negative: str, complaint: str, photo: bool, status) -> Tuple[str, str]:
        user = (f"This prompt made an image I don't like:\n{prompt}\nNegative prompt: {negative}\n"
                f"What's wrong: {complaint}\nWrite an improved prompt and negative prompt that fix "
                "this but keep what the image is of.")
        out = self._chat(user, _schema_single(), photo, 0.6, status)
        return self._clean(out["prompt"]), self._clean(out["negative"])

    def download(self, progress: Callable[[float, str], None]) -> Path:
        folder = core.config_dir() / "models"
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / RECOMMENDED_NAME
        if dest.exists() and dest.stat().st_size > 2_000_000_000:
            return dest
        part = dest.with_suffix(".part")
        req = urllib.request.Request(RECOMMENDED_URL, headers={"User-Agent": "PrivateLM-Laptop"})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            last = 0.0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if time.time() - last > 0.5:
                    last = time.time()
                    progress(got / total if total else 0, f"{got / 1e9:.2f} of {total / 1e9:.2f} GB")
        if total and part.stat().st_size != total:
            raise RuntimeError("The download was cut off. Try again.")
        part.replace(dest)
        return dest

    def shutdown(self):
        proc, self.server = self.server, None
        self.ready = False
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


# ── Window ──────────────────────────────────────────────────────────────────

class HelperWindow:
    """Builder + AI helper. `apply(prompts, negative)` sends the result to the
    Images tab (several prompts: the Queue)."""

    def __init__(self, root: tk.Tk, ai: PromptAI, apply: Callable[[List[str], str], None],
                 selected: Callable[[], Optional[Tuple[str, str]]], current_model: Callable[[], str]):
        self.root = root
        self.ai = ai
        self.apply = apply
        self.selected = selected
        self.current_model = current_model
        self.busy = False
        win = tk.Toplevel(root)
        win.title("Prompt helper")
        win.transient(root)
        win.resizable(False, False)
        self.win = win
        self.build()
        self.rebuild()

    def build(self):
        f = ttk.Frame(self.win, padding=14)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Describe what you want in plain words, pick a few options, and let the "
                          "helper write the prompt.", style="Muted.TLabel", wraplength=520,
                  justify="left").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(f, text="Your idea").grid(row=1, column=0, sticky="nw", pady=3)
        self.idea = tk.Text(f, height=2, width=50, wrap="word",
                            font=("Segoe UI", 10) if os.name == "nt" else ("", 10))
        self.idea.grid(row=1, column=1, columnspan=2, sticky="ew", pady=3)
        self.idea.bind("<KeyRelease>", lambda _: self.rebuild())
        ttk.Label(f, text="e.g. an old fisherman mending his nets on a pier", style="Muted.TLabel")\
            .grid(row=2, column=1, columnspan=2, sticky="w")

        photo = any(w in self.current_model().lower() for w in PRESETS[next(iter(PRESETS))][4])
        self.style_var = tk.StringVar(value="Photo" if photo or not self.current_model() else "Digital painting")
        self.light_var, self.shot_var, self.mood_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.extra_var = tk.StringVar()
        rows = (("Style", self.style_var, list(STYLES)), ("Lighting", self.light_var, LIGHTING),
                ("Framing", self.shot_var, SHOTS), ("Mood", self.mood_var, MOODS))
        for r, (label, var, values) in enumerate(rows, 3):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=3)
            box = ttk.Combobox(f, textvariable=var, values=[v or "(any)" for v in values], width=30,
                               state="readonly")
            box.grid(row=r, column=1, sticky="w", pady=3)
            if not var.get():
                var.set("(any)")
            box.bind("<<ComboboxSelected>>", lambda _: self.rebuild())
        ttk.Label(f, text="Details").grid(row=7, column=0, sticky="w", pady=3)
        extra = ttk.Entry(f, textvariable=self.extra_var)
        extra.grid(row=7, column=1, columnspan=2, sticky="ew", pady=3)
        extra.bind("<KeyRelease>", lambda _: self.rebuild())
        ttk.Label(f, text="optional, e.g. red raincoat, rainy harbor", style="Muted.TLabel")\
            .grid(row=8, column=1, columnspan=2, sticky="w")

        # AI
        ai = ttk.LabelFrame(f, text="AI helper", padding=8)
        ai.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(10, 6))
        ai.columnconfigure(3, weight=1)
        self.improve_btn = ttk.Button(ai, text="✨ Improve with AI", command=self.do_improve)
        self.improve_btn.grid(row=0, column=0, sticky="w")
        self.ideas_btn = ttk.Button(ai, text="Give me ideas:", command=self.do_ideas)
        self.ideas_btn.grid(row=0, column=1, sticky="w", padx=(8, 4))
        self.n_var = tk.StringVar(value="8")
        ttk.Spinbox(ai, textvariable=self.n_var, from_=2, to=20, width=4).grid(row=0, column=2, sticky="w")
        ttk.Label(ai, text="What's wrong with the selected image?").grid(row=1, column=0, columnspan=4,
                                                                          sticky="w", pady=(8, 2))
        self.complaint_var = tk.StringVar()
        ttk.Entry(ai, textvariable=self.complaint_var).grid(row=2, column=0, columnspan=3, sticky="ew")
        self.fix_btn = ttk.Button(ai, text="Fix it", command=self.do_fix)
        self.fix_btn.grid(row=2, column=3, sticky="w", padx=(6, 0))
        ttk.Label(ai, text="e.g. too dark, face looks strange, want it more colorful", style="Muted.TLabel")\
            .grid(row=3, column=0, columnspan=4, sticky="w")
        self.model_var = tk.StringVar(value=self.ai.describe())
        ttk.Label(ai, textvariable=self.model_var, style="Muted.TLabel", wraplength=500, justify="left")\
            .grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 2))
        mrow = ttk.Frame(ai)
        mrow.grid(row=5, column=0, columnspan=4, sticky="w")
        self.download_btn = ttk.Button(mrow, text=f"Download recommended ({RECOMMENDED_SIZE})",
                                       command=self.do_download)
        self.download_btn.pack(side="left")
        ttk.Button(mrow, text="Choose a .gguf…", command=self.choose_model).pack(side="left", padx=(6, 0))
        self.progress = ttk.Progressbar(ai, mode="determinate", maximum=1.0)
        self.ai_status = tk.StringVar(value="")
        ttk.Label(ai, textvariable=self.ai_status, style="Muted.TLabel", wraplength=500, justify="left")\
            .grid(row=7, column=0, columnspan=4, sticky="w")

        # Result
        ttk.Label(f, text="Prompt").grid(row=10, column=0, sticky="nw", pady=3)
        self.result = tk.Text(f, height=6, width=50, wrap="word",
                              font=("Segoe UI", 10) if os.name == "nt" else ("", 10))
        self.result.grid(row=10, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Label(f, text="Several lines = several prompts (they go to the Queue).", style="Muted.TLabel")\
            .grid(row=11, column=1, columnspan=2, sticky="w")
        ttk.Label(f, text="Avoid").grid(row=12, column=0, sticky="w", pady=3)
        self.neg_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.neg_var).grid(row=12, column=1, columnspan=2, sticky="ew", pady=3)

        buttons = ttk.Frame(f)
        buttons.grid(row=13, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Use this prompt", command=self.use).pack(side="left")
        ttk.Button(buttons, text="Close", command=self.win.destroy).pack(side="left", padx=(8, 0))

    def choice(self, var: tk.StringVar) -> str:
        v = var.get()
        return "" if v == "(any)" else v

    def builder(self) -> Tuple[str, str]:
        return build_prompt(self.idea.get("1.0", "end"), self.extra_var.get(), self.style_var.get(),
                            self.choice(self.light_var), self.choice(self.shot_var), self.choice(self.mood_var))

    def rebuild(self):
        if self.busy:
            return
        prompt, negative = self.builder()
        self.set_result([prompt], negative)

    def set_ai_result(self, prompts: List[str], negative: str):
        """AI answers keep the style's standard negatives (bad anatomy, blurry…)."""
        self.set_result(prompts, merge_negative(negative, self.builder()[1]))

    def set_result(self, prompts: List[str], negative: str):
        self.result.delete("1.0", "end")
        self.result.insert("1.0", "\n".join(prompts))
        self.neg_var.set(negative)

    def photographic(self) -> bool:
        return STYLES.get(self.style_var.get(), STYLES["Photo"])[2]

    def run_ai(self, work: Callable, done: Callable):
        if self.busy:
            return
        self.busy = True
        for b in (self.improve_btn, self.ideas_btn, self.fix_btn):
            b.state(["disabled"])
        started = time.time()

        def status(msg):
            self.root.after(0, lambda: self.ai_status.set(msg))

        def thread():
            try:
                out, error = work(status), None
            except Exception as e:  # noqa: BLE001 - shown to the user
                out, error = None, str(e)
            self.root.after(0, lambda: finish(out, error))

        def finish(out, error):
            self.busy = False
            for b in (self.improve_btn, self.ideas_btn, self.fix_btn):
                b.state(["!disabled"])
            self.model_var.set(self.ai.describe())
            if error:
                self.ai_status.set(f"Couldn't: {error}")
            else:
                self.ai_status.set(f"Done in {time.time() - started:.0f} s. Edit it if you like, then "
                                   "Use this prompt.")
                done(out)

        threading.Thread(target=thread, daemon=True).start()

    def do_improve(self):
        idea = " ".join(self.idea.get("1.0", "end").split())
        draft, _ = self.builder()
        self.run_ai(lambda st: self.ai.improve(idea, draft, self.photographic(), st),
                    lambda out: self.set_ai_result([out[0]], out[1]))

    def do_ideas(self):
        idea = " ".join(self.idea.get("1.0", "end").split())
        draft, _ = self.builder()
        try:
            n = max(2, min(20, int(self.n_var.get())))
        except ValueError:
            n = 8
        self.run_ai(lambda st: self.ai.ideas(idea, draft, n, self.photographic(), st),
                    lambda out: self.set_ai_result(out[0], out[1]))

    def do_fix(self):
        sel = self.selected()
        complaint = self.complaint_var.get().strip()
        if not sel:
            messagebox.showinfo("Prompt helper", "Click an image in the gallery first, then say what's "
                                                 "wrong with it.", parent=self.win)
            return
        if not complaint:
            messagebox.showinfo("Prompt helper", "Say what's wrong with the image, e.g. \"too dark\".",
                                parent=self.win)
            return
        prompt, negative = sel
        self.run_ai(lambda st: self.ai.fix(prompt, negative, complaint, self.photographic(), st),
                    lambda out: self.set_result([out[0]], merge_negative(out[1], negative)))

    def do_download(self):
        if self.busy:
            return
        self.busy = True
        self.download_btn.state(["disabled"])
        self.progress.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(6, 2))

        def progress(frac, text):
            self.root.after(0, lambda: (self.progress.configure(value=frac),
                                        self.ai_status.set(f"Downloading… {text}")))

        def thread():
            try:
                path, error = self.ai.download(progress), None
            except Exception as e:  # noqa: BLE001
                path, error = None, str(e)
            self.root.after(0, lambda: finish(path, error))

        def finish(path, error):
            self.busy = False
            self.download_btn.state(["!disabled"])
            self.progress.grid_remove()
            if error:
                self.ai_status.set(f"Download failed: {error}")
                return
            self.ai.shutdown()
            self.ai.model_path = str(path)
            self.model_var.set(self.ai.describe())
            self.ai_status.set("Downloaded. Click Improve with AI.")

        threading.Thread(target=thread, daemon=True).start()

    def choose_model(self):
        path = filedialog.askopenfilename(parent=self.win, title="Choose a chat model",
                                          filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")])
        if path:
            self.ai.shutdown()
            self.ai.model_path = path
            self.model_var.set(self.ai.describe())

    def use(self):
        prompts = [p.strip() for p in self.result.get("1.0", "end").splitlines() if p.strip()]
        if not prompts:
            return
        self.apply(prompts, self.neg_var.get().strip())
        self.win.destroy()
