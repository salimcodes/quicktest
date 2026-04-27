import os
import time
import tempfile
import threading
import warnings
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
ELEVENLABS_API_KEY = "59c79ea1678ff809d9da6aaf6f0743181e052cee1128713b16f99f6cc0ee5262"

# Intron — https://voice.intron.io → Profile → API credentials
INTRON_API_KEY = os.environ.get(
    "INTRON_API_KEY",
    "intr_8923ba2368282f14b3435bbbff0b2684e293924a1ce96dff",
)

# HuggingFace — ONLY needed for the gated Yoruba model (NCAIR1/Yoruba-ASR).
#   1. Accept terms at https://huggingface.co/NCAIR1/Yoruba-ASR
#   2. Create a READ token at https://huggingface.co/settings/tokens
#   3. Paste below or set HF_TOKEN env var
HF_TOKEN = os.environ.get("HF_TOKEN", "LOGAN_ADDS_HERE")

# ---------------------------------------------------------------------------
# Intron endpoints
# ---------------------------------------------------------------------------
INTRON_SYNC_URL = "https://infer.voice.intron.io/file/v1/upload/sync"
INTRON_STATUS_URL = "https://infer.voice.intron.io/file/v1/status/{file_id}"

# ---------------------------------------------------------------------------
# Provider routing
#
# Dropdown values are "<provider>:<langcode>" so the same language can appear
# under multiple providers. Examples:
#   "eleven:ha"  -> ElevenLabs, Hausa
#   "intron:yo"  -> Intron, Yoruba
#   "hf:yo"      -> HuggingFace local, Yoruba
# ---------------------------------------------------------------------------
ELEVENLABS_LANGUAGES = {
    "ha": "Hausa", "ig": "Igbo", "sw": "Swahili", "xh": "Xhosa", "zu": "Zulu",
}
INTRON_LANGUAGES = {"yo": "Yoruba", "tw": "Twi"}
HF_LANGUAGES = {"yo": "Yoruba", "tw": "Twi"}

HF_MODELS = {
    "yo": "NCAIR1/Yoruba-ASR",                     # gated
    "tw": "dkt-py-bot/Whisper-FineTuned-DL-Twi",   # open
}

WHISPER_LANGUAGE_HINTS = {
    "yo": "yoruba",
    "tw": "twi",
}


def parse_selection(selection):
    if not selection or ":" not in selection:
        return None, None
    provider, lang = selection.split(":", 1)
    return provider, lang


# ---------------------------------------------------------------------------
# Lazy HF pipeline cache
# ---------------------------------------------------------------------------
_pipelines = {}
_pipeline_lock = threading.Lock()


def get_pipeline(lang):
    if lang in _pipelines:
        return _pipelines[lang]

    with _pipeline_lock:
        if lang in _pipelines:
            return _pipelines[lang]

        from transformers import pipeline as hf_pipeline

        model_id = HF_MODELS[lang]
        print(f"[init] Loading {model_id} — first run downloads ~1 GB, please wait …")

        kwargs = {"task": "automatic-speech-recognition", "model": model_id}
        if lang == "yo" and HF_TOKEN and not HF_TOKEN.startswith("PASTE_"):
            kwargs["token"] = HF_TOKEN

        pipe = hf_pipeline(**kwargs)
        _configure_asr_pipeline(pipe)
        _pipelines[lang] = pipe
        print(f"[init] {model_id} ready ✓")
        return pipe


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    selection = request.form.get("selection", "eleven:ha")
    provider, lang = parse_selection(selection)

    if provider not in {"eleven", "intron", "hf"}:
        return jsonify({"error": f"Unknown provider in selection: {selection}"}), 400

    audio_bytes = audio_file.read()
    content_type = audio_file.content_type or "audio/webm"
    filename = audio_file.filename or "recording.webm"

    if provider == "eleven":
        if lang not in ELEVENLABS_LANGUAGES:
            return jsonify({"error": f"ElevenLabs does not handle language: {lang}"}), 400
        return transcribe_with_elevenlabs(audio_bytes, content_type, filename, lang)

    if provider == "intron":
        if lang not in INTRON_LANGUAGES:
            return jsonify({"error": f"Intron does not handle language: {lang}"}), 400
        return transcribe_with_intron(audio_bytes, content_type, filename, lang)

    if provider == "hf":
        if lang not in HF_LANGUAGES:
            return jsonify({"error": f"HuggingFace does not handle language: {lang}"}), 400
        return transcribe_local(audio_bytes, lang, filename)


# ---------------------------------------------------------------------------
# ElevenLabs
# ---------------------------------------------------------------------------
def transcribe_with_elevenlabs(audio_bytes, content_type, filename, lang):
    resp = requests.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        data={
            "model_id": "scribe_v2",
            "language_code": lang,
            "tag_audio_events": "false",
        },
        files={"file": (filename, audio_bytes, content_type)},
    )

    print(f"[ElevenLabs] status={resp.status_code} body={resp.text[:200]}")

    if resp.status_code != 200:
        return jsonify({
            "error": f"ElevenLabs API error: {resp.status_code}",
            "detail": resp.text,
        }), 500

    data = resp.json()
    return jsonify({
        "text": data.get("text", ""),
        "language": data.get("language_code", lang),
        "provider": "ElevenLabs Scribe v2",
    })


# ---------------------------------------------------------------------------
# Intron
# ---------------------------------------------------------------------------
def transcribe_with_intron(audio_bytes, content_type, filename, lang):
    if not INTRON_API_KEY or INTRON_API_KEY.startswith("PASTE_"):
        return jsonify({"error": "INTRON_API_KEY is not set"}), 500

    headers = {"Authorization": f"Bearer {INTRON_API_KEY}"}
    form_data = {"audio_file_name": filename, "use_language_asr_input": lang}
    files = {"audio_file_blob": (filename, audio_bytes, content_type)}

    try:
        resp = requests.post(
            INTRON_SYNC_URL, headers=headers, data=form_data, files=files, timeout=120,
        )
    except requests.RequestException as e:
        return jsonify({"error": f"Intron request failed: {e}"}), 500

    print(f"[Intron sync] status={resp.status_code} body={resp.text[:300]}")

    if resp.status_code == 200:
        data = (safe_json(resp).get("data") or {})
        return jsonify({
            "text": data.get("audio_transcript", ""),
            "language": lang,
            "provider": "Intron",
        })

    if resp.status_code == 408:
        file_id = (safe_json(resp).get("data") or {}).get("file_id")
        if not file_id:
            return jsonify({
                "error": "Intron sync timed out and no file_id was returned",
                "detail": resp.text,
            }), 500
        return poll_intron(file_id, lang)

    return jsonify({
        "error": f"Intron API error: {resp.status_code}",
        "detail": resp.text,
    }), 500


def poll_intron(file_id, lang, max_attempts=45, delay_seconds=2):
    headers = {"Authorization": f"Bearer {INTRON_API_KEY}"}
    url = INTRON_STATUS_URL.format(file_id=file_id)

    for attempt in range(1, max_attempts + 1):
        time.sleep(delay_seconds)
        try:
            resp = requests.get(url, headers=headers, timeout=30)
        except requests.RequestException as e:
            return jsonify({"error": f"Intron status request failed: {e}"}), 500

        print(f"[Intron poll #{attempt}] status={resp.status_code} body={resp.text[:200]}")

        if resp.status_code != 200:
            return jsonify({
                "error": f"Intron status error: {resp.status_code}",
                "detail": resp.text,
            }), 500

        data = (safe_json(resp).get("data") or {})
        status = data.get("processing_status")

        if status == "FILE_TRANSCRIBED":
            return jsonify({
                "text": data.get("audio_transcript", ""),
                "language": lang,
                "provider": "Intron",
            })
        if status == "FILE_PROCESSING_FAILED":
            return jsonify({"error": "Intron processing failed", "detail": data}), 500

    return jsonify({"error": "Intron transcription timed out after polling"}), 504


# ---------------------------------------------------------------------------
# HuggingFace local Whisper
# ---------------------------------------------------------------------------
def transcribe_local(audio_bytes, lang, filename="recording.webm"):
    if lang == "yo" and (not HF_TOKEN or HF_TOKEN.startswith("PASTE_")):
        return jsonify({
            "error": (
                "HF_TOKEN is not set. The Yoruba model is gated.\n\n"
                "Fix: 1) Accept terms at https://huggingface.co/NCAIR1/Yoruba-ASR  "
                "2) Create a READ token at https://huggingface.co/settings/tokens  "
                "3) Set HF_TOKEN in app.py or as an env var, then restart Flask."
            )
        }), 500

    try:
        import librosa  # noqa: F401  (used inside _load_audio_16k_mono for resampling)
        import soundfile  # noqa: F401
    except ImportError:
        return jsonify({
            "error": "Missing audio libraries. Run: pip install librosa soundfile"
        }), 500

    suffix = os.path.splitext(filename or "recording.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        pipe = get_pipeline(lang)
    except Exception as e:
        try: os.unlink(tmp_path)
        except OSError: pass
        print(f"[hf/{lang}] model load failed: {e}")
        msg = str(e)
        if any(s in msg.lower() for s in ["gated", "401", "403", "access", "forbidden"]):
            msg += (
                "\n\nThis looks like a HuggingFace access error. Make sure you've accepted "
                "terms at https://huggingface.co/NCAIR1/Yoruba-ASR and that HF_TOKEN is valid."
            )
        return jsonify({"error": f"Failed to load model: {msg}"}), 500

    try:
        audio_array = _load_audio_16k_mono(tmp_path)
        duration = len(audio_array) / 16000
        print(f"[hf/{lang}] audio loaded: {len(audio_array)} samples ({duration:.1f}s)")

        result = _run_asr_pipeline(pipe, audio_array, lang)
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        print(f"[hf/{lang}] transcription: {text!r}")
        return jsonify({
            "text": text,
            "language": lang,
            "provider": f"HuggingFace ({HF_MODELS[lang]})",
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Transcription failed: {type(e).__name__}: {e}"}), 500
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass


def _configure_asr_pipeline(pipe):
    model_config = getattr(getattr(pipe, "model", None), "config", None)
    if getattr(model_config, "model_type", None) != "whisper":
        return

    generation_config = getattr(getattr(pipe, "model", None), "generation_config", None)
    if generation_config is None:
        return

    generation_config.forced_decoder_ids = None


def _run_asr_pipeline(pipe, audio_array, lang):
    model_config = getattr(getattr(pipe, "model", None), "config", None)
    if getattr(model_config, "model_type", None) != "whisper":
        result = pipe(audio_array)
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        if _looks_degenerate_transcript(text):
            raise RuntimeError("ASR produced a repetitive loop instead of a stable transcript.")
        return result

    attempts = []

    primary_kwargs = {
        "task": "transcribe",
        "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.15,
    }
    language_hint = WHISPER_LANGUAGE_HINTS.get(lang)
    if language_hint:
        primary_kwargs["language"] = language_hint

    attempts.append(primary_kwargs)
    attempts.append({
        "task": "transcribe",
        "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.15,
    })

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*custom logits processor.*will take precedence.*",
        )
        last_error = None
        for generate_kwargs in attempts:
            try:
                result = pipe(audio_array, generate_kwargs=generate_kwargs)
            except ValueError as exc:
                if "language" in str(exc).lower() and "language" in generate_kwargs:
                    last_error = exc
                    continue
                raise

            text = result.get("text", "") if isinstance(result, dict) else str(result)
            cleaned = _clean_transcript_text(text)
            if _looks_degenerate_transcript(cleaned):
                last_error = RuntimeError(
                    f"ASR produced a repetitive loop with settings: {generate_kwargs}"
                )
                continue

            if isinstance(result, dict):
                result["text"] = cleaned
            return result

        raise last_error or RuntimeError("ASR failed to produce a stable transcript.")


def _clean_transcript_text(text):
    words = text.split()
    if not words:
        return text.strip()

    cleaned = []
    for word in words:
        if len(cleaned) >= 3 and cleaned[-1] == cleaned[-2] == cleaned[-3] == word:
            continue
        cleaned.append(word)
    return " ".join(cleaned).strip()


def _looks_degenerate_transcript(text):
    words = text.split()
    if len(words) < 12:
        return False

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.2:
        return True

    max_run = 1
    run = 1
    for idx in range(1, len(words)):
        if words[idx] == words[idx - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    if max_run >= 6:
        return True

    phrase = words[:3]
    if len(phrase) == 3:
        repeats = 0
        for idx in range(0, len(words) - 2, 3):
            if words[idx:idx + 3] == phrase:
                repeats += 1
            else:
                break
        if repeats >= 4:
            return True

    return False


def _load_audio_16k_mono(path):
    """Decode any audio file to a 16 kHz mono float32 numpy array.

    Tries soundfile first (works for WAV/FLAC/OGG). Falls back to ffmpeg via
    subprocess for formats like WebM. Surfaces a clear error if ffmpeg is missing.
    """
    import numpy as np
    ext = os.path.splitext(path)[1].lower()
    ffmpeg_first_exts = {".webm", ".m4a", ".mp4", ".aac", ".opus", ".mp3"}

    if ext not in ffmpeg_first_exts:
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            data = data.mean(axis=1)  # mono
            if sr != 16000:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            return data.astype(np.float32)
        except Exception as sf_err:
            print(f"[audio] soundfile failed ({sf_err}); falling back to ffmpeg")
    else:
        print(f"[audio] using ffmpeg for {ext} input")

    # ffmpeg fallback — handles WebM, M4A, etc.
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. Install it and restart Flask. "
            "Windows: https://www.gyan.dev/ffmpeg/builds/ (add bin/ to PATH). "
            "macOS: brew install ffmpeg. Ubuntu: sudo apt install ffmpeg."
        )

    try:
        proc = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", path,
             "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
            capture_output=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg decode failed: {e.stderr.decode(errors='ignore')}") from e

    import numpy as np
    return np.frombuffer(proc.stdout, dtype=np.float32)


def safe_json(resp):
    try:
        return resp.json()
    except ValueError:
        return {}


if __name__ == "__main__":
    app.run(debug=True, port=5000)
