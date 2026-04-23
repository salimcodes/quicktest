import os
import time
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
ELEVENLABS_API_KEY = "59c79ea1678ff809d9da6aaf6f0743181e052cee1128713b16f99f6cc0ee5262"

# Get your Intron key from https://voice.intron.io  (Profile → API credentials)
INTRON_API_KEY = os.environ.get(
    "INTRON_API_KEY",
    "intr_8923ba2368282f14b3435bbbff0b2684e293924a1ce96dff",
)

# ---------------------------------------------------------------------------
# Intron endpoints
# ---------------------------------------------------------------------------
INTRON_SYNC_URL = "https://infer.voice.intron.io/file/v1/upload/sync"
INTRON_STATUS_URL = "https://infer.voice.intron.io/file/v1/status/{file_id}"

# ---------------------------------------------------------------------------
# Language routing
#   ElevenLabs Scribe v2 : ha, ig, sw, xh, zu
#   Intron file API       : yo, tw
# ---------------------------------------------------------------------------
ELEVENLABS_LANGUAGES = {
    "ha": "Hausa",
    "ig": "Igbo",
    "sw": "Swahili",
    "xh": "Xhosa",
    "zu": "Zulu",
}

INTRON_LANGUAGES = {
    "yo": "Yoruba",
    "tw": "Twi",
}

SUPPORTED_LANGUAGES = {**ELEVENLABS_LANGUAGES, **INTRON_LANGUAGES}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    lang = request.form.get("language", "ha")

    if lang not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"Unsupported language: {lang}"}), 400

    audio_bytes = audio_file.read()
    content_type = audio_file.content_type or "audio/webm"
    filename = audio_file.filename or "recording.webm"

    if lang in INTRON_LANGUAGES:
        return transcribe_with_intron(audio_bytes, content_type, filename, lang)
    else:
        return transcribe_with_elevenlabs(audio_bytes, content_type, filename, lang)


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
        return (
            jsonify({"error": f"ElevenLabs API error: {resp.status_code}", "detail": resp.text}),
            500,
        )

    data = resp.json()
    return jsonify({"text": data.get("text", ""), "language": data.get("language_code", lang)})


# ---------------------------------------------------------------------------
# Intron — sync upload, fall back to polling on 408
# ---------------------------------------------------------------------------
def transcribe_with_intron(audio_bytes, content_type, filename, lang):
    if not INTRON_API_KEY or INTRON_API_KEY.startswith("PASTE_"):
        return jsonify({"error": "INTRON_API_KEY is not set in app.py / env"}), 500

    headers = {"Authorization": f"Bearer {INTRON_API_KEY}"}
    form_data = {
        "audio_file_name": filename,
        "use_language_asr_input": lang,
    }
    files = {"audio_file_blob": (filename, audio_bytes, content_type)}

    try:
        resp = requests.post(
            INTRON_SYNC_URL,
            headers=headers,
            data=form_data,
            files=files,
            timeout=120,  # sync API can take up to 60s itself
        )
    except requests.RequestException as e:
        return jsonify({"error": f"Intron request failed: {e}"}), 500

    print(f"[Intron sync] status={resp.status_code} body={resp.text[:300]}")

    # 200 → transcript returned immediately
    if resp.status_code == 200:
        payload = safe_json(resp)
        data = payload.get("data") or {}
        return jsonify({
            "text": data.get("audio_transcript", ""),
            "language": lang,
        })

    # 408 → still processing, but we got a file_id we can poll
    if resp.status_code == 408:
        payload = safe_json(resp)
        file_id = (payload.get("data") or {}).get("file_id")
        if not file_id:
            return jsonify({
                "error": "Intron sync timed out and no file_id was returned",
                "detail": resp.text,
            }), 500
        return poll_intron(file_id, lang)

    # Anything else is a real error
    return jsonify({
        "error": f"Intron API error: {resp.status_code}",
        "detail": resp.text,
    }), 500


def poll_intron(file_id, lang, max_attempts=45, delay_seconds=2):
    """Poll the status endpoint until the file is transcribed or fails."""
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

        payload = safe_json(resp)
        data = payload.get("data") or {}
        status = data.get("processing_status")

        if status == "FILE_TRANSCRIBED":
            return jsonify({
                "text": data.get("audio_transcript", ""),
                "language": lang,
            })
        if status == "FILE_PROCESSING_FAILED":
            return jsonify({
                "error": "Intron processing failed",
                "detail": payload,
            }), 500
        # Otherwise keep polling: FILE_QUEUED / FILE_PENDING / FILE_PROCESSING

    return jsonify({"error": "Intron transcription timed out after polling"}), 504


def safe_json(resp):
    try:
        return resp.json()
    except ValueError:
        return {}


if __name__ == "__main__":
    app.run(debug=True, port=5000)