import os
import time
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# ---------------------------------------------------------------------------
# API keys — HARD-CODED as requested.
# Replace the placeholder below with your real Intron key.
# ---------------------------------------------------------------------------
INTRON_API_KEY = "pbkdf2:sha256:260000$K2m4MYap9K9Aiuna$34d280764f140469b470d2b8ea8a2c9a45872dabfc904127b5fd0c3f482426ce"

# ElevenLabs key can still come from env (or hard-code it here too if you want).
ELEVENLABS_API_KEY = "sk_c5331fec710acfe054e18f9ec7b7184c3d9b81c28528f47f"

# ---------------------------------------------------------------------------
# Language routing
#   - ElevenLabs Scribe v2: ha, ig, sw, xh, zu
#   - Intron:               yo (Yoruba), tw (Twi)
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

INTRON_UPLOAD_SYNC_URL = "https://infer.voice.intron.io/file/v1/upload/sync"
INTRON_STATUS_URL = "https://infer.voice.intron.io/file/v1/status/{file_id}"


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

    if lang in INTRON_LANGUAGES:
        return transcribe_with_intron(audio_file, lang)
    else:
        return transcribe_with_elevenlabs(audio_file, lang)


# ---------------------------------------------------------------------------
# ElevenLabs — unchanged from your original
# ---------------------------------------------------------------------------
def transcribe_with_elevenlabs(audio_file, lang):
    resp = requests.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        data={
            "model_id": "scribe_v2",
            "language_code": lang,
            "tag_audio_events": False,
        },
        files={
            "file": (
                "recording.webm",
                audio_file.read(),
                audio_file.content_type or "audio/webm",
            )
        },
    )

    print("ElevenLabs status:", resp.status_code)
    print("ElevenLabs response:", resp.text)

    if resp.status_code != 200:
        return (
            jsonify(
                {
                    "error": f"ElevenLabs API error: {resp.status_code}",
                    "detail": resp.text,
                }
            ),
            500,
        )

    data = resp.json()
    return jsonify(
        {
            "text": data.get("text", ""),
            "language": data.get("language_code", lang),
        }
    )


# ---------------------------------------------------------------------------
# Intron — sync endpoint, with polling fallback if it 408s
# ---------------------------------------------------------------------------
def transcribe_with_intron(audio_file, lang):
    audio_bytes = audio_file.read()
    content_type = audio_file.content_type or "audio/webm"
    filename = audio_file.filename or "recording.webm"

    headers = {"Authorization": f"Bearer {INTRON_API_KEY}"}
    data = {
        "audio_file_name": filename,
        "use_language_asr_input": lang,
    }
    files = {"audio_file_blob": (filename, audio_bytes, content_type)}

    try:
        resp = requests.post(
            INTRON_UPLOAD_SYNC_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=90,
        )
    except requests.RequestException as e:
        return jsonify({"error": f"Intron request failed: {e}"}), 500

    print("Intron status:", resp.status_code)
    print("Intron response:", resp.text)

    # Happy path: transcript returned directly.
    if resp.status_code == 200:
        payload = resp.json()
        transcript = (payload.get("data") or {}).get("audio_transcript", "") or ""
        return jsonify({"text": transcript, "language": lang})

    # Sync timed out — we still get a file_id, so poll the status endpoint.
    if resp.status_code == 408:
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        file_id = (payload.get("data") or {}).get("file_id")
        if not file_id:
            return (
                jsonify(
                    {
                        "error": "Intron sync timed out and no file_id was returned",
                        "detail": resp.text,
                    }
                ),
                500,
            )
        return poll_intron_until_done(file_id, lang)

    # Anything else is an error.
    return (
        jsonify(
            {
                "error": f"Intron API error: {resp.status_code}",
                "detail": resp.text,
            }
        ),
        500,
    )


def poll_intron_until_done(file_id, lang, max_attempts=30, delay_seconds=2):
    headers = {"Authorization": f"Bearer {INTRON_API_KEY}"}
    url = INTRON_STATUS_URL.format(file_id=file_id)

    for attempt in range(max_attempts):
        time.sleep(delay_seconds)
        try:
            resp = requests.get(url, headers=headers, timeout=30)
        except requests.RequestException as e:
            return jsonify({"error": f"Intron status request failed: {e}"}), 500

        print(f"Intron status poll #{attempt + 1}:", resp.status_code, resp.text[:200])

        if resp.status_code != 200:
            return (
                jsonify(
                    {
                        "error": f"Intron status error: {resp.status_code}",
                        "detail": resp.text,
                    }
                ),
                500,
            )

        payload = resp.json()
        data = payload.get("data") or {}
        status = data.get("processing_status")

        if status == "FILE_TRANSCRIBED":
            return jsonify(
                {"text": data.get("audio_transcript", ""), "language": lang}
            )
        if status == "FILE_PROCESSING_FAILED":
            return jsonify({"error": "Intron processing failed", "detail": payload}), 500
        # Otherwise: FILE_QUEUED / FILE_PENDING / FILE_PROCESSING — keep polling.

    return jsonify({"error": "Intron transcription timed out after polling"}), 504


if __name__ == "__main__":
    app.run(debug=True, port=5000)