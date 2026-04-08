import os
import json
import logging
import time
import uuid
from functools import wraps

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from groq import Groq

from diarisation import SpeakerDiarizer, DiarizationError
from metrics import compute_all_metrics, compute_final_scores, generate_explanations

# ---------------------------
# LOGGING — replaces all print()
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------
# STARTUP VALIDATION
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# FLASK + CONFIG
# ---------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}

# Single shared diarizer instance — models load once at startup.
# num_speakers is now passed per-request via diarize(), so this singleton
# can correctly handle different speaker counts across concurrent calls.
diarizer = SpeakerDiarizer()


# ---------------------------
# RETRY DECORATOR
# Wraps any function with exponential backoff.
# Only retries on the listed exception types.
# ---------------------------
def with_retry(max_attempts=3, initial_delay=2.0, backoff=2.0, exceptions=(Exception,)):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            f"{fn.__name__} failed after {max_attempts} attempts: {exc}"
                        )
                        raise
                    logger.warning(
                        f"{fn.__name__} attempt {attempt}/{max_attempts} failed: {exc}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator


# ---------------------------
# HELPERS
# ---------------------------
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def error_response(message: str, status: int = 400):
    """Consistent JSON error envelope for all failure paths."""
    logger.warning(f"[{status}] {message}")
    return jsonify({"error": message}), status


def _parse_num_speakers(raw: str | None) -> int | None:
    """
    Convert the form value for num_speakers into an int or None.

    Returns None (→ auto-detect) when:
      - the field was left blank / missing
      - the value is not a positive integer
      - the value is below 2 (nonsensical for diarization)
    """
    if not raw or not raw.strip():
        return None
    try:
        n = int(raw.strip())
        return n if n >= 2 else None
    except ValueError:
        logger.warning(f"Non-integer num_speakers value '{raw}' — using auto-detect.")
        return None


# ---------------------------
# LLM JSON PARSING
# Three-stage repair: direct → single-quote fix → brace extraction
# Then validate schema: ensure all speakers present, clamp scores 0–10
# ---------------------------
def _parse_and_validate_llm_json(raw: str, expected_speakers: list) -> dict:
    """
    Parse LLM output into a valid analysis dict.
    Repairs common issues before giving up and using a safe fallback.
    """
    # Strip markdown code fences if the model wrapped output in ```json ... ```
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

    parsed = None

    # Attempt 1: parse as-is
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix single → double quotes (common LLM mistake)
    if parsed is None:
        try:
            parsed = json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            pass

    # Attempt 3: extract the outermost {...} block (handles leading/trailing prose)
    if parsed is None:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            pass

    if parsed is None:
        logger.error("All JSON repair attempts failed. Using fallback analysis.")
        parsed = {
            "summary": raw[:500],
            "intent": "-",
            "action_items": "-",
            "decision_impact": "-",
            "speaker_scores": {},
        }

    # Schema validation: ensure every expected speaker is present
    speaker_scores = parsed.setdefault("speaker_scores", {})
    for sp in expected_speakers:
        if sp not in speaker_scores:
            logger.warning(f"LLM response missing speaker '{sp}'. Inserting neutral defaults.")
            speaker_scores[sp] = {
                "contribution_quality": 5,
                "interaction_score": 5,
                "decision_impact": 5,
            }
        else:
            # Clamp each score to [0, 10] and coerce to int
            for key in ("contribution_quality", "interaction_score", "decision_impact"):
                try:
                    speaker_scores[sp][key] = max(0, min(10, int(speaker_scores[sp].get(key, 5))))
                except (TypeError, ValueError):
                    logger.warning(f"Invalid score for {sp}.{key} — defaulting to 5.")
                    speaker_scores[sp][key] = 5

    return parsed


# ---------------------------
# LLM ANALYSIS — retried up to 3×
# Lower temperature (0.1) for more deterministic JSON output.
# Schema is injected into the prompt so the model knows exact speaker keys.
# ---------------------------
@with_retry(max_attempts=3, initial_delay=2.0, exceptions=(Exception,))
def analyze_with_llm(text: str, topic: str, metrics: dict) -> dict:
    speaker_list = list(metrics.keys())

    # Build an exact schema stub so the model fills in the right speaker keys
    speaker_schema_stub = {
        sp: {"contribution_quality": 0, "interaction_score": 0, "decision_impact": 0}
        for sp in speaker_list
    }

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": (
                    "Return ONLY valid JSON. No preamble. No markdown. No explanation.\n\n"
                    "RULES:\n"
                    "- Double quotes everywhere\n"
                    "- All scores are integers 0–10\n"
                    f"- Include ALL of these speakers: {speaker_list}\n\n"
                    f"Meeting Topic: {topic}\n\n"
                    f"Speaker Metrics:\n{json.dumps(metrics, indent=2)}\n\n"
                    f"Conversation:\n{text}\n\n"
                    "Return this exact structure:\n"
                    + json.dumps(
                        {
                            "summary": "",
                            "intent": "",
                            "action_items": "",
                            "decision_impact": "",
                            "speaker_scores": speaker_schema_stub,
                        },
                        indent=2,
                    )
                ),
            }
        ],
    )

    raw = response.choices[0].message.content.strip()
    logger.debug(f"Raw LLM output:\n{raw}")

    return _parse_and_validate_llm_json(raw, speaker_list)


# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def upload_form():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    # --- Input validation ---
    if "file" not in request.files:
        return error_response("No file part in request.")

    file = request.files["file"]
    if not file or file.filename == "":
        return error_response("No file selected.")

    if not allowed_file(file.filename):
        return error_response(
            f"Unsupported file format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    topic = (request.form.get("topic") or "General conversation").strip()

    # Parse optional num_speakers from the form.
    # Returns None → auto-detect via silhouette score inside diarize().
    num_speakers = _parse_num_speakers(request.form.get("num_speakers"))
    logger.info(
        f"num_speakers from form: "
        f"{'auto-detect' if num_speakers is None else num_speakers}"
    )

    # UUID prefix prevents filename collisions across concurrent requests
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    logger.info(f"Upload saved: {filepath} | Topic: {topic}")

    try:
        # ---- Step 1: Diarization ----
        try:
            # num_speakers=None → auto-detect inside diarize()
            transcript, segments = diarizer.diarize(
                filepath, num_speakers=num_speakers
            )
        except DiarizationError as exc:
            return error_response(f"Diarization failed: {exc}", 422)
        except Exception as exc:
            logger.exception("Unexpected diarization error")
            return error_response(f"Diarization error: {exc}", 500)

        if not transcript.strip():
            return error_response(
                "Transcription produced no text. "
                "Audio may be silent, too short, or in an unsupported language.",
                422,
            )

        # Log how many unique speakers were detected
        unique_speakers = {seg["speaker"] for seg in segments}
        logger.info(
            f"Diarization complete. "
            f"Speakers detected: {sorted(unique_speakers)}. "
            f"Transcript length: {len(transcript)} chars."
        )

        # ---- Step 2: Metrics ----
        try:
            metrics = compute_all_metrics(segments, topic)
        except Exception as exc:
            logger.exception("Metrics computation failed")
            return error_response(f"Metrics computation failed: {exc}", 500)

        # ---- Step 3: LLM analysis (retried internally) ----
        try:
            analysis = analyze_with_llm(transcript, topic, metrics)
        except Exception as exc:
            logger.exception("LLM analysis failed after retries")
            return error_response(f"LLM analysis unavailable: {exc}", 502)

        # ---- Step 4: Scoring + explanations ----
        final_scores = compute_final_scores(metrics, analysis)
        explanations = generate_explanations(metrics, analysis)

        logger.info(f"Analysis complete for {filename}.")

        return jsonify(
            {
                "transcript": transcript,
                "metrics": metrics,
                "analysis": analysis,
                "final_scores": final_scores,
                "explanations": explanations,
            }
        )

    finally:
        # Always remove the uploaded file — even on exception
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Removed upload: {filepath}")
        except Exception as exc:
            logger.warning(f"Could not remove upload {filepath}: {exc}")


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)