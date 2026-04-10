"""
Speech Intelligence API — app.py
Run with: python app.py  OR  gunicorn app:app

This file is a SEPARATE entry point. main.py is untouched.
"""

import os
import uuid
import logging
import threading
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from groq import Groq

# ── Reuse existing modules (unmodified) ──────────────────────────────────────
from diarisation import SpeakerDiarizer, DiarizationError
from metrics import compute_all_metrics, compute_final_scores, generate_explanations
from main import analyze_with_llm, _parse_num_speakers   # retry-wrapped LLM call

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Startup validation ────────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

# ── Flask app (defined ONCE) ──────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads_api"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}

# Single shared diarizer — models load once at startup
diarizer = SpeakerDiarizer()

# ── In-memory job store ───────────────────────────────────────────────────────
# Structure: { job_id: { "status": "pending|processing|done|error", "result": {...} } }
_jobs: dict = {}
_jobs_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _err(msg: str, status: int = 400):
    return jsonify({"error": msg}), status


def _build_response(transcript: str, metrics: dict, analysis: dict,
                    final_scores: dict, explanations: dict) -> dict:
    """
    Assemble the required response shape.
    Every metric is preserved exactly — nothing is dropped or renamed.
    """
    ranking = final_scores.get("ranking", [])
    scores  = final_scores.get("scores", {})
    speaker_scores = analysis.get("speaker_scores", {})

    top = ranking[0]["speaker"] if ranking else None

    speaker_insights = {}
    for sp, m in metrics.items():
        sp_scores = speaker_scores.get(sp, {})
        speaker_insights[sp] = {
            "metrics": {
                # ── compute_speaker_metrics ────────────────────────────────
                "speaking_share_percent":      m.get("speaking_share_percent"),
                "num_turns":                   m.get("num_turns"),
                "avg_words_per_turn":          m.get("avg_words_per_turn"),
                "avg_duration_per_turn_sec":   m.get("avg_duration_per_turn_sec"),
                "questions_asked":             m.get("questions_asked"),
                "vocabulary_richness":         m.get("vocabulary_richness"),
                "filler_rate":                 m.get("filler_rate"),
                # ── contribution_ratio ─────────────────────────────────────
                "short_ratio":                 m.get("short_ratio"),
                "long_ratio":                  m.get("long_ratio"),
                # ── topic_metrics ──────────────────────────────────────────
                "agenda_alignment_percent":    m.get("agenda_alignment_percent"),
                "topic_coverage_percent":      m.get("topic_coverage_percent"),
                # ── sentiment / confidence ─────────────────────────────────
                "sentiment_score":             m.get("sentiment_score"),
                "confidence_score":            m.get("confidence_score"),
            },
            "llm_scores": {
                "contribution_quality": sp_scores.get("contribution_quality"),
                "interaction_score":    sp_scores.get("interaction_score"),
                "decision_impact":      sp_scores.get("decision_impact"),
            },
            "final_score": scores.get(sp),
            "strengths":   explanations.get(sp, {}).get("strengths", []),
            "weaknesses":  explanations.get(sp, {}).get("weaknesses", []),
        }

    return {
        "meeting_summary":  analysis.get("summary", ""),
        "intent":           analysis.get("intent", ""),
        "action_items":     analysis.get("action_items", ""),
        "decision_impact":  analysis.get("decision_impact", ""),
        "top_performer":    top,
        "leaderboard":      [{"speaker": r["speaker"], "score": r["score"], "rank": r["rank"]}
                             for r in ranking],
        "speaker_insights": speaker_insights,
        "transcript":       transcript,
    }


# ── Background worker ─────────────────────────────────────────────────────────

def _process_job(job_id: str, filepath: str, topic: str, num_speakers):
    """Runs in a background thread. Updates _jobs[job_id] when done."""
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "processing"

        # Step 1: Diarize
        transcript, segments = diarizer.diarize(filepath, num_speakers=num_speakers)
        if not transcript.strip():
            raise ValueError("Transcription produced no text.")

        # Step 2: Metrics
        metrics = compute_all_metrics(segments, topic)

        # Step 3: LLM analysis (retried via main.analyze_with_llm)
        analysis = analyze_with_llm(transcript, topic, metrics)

        # Step 4: Scores + explanations
        final_scores = compute_final_scores(metrics, analysis)
        explanations = generate_explanations(metrics, analysis)

        result = _build_response(transcript, metrics, analysis, final_scores, explanations)

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result

        logger.info(f"Job {job_id} completed.")

    except DiarizationError as e:
        logger.error(f"Job {job_id} diarization error: {e}")
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["result"] = {"error": f"Diarization failed: {e}"}

    except Exception as e:
        logger.exception(f"Job {job_id} unexpected error")
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["result"] = {"error": str(e)}

    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Serve the frontend HTML."""
    return send_from_directory(".", "meeting_intelligence.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})


@app.route("/version", methods=["GET"])
def version():
    return jsonify({"version": "1.0"})


@app.route("/analyze-meeting", methods=["POST"])
def analyze_meeting():
    """
    Accept multipart/form-data:
      - file        : audio file (wav / mp3 / m4a)
      - topic       : string (optional, default "General conversation")
      - num_speakers: int (optional, triggers auto-detect if absent)

    Returns immediately with a job_id.
    Poll GET /result/<job_id> for the result.
    """
    if "file" not in request.files:
        return _err("No file part in request.")

    file = request.files["file"]
    if not file or file.filename == "":
        return _err("No file selected.")
    if not allowed_file(file.filename):
        return _err(f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    topic        = (request.form.get("topic") or "General conversation").strip()
    num_speakers = _parse_num_speakers(request.form.get("num_speakers"))

    # Save upload with UUID prefix to prevent collisions
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    logger.info(f"Upload saved: {filepath} | topic={topic} | num_speakers={num_speakers}")

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "result": None}

    thread = threading.Thread(
        target=_process_job,
        args=(job_id, filepath, topic, num_speakers),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "pending"}), 202


@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id: str):
    """
    Returns current status of a job.
    - pending / processing → { "job_id": "...", "status": "processing" }
    - done                 → full result payload
    - error                → { "error": "..." }
    - unknown job_id       → 404
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return _err(f"Job '{job_id}' not found.", 404)

    status = job["status"]

    if status in ("pending", "processing"):
        return jsonify({"job_id": job_id, "status": status}), 202

    if status == "error":
        return jsonify({"job_id": job_id, "status": "error", **job["result"]}), 500

    # done
    return jsonify({"job_id": job_id, "status": "done", **job["result"]}), 200


# ── Interpret route ───────────────────────────────────────────────────────────

@app.route("/interpret", methods=["POST"])
def interpret():
    """Takes analysis JSON, returns LLM business insight using Groq."""
    data = request.get_json(force=True)
    client = Groq()

    system_prompt = (
        "You are given the output of an automated meeting analysis system. "
        "This includes metrics, speaker performance, rankings, summary, and transcript.\n\n"
        "Your task is to interpret this data like a human business analyst.\n"
        "Do not repeat the data. Do not describe the metrics individually.\n\n"
        "Instead, focus on:\n"
        "- What actually happened in the meeting\n"
        "- How effective the meeting was\n"
        "- Who contributed meaningfully vs who didn't\n"
        "- Whether the discussion led to real outcomes or just debate\n"
        "- Any imbalance in participation or dominance\n"
        "- The overall business value of the meeting\n"
        "- Clear, practical suggestions to improve future meetings\n\n"
        "Think critically and infer insights from the data rather than restating it.\n"
        "Respond in a clear, professional, business-oriented manner using markdown headings (##) and bullet points where appropriate."
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": str(data)}
            ]
        )
        insight = completion.choices[0].message.content
        return jsonify({"insight": insight})
    except Exception as e:
        logger.exception("Interpret route error")
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)