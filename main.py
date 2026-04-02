from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import json

from diarisation import SpeakerDiarizer
from groq import Groq
from metrics import compute_all_metrics, compute_final_scores, generate_explanations

# ---------------------------
# INIT
# ---------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)
os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}
NUMBER_SPEAKERS = 2
diarizer = SpeakerDiarizer(num_speakers=NUMBER_SPEAKERS)


# ---------------------------
# HELPERS
# ---------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def transcribe_audio(audio_path):
    # 🔥 IMPORTANT: now returns transcript + segments
    transcript, segments = diarizer.diarize(audio_path)
    print(transcript)
    return transcript, segments


# ---------------------------
# LLM ANALYSIS (JSON OUTPUT)
# ---------------------------
def analyze_with_llm(text, topic, metrics):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""
Return ONLY valid JSON. No explanation.

STRICT RULES:
- Use double quotes
- No extra text
- Do not leave fields empty
- Scores must be 0–10 integers

Context:
Meeting Topic: {topic}

Speaker Metrics:
{metrics}

Conversation:
{text}

Return BOTH:

1. High-level analysis
2. Speaker evaluation

JSON format:
{{
  "summary": "",
  "intent": "",
  "action_items": "",
  "decision_impact": "",

  "speaker_scores": {{
    "SPEAKER 1": {{
      "contribution_quality": 0,
      "interaction_score": 0,
      "decision_impact": 0
    }},
    "SPEAKER 2": {{
      "contribution_quality": 0,
      "interaction_score": 0,
      "decision_impact": 0
    }}
  }}
}}
"""
            }
        ]
    )

    raw_output = response.choices[0].message.content.strip()

    print("\n=== RAW LLM OUTPUT ===\n", raw_output, "\n====================\n")

    try:
        parsed = json.loads(raw_output)
    except:
        try:
            fixed = raw_output.replace("'", '"')
            parsed = json.loads(fixed)
        except:
            parsed = {
                "summary": raw_output,
                "intent": "-",
                "action_items": "-",
                "decision_impact": "-",
                "speaker_scores": {}
            }

    return parsed


# ---------------------------
# ROUTES
# ---------------------------

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    topic = request.form.get("topic", "General conversation")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        print(f"Saved file at: {filepath}")
        print(f"Meeting Topic: {topic}")

        # ---------------------------
        # STEP 1: DIARIZATION
        # ---------------------------
        transcript, segments = transcribe_audio(filepath)

        # ---------------------------
        # STEP 2: METRICS (deterministic)
        # ---------------------------
        metrics = compute_all_metrics(segments, topic)

        # ---------------------------
        # STEP 3: LLM ANALYSIS
        # ---------------------------
        analysis = analyze_with_llm(transcript, topic, metrics)
        final_scores = compute_final_scores(metrics, analysis)
        explanations = generate_explanations(metrics, analysis)

        # ---------------------------
        # FINAL RESPONSE
        # ---------------------------
        return jsonify({
            "transcript": transcript,
            "metrics": metrics,
            "analysis": analysis,
            "final_scores": final_scores,
        "explanations": explanations
        })

    return jsonify({'error': 'Invalid file format'})


# ---------------------------
# RUN
# ---------------------------

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)