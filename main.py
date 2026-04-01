from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from diarisation import SpeakerDiarizer
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)
os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}
NUMBER_SPEAKERS = 2
diarizer = SpeakerDiarizer(num_speakers=NUMBER_SPEAKERS)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(audio_path):
    transcript = diarizer.diarize(audio_path)
    print(transcript)
    return transcript

def analyze_sentiment(text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""
You MUST follow the format EXACTLY. Do NOT add extra text, headings, or spacing.

Rules:
- Output must be plain text (no markdown, no **, no bullet symbols except "-")
- Keep everything in ONE continuous block (no empty lines between sections)
- Do NOT include "Transcript"
- Do NOT change section titles
- Follow spacing exactly as shown

Format:

Emotions(explained in detail):
- Speaker 1: <detailed explanation>
- Speaker 2: <detailed explanation>
Relationship(between the users explained situation wise):
- <detailed explanation>
Overall Sentiment(detailed but easy to understand):
- <detailed explanation>

Now analyze this conversation:

{text}
"""
            }
        ]
    )

    return response.choices[0].message.content

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

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        print(f"Saved file at: {filepath}")

        transcript = transcribe_audio(filepath)
        sentiment_analysis = analyze_sentiment(transcript)

        return jsonify({
            'transcript': transcript,
            'sentiment_analysis': sentiment_analysis
        })

    return jsonify({'error': 'Invalid file format'})


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)