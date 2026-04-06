from pyannote.audio import Audio
from pyannote.core import Segment
import contextlib
import wave
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import datetime
import subprocess
import os
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from groq import Groq


class SpeakerDiarizer:
    def __init__(self, num_speakers=None):
        self.num_speakers = num_speakers
        # Replaced local whisper with Groq API for extreme speed
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            print("Warning: GROQ_API_KEY not found in environment variables.")
        self.groq_client = Groq(api_key=self.groq_api_key, timeout=1200.0)
        self.embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")

    def segment_embedding(self, segment, waveform, sample_rate, duration):
        start = segment["start"]
        end = min(duration, segment["end"])
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        
        # Slice the pre-loaded waveform in-memory to prevent disk thrashing
        clip_waveform = waveform[:, start_sample:end_sample]
        return self.embedding_model(clip_waveform[None])

    def diarize(self, path, num_speakers=None):
        if num_speakers is None:
            num_speakers = self.num_speakers
            
        processed_audio_file = "temp_audio_pyannote.wav"
        groq_audio_file = "temp_audio_groq.mp3"

        subprocess.call([
            'ffmpeg',
            '-i', path,
            '-ac', '1',
            '-ar', '16000', # Downsample to 16kHz for Pyannote
            processed_audio_file,
            '-ac', '1',
            '-ar', '16000',
            '-b:a', '32k', # Highly compress the Groq version to guarantee < 25MB
            groq_audio_file,
            '-y'
        ])

        with open(groq_audio_file, "rb") as file:
            transcription = self.groq_client.audio.transcriptions.create(
                file=(groq_audio_file, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
            
        segments = []
        for s in transcription.segments:
            s_dict = s if isinstance(s, dict) else getattr(s, '__dict__', {})
            if hasattr(s, 'model_dump'): s_dict = s.model_dump()
            segments.append({
                "start": float(s_dict.get("start", 0.0)),
                "end": float(s_dict.get("end", 0.0)),
                "text": str(s_dict.get("text", ""))
            })

        # Load the whole audio file ONCE into a numpy array tensor
        audio = Audio()
        full_waveform, sample_rate = audio(processed_audio_file)
        duration = full_waveform.shape[1] / float(sample_rate)

        embeddings = []
        for segment in segments:
            # Pass the loaded array directly, bypassing disk reads
            embeddings.append(self.segment_embedding(segment, full_waveform, sample_rate, duration))

        os.remove(processed_audio_file)
        embeddings = np.nan_to_num(np.array(embeddings))
        embeddings = np.concatenate(embeddings, axis=0)

        num_embeddings = len(embeddings)
        
        if num_speakers is not None:
            # Fallback for hardcoded speakers
            try:
                clustering = AgglomerativeClustering(n_clusters=min(num_speakers, num_embeddings), metric="cosine", linkage="average")
            except TypeError:
                clustering = AgglomerativeClustering(n_clusters=min(num_speakers, num_embeddings), affinity="cosine", linkage="average")
            labels = clustering.fit_predict(embeddings)
        elif num_embeddings <= 1:
            labels = np.zeros(num_embeddings, dtype=int)
        else:
            # Dynamic speaker detection
            max_speakers = min(10, num_embeddings)
            best_num_speakers = 1
            best_score = -1
            best_labels = np.zeros(num_embeddings, dtype=int)

            for k in range(2, max_speakers + 1):
                try:
                    clustering = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
                except TypeError:
                    clustering = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
                current_labels = clustering.fit_predict(embeddings)
                
                score = silhouette_score(embeddings, current_labels, metric="cosine")
                print(f"Testing {k} speakers... Silhouette Score: {score}")
                
                if score > best_score:
                    best_score = score
                    best_num_speakers = k
                    best_labels = current_labels
            
            if best_score < 0.05:
                print("Poor clustering scores detected. Falling back to 1 speaker.")
                labels = np.zeros(num_embeddings, dtype=int)
            else:
                print(f"Optimal clusters chosen: {best_num_speakers} speakers with score {best_score}")
                labels = best_labels

        for i, segment in enumerate(segments):
            segment["speaker"] = f"SPEAKER {labels[i] + 1}"

        def time(secs):
            return datetime.timedelta(seconds=round(secs))

        transcript = ""
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                transcript += f"\n{segment['speaker']} {time(segment['start'])}\n"
            transcript += segment["text"].strip() + " "

        return transcript, segments