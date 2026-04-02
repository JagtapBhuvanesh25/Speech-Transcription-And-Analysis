from pyannote.audio import Audio
from pyannote.core import Segment
import contextlib
import wave
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import datetime
import subprocess
import whisper
import os
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


class SpeakerDiarizer:
    def __init__(self, num_speakers=2):
        self.num_speakers = num_speakers
        self.model = whisper.load_model("small")
        self.embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")

    def segment_embedding(self, segment, path, duration):
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        audio = Audio()
        waveform, sample_rate = audio.crop(path, clip)
        return self.embedding_model(waveform[None])

    def diarize(self, path):
        processed_audio_file = "temp_audio.wav"

        subprocess.call([
            'ffmpeg',
            '-i', path,
            '-ac', '1',
            processed_audio_file,
            '-y'
        ])

        result = self.model.transcribe(processed_audio_file)
        segments = result["segments"]

        with contextlib.closing(wave.open(processed_audio_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        embeddings = []
        for segment in segments:
            embeddings.append(self.segment_embedding(segment, processed_audio_file, duration))

        os.remove(processed_audio_file)
        embeddings = np.nan_to_num(np.array(embeddings))
        embeddings = np.concatenate(embeddings, axis=0)

        clustering = AgglomerativeClustering(n_clusters=self.num_speakers)
        labels = clustering.fit_predict(embeddings)

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