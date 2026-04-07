import contextlib
import wave
import numpy as np
import datetime
import subprocess
import os
import uuid
import logging
import tempfile

from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import whisper

logger = logging.getLogger(__name__)

# ---------------------------
# CONSTANTS
# ---------------------------

# Segments shorter than this produce unreliable embeddings → skip them
MIN_SEGMENT_DURATION = 0.5  # seconds

# If any cluster holds fewer than this fraction of all segments,
# clustering has collapsed into a dominant speaker → trigger fallback
COLLAPSE_RATIO_THRESHOLD = 0.10


# ---------------------------
# CUSTOM EXCEPTION
# ---------------------------

class DiarizationError(Exception):
    """Raised when diarization cannot be completed."""
    pass


# ---------------------------
# DIARIZER
# ---------------------------

class SpeakerDiarizer:
    def __init__(self, num_speakers=2):
        self.num_speakers = num_speakers
        logger.info("Loading Whisper 'small' model...")
        self.model = whisper.load_model("small")
        logger.info("Loading SpeechBrain speaker embedding model...")
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb"
        )

    # ---------------------------
    # AUDIO CONVERSION
    # ---------------------------

    def _convert_to_wav(self, input_path: str) -> str:
        """
        Convert any audio format to 16kHz mono WAV using ffmpeg.
        Returns path to a unique temp file.
        Raises DiarizationError if ffmpeg fails.
        """
        temp_path = os.path.join(
            tempfile.gettempdir(),
            f"diarize_{uuid.uuid4().hex}.wav"
        )
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", temp_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise DiarizationError(
                f"FFmpeg conversion failed (code {result.returncode}): {result.stderr[-300:]}"
            )
        logger.info(f"Audio converted → {temp_path}")
        return temp_path

    # ---------------------------
    # DURATION
    # ---------------------------

    def _get_duration(self, wav_path: str) -> float:
        with contextlib.closing(wave.open(wav_path, "r")) as f:
            return f.getnframes() / float(f.getframerate())

    # ---------------------------
    # EMBEDDING
    # ---------------------------

    def _compute_embedding(self, segment: dict, wav_path: str, duration: float):
        """
        Compute speaker embedding for one Whisper segment.
        Returns the embedding array, or None if:
          - segment is too short (unreliable embedding)
          - audio crop fails for any reason
        """
        start = segment["start"]
        end = min(duration, segment["end"])

        if (end - start) < MIN_SEGMENT_DURATION:
            logger.debug(
                f"Skipping short segment [{start:.2f}s–{end:.2f}s] "
                f"(duration {end - start:.2f}s < {MIN_SEGMENT_DURATION}s)"
            )
            return None

        try:
            clip = Segment(start, end)
            audio = Audio()
            waveform, _ = audio.crop(wav_path, clip)
            return self.embedding_model(waveform[None])
        except Exception as e:
            logger.warning(f"Embedding failed for segment [{start:.2f}–{end:.2f}]: {e}")
            return None

    # ---------------------------
    # CLUSTERING
    # ---------------------------

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize then cluster embeddings with AgglomerativeClustering.
        After clustering, check for collapse:
          - if any cluster has < COLLAPSE_RATIO_THRESHOLD of samples,
            fall back to index-parity assignment.
        Returns per-embedding integer labels.
        """
        normed = normalize(embeddings, norm="l2")
        clustering = AgglomerativeClustering(
            n_clusters=self.num_speakers,
            metric="euclidean",
            linkage="ward"
        )
        labels = clustering.fit_predict(normed)

        # Collapse detection
        for c in range(self.num_speakers):
            ratio = float(np.sum(labels == c)) / len(labels)
            if ratio < COLLAPSE_RATIO_THRESHOLD:
                logger.warning(
                    f"Cluster {c} collapse detected ({ratio:.1%} of segments). "
                    "Switching to index-parity fallback."
                )
                # Fallback: alternate speakers by position
                # This is crude but prevents all-one-speaker output
                return np.array([i % self.num_speakers for i in range(len(labels))])

        return labels

    # ---------------------------
    # LABEL ASSIGNMENT
    # ---------------------------

    def _assign_all_labels(
        self,
        segments: list,
        valid_indices: list,
        cluster_labels: np.ndarray
    ) -> list:
        """
        Map cluster labels back to all segments including skipped ones.
        Skipped segments inherit the last valid label (forward-fill).
        """
        label_map = {
            valid_indices[i]: int(cluster_labels[i])
            for i in range(len(valid_indices))
        }
        result = []
        last = 0
        for i in range(len(segments)):
            if i in label_map:
                last = label_map[i]
            result.append(last)
        return result

    # ---------------------------
    # TRANSCRIPT BUILDER
    # ---------------------------

    def _build_transcript(self, segments: list) -> str:
        def fmt(secs):
            return str(datetime.timedelta(seconds=round(secs)))

        lines = []
        for i, seg in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != seg["speaker"]:
                lines.append(f"\n{seg['speaker']} {fmt(seg['start'])}")
            lines.append(seg["text"].strip())
        return " ".join(lines).strip()

    # ---------------------------
    # MAIN ENTRY POINT
    # ---------------------------

    def diarize(self, input_path: str):
        """
        Full pipeline: convert → transcribe → embed → cluster → label → transcript.

        Returns: (transcript: str, segments: list[dict])
        Raises: DiarizationError on unrecoverable failure.

        Temp file is always cleaned up in finally block.
        """
        wav_path = None
        try:
            # Step 1: Convert audio to mono 16kHz WAV
            wav_path = self._convert_to_wav(input_path)

            # Step 2: Transcribe
            logger.info("Running Whisper transcription...")
            result = self.model.transcribe(wav_path)
            segments = result.get("segments", [])

            if not segments:
                raise DiarizationError(
                    "Whisper returned no segments. "
                    "Audio may be silent, too short, or unrecognisable."
                )
            logger.info(f"Whisper produced {len(segments)} segments.")

            # Step 3: Audio duration (for safe crop bounds)
            duration = self._get_duration(wav_path)

            # Step 4: Compute embeddings, skipping short segments
            embeddings, valid_indices = [], []
            for i, seg in enumerate(segments):
                emb = self._compute_embedding(seg, wav_path, duration)
                if emb is not None:
                    embeddings.append(emb)
                    valid_indices.append(i)

            logger.info(
                f"{len(valid_indices)}/{len(segments)} segments produced valid embeddings."
            )

            # Step 5: Assign speakers
            if len(valid_indices) < self.num_speakers:
                # Not enough embeddings to cluster meaningfully
                logger.warning(
                    f"Only {len(valid_indices)} valid embeddings for "
                    f"{self.num_speakers} requested speakers. "
                    "Assigning all segments to SPEAKER 1."
                )
                for seg in segments:
                    seg["speaker"] = "SPEAKER 1"
            else:
                # Stack embeddings and cluster
                emb_matrix = np.nan_to_num(np.concatenate(embeddings, axis=0))
                labels = self._cluster(emb_matrix)
                all_labels = self._assign_all_labels(segments, valid_indices, labels)
                for i, seg in enumerate(segments):
                    seg["speaker"] = f"SPEAKER {all_labels[i] + 1}"

            # Step 6: Build transcript string
            transcript = self._build_transcript(segments)
            logger.info("Diarization complete.")
            return transcript, segments

        finally:
            # Always remove temp WAV regardless of success or failure
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                    logger.debug(f"Removed temp file: {wav_path}")
                except Exception as e:
                    logger.warning(f"Could not remove temp file {wav_path}: {e}")