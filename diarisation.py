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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import whisper

logger = logging.getLogger(__name__)

# ---------------------------
# CONSTANTS
# ---------------------------

# Segments shorter than this produce unreliable embeddings → skip them
MIN_SEGMENT_DURATION = 0.5  # seconds

# Auto-detection: search this range when num_speakers is not provided
AUTO_MIN_SPEAKERS = 2
AUTO_MAX_SPEAKERS = 8

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
    """
    Speaker diarizer backed by Whisper (ASR) + SpeechBrain ECAPA-TDNN (embeddings).

    num_speakers is NOT set at construction time — it is passed per call to
    diarize(), which allows a single shared instance to handle recordings with
    different numbers of speakers without reloading the heavyweight models.
    """

    def __init__(self):
        logger.info("Loading Whisper 'small' model...")
        self.model = whisper.load_model("small")
        logger.info("Loading SpeechBrain speaker-embedding model...")
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb"
        )

    # ---------------------------
    # AUDIO CONVERSION
    # ---------------------------

    def _convert_to_wav(self, input_path: str) -> str:
        """
        Convert any audio format to 16 kHz mono WAV using ffmpeg.
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
                f"FFmpeg conversion failed (code {result.returncode}): "
                f"{result.stderr[-300:]}"
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
    # AUTO SPEAKER COUNT DETECTION
    # ---------------------------

    def _estimate_num_speakers(self, embeddings: np.ndarray) -> int:
        """
        Use silhouette score to pick the best number of speakers in the range
        [AUTO_MIN_SPEAKERS, AUTO_MAX_SPEAKERS], capped by available embeddings.

        Silhouette measures how well each point fits its own cluster vs the
        nearest neighbouring cluster — higher is better.  We try every k and
        keep the one with the highest mean silhouette score.

        Falls back to 2 when there are too few embeddings to score reliably.
        """
        n = len(embeddings)
        max_k = min(AUTO_MAX_SPEAKERS, n - 1)   # need at least 1 sample per cluster

        if max_k < AUTO_MIN_SPEAKERS:
            logger.warning(
                f"Only {n} embeddings — not enough to search for speaker count. "
                "Defaulting to 2."
            )
            return 2

        best_score = -1.0
        best_k = AUTO_MIN_SPEAKERS

        for k in range(AUTO_MIN_SPEAKERS, max_k + 1):
            clustering = AgglomerativeClustering(
                n_clusters=k, metric="euclidean", linkage="ward"
            )
            labels = clustering.fit_predict(embeddings)

            # silhouette_score requires at least 2 distinct labels
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(embeddings, labels, metric="euclidean")
            logger.debug(f"  k={k}  silhouette={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

        logger.info(
            f"Auto-detected {best_k} speaker(s) "
            f"(best silhouette={best_score:.4f})"
        )
        return best_k

    # ---------------------------
    # CLUSTERING
    # ---------------------------

    def _cluster(self, embeddings: np.ndarray, num_speakers: int) -> np.ndarray:
        """
        L2-normalise → cluster with AgglomerativeClustering.

        Collapse detection: if any cluster contains fewer than
        COLLAPSE_RATIO_THRESHOLD of all samples, fall back to index-parity
        assignment (crude but prevents all-one-speaker output).

        Returns per-embedding integer labels.
        """
        normed = normalize(embeddings, norm="l2")
        clustering = AgglomerativeClustering(
            n_clusters=num_speakers,
            metric="euclidean",
            linkage="ward"
        )
        labels = clustering.fit_predict(normed)

        for c in range(num_speakers):
            ratio = float(np.sum(labels == c)) / len(labels)
            if ratio < COLLAPSE_RATIO_THRESHOLD:
                logger.warning(
                    f"Cluster {c} collapse detected ({ratio:.1%} of segments). "
                    "Switching to index-parity fallback."
                )
                return np.array([i % num_speakers for i in range(len(labels))])

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
        """
        Build a readable transcript string.

        Consecutive segments from the same speaker are merged under one header
        to avoid repeated SPEAKER N / timestamp lines for every Whisper chunk.
        Format:

            SPEAKER 1 0:00:03
            Hello everyone, today we're going to discuss …

            SPEAKER 2 0:00:15
            Thanks for the intro. I wanted to start by …
        """
        def fmt(secs: float) -> str:
            return str(datetime.timedelta(seconds=round(secs)))

        lines = []
        current_speaker = None
        current_texts = []
        current_start = 0.0

        for seg in segments:
            sp = seg["speaker"]
            text = seg["text"].strip()
            if not text:
                continue

            if sp != current_speaker:
                # Flush previous speaker block
                if current_speaker is not None and current_texts:
                    lines.append(
                        f"\n{current_speaker} {fmt(current_start)}\n"
                        + " ".join(current_texts)
                    )
                current_speaker = sp
                current_texts = [text]
                current_start = seg["start"]
            else:
                current_texts.append(text)

        # Flush final block
        if current_speaker and current_texts:
            lines.append(
                f"\n{current_speaker} {fmt(current_start)}\n"
                + " ".join(current_texts)
            )

        return "\n".join(lines).strip()

    # ---------------------------
    # MAIN ENTRY POINT
    # ---------------------------

    def diarize(self, input_path: str, num_speakers: int | None = None):
        """
        Full pipeline: convert → transcribe → embed → (auto-detect or cluster)
                       → label → transcript.

        Parameters
        ----------
        input_path : str
            Path to the audio file (any format supported by ffmpeg).
        num_speakers : int or None
            Number of speakers.  Pass None (or omit) to auto-detect via
            silhouette-score search over [AUTO_MIN_SPEAKERS, AUTO_MAX_SPEAKERS].

        Returns
        -------
        transcript : str
        segments   : list[dict]

        Raises
        ------
        DiarizationError
            On unrecoverable failures (bad audio, ffmpeg error, etc.).
        """
        wav_path = None
        try:
            # ── Step 1: Convert audio to mono 16 kHz WAV ──────────────────
            wav_path = self._convert_to_wav(input_path)

            # ── Step 2: Transcribe ─────────────────────────────────────────
            logger.info("Running Whisper transcription...")
            result = self.model.transcribe(wav_path)
            segments = result.get("segments", [])

            if not segments:
                raise DiarizationError(
                    "Whisper returned no segments. "
                    "Audio may be silent, too short, or unrecognisable."
                )
            logger.info(f"Whisper produced {len(segments)} segment(s).")

            # ── Step 3: Audio duration (for safe crop bounds) ──────────────
            duration = self._get_duration(wav_path)

            # ── Step 4: Compute embeddings, skipping short segments ────────
            embeddings, valid_indices = [], []
            for i, seg in enumerate(segments):
                emb = self._compute_embedding(seg, wav_path, duration)
                if emb is not None:
                    embeddings.append(emb)
                    valid_indices.append(i)

            logger.info(
                f"{len(valid_indices)}/{len(segments)} segment(s) "
                "produced valid embeddings."
            )

            # ── Step 5: Resolve num_speakers ───────────────────────────────
            if len(valid_indices) == 0:
                # No usable embeddings at all — assign everything to one speaker
                logger.warning("No valid embeddings. Assigning all segments to SPEAKER 1.")
                for seg in segments:
                    seg["speaker"] = "SPEAKER 1"
                transcript = self._build_transcript(segments)
                return transcript, segments

            emb_matrix = np.nan_to_num(np.concatenate(embeddings, axis=0))

            if num_speakers is None:
                logger.info("num_speakers not provided — running auto-detection...")
                num_speakers = self._estimate_num_speakers(
                    normalize(emb_matrix, norm="l2")
                )
            else:
                # Clamp to what's actually achievable given the embeddings we have
                clamped = min(num_speakers, len(valid_indices))
                if clamped != num_speakers:
                    logger.warning(
                        f"Requested {num_speakers} speakers but only "
                        f"{len(valid_indices)} valid embeddings exist. "
                        f"Clamping to {clamped}."
                    )
                    num_speakers = clamped

            logger.info(f"Using num_speakers={num_speakers}")

            # ── Step 6: Cluster or trivially assign ───────────────────────
            if num_speakers == 1 or len(valid_indices) < num_speakers:
                logger.warning(
                    f"Only {len(valid_indices)} valid embeddings for "
                    f"{num_speakers} speaker(s). Assigning all to SPEAKER 1."
                )
                for seg in segments:
                    seg["speaker"] = "SPEAKER 1"
            else:
                labels = self._cluster(emb_matrix, num_speakers)
                all_labels = self._assign_all_labels(segments, valid_indices, labels)
                for i, seg in enumerate(segments):
                    seg["speaker"] = f"SPEAKER {all_labels[i] + 1}"

            # ── Step 7: Build transcript ───────────────────────────────────
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