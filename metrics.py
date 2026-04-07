import re
import json
import os
import logging
import time
from collections import defaultdict
from functools import wraps

from groq import Groq

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# CONSTANTS
# ---------------------------

FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "actually"}


# ---------------------------
# RETRY DECORATOR
# Mirrors the one in main.py — kept local so metrics.py stays self-contained.
# ---------------------------

def _with_retry(max_attempts=3, initial_delay=2.0, backoff=2.0, exceptions=(Exception,)):
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

def tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b", text.lower())


def _parse_topic_json(raw: str, num_segments: int) -> dict:
    """
    Repair and parse the JSON object returned by the topic-relevance LLM call.
    Falls back to a neutral score dict (0.5 for all segments) on total failure.
    """
    # Strip markdown fences
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

    scores = None

    # Attempt 1: direct parse
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix unquoted integer keys (0: 0.8 → "0": 0.8)
    #            and trailing commas (,}) which LLMs commonly produce
    if scores is None:
        try:
            fixed = re.sub(r"(\b\d+)\s*:", r'"\1":', raw)   # quote keys
            fixed = re.sub(r",\s*}", "}", fixed)             # trailing comma
            scores = json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # Attempt 3: extract outermost {...} block
    if scores is None:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            scores = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            pass

    if scores is None:
        logger.error("Topic-relevance JSON parse failed on all attempts. Using 0.5 fallback.")
        return {str(i): 0.5 for i in range(num_segments)}

    # Normalise: string keys, float values, clamp 0–1
    safe = {}
    for k, v in scores.items():
        try:
            safe[str(k)] = max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            safe[str(k)] = 0.5
    return safe


# ---------------------------
# CORE METRICS (unchanged logic, logging only)
# ---------------------------

def compute_speaker_metrics(segments: list) -> dict:
    speaker_data = defaultdict(lambda: {
        "turns": 0,
        "total_time": 0.0,
        "total_words": 0,
        "questions": 0,
        "sentences": 0,
        "filler_count": 0,
        "word_set": set(),
    })
    total_time = 0.0

    for seg in segments:
        sp = seg["speaker"]
        text = seg["text"]
        duration = seg["end"] - seg["start"]
        words = tokenize(text)

        total_time += duration
        speaker_data[sp]["turns"] += 1
        speaker_data[sp]["total_time"] += duration
        speaker_data[sp]["total_words"] += len(words)
        speaker_data[sp]["sentences"] += (
            text.count(".") + text.count("?") + text.count("!")
        )
        if "?" in text:
            speaker_data[sp]["questions"] += 1
        speaker_data[sp]["filler_count"] += sum(
            1 for w in words if w in FILLER_WORDS
        )
        speaker_data[sp]["word_set"].update(words)

    results = {}
    for sp, data in speaker_data.items():
        turns = data["turns"]
        total_words = data["total_words"]
        sp_time = data["total_time"]

        results[sp] = {
            "speaking_share_percent": round((sp_time / total_time * 100) if total_time else 0, 2),
            "num_turns": turns,
            "avg_words_per_turn": round(total_words / turns if turns else 0, 2),
            "avg_duration_per_turn_sec": round(sp_time / turns if turns else 0, 2),
            "questions_asked": data["questions"],
            "vocabulary_richness": round(len(data["word_set"]) / total_words if total_words else 0, 3),
            "filler_rate": round(data["filler_count"] / total_words if total_words else 0, 3),
        }

    return results


def contribution_ratio(segments: list, threshold: int = 10) -> dict:
    result = defaultdict(lambda: {"short": 0, "long": 0})
    for seg in segments:
        words = tokenize(seg["text"])
        bucket = "short" if len(words) <= threshold else "long"
        result[seg["speaker"]][bucket] += 1

    ratios = {}
    for sp, data in result.items():
        total = data["short"] + data["long"]
        ratios[sp] = {
            "short_ratio": round(data["short"] / total, 2) if total else 0,
            "long_ratio": round(data["long"] / total, 2) if total else 0,
        }
    return ratios


def sentiment_score(segments: list) -> dict:
    result = defaultdict(lambda: {"score": 0, "count": 0})
    for seg in segments:
        sp = seg["speaker"]
        text = seg["text"].lower()
        score = 0
        if any(w in text for w in ["great", "good", "nice", "love"]):
            score += 1
        if any(w in text for w in ["problem", "issue", "bad", "not good"]):
            score -= 1
        result[sp]["score"] += score
        result[sp]["count"] += 1

    return {
        sp: round(d["score"] / d["count"], 2) if d["count"] else 0
        for sp, d in result.items()
    }


def confidence_score(segments: list) -> dict:
    assertive = {"will", "definitely", "sure", "of course", "take", "i'll"}
    uncertain = {"maybe", "i think", "not sure", "probably", "might"}

    result = defaultdict(lambda: {"score": 0, "count": 0})
    for seg in segments:
        sp = seg["speaker"]
        text = seg["text"].lower()
        score = 0
        if any(w in text for w in assertive):
            score += 1
        if any(w in text for w in uncertain):
            score -= 1
        if any(w in text for w in ["i'll take", "i will", "done", "confirm"]):
            score += 2
        result[sp]["score"] += score
        result[sp]["count"] += 1

    return {
        sp: round(d["score"] / d["count"], 2) if d["count"] else 0
        for sp, d in result.items()
    }


# ---------------------------
# TOPIC METRICS — now retried
# ---------------------------

@_with_retry(max_attempts=3, initial_delay=2.0, exceptions=(Exception,))
def topic_metrics(segments: list, topic: str) -> dict:
    """
    Calls the LLM to rate each segment's relevance to the topic (0–1).
    Retried up to 3× on failure.
    Falls back to a neutral 0.5 score per segment if JSON cannot be parsed.
    """
    if not segments:
        return {}

    text_blocks = "\n".join(
        f"{i}. {seg['speaker']}: {seg['text']}"
        for i, seg in enumerate(segments)
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": (
                    "Rate each line's relevance to the topic from 0 to 1.\n\n"
                    "STRICT:\n"
                    "- Return ONLY valid JSON\n"
                    "- Keys MUST be quoted strings of the line index\n"
                    "- Values MUST be numbers between 0 and 1\n"
                    "- No explanation, no markdown\n\n"
                    f"Topic: {topic}\n\n"
                    f"{text_blocks}\n\n"
                    'Expected format: {"0": 0.8, "1": 0.2, ...}'
                ),
            }
        ],
    )

    raw = response.choices[0].message.content.strip()
    logger.debug(f"Raw topic-relevance output:\n{raw}")

    scores = _parse_topic_json(raw, len(segments))

    # Aggregate per speaker
    speaker_data: dict = {}
    for i, seg in enumerate(segments):
        sp = seg["speaker"]
        score = scores.get(str(i), 0.5)
        if sp not in speaker_data:
            speaker_data[sp] = {"scores": [], "relevant": 0, "total": 0}
        speaker_data[sp]["scores"].append(score)
        speaker_data[sp]["total"] += 1
        if score > 0.5:
            speaker_data[sp]["relevant"] += 1

    final = {}
    for sp, data in speaker_data.items():
        avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        coverage = (data["relevant"] / data["total"] * 100) if data["total"] else 0
        final[sp] = {
            "agenda_alignment_percent": round(avg * 100, 2),
            "topic_coverage_percent": round(coverage, 2),
        }

    return final


# ---------------------------
# FINAL SCORE (unchanged)
# ---------------------------

def compute_final_scores(metrics: dict, analysis: dict) -> dict:
    speaker_scores = analysis.get("speaker_scores", {})
    final_scores = {}

    for sp, m in metrics.items():
        s = speaker_scores.get(sp, {})

        speaking   = m.get("speaking_share_percent", 0) / 100
        alignment  = m.get("agenda_alignment_percent", 0) / 100
        coverage   = m.get("topic_coverage_percent", 0) / 100
        confidence = (m.get("confidence_score", 0) + 1) / 2
        sentiment  = (m.get("sentiment_score", 0) + 1) / 2
        contrib    = s.get("contribution_quality", 0) / 10
        interaction= s.get("interaction_score", 0) / 10
        decision   = s.get("decision_impact", 0) / 10

        score = (
            speaking    * 0.15
            + alignment * 0.20
            + coverage  * 0.15
            + confidence* 0.10
            + sentiment * 0.05
            + contrib   * 0.15
            + interaction*0.10
            + decision  * 0.10
        )
        final_scores[sp] = round(score * 100, 2)

    ranking = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return {
        "scores": final_scores,
        "ranking": [
            {"speaker": sp, "score": sc, "rank": i + 1}
            for i, (sp, sc) in enumerate(ranking)
        ],
    }


# ---------------------------
# EXPLANATIONS (unchanged)
# ---------------------------

def generate_explanations(metrics: dict, analysis: dict) -> dict:
    speaker_scores = analysis.get("speaker_scores", {})
    explanations = {}

    for sp, m in metrics.items():
        s = speaker_scores.get(sp, {})
        strengths, weaknesses = [], []

        if m.get("speaking_share_percent", 0) > 55:
            strengths.append("High participation")
        if m.get("agenda_alignment_percent", 0) > 60:
            strengths.append("Strong alignment with topic")
        if m.get("topic_coverage_percent", 0) > 70:
            strengths.append("Consistently on-topic")
        if m.get("confidence_score", 0) > 0.3:
            strengths.append("Confident communication")
        if s.get("contribution_quality", 0) >= 7:
            strengths.append("High-quality contributions")
        if s.get("interaction_score", 0) >= 7:
            strengths.append("Good interaction with others")

        if m.get("speaking_share_percent", 0) < 30:
            weaknesses.append("Low participation")
        if m.get("agenda_alignment_percent", 0) < 30:
            weaknesses.append("Poor topic alignment")
        if m.get("topic_coverage_percent", 0) < 40:
            weaknesses.append("Frequent off-topic responses")
        if m.get("confidence_score", 0) < 0:
            weaknesses.append("Uncertain communication")
        if s.get("contribution_quality", 0) <= 4:
            weaknesses.append("Low contribution quality")
        if s.get("interaction_score", 0) <= 4:
            weaknesses.append("Limited interaction")

        explanations[sp] = {
            "strengths": strengths or ["No strong signals"],
            "weaknesses": weaknesses or ["No major issues"],
        }

    return explanations


# ---------------------------
# MAIN WRAPPER
# ---------------------------

def compute_all_metrics(segments: list, topic: str) -> dict:
    if not segments:
        logger.warning("compute_all_metrics called with empty segments list.")
        return {}

    base     = compute_speaker_metrics(segments)
    contrib  = contribution_ratio(segments)
    sentiment= sentiment_score(segments)
    confidence = confidence_score(segments)
    topic_data = topic_metrics(segments, topic)

    final = {}
    for sp in base:
        final[sp] = base[sp]

        if sp in contrib:
            final[sp].update(contrib[sp])

        final[sp]["agenda_alignment_percent"] = (
            topic_data.get(sp, {}).get("agenda_alignment_percent", 0)
        )
        final[sp]["topic_coverage_percent"] = (
            topic_data.get(sp, {}).get("topic_coverage_percent", 0)
        )
        final[sp]["sentiment_score"]  = sentiment.get(sp, 0)
        final[sp]["confidence_score"] = confidence.get(sp, 0)

    return final