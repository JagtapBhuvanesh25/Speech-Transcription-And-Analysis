import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
import json
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# BASIC HELPERS
# ---------------------------

FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "actually"}

POSITIVE_WORDS = {"good", "great", "nice", "excellent", "happy", "love"}
NEGATIVE_WORDS = {"bad", "poor", "sad", "angry", "issue", "problem"}


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


# ---------------------------
# CORE METRICS
# ---------------------------

def compute_speaker_metrics(segments):
    speaker_data = defaultdict(lambda: {
        "turns": 0,
        "total_time": 0,
        "total_words": 0,
        "questions": 0,
        "sentences": 0,
        "filler_count": 0,
        "word_set": set()
    })

    total_conversation_time = 0

    for seg in segments:
        sp = seg["speaker"]
        text = seg["text"]

        duration = seg["end"] - seg["start"]
        words = tokenize(text)

        total_conversation_time += duration

        speaker_data[sp]["turns"] += 1
        speaker_data[sp]["total_time"] += duration
        speaker_data[sp]["total_words"] += len(words)
        speaker_data[sp]["sentences"] += text.count(".") + text.count("?") + text.count("!")

        if "?" in text:
            speaker_data[sp]["questions"] += 1

        # filler words
        speaker_data[sp]["filler_count"] += sum(1 for w in words if w in FILLER_WORDS)

        # vocabulary
        speaker_data[sp]["word_set"].update(words)

    # ---------------------------
    # POST PROCESSING
    # ---------------------------

    results = {}

    for sp, data in speaker_data.items():
        turns = data["turns"]
        total_words = data["total_words"]
        total_time = data["total_time"]

        avg_words = total_words / turns if turns else 0
        avg_duration = total_time / turns if turns else 0

        speaking_share = (total_time / total_conversation_time) * 100 if total_conversation_time else 0

        vocab_richness = len(data["word_set"]) / total_words if total_words else 0

        filler_rate = data["filler_count"] / total_words if total_words else 0

        results[sp] = {
            "speaking_share_percent": round(speaking_share, 2),
            "num_turns": turns,
            "avg_words_per_turn": round(avg_words, 2),
            "avg_duration_per_turn_sec": round(avg_duration, 2),
            "questions_asked": data["questions"],
            "vocabulary_richness": round(vocab_richness, 3),
            "filler_rate": round(filler_rate, 3)
        }

    return results


# ---------------------------
# SHORT vs LONG CONTRIBUTIONS
# ---------------------------

def contribution_ratio(segments, threshold=10):
    result = defaultdict(lambda: {"short": 0, "long": 0})

    for seg in segments:
        sp = seg["speaker"]
        words = tokenize(seg["text"])

        if len(words) <= threshold:
            result[sp]["short"] += 1
        else:
            result[sp]["long"] += 1

    # convert to ratio
    ratios = {}
    for sp, data in result.items():
        total = data["short"] + data["long"]
        if total == 0:
            ratios[sp] = {"short_ratio": 0, "long_ratio": 0}
        else:
            ratios[sp] = {
                "short_ratio": round(data["short"] / total, 2),
                "long_ratio": round(data["long"] / total, 2)
            }

    return ratios


# ---------------------------
# SENTIMENT SCORE (-1 to +1)
# ---------------------------

def sentiment_score(segments):
    result = defaultdict(lambda: {"score": 0, "count": 0})

    for seg in segments:
        sp = seg["speaker"]
        text = seg["text"].lower()

        score = 0

        # stronger signals
        if any(w in text for w in ["great", "good", "nice", "love"]):
            score += 1
        if any(w in text for w in ["problem", "issue", "bad", "not good"]):
            score -= 1

        # normalize per segment instead of word count
        result[sp]["score"] += score
        result[sp]["count"] += 1

    final = {}
    for sp, data in result.items():
        if data["count"] == 0:
            final[sp] = 0
        else:
            final[sp] = round(data["score"] / data["count"], 2)

    return final


# ---------------------------
# CONFIDENCE SCORE (-1 to +1)
# ---------------------------

def confidence_score(segments):
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

        # 🔥 boost if decisive action
        if any(w in text for w in ["i'll take", "i will", "done", "confirm"]):
            score += 2

        result[sp]["score"] += score
        result[sp]["count"] += 1

    final = {}
    for sp, data in result.items():
        if data["count"] == 0:
            final[sp] = 0
        else:
            final[sp] = round(data["score"] / data["count"], 2)

    return final


# ---------------------------
# TOPIC METRICS
# ---------------------------

def topic_metrics(segments, topic):
    text_blocks = "\n".join([
        f"{i}. {seg['speaker']}: {seg['text']}"
        for i, seg in enumerate(segments)
    ])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""
Rate each line's relevance to the topic from 0 to 1.

STRICT:
- Return ONLY valid JSON
- Keys MUST be strings
- Values MUST be numbers (0 to 1)
- No explanation

Topic: {topic}

{text_blocks}

Expected format:
{{
  "0": 0.8,
  "1": 0.2
}}
"""
            }
        ]
    )

    raw = response.choices[0].message.content.strip()

    print("\nRAW LLM OUTPUT:\n", raw)

    # -----------------------
    # 🔥 ROBUST PARSING
    # -----------------------
    import re

    try:
        scores = json.loads(raw)
    except:
        try:
            # fix unquoted keys: 0: 0.8 → "0": 0.8
            fixed = re.sub(r'(\d+)\s*:', r'"\1":', raw)

            # remove trailing commas
            fixed = re.sub(r',\s*}', '}', fixed)

            scores = json.loads(fixed)
        except:
            print("JSON PARSE FAILED")
            scores = {}

    # -----------------------
    # 🔥 SAFETY: convert keys
    # -----------------------
    scores = {str(k): float(v) for k, v in scores.items()}

    # -----------------------
    # PROCESS
    # -----------------------
    speaker_data = {}

    for i, seg in enumerate(segments):
        sp = seg["speaker"]
        score = scores.get(str(i), 0)

        if sp not in speaker_data:
            speaker_data[sp] = {
                "scores": [],
                "relevant": 0,
                "total": 0
            }

        speaker_data[sp]["scores"].append(score)
        speaker_data[sp]["total"] += 1

        if score > 0.5:
            speaker_data[sp]["relevant"] += 1

    final = {}

    for sp, data in speaker_data.items():
        avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        coverage = (data["relevant"] / data["total"]) * 100 if data["total"] else 0

        final[sp] = {
            "agenda_alignment_percent": round(avg * 100, 2),
            "topic_coverage_percent": round(coverage, 2)
        }

    return final

# -----------------------
# COMPUTE FINAL sCORE
# -----------------------
        
def compute_final_scores(metrics, analysis):
    final_scores = {}

    speaker_scores = analysis.get("speaker_scores", {})

    for sp, m in metrics.items():
        s = speaker_scores.get(sp, {})

        # -----------------------
        # NORMALIZATION (0–1)
        # -----------------------
        speaking = (m.get("speaking_share_percent", 0) / 100)
        alignment = (m.get("agenda_alignment_percent", 0) / 100)
        coverage = (m.get("topic_coverage_percent", 0) / 100)

        # confidence is already small → normalize to 0–1
        confidence = (m.get("confidence_score", 0) + 1) / 2

        sentiment = (m.get("sentiment_score", 0) + 1) / 2

        # LLM scores (0–10 → 0–1)
        contrib = (s.get("contribution_quality", 0) / 10)
        interaction = (s.get("interaction_score", 0) / 10)
        decision = (s.get("decision_impact", 0) / 10)

        # -----------------------
        # WEIGHTS (balanced)
        # -----------------------
        score = (
            speaking * 0.15 +
            alignment * 0.20 +
            coverage * 0.15 +
            confidence * 0.10 +
            sentiment * 0.05 +
            contrib * 0.15 +
            interaction * 0.10 +
            decision * 0.10
        )

        final_scores[sp] = round(score * 100, 2)  # convert to %

    # -----------------------
    # RANKING
    # -----------------------
    ranking = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    ranked_output = []
    for i, (sp, score) in enumerate(ranking, start=1):
        ranked_output.append({
            "speaker": sp,
            "score": score,
            "rank": i
        })

    return {
        "scores": final_scores,
        "ranking": ranked_output
    }

# ---------------------------
# GENERATE EXPLAINATION
# ---------------------------

def generate_explanations(metrics, analysis):
    explanations = {}

    speaker_scores = analysis.get("speaker_scores", {})

    for sp, m in metrics.items():
        s = speaker_scores.get(sp, {})

        strengths = []
        weaknesses = []

        # -------- Strength rules --------
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

        # -------- Weakness rules --------
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
            "strengths": strengths if strengths else ["No strong signals"],
            "weaknesses": weaknesses if weaknesses else ["No major issues"]
        }

    return explanations


# ---------------------------
# MAIN WRAPPER
# ---------------------------

def compute_all_metrics(segments, topic):
    base = compute_speaker_metrics(segments)
    contrib = contribution_ratio(segments)
    sentiment = sentiment_score(segments)

    # Phase 2 + improved topic metrics
    confidence = confidence_score(segments)
    topic_data = topic_metrics(segments, topic)

    final = {}

    for sp in base:
        final[sp] = base[sp]

        # contribution ratios
        if sp in contrib:
            final[sp].update(contrib[sp])

        # 🔥 FIXED: use topic_data instead of old vars
        final[sp]["agenda_alignment_percent"] = topic_data.get(sp, {}).get("agenda_alignment_percent", 0)
        final[sp]["topic_coverage_percent"] = topic_data.get(sp, {}).get("topic_coverage_percent", 0)

        # other metrics
        final[sp]["sentiment_score"] = sentiment.get(sp, 0)
        final[sp]["confidence_score"] = confidence.get(sp, 0)

    return final