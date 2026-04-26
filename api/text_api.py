from flask import Flask, jsonify
from transformers import pipeline
import re

app = Flask(__name__)

HARDCODED_TEXT = (
    "Hello, and welcome to our session today, it is a pleasure to meet you. "
    "I am looking forward to learning more about your professional journey and discussing "
    "how your skills align with our company's mission."
)

# Load emotion classification pipeline once at startup
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False,
    top_k=1,
)


def split_into_sentences(text: str) -> list[str]:
    # Simple sentence splitter based on punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def classify_emotion(sentence: str) -> dict:
    # Depending on transformers version/top_k settings, the pipeline may return
    # either [{'label': ..., 'score': ...}] or [[{'label': ..., 'score': ...}]].
    result = emotion_classifier(sentence)
    if result and isinstance(result[0], list):
        result = result[0]
    return result[0]


@app.get("/")
def get_text():
    sentences = split_into_sentences(HARDCODED_TEXT)

    results = []

    for sentence in sentences:
        prediction = classify_emotion(sentence)
        results.append(
            {
                "sentence": sentence,
                "emotion": prediction["label"],
                "score": round(float(prediction["score"]), 4),
            }
        )

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
