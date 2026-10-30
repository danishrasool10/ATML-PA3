"""VQAv2 complementary-pair filtering.

Kept pairs: both questions exist, different images, same question text and answer_type, each image's
10 normalised answers have a consensus answer given by >= min_agree annotators, and the two consensus
answers differ.  Training target = most frequent *original* answer string inside the consensus group
(ties: lexicographically smallest); all ten raw answers are kept for scoring.
"""
from __future__ import annotations

from collections import Counter

from .normalize import normalize_answer


def consensus(raw_answers: list[str], min_agree: int) -> tuple[str, str] | None:
    """(canonical normalised answer, original target string) or None."""
    assert len(raw_answers) == 10, len(raw_answers)
    norm = [normalize_answer(a) for a in raw_answers]
    canon, n = Counter(norm).most_common(1)[0]
    if n < min_agree or canon == "":
        return None
    raw = Counter(a.strip() for a, z in zip(raw_answers, norm) if z == canon)
    target = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return canon, target


def candidate_pairs(pairs: list[list[int]], questions: dict, annotations: dict, split: str,
                    min_agree: int) -> list[dict]:
    """questions: qid -> {image_id, question}; annotations: qid -> {answers: [10 raw], answer_type}."""
    out = []
    for q1, q2 in pairs:
        if q1 not in questions or q2 not in questions or q1 not in annotations or q2 not in annotations:
            continue
        a, b = questions[q1], questions[q2]
        if a["image_id"] == b["image_id"]:
            continue
        if " ".join(a["question"].split()).lower() != " ".join(b["question"].split()).lower():
            continue
        ta, tb = annotations[q1]["answer_type"], annotations[q2]["answer_type"]
        if ta != tb:
            continue
        ca = consensus(annotations[q1]["answers"], min_agree)
        cb = consensus(annotations[q2]["answers"], min_agree)
        if ca is None or cb is None or ca[0] == cb[0]:
            continue
        exs = []
        for qid, q, (canon, target) in ((q1, a, ca), (q2, b, cb)):
            exs.append({"qid": qid, "image": f"{split}:{q['image_id']}", "image_id": q["image_id"],
                        "target": target, "canonical": canon, "answers": list(annotations[qid]["answers"])})
        out.append({"kind": ta, "question": " ".join(a["question"].split()), "examples": exs,
                    "source_pair": [q1, q2]})
    return out


def load_vqa(questions_json: dict, annotations_json: dict) -> tuple[dict, dict]:
    qs = {q["question_id"]: {"image_id": q["image_id"], "question": q["question"]}
          for q in questions_json["questions"]}
    anns = {a["question_id"]: {"answers": [x["answer"] for x in a["answers"]], "answer_type": a["answer_type"]}
            for a in annotations_json["annotations"]}
    return qs, anns
