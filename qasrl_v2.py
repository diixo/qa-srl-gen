# collapsed dataset into flat form: data/qasrl-v2
# format: https://github.com/uwnlp/qasrl-bank/blob/master/FORMAT.md
# description: https://arxiv.org/html/1805.05377v1


import gzip
import json
import math
from collections import Counter
from pathlib import Path


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def flatten_qasrl(
    input_path,
    output_path,
    min_valid_ratio=1.0,
    min_span_ratio=0.5,
):
    count = 0

    with open(output_path, "w", encoding="utf-8") as output:
        for sentence in read_jsonl_gz(input_path):
            tokens = sentence["sentenceTokens"]

            for verb_entry in sentence["verbEntries"].values():
                verb_index = verb_entry["verbIndex"]
                forms = verb_entry["verbInflectedForms"]

                for question_label in verb_entry["questionLabels"].values():
                    judgments = question_label["answerJudgments"]

                    if not judgments:
                        continue

                    valid_judgments = [
                        judgment
                        for judgment in judgments
                        if judgment["isValid"]
                    ]

                    valid_ratio = (
                        len(valid_judgments) / len(judgments)
                    )

                    if valid_ratio < min_valid_ratio:
                        continue

                    span_votes = Counter()

                    for judgment in valid_judgments:
                        for span in judgment.get("spans") or []:
                            span_votes[tuple(span)] += 1

                    required_span_votes = max(
                        1,
                        math.ceil(
                            len(valid_judgments) * min_span_ratio
                        ),
                    )

                    answers = []

                    for (start, end), votes in sorted(
                        span_votes.items()
                    ):
                        if votes < required_span_votes:
                            continue

                        answers.append({
                            "start": start,
                            "end": end,
                            "tokens": tokens[start:end],
                            "text": " ".join(tokens[start:end]),
                            "votes": votes,
                        })

                    if not answers:
                        continue

                    record = {
                        "sentence_id": sentence["sentenceId"],
                        "tokens": tokens,
                        "sentence": " ".join(tokens),
                        "predicate": {
                            "index": verb_index,
                            "text": tokens[verb_index],
                            "lemma": forms["stem"],
                            "forms": forms,
                        },
                        "question": question_label["questionString"],
                        "question_slots": question_label["questionSlots"],
                        "question_features": {
                            "tense": question_label["tense"],
                            "perfect": question_label["isPerfect"],
                            "progressive": question_label["isProgressive"],
                            "negated": question_label["isNegated"],
                            "passive": question_label["isPassive"],
                        },
                        "answers": answers,
                        "valid_votes": len(valid_judgments),
                        "total_votes": len(judgments),
                    }

                    output.write(
                        json.dumps(record, ensure_ascii=False)
                        + "\n"
                    )
                    count += 1

    return count


count = flatten_qasrl(
    input_path=Path("data/qasrl-v2/expanded/train.jsonl.gz"),
    output_path=Path("data/qasrl_train_flat.jsonl"),
    min_valid_ratio=1.0,
    min_span_ratio=0.5,
)

print("Создано записей:", count)
