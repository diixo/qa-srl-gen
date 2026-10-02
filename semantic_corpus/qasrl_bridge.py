"""QA-SRL Bank as annotated real text.

The bank is the one corpus here that is real text *and* already annotated:
710 374 questions over Wikipedia, Wikinews and TQA sentences, each with a
predicate, a natural-language question and answer spans that human
annotators voted on. Everything the annotator branch cannot produce without
a model — which token is the predicate, which spans are its arguments, how
to word the question — is already there.

This module converts it into the canonical representation so it enters the
same pipeline as everything else. Nothing is inferred and nothing costs
anything.

What the bank gives and what it does not
----------------------------------------
It gives predicates, argument spans and questions. It does **not** give
entity types: *an example* is an argument of *see*, and no annotator ever
said whether it is a physical object or an abstract one. So mentions come
out with an empty label set, which is the honest record — an empty set means
"not annotated", and anything else would be this module inventing types.

The normalised role is derived only for explicit location/time questions.
Other questions keep their edge with an unknown (null) normalised role. The
handoff's rule is that the question is the primary record and the role is
the derived one, which is exactly this ordering.

Text is the token sequence joined with single spaces. The bank ships tokens,
not raw text, so that join *is* the source text as far as this corpus is
concerned, and every offset is measured against it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .documents import (
    AnnotationRun,
    Document,
    EntityMention,
    ONTOLOGY_VERSION,
    Predicate,
    RelationEdge,
    TextSpan,
    iter_sentence_spans,
)
from .ontology import Label, LabelSet, Relation, ReviewStatus
from .qasrl_core.bank_reader import read_bank
from .qasrl_core.frame import (
    OBJ,
    OBJ2,
    SUBJ,
    ArgumentSlot,
    Locative,
    Noun,
    Prep,
    frame_from_slots,
)
from .qasrl_core.models import QuestionLabel, Sentence, Span, VerbEntry
from .semantic_annotator.lexicons import STATIVE_VERBS

__all__ = [
    "BANK_SOURCE",
    "token_spans",
    "sentence_to_document",
    "sentence_to_run",
    "sentence_to_canonical",
    "bank_qa_examples",
    "iter_bank_canonical",
    "relation_for",
]

BANK_SOURCE = "qasrl-bank"
BANK_LICENSE = "QA-SRL Bank 2.0"

#: Adverbial questions whose role the relation inventory does not name.
#: ``why`` and ``how`` keep their question and get no normalised edge rather
#: than being forced into a role that does not mean what they ask.
_ADVERBIAL_RELATIONS: Mapping[str, Relation] = {
    "where": Relation.LOCATION_OF,
    "when": Relation.TIME_OF,
}


def token_spans(tokens: Sequence[str]) -> tuple[TextSpan, ...]:
    """Character spans of each token in the single-space join."""
    spans: list[TextSpan] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index:
            cursor += 1
        spans.append(TextSpan(cursor, cursor + len(token), index, index + 1))
        cursor += len(token)
    return tuple(spans)


def sentence_to_document(
    sentence: Sentence, *, split: str | None = None
) -> Document:
    """Wrap a bank sentence as a document."""
    text = " ".join(sentence.sentence_tokens)
    return Document(
        document_id=sentence.sentence_id,
        text=text,
        source=BANK_SOURCE,
        license=BANK_LICENSE,
        split=split,
        metadata={
            "tokens": len(sentence.sentence_tokens),
            "predicates": len(sentence.verb_entries),
        },
    )


def relation_for(
    label: QuestionLabel, entry: VerbEntry
) -> Relation | None:
    """Normalize only explicit location/time questions.

    Subject/object position and a preposition do not determine a semantic
    role: "to Kyiv" is not a recipient, and a passive subject can be one.
    """
    try:
        _frame, answer_slot = frame_from_slots(
            label.question_slots, entry.verb_inflected_forms
        )
    except ValueError:
        return None

    if answer_slot.is_adverbial and answer_slot.wh:
        return _ADVERBIAL_RELATIONS.get(answer_slot.wh)
    if answer_slot == OBJ2 and isinstance(_frame.args.get(OBJ2), Locative):
        return Relation.LOCATION_OF
    return None


def sentence_to_run(
    sentence: Sentence,
    document: Document,
    *,
    run_id: str,
    min_votes: int = 2,
    status: ReviewStatus = ReviewStatus.VERIFIED,
) -> AnnotationRun:
    """Build the annotation run a bank sentence already contains.

    *min_votes* is the agreement an answer span needs to be kept. The
    published evaluations use a strict majority, and a span one annotator
    highlighted alone is a claim, not a fact — so the default keeps only
    spans two people marked.
    """
    spans = token_spans(sentence.sentence_tokens)
    mentions: list[EntityMention] = []
    predicates: list[Predicate] = []
    relations: list[RelationEdge] = []
    seen_mentions: dict[tuple[int, int], str] = {}

    for key, entry in sorted(sentence.verb_entries.items()):
        if not 0 <= entry.verb_index < len(spans):
            continue
        predicate_id = f"{document.document_id}#p{entry.verb_index}"
        verb_span = spans[entry.verb_index]
        predicates.append(
            Predicate(
                predicate_id=predicate_id,
                document_id=document.document_id,
                span=TextSpan(verb_span.start_char, verb_span.end_char),
                exact_text=sentence.sentence_tokens[entry.verb_index],
                lemma=entry.verb_inflected_forms.stem,
                predicate_type=None,
                polarity=None,
                # The bank records tense per question, not per clause, so a
                # single value for the predicate would be a fabrication.
                tense=None,
                confidence=None,
                source=BANK_SOURCE,
                review_status=status,
            )
        )

        for label in entry.question_labels.values():
            for answer, votes in sorted(
                label.span_votes().items(), key=lambda kv: (-kv[1], kv[0])
            ):
                if votes < min_votes or not answer.fits(sentence.sentence_tokens):
                    continue
                char_span = TextSpan(
                    spans[answer.start].start_char, spans[answer.end - 1].end_char
                )
                mention_key = (char_span.start_char, char_span.end_char)
                mention_id = seen_mentions.get(mention_key)
                if mention_id is None:
                    mention_id = f"{document.document_id}#m{len(seen_mentions)}"
                    seen_mentions[mention_key] = mention_id
                    mentions.append(
                        EntityMention(
                            mention_id=mention_id,
                            document_id=document.document_id,
                            span=char_span,
                            exact_text=char_span.text_in(document.text),
                            # The bank never says what kind of thing this is.
                            labels=LabelSet(frozenset()),
                            confidence=votes / max(1, len({j.source_id for j in label.answer_judgments})),
                            source=BANK_SOURCE,
                            review_status=status,
                        )
                    )
                relation = relation_for(label, entry)
                relations.append(
                    RelationEdge(
                        source_id=mention_id,
                        relation=relation,
                        target_id=predicate_id,
                        question=label.question_string,
                        confidence=None,
                        source=BANK_SOURCE,
                    )
                )

    return AnnotationRun(
        run_id=run_id,
        ontology_version=ONTOLOGY_VERSION,
        model_name=BANK_SOURCE,
        prompt_version="n/a",
        synthetic=False,
    ).extended(mentions=mentions, predicates=predicates, relations=relations)


def sentence_to_canonical(
    sentence: Sentence,
    *,
    run_id: str | None = None,
    split: str | None = None,
    min_votes: int = 2,
) -> tuple[Document, AnnotationRun]:
    """A bank sentence as a document and the run describing it."""
    document = sentence_to_document(sentence, split=split)
    run = sentence_to_run(
        sentence,
        document,
        run_id=run_id or f"{document.document_id}-bank",
        min_votes=min_votes,
    )
    return document, run


def bank_qa_examples(
    sentence: Sentence,
    document: Document | None = None,
    *,
    min_votes: int = 2,
    run_id: str | None = None,
):
    """The QA examples the bank already contains, worded by its annotators.

    These need no template: a human wrote the question and other humans
    voted on the answer. That makes them the only atomic QA in the project
    whose wording was not produced by a rule.
    """
    from .question_generator.answers import QAExample, QAKind, phrase_answer

    document = document or sentence_to_document(sentence)
    spans = token_spans(sentence.sentence_tokens)
    examples = []
    for entry, label in sentence.question_labels():
        accepted = [
            (span, votes)
            for span, votes in sorted(
                label.span_votes().items(), key=lambda kv: (-kv[1], kv[0])
            )
            if votes >= min_votes and span.fits(sentence.sentence_tokens)
        ]
        if not accepted:
            continue
        best, votes = accepted[0]
        char_span = TextSpan(spans[best.start].start_char, spans[best.end - 1].end_char)
        examples.append(
            QAExample(
                context=document.text,
                question=label.question_string,
                answer=phrase_answer(char_span.text_in(document.text)),
                kind=QAKind.ATOMIC,
                document_id=document.document_id,
                run_id=run_id,
                evidence=(char_span,),
                metadata={
                    "predicate": entry.verb_inflected_forms.stem,
                    "predicate_index": entry.verb_index,
                    "votes": votes,
                    "judgments": len(label.answer_judgments),
                    "source": BANK_SOURCE,
                    "alternative_answers": [
                        TextSpan(
                            spans[s.start].start_char, spans[s.end - 1].end_char
                        ).text_in(document.text)
                        for s, _ in accepted[1:]
                    ],
                },
            )
        )
    return examples


def iter_bank_canonical(
    path: Path | str,
    *,
    split: str | None = None,
    min_votes: int = 2,
    limit: int | None = None,
) -> Iterator[tuple[Document, AnnotationRun]]:
    """Stream a bank file as canonical documents and runs."""
    for index, sentence in enumerate(read_bank(path)):
        if limit is not None and index >= limit:
            return
        yield sentence_to_canonical(sentence, split=split, min_votes=min_votes)
