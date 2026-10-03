"""Python core of QA-SRL: bank models, inflections, questions and validation.

Adapted from ``julianmichael/qasrl`` (MIT); see THIRD_PARTY_NOTICES.md.
"""

from .models import (
    AnswerJudgment,
    InflectedForms,
    QuestionLabel,
    QuestionSlots,
    Sentence,
    Span,
    VerbEntry,
    VerbForm,
)
from .question_parser import QuestionParseError, parse_question, parse_question_all
from .question_renderer import render_question
from .state_machine import TenseFeatures, VerbChain, build_verb_chain, features_for_chain
from .frame import ArgumentSlot, ArgStructure, Frame, Noun, Prep, Locative, SUBJ, OBJ, OBJ2, adv
from .tense import Finite, NonFinite, Tense
from .clausal_question import ClausalQuestion
from .template_state_machine import TemplateStateMachine
from .question_processor import (
    AggregatedInvalidState, CompleteState, InProgressState, ProcessingState, QuestionProcessor,
)
from .autocomplete import Autocomplete, Complete, Incomplete, Suggestion
from .question_label_mapper import QuestionLabelMapper
from .slot_based_label import (
    SurfaceQuestionSlots, get_slots_for_question,
    get_verb_tense_abstracted_slots_for_question, instantiate_verb_for_tense_slots,
)
from .clause_resolution import get_resolved_frame_pairs, get_resolved_structures
from .discrete_label import DiscreteLabel, NounRole, AdvRole, get_discrete_labels, get_all_discrete_labels
from .dataset import Dataset, MergeResult, ConsolidatedDataset, ConsolidatedSentence
from .bank_index import (
    Domain, DatasetPartition, DocumentId, SentenceId, DocumentMetadata,
    Document as BankDocument, DataIndex, read_index,
)
from .bank_sources import AnnotationRound, AnswerSource, QuestionSource, QuestionSourceKind, filter_expanded_to_orig
from .bank_reader import read_consolidated_bank
from .qanom import QANomSentence, QANomData, read_qanom, reprocess_qanom_sentence, reprocess_qanom, read_reformatted_qanom_data

__all__ = [
    "AnswerJudgment",
    "InflectedForms",
    "QuestionLabel",
    "QuestionSlots",
    "Sentence",
    "Span",
    "VerbEntry",
    "VerbForm",
    "QuestionParseError",
    "parse_question",
    "parse_question_all",
    "render_question",
    "TenseFeatures",
    "VerbChain",
    "build_verb_chain",
    "features_for_chain",
    "ArgumentSlot", "ArgStructure", "Frame", "Noun", "Prep", "Locative", "SUBJ", "OBJ", "OBJ2", "adv",
    "Finite", "NonFinite", "Tense", "ClausalQuestion",
    "TemplateStateMachine", "QuestionProcessor", "ProcessingState",
    "CompleteState", "InProgressState", "AggregatedInvalidState",
    "Autocomplete", "Complete", "Incomplete", "Suggestion", "QuestionLabelMapper",
    "SurfaceQuestionSlots", "get_slots_for_question", "get_verb_tense_abstracted_slots_for_question",
    "instantiate_verb_for_tense_slots", "get_resolved_frame_pairs", "get_resolved_structures",
    "DiscreteLabel", "NounRole", "AdvRole", "get_discrete_labels", "get_all_discrete_labels",
    "Dataset", "MergeResult", "ConsolidatedDataset", "ConsolidatedSentence",
    "Domain", "DatasetPartition", "DocumentId", "SentenceId", "DocumentMetadata", "DataIndex", "read_index",
    "BankDocument", "AnnotationRound", "AnswerSource", "QuestionSource", "QuestionSourceKind",
    "filter_expanded_to_orig", "read_consolidated_bank", "QANomSentence", "QANomData", "read_qanom",
    "reprocess_qanom_sentence", "reprocess_qanom", "read_reformatted_qanom_data",
]
