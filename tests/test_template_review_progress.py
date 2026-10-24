import pytest

from semantic_corpus.template_review_progress import count_coverage


def test_occurrence_decision_does_not_approve_repeated_template():
    rows = [dict(idx=str(i), template_id=t, text='Hello')
            for i, t in enumerate(['a', 'a', 'b', 'c', 'd'])]
    result = count_coverage(rows, {'0': 'context_adapter'},
                            {'2': {'Hello'}, '4': {'Different text'}}, {'3'})
    assert result['remaining_occurrences'] == 2
    assert result['templates_with_remaining_occurrences'] == 2
    assert result['templates_without_decisions'] == 1
    assert result['templates_partially_resolved'] == 1
    assert result['templates_all_occurrences_resolved'] == 2
    assert result['occurrence_outcomes']['closed_restricted'] == 1


def test_duplicate_addresses_are_not_counted_twice():
    row = dict(idx='train:1:1:1', template_id='a', text='Hi')
    with pytest.raises(ValueError, match='Duplicate'):
        count_coverage([row, row], {}, {}, set())
