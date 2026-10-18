"""Validate explicit fragment adapters and append them to the central catalog.

This is not an automatic interpreter. The model supplies each adapted dialogue
pair and its meaning; code checks addresses, prior context, schema and export.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantic_corpus.template_bank import fingerprint, read_rows
from semantic_corpus.dailydialog_quarantine import check_fragment_catalog
from semantic_corpus.dialogue_generator.fragments import (
    CATALOG, fragment_episode, validate_fragment_patterns,
)
from semantic_corpus.dialogue_generator.extraction_text import decode_update, parse_prompt
from semantic_corpus.dialogue_generator.export import extraction_row, response_row


def admit(review_file, contexts_file, *, catalog=CATALOG, publish=False):
    proposal = json.loads(Path(review_file).read_text(encoding='utf-8'))
    if proposal.get('format') != 'reviewed-fragment-adapters-v1':
        raise ValueError('Unsupported fragment adapter review')
    patterns = deepcopy(proposal['patterns'])
    if not patterns or any(p.get('review', {}).get('method') !=
                           'direct_assistant_fragment_adapter' for p in patterns):
        raise ValueError('Every proposed fragment needs an explicit contextual review')
    wanted = {f"{p['source']['split']}:{p['source']['dialogue']}" for p in patterns}
    contexts = {r['id']:r['turns'] for r in read_rows(contexts_file) if r['id'] in wanted}
    if set(contexts) != wanted:
        raise ValueError('Fragment source dialogues missing from current candidate')
    for pattern in patterns:
        source = pattern['source']
        context = contexts[f"{source['split']}:{source['dialogue']}"]
        if source.get('context', context) != context:
            raise ValueError('Fragment source changed')
        source['context'] = context
    validate_fragment_patterns(patterns)
    original_bytes = Path(catalog).read_bytes()
    original = json.loads(original_bytes)
    if original.get('format') != 'reviewed-dialogue-fragments-v1':
        raise ValueError('Unsupported reviewed fragment catalog')
    validate_fragment_patterns(original['patterns'])
    indexed = {p['id']:p for p in original['patterns']}
    additions = []
    for pattern in patterns:
        if pattern['id'] in indexed:
            if pattern != indexed[pattern['id']]:
                raise ValueError('An existing fragment adapter cannot be overwritten')
        else:
            additions.append(pattern)
    result = {**original, 'patterns': [*original['patterns'], *additions]}
    validate_fragment_patterns(result['patterns'])
    check_fragment_catalog(result['patterns'], catalog)
    episodes = transitions = 0
    for pattern in patterns:
        for user_only in (True, False):
            context = ('the kitchen', 'tired', 'tea') if user_only else ('Anna', 'Rome')
            episode = fragment_episode(pattern, context, split='train', user_only=user_only)
            episode.validate()
            for transition in episode.transitions:
                row = extraction_row(episode, transition)
                prompt, target = (m['content'] for m in row['messages'][1:])
                memory, turn, previous = parse_prompt(prompt)
                decoded = decode_update(target, transition.before, transition.turn, transition.previous_assistant)
                if decoded != transition.delta or target.startswith(('{', '[')):
                    raise ValueError('Fragment extraction roundtrip failed')
                decode_update(target, memory, turn, previous)
                transitions += 1
            for view in ('history', 'memory'):
                response_row(episode, view)
            episodes += 1
    if publish and additions:
        # Refuse a concurrent edit instead of overwriting another reviewer's work.
        text = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',
                    dir=Path(catalog).parent,prefix='.fragments-',suffix='.tmp',delete=False) as sink:
                temporary = Path(sink.name)
                sink.write(text)
                sink.flush()
                os.fsync(sink.fileno())
            if Path(catalog).read_bytes() != original_bytes:
                raise ValueError('Fragment catalog changed during admission; retry')
            os.replace(temporary, catalog)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return dict(status='passed',published=publish,new_patterns=len(additions),
                checked_patterns=len(patterns),catalog_patterns=len(result['patterns']),
                source_dialogues=len(contexts),episodes_checked=episodes,transitions_checked=transitions,
                previous_patterns_preserved=len(original['patterns']),
                previous_catalog_sha256=fingerprint(original),result_catalog_sha256=fingerprint(result),
                template_bank_changed=False,api_calls=0,parser_inference_calls=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews',type=Path,required=True)
    parser.add_argument('--contexts',type=Path,required=True)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():
        parser.error('choose a new report path')
    report=admit(args.reviews,args.contexts,publish=args.publish)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
