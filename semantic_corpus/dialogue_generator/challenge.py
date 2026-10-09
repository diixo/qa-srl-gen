"""Freeze a test-only contrast set using surfaces absent from normal exports."""

import argparse
from dataclasses import replace
from itertools import islice
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .actions import ACTION_BY_RELATION, CONTRAST_TASKS, action_question
from .export import _sha256, prompt_key, response_row
from .generator import DialogueGenerator, fingerprint


def build_challenge(out: Path, *, groups_per_task=30, seed=20261009):
    if type(groups_per_task) is not int or groups_per_task <= 0:
        raise ValueError('groups_per_task must be positive')
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Choose a new challenge directory: {out}')
    rows, seen = [], set()
    generator = DialogueGenerator(seed=seed, user_assistant_only=True)
    for task in CONTRAST_TASKS:
        episodes = iter(generator.generate(task, groups_per_task * 60, split='test'))
        accepted = 0
        while accepted < groups_per_task:
            group = list(islice(episodes, 3))
            if len(group) != 3:
                raise ValueError('Not enough unique contrast groups')
            style = 3 + int(fingerprint(group[0].contrast_group)[:8], 16) % 2
            candidates = []
            for episode in group:
                question = action_question(ACTION_BY_RELATION[episode.query.relation], episode.query.value, style)
                # The question turn carries no delta. Change its surface consistently
                # in both the dialogue and transition audit before validation.
                turn = replace(episode.turns[-1], text=question)
                changed = replace(episode, memory_style=style, turns=episode.turns[:-1] + (turn,),
                                  transitions=episode.transitions[:-1] + (replace(episode.transitions[-1], turn=turn),))
                changed.validate()
                candidates.append(response_row(changed, 'memory'))
            keys = {prompt_key(row) for row in candidates}
            if len(keys) != 3 or keys & seen:
                continue
            seen.update(keys)
            rows.extend(candidates)
            accepted += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.challenge-', dir=out.parent) as temporary:
        staging = Path(temporary) / 'dataset'
        staging.mkdir()
        (staging / 'test.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        report = {'kind': 'held-out-memory-surfaces-v1', 'seed': seed, 'groups_per_task': groups_per_task,
                  'count': len(rows), 'styles': [3, 4], 'split': 'test',
                  'test_sha256': _sha256(staging / 'test.json'),
                  'resources': {path.name: _sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
                  'note': 'Unseen wording, shared action vocabulary and task families; no training split.'}
        (staging / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        staging.rename(out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--groups-per-task', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261009)
    args = parser.parse_args()
    try:
        result = build_challenge(**vars(args))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Frozen {result['count']} challenge records in {args.out}")


if __name__ == '__main__':
    main()
