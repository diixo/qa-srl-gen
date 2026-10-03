"""Compare HEAD with the working tree on saved DailyDialog records; no source edits."""

import argparse
from collections import defaultdict
import hashlib
from itertools import islice
import json
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantic_corpus.storage.repository import CorpusStore


def load_version(name, source):
    module = ModuleType('semantic_corpus.storage.' + name)
    sys.modules[module.__name__] = module
    exec(compile(source, name + '.py', 'exec'), module.__dict__)
    return module.CorpusStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out', type=Path, default=ROOT / 'artifacts/add-examples-review',
        help='New directory for benchmark stores and report (must not exist).',
    )
    args = parser.parse_args()
    output = args.out
    if output.exists():
        parser.error(f'Output directory already exists: {output}. Choose another --out path.')
    output.mkdir(parents=True, exist_ok=False)
    head_source = subprocess.check_output(
        ['git', 'show', 'HEAD:semantic_corpus/storage/repository.py'], cwd=ROOT,
    ).decode('utf-8')
    work_source = (ROOT / 'semantic_corpus/storage/repository.py').read_text(encoding='utf-8')
    versions = {'head': load_version('_review_head', head_source),
                'working': load_version('_review_work', work_source)}
    (output / 'head_repository.py').write_text(head_source, encoding='utf-8')
    (output / 'working_repository.py').write_text(work_source, encoding='utf-8')
    source = CorpusStore(ROOT / 'artifacts/dailydialog-audit/store', create=False)
    documents = list(islice(source.documents(), 500))
    wanted = {document.document_id for document in documents}
    runs = list(islice(source.runs(), 500))
    grouped = defaultdict(list)
    for example in source.examples():
        if example.document_id not in wanted:
            break
        grouped[example.document_id].append(example)
    triples = []
    for document, run in zip(documents, runs):
        examples = grouped[document.document_id]
        assert examples and all(e.run_id == run.run_id for e in examples)
        triples.append((document, run, examples))
    qa_count = sum(len(examples) for _, _, examples in triples)
    print(f'Loaded {len(triples)} documents and {qa_count} real QA records', flush=True)

    results = []
    hashes = {}
    for scenario in ('interleaved', 'bulk', 'reopened'):
        for repeat in range(3):
            order = ('head', 'working') if repeat % 2 == 0 else ('working', 'head')
            for version in order:
                cls = versions[version]
                path = output / f'{scenario}-{repeat}-{version}'
                store = cls(path)
                setup_seconds = 0.0
                qa_seconds = 0.0
                reread_seconds = 0.0
                header_calls = 0
                written = 0

                def instrument(target):
                    original = target._rows_for
                    def counted(table, key, value):
                        nonlocal header_calls
                        if table == 'annotation_runs':
                            header_calls += 1
                        return original(table, key, value)
                    target._rows_for = counted

                def store_data(document, run):
                    nonlocal setup_seconds
                    begin = perf_counter()
                    store.add_document(document)
                    store.add_run(run, document)
                    setup_seconds += perf_counter() - begin

                if scenario != 'interleaved':
                    for document, run, _ in triples:
                        store_data(document, run)
                if scenario == 'reopened':
                    store = cls(path, create=False)
                instrument(store)
                for document, run, examples in triples:
                    if scenario == 'interleaved':
                        store_data(document, run)
                    begin = perf_counter()
                    written += store.add_examples(examples, generation_run_id=run.run_id)
                    qa_seconds += perf_counter() - begin
                first_pass_headers = header_calls
                if scenario == 'reopened':
                    begin = perf_counter()
                    for _, run, examples in triples:
                        assert store.add_examples(examples, generation_run_id=run.run_id) == 0
                    reread_seconds = perf_counter() - begin
                assert written == qa_count
                digest = hashlib.sha256(store.path_for('qa_examples').read_bytes()).hexdigest()
                if scenario in hashes:
                    assert digest == hashes[scenario]
                hashes[scenario] = digest
                cache = getattr(store, '_run_documents', getattr(store, '_recent_run_documents', {}))
                cache_container_bytes = sys.getsizeof(cache) + sum(sys.getsizeof(v) for v in cache.values())
                result = dict(scenario=scenario, repeat=repeat, version=version,
                    setup_seconds=round(setup_seconds, 6), qa_seconds=round(qa_seconds, 6),
                    second_pass_seconds=round(reread_seconds, 6),
                    header_reads=first_pass_headers, second_pass_header_reads=header_calls-first_pass_headers,
                    cache_entries=len(cache), cache_container_bytes=cache_container_bytes,
                    written=written, qa_sha256=digest)
                results.append(result)
                print(json.dumps(result), flush=True)
    summary = {}
    for scenario in ('interleaved', 'bulk', 'reopened'):
        summary[scenario] = {}
        for version in versions:
            rows = [r for r in results if r['scenario'] == scenario and r['version'] == version]
            summary[scenario][version] = {
                field: statistics.median(r[field] for r in rows)
                for field in ('setup_seconds','qa_seconds','second_pass_seconds','header_reads',
                              'second_pass_header_reads','cache_entries','cache_container_bytes')
            }
    report = {'head_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
              'documents':len(triples), 'qa_records':qa_count, 'results':results, 'medians':summary,
              'source_sha256':{version:hashlib.sha256(content.encode()).hexdigest()
                  for version,content in [('head',head_source),('working',work_source)]}}
    (output / 'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('MEDIANS ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
