# QA-SRL grammar reference corpus

`question-strings.txt` is an unmodified copy of a committed test resource from
[julianmichael/qasrl](https://github.com/julianmichael/qasrl), revision
`16ab49490f2df837ce7069bb1344f09e730d9a4a` (2023-09-22).

- [Original resource](https://github.com/julianmichael/qasrl/blob/16ab49490f2df837ce7069bb1344f09e730d9a4a/qasrl/test/resources/question-strings.txt)
- [Original Scala tests and expected ambiguity counts](https://github.com/julianmichael/qasrl/blob/16ab49490f2df837ce7069bb1344f09e730d9a4a/qasrl/test/src-jvm/qasrl/test/QuestionTests.scala)
- License: MIT, Copyright (c) 2017 Julian Michael; the upstream notice is
  preserved in [LICENSE](LICENSE).
- Size: 554,688 bytes; 13,616 questions; UTF-8 with LF line endings.
- SHA-256: `24662edadba126f09750125f1225738d7a77fe2129539643a20a927eefe29cac`
- Upstream Git blob: `9aeba6d78459d1d0bfcee62874c6204a8e679020`

The upstream Git tree and the local file's blob hash were compared on
2026-10-03. The fixture's `.gitattributes` preserves LF on Windows checkouts.

`tests/test_incremental_port.py::test_upstream_questions_exact_ambiguity_histogram`
checks parsing, question reconstruction and the expected ambiguity histogram:
10,882 questions with one reading, 2,599 with two, and 135 with three.
The test requires this fixture and fails if it is missing. It neither reads
`artifacts/` nor downloads anything, and it does not require a Scala runtime.

Run from the repository root:

```bash
python -m pytest -q tests/test_incremental_port.py
```
