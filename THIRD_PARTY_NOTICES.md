# Third-party notices

## QA-SRL (julianmichael/qasrl)

<https://github.com/julianmichael/qasrl>

The QA-SRL data formats, the seven-slot question template, the verb-form
inventory and the auxiliary-chain grammar implemented in
`semantic_corpus/qasrl_core/` originate in that project and in the paper
*Large-Scale QA-SRL Parsing* (Fitzgerald, Michael, He, Zettlemoyer, ACL 2018).

**Nature of the adaptation.** The modules here were first written from the
published format specification
(<https://github.com/uwnlp/qasrl-bank/blob/master/FORMAT.md>), the paper, and
the released QA-SRL Bank 2.0 data. The upstream source was then audited at
commit `16ab4949` (2023-09-22) and the grammar corrected against it; the
rules carried over from the source are documented file by file in
SCALA_TO_PYTHON.md. The subsequent port directly adapts algorithms and models
from the core grammar, frame rendering, labeling, dataset operations, Bank
metadata/provenance and QANom reformatting into Python. Field names, slot names
and verb-form keys are kept compatible with the originals. These adaptations
remain subject to the upstream MIT notice below. The unmodified, committed
upstream test resource `qasrl/test/resources/question-strings.txt` is bundled
under `tests/fixtures/upstream-qasrl/`, with its original license and provenance.
The full local reference source remains under ignored `artifacts/` and is not
required to run the conformance test.

The upstream project is MIT licensed. Its notice:

```text
MIT License

Copyright (c) 2017 Julian Michael

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Data under `data/`

These are datasets, not code, and carry their own terms:

* **QA-SRL Bank 2.0** (`data/qasrl-v2/`, `data/qasrl-v2.tar`) — from
  <https://qasrl.org/data/qasrl-v2.tar>. Built over Wikipedia, Wikinews and
  TQA text.
* **QA-SRL 1.0 / wiki1** (`data/qa-srl-2.0/*.qa`) — the earlier tab-separated
  release.
* **Wiktionary scrape** (`data/wiktionary/`, `data/wiktionary.tar.gz`) —
  distributed with the QA-SRL project; derived from Wiktionary, which is
  licensed CC BY-SA. SHA-256 of the archive:
  `3dbfe046fe0dffbb91c5df3fe513589998655a7032a91a4125a6b8db71314ada`.
  The extraction scripts inside it are Python 2 and are not used here.
* **DailyDialog** (`data/dailydialog/`) — research use, CC BY-NC-SA 4.0.

Verify the licence terms of each dataset before redistributing anything
derived from it.
