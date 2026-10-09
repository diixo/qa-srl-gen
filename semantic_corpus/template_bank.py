"""Data-driven generation from the complete corpus template bank.

These are executable surface templates, not hand-verified semantic frames.
Every lexical verb and particle retains its observed position. Alternatives
are retained in the bank rather than conflated with verified role annotations.
No NLP model is needed to read, select or render a compiled bank.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import random

FORMAT = "corpus-templates-v1"
DEFAULT_BANK = Path(__file__).resolve().parents[1] / "data" / "eng-base-templates"


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class CorpusTemplate:
    def __init__(self, record):
        self.record = record
        self.template_id = record["template_id"]
        self.slots = record["slots"]

    @property
    def verbs(self):
        return tuple(self.record["verbs"])

    @property
    def text(self):
        return "".join(part.get("literal", "{" + part.get("slot", "") + "}")
                       for part in self.record["parts"])

    def render(self, bindings=None):
        """Render all slots, returning exact new spans, including separated verbs.

        A partial explicit replacement is allowed; unspecified slots retain the
        representative source binding. This is surface substitution, not a
        claim that arbitrary user-supplied replacements are semantically valid.
        """
        values = dict(self.record["defaults"])
        if bindings:
            unknown = set(bindings) - set(self.slots)
            if unknown:
                raise ValueError(f"unknown template slots: {sorted(unknown)}")
            values.update(bindings)
        fragments, spans, cursor, part_spans = [], {}, 0, []
        for part in self.record["parts"]:
            if "literal" in part:
                text = part["literal"]
            else:
                key = part["slot"]
                text = values[key]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"slot {key!r} requires a nonempty string")
                spans[key] = [cursor, cursor + len(text)]
            fragments.append(text)
            part_spans.append([cursor, cursor + len(text)])
            cursor += len(text)
        result = "".join(fragments)
        for key, (start, end) in spans.items():
            if result[start:end] != values[key]:
                raise AssertionError("template produced inconsistent offsets")
        groups = []
        for group in self.record["predicate_groups"]:
            variants = []
            for construction in group["constructions"]:
                markers = []
                for marker in construction["markers"]:
                    span = (spans[marker["slot"]] if "slot" in marker else
                            [part_spans[marker["part"]][0] + marker["start"],
                             part_spans[marker["part"]][0] + marker["end"]])
                    markers.append({"span": span, "text": result[span[0]:span[1]]})
                variants.append({**construction, "markers": markers})
            groups.append({**group, "head_span": spans.get(group["head_slot"]), "constructions": variants})
        attributes = [{**a, "span": spans[a["slot"]],
                       "target_span": spans.get(a["target_slot"])} for a in self.record.get("attributes", ())]
        return {"template_id": self.template_id, "text": result, "bindings": values, "slots": self.slots,
                "spans": spans, "verbs": list(self.verbs),
                "predicate_groups": groups, "attributes": attributes,
                "review_status": "UNREVIEWED"}

    def attested(self, adjective_forms=None):
        attested = self.record["attested_bindings"]
        adjective_slots = {name for name, slot in self.slots.items()
                           if slot["pos"] == "ADJ" or slot["tag"] in ("JJ", "JJR", "JJS")}
        if adjective_forms is not None:
            attested = [b for b in attested if any(b[name] in adjective_forms for name in adjective_slots)]
            if not attested:
                raise ValueError("no attested binding contains the requested adjective")
        return attested

    def sample(self, *, seed=0, variant=0, recombine=False, adjective_forms=None):
        rng = random.Random(fingerprint([seed, self.template_id, variant]))
        attested = self.attested(adjective_forms)
        if recombine:
            # Pools are local to this exact lexical/dependency/morphology shape;
            # predicates and markers are never replaced by a random other verb.
            values = {name: rng.choice(pool) for name, pool in self.record["fillers"].items()}
            if adjective_forms is not None:
                original = rng.choice(attested)
                values.update({name: original[name] for name, slot in self.slots.items()
                               if slot["pos"] == "ADJ" or slot["tag"] in ("JJ", "JJR", "JJS")})
        else:
            # This is an attested set of bindings, including cross-slot agreement.
            values = rng.choice(attested)
        return self.render(values)


class TemplateBank:
    def __init__(self, path=DEFAULT_BANK):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if (not isinstance(self.manifest, dict) or self.manifest.get("format") != FORMAT
                or self.manifest.get("status") != "complete"):
            raise ValueError("a complete corpus-templates-v1 bank is required")
        if not isinstance(self.manifest.get("counts", {}), dict):
            raise ValueError("template bank counts must be a JSON object")

    def templates(self, *, lemma=None, adjective=None, template_id=None):
        if template_id is not None and (len(template_id) != 64 or
                any(c not in "0123456789abcdef" for c in template_id)):
            raise ValueError("template_id must be a SHA-256 identifier")
        files = ([self.path / "templates" / f"{template_id[:2]}.jsonl.gz"] if template_id
                 else sorted((self.path / "templates").glob("*.jsonl.gz")))
        for file in files:
            if not file.exists():
                continue
            for row in read_rows(file):
                if template_id is not None and row["template_id"] != template_id:
                    continue
                if lemma is not None and lemma not in row["verbs"] and lemma not in row["constructions"]:
                    continue
                if adjective is not None and adjective not in row.get("adjective_lemmas", ()):
                    continue
                yield CorpusTemplate(row)

    def verbs(self):
        return read_rows(self.path / "verbs.jsonl.gz")

    def adjectives(self):
        return read_rows(self.path / "adjectives.jsonl.gz")

    def generate(self, *, count=1, seed=0, lemma=None, adjective=None, template_id=None, recombine=False,
                 exclude_controlled=False):
        """Exactly count outputs for EVERY selected template; no hidden top-N."""
        if count < 1:
            raise ValueError("count per template must be positive")
        adjective_forms = None
        if adjective is not None:
            adjective_forms = next((set(row["forms"]) for row in self.adjectives()
                                    if row["lemma"] == adjective), set())
            if not adjective_forms:
                raise ValueError(f"no adjective {adjective!r} in the bank")
        found = False
        for template in self.templates(lemma=lemma, adjective=adjective, template_id=template_id):
            if exclude_controlled:
                from .controlled_verbs import excluded_predicates
                if excluded_predicates(template.record):
                    continue
            found = True
            if not recombine:
                # Enumerate without replacement: an available binding must not
                # be missed merely because repeated random draws collided.
                choices = list(template.attested(adjective_forms))
                random.Random(fingerprint([seed, template.template_id])).shuffle(choices)
                if len(choices) < count:
                    raise ValueError(f"template {template.template_id}: {len(choices)} attested bindings "
                                     f"available for {count} requested; reduce count or enable recombination")
                for binding in choices[:count]:
                    yield template.render(binding)
                continue
            seen = set()
            for variant in range(count * 20 + 100):
                result = template.sample(seed=seed, variant=variant, recombine=recombine,
                                         adjective_forms=adjective_forms)
                if result["text"] in seen:
                    continue
                seen.add(result["text"])
                yield result
                if len(seen) == count:
                    break
            else:
                raise ValueError(f"template {template.template_id}: found {len(seen)}/{count} distinct outputs "
                                 "within the sampling budget; "
                                 "reduce count or explicitly enable recombination")
        if not found:
            raise ValueError("no templates match the selection")


def generate_file(bank, output, *, count=1, seed=0, lemma=None, adjective=None, template_id=None,
                  recombine=False, progress=None):
    """Stream the whole selected bank to a new file, refusing overwrites."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if output.suffix == ".gz" else open
    written = 0
    with opener(output, "xt", encoding="utf-8") as stream:
        for row in bank.generate(count=count, seed=seed, lemma=lemma, adjective=adjective, template_id=template_id,
                                 recombine=recombine):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if progress and written % 10000 == 0:
                progress(written)
    return written
