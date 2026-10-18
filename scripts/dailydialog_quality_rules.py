"""Narrow syntax analogues of v36 QC findings; never semantic admission.

Complex coordination, apposition, conditional scope and noun-phrase attachment
remain contextual review candidates. Existing reviews and warning decisions
are excluded by the caller in addition to the guards below.
"""
from scripts.dailydialog_offline_rules import apply_edits

VERSION = 'hybrid_quality_patterns_v1'


def proposals(payload):
    ledger = payload.get('ledger', {})
    ts = ledger.get('tokens', [])
    if (payload.get('method') != 'local_spacy_syntax'
            or payload.get('review_status') != 'UNREVIEWED'
            or not ts or any('dep' not in t for t in ts)
            or any(c in ledger.get('text', '') for c in ('"', '\u201c', '\u201d', '\ufffd'))
            or any(k in ledger for k in ('source_revision', 'attribute_revision', 'attribute_target_revision', 'context_review'))
            or payload.get('record', {}).get('usage_constraints')):
        return []
    children = {i: [] for i in range(len(ts))}
    for t in ts:
        if t['head'] != t['i']:
            children[t['head']].append(t)
    edits = {}

    def change(t, rule, **fields):
        fields = {k:v for k,v in fields.items() if t[k] != v}
        if not fields:
            return
        edit = edits.setdefault(t['i'], dict(token=t['i'], rules=[], set={}))
        if any(k in edit['set'] and edit['set'][k] != v for k,v in fields.items()):
            raise ValueError('Conflicting QC grammar rules')
        edit['set'].update(fields)
        if rule not in edit['rules']:
            edit['rules'].append(rule)

    for t in ts:
        i, w, head = t['i'], t['text'].lower(), ts[t['head']]
        if (w in {'chequing','checking'} and t['pos']=='VERB'
                and t['dep']=='amod' and head['text'].lower() in {'account','accounts'}
                and not children[i]):
            change(t, 'bank_account_nominal', pos='NOUN', tag='NN', dep='compound',
                   parser_lemma=w, morph='Number=Sing')
        if (w=='ground' and t['pos']=='NOUN' and t['dep']=='compound'
                and head['text'].lower()=='beef' and not children[i]):
            change(t, 'ground_beef_modifier', pos='ADJ', tag='JJ', dep='amod',
                   parser_lemma='ground', morph='Degree=Pos')
        if w in {'ok','okay'} and t['pos']=='PROPN' and t['dep'] in {'acomp','attr'} and head['parser_lemma']=='be':
            change(t, 'okay_evaluation', pos='ADJ', tag='JJ', dep='acomp',
                   parser_lemma=w, morph='Degree=Pos')
        if w=='birdseed' and t['pos']=='ADJ' and t['dep']=='attr' and head['parser_lemma']=='be':
            change(t, 'birdseed_material_noun', pos='NOUN', tag='NN',
                   parser_lemma='birdseed', morph='Number=Sing')
        if (w=='on' and t['dep']=='prep' and head['parser_lemma']=='try'
                and not children[i] and any(c['dep']=='dobj' for c in children[t['head']])):
            change(t, 'try_on_particle', pos='ADP', tag='RP', dep='prt', morph='')
        if w=='in' and t['dep']=='advmod' and head['parser_lemma']=='move' and not children[i]:
            # Particle syntax also applies to a weather front; no residential
            # location or completed move is inferred from the construction.
            change(t, 'move_in_particle', pos='ADP', tag='RP', dep='prt', morph='')
        if t['dep']=='xcomp' and t['tag']=='VB' and head['parser_lemma']=='stop':
            infinitive_to=[c for c in children[i] if c['text'].lower()=='to'
                           and c['tag']=='TO' and c['dep']=='aux' and c['pos']=='PART']
            if len(infinitive_to)==1 and not any(c['dep'] in {'nsubj','nsubjpass','mark'} for c in children[i]):
                change(t, 'stop_infinitive_purpose', dep='advcl')
                change(infinitive_to[0], 'stop_infinitive_purpose', dep='mark')
        if head['parser_lemma']=='be' and t['dep']=='xcomp' and t['tag']=='VB':
            adjectives=[c for c in children[t['head']] if c['dep']=='acomp'
                        and c['pos']=='ADJ' and c['text'].lower() in {'hard','easy','difficult'}]
            infinitives=[c for c in children[t['head']] if c['dep']=='xcomp']
            if (len(adjectives)==len(infinitives)==1
                    and not any(c['dep']=='conj' for c in children[adjectives[0]['i']])
                    and any(c['text'].lower()=='to' and c['dep']=='aux' and c['tag']=='TO' for c in children[i])
                    and not any(c['dep'] in {'nsubj','nsubjpass','mark'} for c in children[i])):
                change(t, 'evaluative_infinitive_attachment', head=adjectives[0]['i'])
                # Do not label "it" expletive without resolving its context.
        if (w=='dear' and t['pos']=='ADJ'
                and [x['text'].lower() for x in ts[:i] if x['pos']!='PUNCT']==['oh']
                and not any(c['dep'] not in {'punct','intj'} for c in children[i])):
            if t['dep']=='ROOT':
                change(t, 'oh_dear_interjection', pos='INTJ', tag='UH', morph='')
            elif head['pos'] in {'VERB','AUX','INTJ'}:
                change(t, 'oh_dear_interjection', pos='INTJ', tag='UH', dep='intj', morph='')
        # Only a bare greeting + one delimited name/title. No noun apposition,
        # "Oh God", or names in assertions are promoted by this rule.
        if t['pos']=='PROPN' and t['dep']=='npadvmod' and i==2 and ts[1]['text']==',':
            if (ts[0]['text'].lower() in {'hello','hi','hey'} and t['head']==0
                    and ts[0]['dep']=='ROOT' and ts[0]['pos']=='INTJ'
                    and not children[i] and all(x['pos']=='PUNCT' for x in ts[i+1:])):
                person=any(e['label']=='PERSON' and e['span']==t['span'] for e in payload.get('entities',[]))
                if person or w in {'sir','madam','miss'}:
                    change(t, 'greeting_vocative', dep='vocative')
    return list(edits.values())


def repair(payload, audit, vocab, *, protected=False):
    if protected:
        return None
    return apply_edits(payload, proposals(payload), audit, vocab, version=VERSION)
