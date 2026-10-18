"""Property bearers must follow clause structure rather than nearby subjects."""
import pytest

from semantic_corpus.template_compiler import adjective_target
from tests.test_template_bank import parsed


@pytest.mark.parametrize('words,heads,deps,pos,tags,lemmas,targets', [
    (['I','know','much'], [1,1,1], ['nsubj','ROOT','dobj'], ['PRON','VERB','ADJ'],
     ['PRP','VBP','JJ'], ['i','know','much'], {2:None}),
    (['I','work','at','least'], [1,1,1,2], ['nsubj','ROOT','prep','amod'],
     ['PRON','VERB','ADP','ADJ'], ['PRP','VBP','IN','JJS'], ['i','work','at','least'], {3:None}),
    (['I','find','it','interesting'], [1,1,1,1], ['nsubj','ROOT','dobj','oprd'],
     ['PRON','VERB','PRON','ADJ'], ['PRP','VBP','PRP','JJ'], ['i','find','it','interesting'], {3:2}),
    (['Keep','the','door','open'], [0,2,0,0], ['ROOT','det','dobj','oprd'],
     ['VERB','DET','NOUN','ADJ'], ['VB','DT','NN','JJ'], ['keep','the','door','open'], {3:2}),
    (['She','is','calm','and','happy'], [1,1,1,2,2], ['nsubj','ROOT','acomp','cc','conj'],
     ['PRON','AUX','ADJ','CCONJ','ADJ'], ['PRP','VBZ','JJ','CC','JJ'], ['she','be','calm','and','happy'], {2:0,4:0}),
    (['I','find','him','calm','and','happy'], [1,1,1,1,3,3], ['nsubj','ROOT','dobj','oprd','cc','conj'],
     ['PRON','VERB','PRON','ADJ','CCONJ','ADJ'], ['PRP','VBP','PRP','JJ','CC','JJ'], ['i','find','he','calm','and','happy'], {3:2,5:2}),
    (['Walking','is','nice'], [1,1,1], ['nsubj','ROOT','acomp'], ['VERB','AUX','ADJ'],
     ['VBG','VBZ','JJ'], ['walk','be','nice'], {2:None}),
    (['He','ate','it','raw'], [1,1,1,1], ['nsubj','ROOT','dobj','advcl'],
     ['PRON','VERB','PRON','ADJ'], ['PRP','VBD','PRP','JJ'], ['he','eat','it','raw'], {3:None}),
    (['It','is','more','and','more','difficult'], [1,1,5,2,2,1], ['nsubj','ROOT','advmod','cc','conj','acomp'],
     ['PRON','AUX','ADJ','CCONJ','ADJ','ADJ'], ['PRP','VBZ','JJR','CC','JJR','JJ'], ['it','be','much','and','much','difficult'], {2:None,4:None,5:0}),
    (['I','feel','happy'], [1,1,1], ['nsubj','ROOT','acomp'], ['PRON','VERB','ADJ'],
     ['PRP','VBP','JJ'], ['i','feel','happy'], {2:0}),
])
def test_supported_and_unresolved_bearers(words,heads,deps,pos,tags,lemmas,targets):
    doc=parsed(words,heads,deps,pos,tags,lemmas)
    for i, expected in targets.items():
        target=adjective_target(doc[i])
        assert (None if target is None else target.i)==expected
