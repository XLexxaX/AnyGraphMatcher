from cle.tokenization.spacytoken import tokenize
#from cle.tokenization.nltktoken import tokenize

def test_camel_case():
    assert list(tokenize("hello")) == [['hello']]

    # snake case
    assert list(tokenize("MA_0002738")) == [['ma', '0002738']]
    assert list(tokenize("this_is_word_word")) == [['this', 'is', 'one', 'word']]
    assert list(tokenize("Right_Coronary_Artery")) == [['right', 'coronary', 'artery']]

    # camel case
    assert list(tokenize("ThisIsOneWord")) == [['this', 'is', 'one', 'word']]
    assert list(tokenize("ThisIsOneWord in a full sentence. Second sentence here.")) == [['this', 'is', 'one', 'word', 'in', 'a', 'full', 'sentence', '.'], ['second', 'sentence', 'here', '.']]

    assert list(tokenize("body cavity/lining")) == [['body', 'cavity', '/', 'lining']]
    assert list(tokenize("abdomen/pelvis/perineum muscle")) == [['abdomen', '/', 'pelvis', '/', 'perineum', 'muscle']]

    assert list(tokenize("Meta-Gutachater (Gutachter der Gutachten begutachtet)")) == [['meta', '-', 'gutachater', '(', 'gutachter', 'der', 'gutachten', 'begutachtet', ')']]
