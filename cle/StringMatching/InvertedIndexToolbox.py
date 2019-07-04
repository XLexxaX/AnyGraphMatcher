

class InvertedIndex:

    def __init__(self, n=3):
        self.iindex = dict()
        self.N = n

    def addToIndex(self, value, index):
        ngrams = getNGrams(value, self.N)
        for gram in ngrams:
            if gram in self.iindex:
                s = self.iindex[gram]
                s.add(index)
                self.iindex[gram] = s
            else:
                s = set()
                s.add(index)
                self.iindex[gram] = s

    def getIndicesForNGram(self, ngram):
        if ngram in self.iindex.keys():
            return self.iindex[ngram]
        else:
            return set()

    def getIndicesForValue(self, value):
        ngrams = getNGrams(value, self.N)
        indices = list()
        for gram in ngrams:
            indices = indices + list(self.getIndicesForNGram(gram))
        return indices

def getNGrams(value, n=3):
    return [value[i:i+n] for i in range(len(value)-n+1)]
