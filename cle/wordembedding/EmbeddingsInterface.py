import numpy as np
import fileinput
import re

class EmbeddingsInterface:

    def __init__(self, pEmbeddings, dim):
        self.embeddings = pEmbeddings
        self.dim = dim

    def get_embedding(self, word):
        if self.embeddings is None:
            return []

        try:
            return list(self.embeddings[word])
        except KeyError:
            return list(np.zeros(self.dim))

    def line_pre_adder(self, filename, line_to_prepend):
        f = fileinput.input(filename, inplace=1)
        for xline in f:
            if f.isfirstline():
                print(line_to_prepend.lower() + xline.lower(), end='')
            else:
                print(xline.lower(), end='')

    def add_embedding(self, word, embedding):
        self.embeddings[word] = embedding

    def fill_vocab_from_triples(self, triples_path):
        self.embeddings = dict()
        NODEID_REGEXER = re.compile("^<[^<^>]*>")
        for line in open(triples_path):
            try:
                nodeid = NODEID_REGEXER.findall(line)[0].replace("\"", "")
                self.embeddings[nodeid] = []
            except:
                pass

    def write_to_file(self, path):
        embfile = open(path, "w+")
        ctr = 0
        dim = None
        for word in self.embeddings.keys():
            embfile.write(word + " " + str(self.embeddings[word]).replace(",", "").replace("[", "").replace("]", "") + " \n")
            ctr = ctr + 1
            if dim is None:
                dim = len(self.embeddings[word])
        self.line_pre_adder(path, str(ctr) + " " + str(dim) + " \n")
