from cle.crosslingual.muse.EmbeddingProcess import W2VMock, SimpleEmbeddingCombiner, PCACombiner, MUSEEmbeddingAligner, Gensim_Doc2Vec

import os
import re
import fileinput

def line_pre_adder(self, filename, line_to_prepend):
        f = fileinput.input(filename, inplace=1)
        for xline in f:
            if f.isfirstline():
                print(line_to_prepend.lower() + xline.lower(), end='')
            else:
                print(xline.lower(), end='')

def embed(self, triples_path, embpath, w2vi):
        vocab = set()
        NODEID_REGEXER = re.compile("^<[^<^>]*>")
        for line in open(triples_path):
            try:
                vocab.add(NODEID_REGEXER.findall(line)[0].replace("\"", ""))
            except:
                pass


        Gensim_Doc2Vec.reset()
        Gensim_Doc2Vec.init(triples_path, self.dim)
        Gensim_Doc2Vec.mock_init()

        embfile = open(embpath, "w+")
        ctr = 0
        for word in vocab:
                a = Gensim_Doc2Vec.get_embedding(word)
                b = w2vi.get_embedding(word)
                emb = a+b #SimpleEmbeddingCombiner.combine(a + b)
                embfile.write(word + " " + str(emb).replace(",","").replace("[","").replace("]","") + " \n")
                ctr = ctr + 1

        embfile.flush()
        embfile.close()
        self.line_pre_adder(embpath, str(ctr)+" " + str(self.dim) + " \n")
        PCACombiner.reduce_dimensionality(embpath, self.dim)

def prepare_output_dir(self):
        for root, dirs, files in os.walk("crosslingual/muse/MUSE/dumped/debug", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))



