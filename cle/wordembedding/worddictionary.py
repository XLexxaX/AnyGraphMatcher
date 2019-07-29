from six.moves import zip as izip
from six.moves import xrange
import logging
from operator import itemgetter

from gensim.corpora.dictionary import Dictionary

class WordDictionary():

    def __init__(self, sentences):
        self.current_max_id = 0
        self.token2id = {}
        self.id2token = {}
        self.word_freq = {}
        self.add_sentences(sentences)

    def add_sentences(self, sentences):
        for i, sentence in enumerate(sentences):
            if i % 10000 == 0:
                logging.info("adding sentence #%i", i)

            for word in sentence:
                word_id = self.token2id.get(word, None)
                if word_id is None:
                    word_id = self.current_max_id
                    self.token2id[word] = word_id
                    self.current_max_id += 1
                self.word_freq[word_id] = self.word_freq.get(word_id, 0) + 1

        logging.info("built WordDictionary from %i sentences (%i distinct words)", i, len(self.token2id))

    def get_word_ids(self, sentence, return_missing=False):# , allow_update=False, return_missing=False):
        #if allow_update:
        word_ids = []
        for word in sentence:
            word_id = self.token2id.get(word, None)
            if word_id is not None or return_missing:
                word_ids.append(word_id)# if there are only word ids in list, if they exist in the vocab then the window might increase later
        return word_ids

    def filter_extremes(self, min_count=None, n_most_frequent=None):
        if min_count is not None:
            self.filter_infrequent(min_count, compactify=False)
        if n_most_frequent is not None:
            self.filter_n_most_frequent(n_most_frequent, compactify=False)
        if min_count is not None or n_most_frequent is not None:
            self.compactify()

    def filter_n_most_frequent(self, remove_n=100, compactify=True):
        """Filter out the 'remove_n' most frequent tokens that appear in the documents."""
        most_frequent_items = sorted(self.word_freq.items(), key=itemgetter(1), reverse=True)[:remove_n]
        most_frequent_ids = {word_id for word_id, freq in most_frequent_items}
        # for freq_id in most_frequent_ids:
        #     CONFIGURATION.log("remove id" + str(freq_id) + " " + self.get_word_by_id(freq_id) + "(" + str(self.word_freq[freq_id]) + ")")
        logging.info("discarding %i most_frequent tokens", len(most_frequent_ids))
        self.filter_ids(most_frequent_ids, compactify)

    def filter_infrequent(self, no_below_or_equals_freq=10, compactify=True):
        infrequent_ids = { word_id for word_id, freq in self.word_freq.items() if freq <= no_below_or_equals_freq }
        logging.info("discarding %i infrequent tokens", len(infrequent_ids))
        self.filter_ids(infrequent_ids, compactify)


    def filter_ids(self, ids, compactify=True):
        self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid not in ids}
        self.word_freq = {tokenid: freq for tokenid, freq in self.word_freq.items() if tokenid not in ids}
        if compactify:
            self.compactify()

    def get_word_by_id(self, id):
        if len(self.id2token) != len(self.token2id):
            # the word->id mapping has changed (presumably via add_documents);
            # recompute id->word accordingly
            self.id2token = {v: k for (k, v) in self.token2id.items()}
        return self.id2token[id]  # will throw for non-existent ids

    def compactify(self):
        """Assign new word ids to all words, shrinking gaps."""

        # build mapping from old id -> new id
        idmap = dict(izip(sorted(self.token2id.values()), xrange(len(self.token2id))))

        # reassign mappings to new ids
        self.token2id = {token: idmap[tokenid] for token, tokenid in self.token2id.items()}
        self.id2token = {}
        self.word_freq = {idmap[tokenid]: freq for tokenid, freq in self.word_freq.items()}