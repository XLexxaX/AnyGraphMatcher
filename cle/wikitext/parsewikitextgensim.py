from gensim.corpora.wikicorpus import filter_wiki

def get_raw_text_and_links_from_markup(raw):
    return filter_wiki(raw), None