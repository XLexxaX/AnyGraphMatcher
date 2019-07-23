from googletrans import Translator
import csv


#changes to googletrans: https://stackoverflow.com/questions/52455774/googletrans-stopped-working-with-error-nonetype-object-has-no-attribute-group/52487148#52487148

def get_lexicon_word_list(word_sorted_list_src, word_sorted_list_tgt, src_language, tgt_language, max_translations=5000):
    lexicon = list()
    word_set_lower_tgt = set([word.lower() for word in word_sorted_list_tgt])
    translator = Translator()
    for word in word_sorted_list_src:
        translation = translator.translate(word.lower(), src=src_language, dest=tgt_language).text
        if translation.lower() in word_set_lower_tgt:
            lexicon.append((word.lower(), translation.lower()))
        if len(lexicon) == max_translations:
            break
    return lexicon


def get_lexicon_keyed_vectors(word_vector_src, word_vector_tgt, src_language, tgt_language, max_translations=5000):
    """ Given source and target keyed vector, returns a lexicon
    for the k most frequent source words using Google Translate"""

    def get_sorted_words(keyed_vector):
        return sorted(keyed_vector.vocab, key=lambda word: keyed_vector.vocab[word].count, reverse=True)

    return get_lexicon_word_list(get_sorted_words(word_vector_src), get_sorted_words(word_vector_tgt),
                                 src_language, tgt_language, max_translations)


def write_lexicon_to_file(lexicon, out_file):
    with open(out_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for src, tgt in lexicon:
            writer.writerow([src, tgt])



# def __convert_train_test(in_train, in_test, out_file):
#     with open(out_file, 'w', newline='', encoding='utf-8') as csv_file:
#         writer = csv.writer(csv_file)
#
#         with open(in_train, "rb") as lex:
#             for (source, target) in pickle.load(lex):
#                 writer.writerow([source, target])
#
#         with open(in_test, "rb") as lex:
#             for (source, target) in pickle.load(lex):
#                writer.writerow([source, target])