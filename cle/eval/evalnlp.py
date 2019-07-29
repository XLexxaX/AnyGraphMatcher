
from eval.rankingeval import RankingEvaluation
import os
import logging


logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))

def generate_eval_files(selection_sub_tracks=None, type_of_text_files='.tokenized.lowercased.processed.colloq.unique'):
    """Selection subtracks is a set/list of language pairs/tuples like [('en', 'fr')]"""
    gold_files = os.listdir(os.path.join(package_directory, '..', '..', 'data', 'wmt11', 'lexicon_wmt11'))
    if selection_sub_tracks:
        gold_files = [file_name for file_name in gold_files
                     if (file_name.split('_')[1], file_name.split('_')[2]) in selection_sub_tracks]

    for gold_file in gold_files:
        source_language, target_language = gold_file.split('_')[1], gold_file.split('_')[2]

        source_path = os.path.join(package_directory, '..', '..', 'data', 'wmt11', 'training-monolingual',
                                              'news.2011.' + source_language + '.shuffled' + type_of_text_files)
        target_path = os.path.join(package_directory, '..', '..', 'data', 'wmt11', 'training-monolingual',
                                              'news.2011.' + target_language + '.shuffled' + type_of_text_files)

        yield source_path, target_path, gold_file


def get_ranking(system_ranking, gold_not_null_list, ranking_metrics=[], macro=True):
    ranking_eval = RankingEvaluation()
    for system, gold in zip(system_ranking, gold_not_null_list):
        for i, (src, dst, rel, conf) in enumerate(gold):
            ranking_eval.add_ranking(system[i], dst)
        ranking_eval.close_track()

    return [ranking_eval.get_eval(ranking_metric, macro) for ranking_metric in ranking_metrics]


def evaluate(system, selection_sub_tracks=None, type_of_text_files='.tokenized.lowercased.processed.colloq.unique'):

    for src, tgt, gold in generate_eval_files(selection_sub_tracks, type_of_text_files):
        initial_mapping = None
        ranking = system(src, tgt, initial_mapping)









if __name__ == '__main__':
    import pprint

    pprint.pCONFIGURATION.log(list(generate_eval_files()))
