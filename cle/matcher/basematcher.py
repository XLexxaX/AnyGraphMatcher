
class BaseMatcher:

    def __init__(self):
        self.src_kg = None
        self.dst_kg = None
        self.initial_mapping = None

    def set_kg(self, src_kg, dst_kg):
        self.src_kg = src_kg
        self.dst_kg = dst_kg

    def set_initial_mapping(self, initial_mapping):
        self.initial_mapping = initial_mapping

    def generate_lexicon_from_initial_mapping(self):
        #return [(src, dst) for src, dst, rel, confidence in self.initial_mapping]
        for src, dst, rel, confidence in self.initial_mapping:
            yield (src, dst)

    def compute_mapping(self):
        pass

    def get_mapping(self):
        pass

    def get_mapping_with_ranking(self, elements, topn=20):
        pass


    #def get_mapping_with_one_ranking(self, element, topn=20):
    #    pass