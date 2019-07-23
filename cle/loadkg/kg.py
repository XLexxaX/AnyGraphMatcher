from itertools import chain

class KG:

    def __init__(self, kg_generator_function):
        self.kg_generator_function = kg_generator_function

    def get_object_and_literal_triples(self):
        return self.kg_generator_function()

    def get_object_triples(self):
        object_triples, _ = self.kg_generator_function()
        return object_triples

    def get_literal_triples(self):
        _, literal_triples = self.kg_generator_function()
        return literal_triples

    def get_literal_triples_with_fragments(self):
        for s, p, o in self.get_literal_triples():
            yield s, p, o
            fragments = s.split('#')
            if len(fragments) > 1:
                yield s, 'http://purl.org/dc/terms/identifier', fragments[-1]
        for s, p, o in self.get_object_triples():
            fragments = s.split('#')
            if len(fragments) > 1:
                yield s, 'http://purl.org/dc/terms/identifier', fragments[-1]


class KGInMemory(KG):

    def __init__(self, kg_generator_function):
        obj_triples, lit_triples = kg_generator_function()
        self.object_triples, self.literal_triples = list(obj_triples), list(lit_triples)
        #self.object_triples, self.literal_triples = kg_generator_function()

    def get_object_and_literal_triples(self):
        return self.object_triples, self.literal_triples

    def get_object_triples(self):
        return self.object_triples

    def get_literal_triples(self):
        return self.literal_triples


class CombinedKG(KG):

    def __init__(self, kg_one, kg_two):
        self.kg_one, self.kg_two = kg_one, kg_two

    def get_object_and_literal_triples(self):
        return chain(self.kg_one.object_triples, self.kg_two.object_triples), \
               chain(self.kg_one.literal_triples, self.kg_two.literal_triples)

    def get_object_triples(self):
        return chain(self.kg_one.object_triples, self.kg_two.object_triples)

    def get_literal_triples(self):
        return chain(self.kg_one.literal_triples, self.kg_two.literal_triples)