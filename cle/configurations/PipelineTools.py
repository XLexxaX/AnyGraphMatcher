class PipelineDataTuple(list):

    # Possible input looks like following:
    # list ( Tuple ( any, any, ... ), Tuple ( any, any, ... ), ... )
    # or
    # Tuple ( any, any, ... ), Tuple ( any, any, ... ) , ...
    #
    # All formats should boil down to the same tuple-object with Tuple.elems = [any,any,...]
    # I.e. make sure no nested Tuples can be created. If this is attemped, automatically flatten the tuple-elements.
    def __init__(self, *content):
        self.elems = list()
        for e in content:
            self.elems.append(e)
        #while True in [self.clean_tuples(e, e) for e in self.elems]:
        #    pass

    def clean_tuples(self, search_item, list_item, i=0, searchdepth=10):
        if i < searchdepth:
                if isinstance(search_item, PipelineDataTuple):
                    self.elems.remove(list_item)
                    for e in search_item.elems:
                        self.elems.append(e)
                    return True
                else:
                    try:
                        for t in search_item:
                            self.clean_tuples(t, list_item, i+1)
                    except TypeError:
                        return False
        else:
            return False

    def __repr__(self):
        return self.elems

    def get(self, i):
        try:
            return self.elems[i]
        except IndexError:
            return None

    def __str__(self):
        text = ""
        for e in self.elems:
            text = text + str(e)
        return text


class Step:

    def __init__(self, func, input_step, args):
        self.func = func # function
        self.input_step = input_step # Tuple(Step)
        self.args = args # Tuple(arbitrary)
        self.next_step = None # Step
        self.output = None
        self.persist_output = False


class Pipeline:

    def __init__(self):
        self.steps = list()

    def append_step(self, func, input_step, args):
        new_step = Step(func, input_step, args)
        if input_step is not None:
            for i_s in input_step.elems:
                i_s.persist_output = True
        if len(self.steps) > 0:
            self.get_last_step().next_step = new_step
        self.steps.append(new_step)
        return new_step

    def get_last_step(self):
        return self.steps[len(self.steps) - 1] if len(self.steps) > 0 else None

    def get_first_step(self):
        return self.steps[0] if len(self.steps) > 0 else None
