import os
import datetime
import re
from cle.configurations.InternalGoldStandard import InternalGoldStandard
from cle.configurations.InternalGoldStandard import Values
import hashlib
import re



class Configuration:
    def __init__(self, name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim, pipeline,
                 src_properties, tgt_properties, use_streams=False, use_cache=True, calc_PLUS_SCORE=True):
        assert type(gold_mapping) is InternalGoldStandard, "Gold mapping must be provided as an InternalGoldStandard"
        self.name, self.src_corpus, self.tgt_corpus, self.src_triples, self.tgt_triples, self.gold_mapping, \
            self.dim, self.pipeline, self.src_properties, self.tgt_properties, self.use_streams, self.use_cache, \
            self.calc_PLUS_SCORE \
            = \
            name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim, pipeline,\
            src_properties, tgt_properties, use_streams, use_cache, calc_PLUS_SCORE
        self.gold_mapping.prepared_trainsets = list()
        self.gold_mapping.prepared_testsets = list()
        self.rundir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..','result_data',
                                                   self.name+'_'+re.sub(r"[-:\.\s]","_",str(datetime.datetime.now())))) + str(os.sep)
        self.projectdir = os.path.join(os.path.dirname( __file__ ), '..')
        self.logfile = os.path.abspath(os.path.join(self.rundir, 'results.log'))
        self.musedir = os.path.join(self.projectdir, "crosslingual","muse","MUSE")
        self.logs_ = None
        self.cachedir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','..','cache')) + str(os.sep)
        self.match_cross_product = len(self.gold_mapping.raw_testsets) == 0


    def log(self, text):
        if self.logs_ is not None:
            self.logs_.write(str(text))


    def to_string(self):
        #return '' + str(self.src_corpus) + '\n' + str(self.tgt_corpus) + '\n' + str(self.src_triples)\
        #       + '\n' + str(self.tgt_triples) + '\n' + str(self.gold_mapping) + '\n' + str(self.dim) + '\n' + str(self.pipeline)\
        #       + '\n' + str(self.src_properties) + '\n' + str(self.tgt_properties) + '\n#####################\n'
        return "\n".join(['Step: '+str(step.func.__code__).split("file ")[1].split(", line")[0].replace("\"","") for step in self.pipeline.steps])
