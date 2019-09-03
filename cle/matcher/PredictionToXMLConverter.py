import pandas as pd
import numpy as np
import ntpath

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from configurations.PipelineTools import PipelineDataTuple
from matcher import Matchdata_Saver
import sys
import os
from joblib import dump, load

import pandas as pd
import os
import shutil
import numpy as np

global CONFIGURATION

def exec(matchings_filename):

    married_matches = pd.read_csv(CONFIGURATION.rundir + matchings_filename, sep="\t", encoding="UTF-8")
    starttag = '<?xml version="1.0" encoding="utf-8"?>\n<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"\n  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n  xmlns:xsd="http://www.w3.org/2001/XMLSchema#">\n<Alignment>\n  <xml>yes</xml>\n  <level>0</level>\n  <type>??</type>\n  <onto1>\n    <Ontology rdf:about="darkscape">\n      <location>http://darkscape.wikia.com</location>\n    </Ontology>\n  </onto1>\n  <onto2>\n    <Ontology rdf:about="oldschoolrunescape">\n      <location>http://oldschoolrunescape.wikia.com</location>\n    </Ontology>\n  </onto2>\n'
    endtag = '</Alignment>\n</rdf:RDF>'
    os.mkdir(CONFIGURATION.rundir + matchings_filename.replace(".csv",""))
    with open(CONFIGURATION.rundir + matchings_filename.replace(".csv","") + str(os.sep) + 'darkscape~oldschoolrunescape~results.xml', "w+", encoding="UTF-8") as f:
        f.write(starttag)
        for index, row in married_matches.iterrows():
            f.write(create_elem(str(row.src_id).replace("&","&amp;"), str(row.tgt_id).replace("&","&amp;"))+"\n")
        f.write(endtag)

    return PipelineDataTuple(None)


def create_elem(src_id, tgt_id):
    elem = '<map>\n<Cell>\n<entity1 rdf:resource="'+src_id+'"/>\n'
    elem = elem + '<entity2 rdf:resource="'+tgt_id+'"/>\n<relation>=</relation>\n'
    elem = elem + '<measure rdf:datatype="xsd:float">1.0</measure>\n</Cell>\n</map>'
    return elem

def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    #graph1 = main_input.get(0)
    #graph2 = main_input.get(1)
    matchings_filename = args.get(0)
    #assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    #assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.gold_mapping is not None, "Path to gold standard file not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.logfile is not None, "Path to logfile not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.name is not None, "Test config name not found in " + os.path.basename(sys.argv[0])
    assert matchings_filename is not None, "Matchings filename not found in " + os.path.basename(sys.argv[0])
    return exec(matchings_filename)
