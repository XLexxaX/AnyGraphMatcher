import pandas as pd
import numpy as np
import ntpath

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from pandas_ml import ConfusionMatrix

from configurations.PipelineTools import PipelineDataTuple
from matcher import PredictionToXMLConverter
from visualization import EmbeddingSaver
import sys
import os
from joblib import dump, load
from sklearn.metrics.pairwise import *
import re
from gensim.models import Doc2Vec, Word2Vec


global CONFIGURATION

def exec(graph1, graph2):


    graph1 = None
    graph2 = None

    return PipelineDataTuple(graph1, graph2)


def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.logfile is not None, "Path to logfile not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.name is not None, "Test config name not found in " + os.path.basename(sys.argv[0])
    return exec(graph1, graph2)
