import os
import fileinput
from cle.matcher.DatasetHelperTools import sample_gold_data
from shutil import copyfile
from cle.wordembedding import MUSEEmbeddingAligner
from cle.configurations.PipelineTools import PipelineDataTuple
import numpy as np
import pandas as pd
import sys
import pandas as pd
global CONFIGURATION

def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.gold_mapping is not None, "Path to gold standard file not found in " + os.path.basename(sys.argv[0])
    return execute(graph1, graph2)

def execute(graph1, graph2):
    for root, dir, files in os.walk(os.path.join(CONFIGURATION.musedir,"data","dumped"), topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dir:
            os.rmdir(os.path.join(root, name))

    try:
        os.remove(os.path.join(CONFIGURATION.musedir,"data","crosslingual","dictionaries","src-tgt.txt"))
    except FileNotFoundError:
        pass
    try:
        os.remove(os.path.join(CONFIGURATION.musedir, "data","embeddings1.vec"))
    except FileNotFoundError:
        pass
    try:
        os.remove(os.path.join(CONFIGURATION.musedir, "data","embeddings2.vec"))
    except FileNotFoundError:
        pass

    f = open(os.path.join(CONFIGURATION.musedir, "data","embeddings1.vec"), "w+")
    ctr = 0
    dim = None
    for descriptor, resource in graph1.elements.items():
        f.write(descriptor + " " + str(resource.embeddings).replace("[","").replace("]","").replace(",","") + " \n")
        ctr = ctr + 1
        if dim is None:
            dim = len(resource.embeddings[0])
    f.close()
    line_pre_adder(os.path.join(CONFIGURATION.musedir, "data","embeddings1.vec"), str(ctr) + " " + str(dim) + "\n")

    f = open(os.path.join(CONFIGURATION.musedir, "data","embeddings2.vec"), "w+")
    ctr = 0
    dim = None
    for descriptor, resource in graph2.elements.items():
        f.write(descriptor + " " + str(resource.embeddings).replace("[","").replace("]","").replace(",","") + " \n")
        ctr = ctr + 1
        if dim is None:
            dim = len(resource.embeddings[0])
    f.close()
    line_pre_adder(os.path.join(CONFIGURATION.musedir, "data","embeddings2.vec"), str(ctr) + " " + str(dim) + "\n")

    gs = None
    for path_to_gs in CONFIGURATION.gold_mapping.raw_trainsets:
        if gs is None:
            gs = pd.read_csv(path_to_gs, header=None, delimiter='\t')
        else:
            tmp_gs = pd.read_csv(path_to_gs, header=None, delimiter='\t')
            gs = gs.append(tmp_gs, ignore_index=True)
    gs = gs.loc[gs[2]==1]
    gs.to_csv(os.path.join(CONFIGURATION.musedir, "data","crosslingual","dictionaries","src-tgt.txt"), header=False, index=False, sep='\t')
    gs.to_csv(os.path.join(CONFIGURATION.musedir, "data","crosslingual","dictionaries","src-tgt.0-5000.txt"), header=False, index=False, sep='\t')



    align(os.path.join(CONFIGURATION.musedir,"data","embeddings1.vec"),os.path.join(CONFIGURATION.musedir,"data","embeddings2.vec"),dim)

    for root, dirs, files in os.walk(os.path.join(CONFIGURATION.musedir, "dumped","debug"), topdown=False):
        for dir in dirs:
            emb_dir = root + str(os.sep) + dir

    ctr = 0
    for line in open(emb_dir + str(os.sep) +"vectors-src.txt","r"):
        if ctr < 1:
            ctr = ctr + 1
            continue
        line = line.split()
        try:
            tmp = list()
            tmp.append(np.array(line[1:len(line)]).astype(float).tolist())
            graph1.elements[line[0]].embeddings = tmp
        except KeyError:
            print("key not found for " + line[0])

    ctr = 0
    for line in open(emb_dir + str(os.sep) + "vectors-tgt.txt","r"):
        if ctr < 1:
            ctr = ctr + 1
            continue
        line = line.split()
        try:
            tmp = list()
            tmp.append(np.array(line[1:len(line)]).astype(float).tolist())
            graph2.elements[line[0]].embeddings = tmp
        except KeyError:
            print("key not found for " + line[0])

    return PipelineDataTuple(graph1, graph2)


def line_pre_adder(filename, line_to_prepend):
        f = fileinput.input(filename, inplace=1)
        for xline in f:
            if f.isfirstline():
                print(line_to_prepend.lower() + xline.lower(), end='')
            else:
                print(xline.lower(), end='')

def align(src_emb, tgt_emb, dim):

    dico_train = os.path.join(CONFIGURATION.musedir,"data","crosslingual","dictionaries","src-tgt.txt")
    #dictionary will be stored in: 'MUSE/data/crosslingual/dictionaries/src-tgt.0-5000.txt'
    prepare_output_dir()

    sample_gold_data(dico_train)
    copyfile(os.path.join(CONFIGURATION.musedir,"data","crosslingual","dictionaries","src-tgt.txt.sampled"), os.path.join(CONFIGURATION.musedir,"data","crosslingual","dictionaries","src-tgt.txt"))
    os.remove(os.path.join(CONFIGURATION.musedir,"data","crosslingual","dictionaries","src-tgt.txt.sampled"))
    MUSEEmbeddingAligner.align(src_emb, tgt_emb, "src", "tgt", dim, dico_train)
    print("\n\n ------- Embedding creation process finished -------")



def prepare_output_dir():
        for root, dirs, files in os.walk(os.path.join(CONFIGURATION.musedir,"dumped","debug"), topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))



