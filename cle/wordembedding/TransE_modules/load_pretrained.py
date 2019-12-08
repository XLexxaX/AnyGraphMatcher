import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import shutil

def load(traindata):
    #traindata ="C:/Users/allue/oaei_track_cache/tmpdata/"
    #D:/Development/Code/melt/examples/simpleJavaMatcher/oaei-resources/AnyGraphMatcher/result_data/cli_task_2019_11_01_12_55_35_637929/



    e = pd.read_csv(traindata + "entity_embeddings.nt", encoding="UTF-8",header=None, sep="\t", index_col=False)
    #e[101] = e.index.values
    e[[0,1]].to_csv(path_or_buf=traindata+"entity2id_tmp.txt", encoding="UTF-8", sep="\t", columns=None, header=False, index=False)
    with open(traindata+"entity2id_tmp.txt", encoding="UTF-8", mode="r") as f:
        with open(traindata+"entity2id.txt", encoding="UTF-8", mode="w+") as f2:
            f2.write(str(len(e))+"\n"+f.read())
    os.remove(traindata+"entity2id_tmp.txt")

    weight = torch.FloatTensor(e.loc[:, 1:].to_numpy())
    pre_trained_embeddings_in_torch_format = nn.Embedding.from_pretrained(weight)
    rows, cols = len(e),100
    print(str(rows), "-------------", str(cols))
    #embedding = pre_trained_embeddings_in_torch_format# torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding = nn.Embedding(rows, cols)
    embedding.weight=nn.Parameter(weight)
    #embedding.load_state_dict({'weight': e2.loc[:, 1:].to_numpy()})
    #embedding.weight.requires_grad = True
    #embedding.weight = torch.nn.Parameter(pre_trained_embeddings_in_torch_format)
    del e
    e2 =  pd.read_csv(traindata + "relation_embeddings.nt", encoding="UTF-8",header=None, sep="\t", index_col=False)
    #e2[101] = e2.index.values
    e2[[0,1]].to_csv(path_or_buf=traindata+"relation2id_tmp.txt", encoding="UTF-8", sep="\t", columns=None, header=False, index=False)
    with open(traindata+"relation2id_tmp.txt", encoding="UTF-8", mode="r") as f:
        with open(traindata+"relation2id.txt", encoding="UTF-8", mode="w+") as f2:
            f2.write(str(len(e2))+"\n"+f.read())
    os.remove(traindata+"relation2id_tmp.txt")
    weight2 = torch.FloatTensor(e2.loc[:, 1:].to_numpy())
    pre_trained_embeddings_in_torch_format2 = nn.Embedding.from_pretrained(weight2)
    rows2, cols2 = len(e2),100
    print(str(rows2), "-------------", str(cols2))
    #embedding2 = pre_trained_embeddings_in_torch_format2#torch.nn.Embedding(num_embeddings=rows2, embedding_dim=cols2)
    embedding2 = nn.Embedding(rows2, cols2)
    embedding2.weight = nn.Parameter(weight2)
    #embedding2.load_state_dict({'weight': e2.loc[:, 1:].to_numpy()})
    #embedding2.weight.requires_grad = True
    #embedding2.weight = torch.nn.Parameter(pre_trained_embeddings_in_torch_format2)

    rundir=traindata
    shutil.copy(rundir+"entity2id.txt", rundir+"data/outputData/entity2id.txt")
    shutil.copy(rundir+"relation2id.txt", rundir+"data/outputData/relation2id.txt")

    return embedding, embedding2
