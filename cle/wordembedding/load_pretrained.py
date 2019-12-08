import gensim
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import shutil

def load():
    modelpath ="D:/Development/Code/melt/examples/simpleJavaMatcher/oaei-resources/AnyGraphMatcher/result_data/cli_task_2019_11_10_21_00_25_192916/w2v.model"
    traindata ="C:/Users/allue/oaei_track_cache/tmpdata/"
    #D:/Development/Code/melt/examples/simpleJavaMatcher/oaei-resources/AnyGraphMatcher/result_data/cli_task_2019_11_01_12_55_35_637929/
    m = gensim.models.Doc2Vec.load(modelpath)

    embeddings_vocab = dict()
    e_ctr=-1
    r_ctr=-1
    # with open(traindata + "relation_embeddings.nt", mode="w+", encoding="UTF-8") as f_r_embs:
    #     with open(traindata + "entity_embeddings.nt", mode="w+", encoding="UTF-8") as f_e_embs:
    #         with open(traindata + "train2id_tmp.txt", mode="w+", encoding="UTF-8") as f_resourcified:
    #             with open(traindata + "graph_triples_source.nt", mode="r", encoding="UTF-8") as f:
    #                for line in f:
    #                         line = line.replace("\n","").split(" ")
    #                         s = "s_"+line[0]
    #                         p = "p_"+line[1]
    #                         o = ["o_"]+line[2:]
    #                         if not s in embeddings_vocab.keys():
    #                             e_ctr=e_ctr+1
    #                             embeddings_vocab[s] = e_ctr
    #                             f_e_embs.write(s + "\t" + str(e_ctr) + "\t" +  "\t".join([str(x) for x in m.infer_vector([s])]) + "\n")
    #                         if not p in embeddings_vocab.keys():
    #                             r_ctr=r_ctr+1
    #                             embeddings_vocab[p] = r_ctr
    #                             f_r_embs.write(p + "\t" + str(r_ctr) + "\t" +  "\t".join([str(x) for x in m.infer_vector([p])]) + "\n")
    #                         if not "".join(o) in embeddings_vocab.keys():
    #                             e_ctr=e_ctr+1
    #                             embeddings_vocab["".join(o)] = e_ctr
    #                             f_e_embs.write(str(embeddings_vocab["".join(o)]) + "\t" + str(e_ctr) + "\t" +  "\t".join([str(x) for x in m.infer_vector(o)]) + "\n")
    #                         f_resourcified.write(str(embeddings_vocab[s])+ " " + str(embeddings_vocab["".join(o)])+" "+str(embeddings_vocab[p]) + "\n")
    #
    #                         if e_ctr%1000 == 0:
    #                             print("Written "+str(e_ctr)+" docs", end="\r")
    #
    #
    #         with open(traindata + "graph_triples_target.nt", mode="r", encoding="UTF-8") as f:
    #             with open(traindata + "train2id_tmp.txt", mode="a+", encoding="UTF-8") as f_resourcified:
    #                 for line in f:
    #                     line = line.split(" ")
    #                     s = "s_"+line[0]
    #                     p = "p_"+line[1]
    #                     o = ["o_"]+line[2:]
    #                     if not s in embeddings_vocab.keys():
    #                         e_ctr=e_ctr+1
    #                         embeddings_vocab[s] = e_ctr
    #                         f_e_embs.write(s + "\t" + str(e_ctr) + "\t" + "\t".join([str(x) for x in m.infer_vector([s])]) + "\n")
    #                     if not p in embeddings_vocab.keys():
    #                         r_ctr=r_ctr+1
    #                         embeddings_vocab[p] = r_ctr
    #                         f_r_embs.write(p + "\t" + str(r_ctr) + "\t" +  "\t".join([str(x) for x in m.infer_vector([p])]) + "\n")
    #                     if not "".join(o) in embeddings_vocab.keys():
    #                         e_ctr=e_ctr+1
    #                         embeddings_vocab["".join(o)] = e_ctr
    #                         f_e_embs.write(str(embeddings_vocab["".join(o)]) + "\t" + str(e_ctr) + "\t" +  "\t".join([str(x) for x in m.infer_vector(o)]) + "\n")
    #                     f_resourcified.write(str(embeddings_vocab[s])+ " " + str(embeddings_vocab["".join(o)])+" "+str(embeddings_vocab[p]) + "\n")
    #
    #                     if e_ctr%1000 == 0:
    #                         print("Written "+str(e_ctr)+" docs", end="\r")
    #
    # with open(traindata+"train2id_tmp.txt", encoding="UTF-8", mode="r") as f:
    #     e_ctr =0
    #     for line in f:
    #         e_ctr += 1
    # with open(traindata+"train2id_tmp.txt", encoding="UTF-8", mode="r") as f:
    #     with open(traindata+"train2id.txt", encoding="UTF-8", mode="w+") as f2:
    #         f2.write(str(e_ctr)+"\n"+f.read())
    # os.remove(traindata+"train2id_tmp.txt")

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
    return embedding, embedding2
