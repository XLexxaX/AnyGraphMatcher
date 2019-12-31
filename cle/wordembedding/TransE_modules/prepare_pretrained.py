import gensim
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import shutil
import random
import decimal

def add_line_count(filepath):
    e_ctr =0
    with open(filepath, encoding="UTF-8", mode="r") as f:
        for line in f:
            e_ctr += 1
    content=""
    with open(filepath, encoding="UTF-8", mode="r") as f:
        content = f.read()
    with open(filepath+".tmp", encoding="UTF-8", mode="w+") as f:
        f.write(str(e_ctr) + '\n' + content)
    os.remove(filepath)
    shutil.copy(filepath+".tmp", filepath)
    os.remove(filepath+".tmp")


def load(rundir, w2vmodelpath, source_triples_path,target_triples_path, dim):
    #modelpath ="D:\\Development\\Code\\melt\\examples\\simpleJavaMatcher\\oaei-resources\\AnyGraphMatcher\\result_data\\cli_task_2019_11_10_21_00_25_192916\\w2v.model"
    modelpath = w2vmodelpath

    os.mkdir(rundir+"data\\")
    os.mkdir(rundir+"data\\FB15K\\")
    os.mkdir(rundir+"data\\outputData\\")

    rundir = rundir + "data\\FB15K\\"
    #D:\\Development\\Code\\melt\\examples\\simpleJavaMatcher\\oaei-resources\\AnyGraphMatcher\\result_data\\cli_task_2019_11_01_12_55_35_637929\\



    m = gensim.models.Doc2Vec.load(modelpath)

    embeddings_vocab = dict()
    e_ctr=-1
    r_ctr=-1
    with open(rundir + "relation_embeddings.nt", mode="w+", encoding="UTF-8") as f_r_embs:
        with open(rundir + "entity_embeddings.nt", mode="w+", encoding="UTF-8") as f_e_embs:
            with open(rundir + "train2id.txt", mode="w+", encoding="UTF-8") as f_resourcified:
                with open(source_triples_path, mode="r", encoding="UTF-8") as f:
                   for line in f:
                            line = line.replace("\n","").split(" ")
                            s = "s/"+line[0]
                            p = "p/"+line[1]
                            o = ["o/"]+line[2:]
                            if not s in embeddings_vocab.keys():
                                e_ctr=e_ctr+1
                                embeddings_vocab[s] = e_ctr
                                f_e_embs.write(s + "\t" + str(e_ctr) + "\t" +  "\t".join([str(x) for x in m.infer_vector([s])]) + "\n")
                            if not p in embeddings_vocab.keys():
                                r_ctr=r_ctr+1
                                embeddings_vocab[p] = r_ctr
                                f_r_embs.write(p + "\t" + str(r_ctr) + "\t" +  "\t".join([str(x) for x in m.infer_vector([p])]) + "\n")
                            if not "".join(o) in embeddings_vocab.keys():
                                e_ctr=e_ctr+1
                                embeddings_vocab["".join(o)] = e_ctr
                                f_e_embs.write(str(embeddings_vocab["".join(o)]) + "\t" + str(e_ctr) + "\t" +  "\t".join([str(x) for x in m.infer_vector(o)]) + "\n")

                            f_resourcified.write(str(embeddings_vocab[s])+ " " + str(embeddings_vocab["".join(o)])+" "+str(embeddings_vocab[p]) + "\n")

                            if e_ctr%1000 == 0:
                                print("Written "+str(e_ctr)+" docs", end="\r")


            with open(target_triples_path, mode="r", encoding="UTF-8") as f:
                with open(rundir + "train2id.txt", mode="a+", encoding="UTF-8") as f_resourcified:
                    for line in f:
                        line = line.replace("\n","").split(" ")
                        s = "s/"+line[0]
                        p = "p/"+line[1]
                        o = ["o/"]+line[2:]
                        if not s in embeddings_vocab.keys():
                            e_ctr=e_ctr+1
                            embeddings_vocab[s] = e_ctr
                            f_e_embs.write(s + "\t" + str(e_ctr) + "\t" + "\t".join([str(x) for x in m.infer_vector([s])]) + "\n")
                        if not p in embeddings_vocab.keys():
                            r_ctr=r_ctr+1
                            embeddings_vocab[p] = r_ctr
                            f_r_embs.write(p + "\t" + str(r_ctr) + "\t" +  "\t".join([str(x) for x in m.infer_vector([p])]) + "\n")
                        if not "".join(o) in embeddings_vocab.keys():
                            e_ctr=e_ctr+1
                            embeddings_vocab["".join(o)] = e_ctr
                            f_e_embs.write(str(embeddings_vocab["".join(o)]) + "\t" + str(e_ctr) + "\t" +  "\t".join([str(x) for x in m.infer_vector(o)]) + "\n")

                        f_resourcified.write(str(embeddings_vocab[s])+ " " + str(embeddings_vocab["".join(o)])+" "+str(embeddings_vocab[p]) + "\n")

                        if e_ctr%1000 == 0:
                            print("Written "+str(e_ctr)+" docs", end="\r")


    shutil.copy(rundir+"entity_embeddings.nt", rundir+"entity2id.txt")
    shutil.copy(rundir+"relation_embeddings.nt", rundir+"relation2id.txt")
    #os.remove(rundir+"entity_embeddings.nt")
    #os.remove(rundir+"relation_embeddings.nt")

    add_line_count(rundir+"train2id.txt")
    add_line_count(rundir+"entity2id.txt")
    add_line_count(rundir+"relation2id.txt")


    #os.remove(rundir+"train2id.txt")
    #os.remove(rundir+"entity2id.txt")
    #os.remove(rundir+"relation2id.txt")
