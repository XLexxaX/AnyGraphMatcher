
from gensim.models import Doc2Vec, Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import editdistance
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from sklearn.linear_model import LogisticRegression




def exec(possible_matches, CONFIGURATION, graph1, graph2):

        # In[270]:
        additional_features = None
        progress = 0

        def mergedf(df1, df2):
            if df1 is None:
                return df2
            else:
                return df1.append(df2, ignore_index=True)

        basedir = CONFIGURATION.rundir
        current_process_dir = basedir
        dirpath = basedir

        documents_ids_A = dict()
        documents_ids_B = dict()
        all_possible_matches = dict()
        all_nodeids = set()
        possible_matches.loc[:, 'tgt_category'] = ''
        possible_matches.loc[:, 'src_category'] = ''
        for index, row in possible_matches.iterrows():
            src = row.src_id
            tgt = row.tgt_id

            try:
                src_category = \
                    graph1.elements[src].relations['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'].descriptor
            except:
                src_category = ""
            if not src_category=='http://www.w3.org/2002/07/owl#class' and \
                    not src_category=='http://www.w3.org/1999/02/22-rdf-syntax-ns#property':
                possible_matches.loc[index, 'src_category'] = 'RESOURCE'
            else:
                possible_matches.loc[index, 'src_category'] = src_category

            try:
                tgt_category = \
                    graph2.elements[tgt].relations['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'].descriptor
            except:
                tgt_category = ""
            if not tgt_category=='http://www.w3.org/2002/07/owl#class' and \
                    not tgt_category=='http://www.w3.org/1999/02/22-rdf-syntax-ns#property':
                possible_matches.loc[index, 'tgt_category'] = 'RESOURCE'
            else:
                possible_matches.loc[index, 'tgt_category'] = tgt_category

            if not possible_matches.loc[index, 'src_category'] == possible_matches.loc[index, 'tgt_category']:
                continue

            all_nodeids.add(src)
            if src in all_possible_matches.keys():
                all_possible_matches[src].add(tgt)
            else:
                all_possible_matches[src] = set([tgt])

            if tgt in all_possible_matches.keys():
                all_possible_matches[tgt].add(src)
            else:
                all_possible_matches[tgt] = set([src])

        possible_matches = possible_matches.loc[possible_matches.src_category == possible_matches.tgt_category]

        #resourceclasses = pd.read_csv(dirpath+'stratified_embeddings.csv')
        #resources = resourceclasses.loc[~(resourceclasses.category=='http://www.w3.org/1999/02/22-rdf-syntax-ns#property') & ~(resourceclasses.category=='http://www.w3.org/2002/07/owl#class')]
        #classes = resourceclasses.loc[(resourceclasses.category=='http://www.w3.org/2002/07/owl#class')]
        #properties = resourceclasses.loc[(resourceclasses.category=='http://www.w3.org/1999/02/22-rdf-syntax-ns#property')]
        #resources.head()


        #resources = possible_matches.loc[~(possible_matches.category=='http://www.w3.org/1999/02/22-rdf-syntax-ns#property') & ~(possible_matches.category=='http://www.w3.org/2002/07/owl#class')]
        #classes = possible_matches.loc[(possible_matches.category=='http://www.w3.org/2002/07/owl#class')]
        #properties = possible_matches.loc[(possible_matches.category=='http://www.w3.org/1999/02/22-rdf-syntax-ns#property')]



        def get_possible_matches(nid):
                final_matches = list(all_possible_matches[nid])
                #if nid in resources.label.tolist():
                #    for m in matches:
                #        if m in resources.label.tolist():
                #            pass
                #            final_matches.append(m)
#
                #if nid in classes.label.tolist():
                #    for m in matches:
                #        if m in classes.label.tolist():
                #            final_matches.append(m)
#
                #if nid in properties.label.tolist():
                #    for m in matches:
                #        if m in properties.label.tolist():
                #            final_matches.append(m)

                return final_matches


        # In[311]:


        def get_training_material(nid):
            res = list()
            with open(dirpath+"w2v_training_material.csv", mode="r", encoding="UTF-8") as f:
                for line in f:
                    if nodeid in line.split(" "):
                        res = res + line.split(" ")
                return list(set(res))
        model = Word2Vec.load(dirpath+"w2v.model")

        total = len(all_nodeids)
        matchings = None
        with open(dirpath+'additional_features.csv', mode="w+", encoding="UTF-8") as f:
            for nodeid in all_nodeids:

                possible_matches_for_nodeid = possible_matches.loc[((possible_matches.src_id==nodeid) & (possible_matches.tgt_id.isin(get_possible_matches(nodeid))))]



                progress += 1
                if len(get_possible_matches(nodeid))<1:
                    continue

                # In[312]:



                #model.docvecs.most_similar(0)


                # In[313]:


                #print('Closest in general:')
                #for val in model.docvecs.most_similar(i):
                #    try:
                #        print(documents_ids_A[int(val[0])])
                #    except:
                #        try:
                #            print(documents_ids_B[int(val[0])])
                #        except:
                #            print(str(val[0]) + " not found")


                # In[314]:


                #print('Closest in terms of cosine similarity:')
                #vecs = model.docvecs.doctag_syn0[np.array(get_possible_matches(nodeid))]
                #vecs = model.wv[get_possible_matches(nodeid)]
                #x = cosine_similarity(np.array([model.wv[nodeid]]), vecs)
                #x = np.concatenate((x, np.array([get_possible_matches(nodeid)])), axis=0)
                #sorted_x = pd.DataFrame(x).T.sort_values(by=[0], ascending=False)
                #sorted_x.loc[:,'cos_score'] = 0
                ctr = 1
                #sorted_x.columns = ['cos_sim' if col==0 else col for col in sorted_x.columns]
                #sorted_x.columns = ['cos_sim' if col==0 else col for col in sorted_x.columns]
                #sorted_x['cos_sim'] = sorted_x['cos_sim'].astype('float64')
                sorted_x = possible_matches_for_nodeid.sort_values(by=['src_tgt_angle'], ascending=False)
                sorted_x.loc[:,'cos_score'] = 0
                maximum = sorted_x.head(1).src_tgt_angle.values[0]
                sorted_x.loc[:,'diff_to_max'] = 1.0 - sorted_x.loc[:, 'src_tgt_angle'] / maximum
                for index, row in sorted_x.iterrows():
                    #print(row[1] + " - " + str(row['cos_sim']))
                    sorted_x.loc[index, 'cos_score'] = row['cos_score'] + 1/ctr
                    ctr += 1


                # In[315]:


                #print('Closest in terms of Euclidean distance:')print('Closest in terms of Euclidean distance:')
                sorted_x2 = sorted_x
                #vecs = model.wv[get_possible_matches(nodeid)]
                #x = euclidean_distances(np.array([model.wv[nodeid]]), vecs)
                #x = np.concatenate((x, np.array([get_possible_matches(nodeid)])), axis=0)
                #sorted_x = pd.DataFrame(x).T.sort_values(by=[0], ascending=True)
                sorted_x = possible_matches_for_nodeid.sort_values(by=['src_tgt_veclen'], ascending=True)
                sorted_x.loc[:,'euclid_score'] = 0
                ctr = 1
                #sorted_x.columns = ['euclid_sim' if col==0 else col for col in sorted_x.columns]
                for index, row in sorted_x.iterrows():
                    #print(row[1] + " - " + str(row['euclid_sim']))
                    sorted_x.loc[index, 'euclid_score'] = row['euclid_score'] + 1/ctr
                    ctr += 1



                ##print('Closest in terms of syntax:')
                #sorted_x3 = sorted_x
                ##vecs = model.wv[get_possible_matches(nodeid)]
                #def edits(v1, v2s):
                #    res = list()
                #    v1 = v1.split("/")[-1]
                #    for v2 in v2s:
                #        v2 = v2.split("/")[-1]
                #        res.append(editdistance.eval(v1, v2)/min(len(v1), len(v2)))
                #    return np.array([res])
                ##x = edits(nodeid, get_possible_matches(nodeid))
                ##x = np.concatenate((x, np.array([get_possible_matches(nodeid)])), axis=0)
                ##sorted_x = pd.DataFrame(x).T.sort_values(by=[0], ascending=True)
                #sorted_x = possible_matches_for_nodeid.sort_values(by=['syntactic_diff'], ascending=True)
                #sorted_x.loc[:,'syntax_score'] = 0
                #ctr = 1
                ##sorted_x.columns = ['syntax_diff' if col==0 else col for col in sorted_x.columns]
                #for index, row in sorted_x.iterrows():
                #    #print(row[1] + " - " + str(row['syntax_diff']))
                #    sorted_x.loc[index, 'syntax_score'] = row['syntax_score'] + 1/ctr
                #    ctr += 1



                sorted_x3 = sorted_x
                sorted_x = possible_matches_for_nodeid
                ctr = 1
                sorted_x.loc[:,'probability_score'] = 0
                sorted_x.loc[:,'probability'] = 0
                #print('Closest in terms of output probability:')
                for tuple in model.predict_output_word(get_training_material(nodeid), topn=99999999):
                    if tuple[0] in get_possible_matches(nodeid):
                        sorted_x.loc[sorted_x.tgt_id==tuple[0], 'probability'] = float(tuple[1])
                        sorted_x.loc[sorted_x.tgt_id==tuple[0], 'probability_score'] = 1/ctr
                        ctr = ctr + 1


                # In[316]:


                #print('Closest in sum:')
                x = sorted_x[['probability_score','probability']].merge(sorted_x3['euclid_score'].to_frame().merge(sorted_x2, left_index=True, right_index=True), left_index=True, right_index=True)
                x.loc[:,'total_score'] = x['cos_score'] + x['euclid_score'] + x['probability_score']
                sorted_x = x.sort_values(by=['total_score'], ascending=False)
                #sorted_x.columns = ['tgt_id' if col==1 else col for col in sorted_x.columns]
                for index, row in sorted_x.iterrows():#sorted_x.loc[sorted_x.total_score == max(sorted_x.total_score.values),:].iterrows():
                    matching_pair = pd.DataFrame([sorted_x.loc[index]])
                    matching_pair.loc[:,'src_id'] = nodeid
                    #print(nodeid + "\t" + row[1] + "\t" + str(row['total_score']) + "\t" + str(row['cos_score']) + "\t" + str(row['euclid_score']))
                    matchings = mergedf(matchings, matching_pair)

                print("         Computing rank-features: " + str(int(100*progress/total)) + "%.", end='\r')



        matchings.loc[:,'universe'] = matchings.loc[:,'src_tgt_angle'] + matchings.loc[:,'src_tgt_veclen'] - matchings.loc[:,'syntactic_diff']
        #matchings = matchings.loc[matchings.syntax_diff < 0.3]


        #possible_matches = possible_matches.merge(matchings, left_on=['src_id', 'tgt_id'], right_on=['src_id', 'tgt_id'])


        matchings.to_csv(dirpath+"additional_features.csv")
        print("         Computing rank-features: 100%")

        return matchings
