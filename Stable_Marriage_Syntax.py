
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import *


# In[2]:


def match(basedir):
	gs = pd.read_csv("C:/Users/D072202/RData2Graph/rdata2graph/data/oaei_data/oaei_gold_standard5best.csv", encoding="UTF-8", sep="\t", header=None)
	gs.columns = ['src_id','tgt_id','prediction']
	embs = pd.read_csv(basedir+"stratified_embeddings.csv", encoding="UTF-8", sep=",")
	embs = embs[[col for col in embs.columns if re.match('x\d+', col) is not None]+['label']]
	embs.columns = ["src_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if re.match('x\d+', col) is not None]] + ['label']
	gs = gs.merge(embs, left_on=['src_id'], right_on=['label'])
	embs.columns = ["tgt_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if re.match('src_\d+', col) is not None]] + ['label']
	gs = gs.merge(embs, left_on=['tgt_id'], right_on=['label'])
	gs.head()


	# In[3]:


	def extend_features(df):
		src_pattern = "src_\d+"
		tgt_pattern = "tgt_\d+"
		src_dim = int(len([elem for elem in [re.match(src_pattern, elem) is not None for elem in df.columns.values.tolist()] if elem==True]))
		tgt_dim = int(len([elem for elem in [re.match(tgt_pattern, elem) is not None for elem in df.columns.values.tolist()] if elem==True]))


		def dotproduct(v1, v2):
			result = list()
			for i in range(len(v1)):
				result.append([np.dot(v1[i], v2[i])])
			return np.array(result)

		def length(v):
			return np.sqrt(dotproduct(v, v))

		def angle(v1, v2):
			return np.arctan(dotproduct(v1, v2) / (length(v1) * length(v2)))

		a = np.array(df[["src_" + str(i) for i in range(src_dim)]].values.tolist())
		b = np.array(df[["tgt_" + str(i) for i in range(tgt_dim)]].values.tolist())
		#print(".")
		df['src_tgt_angle'] = paired_cosine_distances(a, b)
		#print(".")
		#src_origin = np.full((len(df), src_dim), 0.0000001)
		#tgt_origin = np.full((len(df), tgt_dim), 0.0000001)
		#df['src_angle_to_origin'] = cosine_similarity(tgt_origin,a).diagonal()
		##print(".")
		#df['tgt_angle_to_origin'] = cosine_similarity(src_origin,b).diagonal()
		df['src_veclen'] = length(a)
		df['tgt_veclen'] = length(b)
		df['src_tgt_veclen'] = paired_euclidean_distances(a,b)#.diagonal()#length(a-b)
		df.head()

		df.fillna(0, inplace = True)
		return df


	# In[4]:


	gs = extend_features(gs)
	len(gs)


	# In[5]:


	memo = {}
	def lev(s,t):
		return levenshtein(s,t)/max(len(s),len(t))

	def levenshtein(s, t):
		if s == "":
			return len(t)
		if t == "":
			return len(s)
		cost = 0 if s[-1] == t[-1] else 1

		i1 = (s[:-1], t)
		if not i1 in memo:
			memo[i1] = levenshtein(*i1)
		i2 = (s, t[:-1])
		if not i2 in memo:
			memo[i2] = levenshtein(*i2)
		i3 = (s[:-1], t[:-1])
		if not i3 in memo:
			memo[i3] = levenshtein(*i3)
		res = min([memo[i1]+1, memo[i2]+1, memo[i3]+cost])

		return res
	gs['syntactic_diff'] = gs.apply(lambda row: lev(row['src_id'], row['tgt_id']), axis=1)


	# In[6]:


	len(set(gs.src_id))
	from gensim.models import Doc2Vec, Word2Vec
	model = Word2Vec.load(basedir+"w2v.model")
	#def get_training_material(nid):
	#            res = list()
	#            with open(basedir+"w2v_training_material.csv", mode="r", encoding="UTF-8") as f:
	#                for line in f:
	#                    if nodeid in line.split(" "):
	#                        res = res + line.split(" ")
	#                return list(set(res))

	def mergedf(df1, df2):
				if df1 is None:
					return df2
				else:
					return df1.append(df2, ignore_index=True)


	# In[7]:


	progress = 0
	matchings = None
	total = len(set(gs.src_id))
	for nodeid in set(gs.src_id):#.union(gs.tgt_id)
					possible_matches_for_nodeid = gs.loc[gs.src_id==nodeid]
					#possible_matches.loc[((possible_matches.src_id==nodeid) & (possible_matches.tgt_id.isin(get_possible_matches(nodeid))))]



					progress += 1
					if len(possible_matches_for_nodeid)<1:
						continue
					
					##print(str(progress), end="\r")
					#print("         Computing rank-features: " + str(int(100*progress/total)) + "%.", end='\r')
					# In[312]:



					#model.docvecs.most_similar(0)


					# In[313]:


					##print('Closest in general:')
					#for val in model.docvecs.most_similar(i):
					#    try:
					#        #print(documents_ids_A[int(val[0])])
					#    except:
					#        try:
					#            #print(documents_ids_B[int(val[0])])
					#        except:
					#            #print(str(val[0]) + " not found")


					# In[314]:


					##print('Closest in terms of cosine similarity:')
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
					sorted_x = possible_matches_for_nodeid.sort_values(by=['syntactic_diff'], ascending=False)
					sorted_x.loc[:,'syntax_score'] = 0
					maximum = sorted_x.head(1).src_tgt_angle.values[0]
					#sorted_x.loc[:,'diff_to_max'] = 1.0 - sorted_x.loc[:, 'src_tgt_angle'] / maximum
					for index, row in sorted_x.iterrows():
						##print(row[1] + " - " + str(row['cos_sim']))
						sorted_x.loc[index, 'syntax_score'] = row['syntax_score'] + 1/ctr
						ctr += 1


					# In[315]:


					##print('Closest in terms of Euclidean distance:')#print('Closest in terms of Euclidean distance:')
					#sorted_x2 = sorted_x
					#vecs = model.wv[get_possible_matches(nodeid)]
					#x = euclidean_distances(np.array([model.wv[nodeid]]), vecs)
					#x = np.concatenate((x, np.array([get_possible_matches(nodeid)])), axis=0)
					#sorted_x = pd.DataFrame(x).T.sort_values(by=[0], ascending=True)
					#sorted_x = possible_matches_for_nodeid.sort_values(by=['src_tgt_veclen'], ascending=True)
					#sorted_x.loc[:,'euclid_score'] = 0
					#ctr = 1
					##sorted_x.columns = ['euclid_sim' if col==0 else col for col in sorted_x.columns]
					#for index, row in sorted_x.iterrows():
					#    ##print(row[1] + " - " + str(row['euclid_sim']))
					#    sorted_x.loc[index, 'euclid_score'] = row['euclid_score'] + 1/ctr
					#    ctr += 1



					###print('Closest in terms of syntax:')
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
					#    ##print(row[1] + " - " + str(row['syntax_diff']))
					#    sorted_x.loc[index, 'syntax_score'] = row['syntax_score'] + 1/ctr
					#    ctr += 1



					#sorted_x3 = sorted_x
					##sorted_x = possible_matches_for_nodeid
					#ctr = 1
					#sorted_x.loc[:,'probability_score'] = 0
					#sorted_x.loc[:,'probability'] = 0
					##print('Closest in terms of output probability:')
					#for tuple in model.predict_output_word(get_training_material(nodeid), topn=99999999):
					#    if tuple[0] in possible_matches_for_nodeid.tgt_id.to_list():
					#        sorted_x.loc[sorted_x.tgt_id==tuple[0], 'probability'] = float(tuple[1])
					#        sorted_x.loc[sorted_x.tgt_id==tuple[0], 'probability_score'] = 1/ctr
					#        ctr = ctr + 1


					# In[316]:


					##print('Closest in sum:')
					#x = sorted_x[['probability_score','probability']].merge(sorted_x3['euclid_score'].to_frame().merge(sorted_x2, left_index=True, right_index=True), left_index=True, right_index=True)
					x = sorted_x
					x.loc[:,'total_score'] = x['syntax_score'] #x['cos_score'] + x['euclid_score'] + x['probability_score']
					sorted_x = x.sort_values(by=['total_score'], ascending=False)
					#sorted_x.columns = ['tgt_id' if col==1 else col for col in sorted_x.columns]
					for index, row in sorted_x.iterrows():#sorted_x.loc[sorted_x.total_score == max(sorted_x.total_score.values),:].iterrows():
						matching_pair = pd.DataFrame([sorted_x.loc[index]])
						matching_pair.loc[:,'src_id'] = nodeid
						##print(nodeid + "\t" + row[1] + "\t" + str(row['total_score']) + "\t" + str(row['cos_score']) + "\t" + str(row['euclid_score']))
						matchings = mergedf(matchings, matching_pair)
					


	# In[ ]:


	matchings_saved=matchings
	matchings = matchings.sort_values(by=['total_score','src_tgt_angle'], ascending=[False, False])
	married_matchings = None
	ctr = 0
	while len(matchings) > 0:
					ctr += 1
					row = matchings.head(1)
					married_matchings = mergedf(married_matchings, pd.DataFrame(row))
					matchings = matchings.loc[~(matchings.src_id == row.src_id.values[0]) & ~(matchings.tgt_id == row.tgt_id.values[0])]


	# In[ ]:


	import os
	def create_elem(src_id, tgt_id):
		elem = '<map>\n<Cell>\n<entity1 rdf:resource="'+src_id+'"/>\n'
		elem = elem + '<entity2 rdf:resource="'+tgt_id+'"/>\n<relation>=</relation>\n'
		elem = elem + '<measure rdf:datatype="xsd:float">1.0</measure>\n</Cell>\n</map>'
		return elem

	matchings_filename ="postsyntaxmarried_matchings.csv"
	married_matches = pd.read_csv(basedir + matchings_filename, sep="\t", encoding="UTF-8")
	starttag = '<?xml version="1.0" encoding="utf-8"?>\n<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"\n  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n  xmlns:xsd="http://www.w3.org/2001/XMLSchema#">\n<Alignment>\n  <xml>yes</xml>\n  <level>0</level>\n  <type>??</type>\n  <onto1>\n    <Ontology rdf:about="darkscape">\n      <location>http://darkscape.wikia.com</location>\n    </Ontology>\n  </onto1>\n  <onto2>\n    <Ontology rdf:about="oldschoolrunescape">\n      <location>http://oldschoolrunescape.wikia.com</location>\n    </Ontology>\n  </onto2>\n'
	endtag = '</Alignment>\n</rdf:RDF>'
	os.mkdir(basedir + matchings_filename.replace(".csv",""))
	with open(basedir + matchings_filename.replace(".csv","") + str(os.sep) + 'darkscape~oldschoolrunescape~results.xml', "w+", encoding="UTF-8") as f:
			f.write(starttag)
			for index, row in married_matches.iterrows():
				f.write(create_elem(str(row.src_id).replace("&","&amp;"), str(row.tgt_id).replace("&","&amp;"))+"\n")
			f.write(endtag)

