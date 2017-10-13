'''Reading files from the folder using os library'''
import os
DIR = "/home/sunny/ML Python/ch03/toy/"
posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]


'''Counting words and represting them as vector'''
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem													# NLTK is used for stemming of words


'''this section works on stemming the words using porter stemmer where are overwrite method build_analyzer'''
english_stemmer =  nltk.stem.PorterStemmer()
class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedCountVectorizer(min_df = 1, stop_words='english')		#min_df parameter will drop words whose count are less than min_df

#content = ["How to format my hard disk", "Hard disk format problems"]
X_train = vectorizer.fit_transform(posts)
#print vectorizer.get_feature_names()
num_samples, num_features = X_train.shape
#print num_samples, num_features
#print X.toarray().transpose()

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post]) #transform method are sparse it will use coo_matrix

import scipy as sp
#function to find distance between two vectors
def dist_raw(v1, v2):
	delta = v1 -v2
	return sp.linalg.norm(delta.toarray())

#fuvtion to find distance between two vectors by normalizing the count vectors
def dist_norm(v1, v2):
	v1_normalized = v1/sp.linalg.norm(v1.toarray())
	v2_normalized = v2/sp.linalg.norm(v2.toarray())
	delta = v1_normalized - v2_normalized
	return sp.linalg.norm(delta.toarray())

'''Find the best match with the minimum distance'''
import sys
best_doc = None
best_dist = sys.maxint
best_i = None
for i in range(0, num_samples):
	post = posts[i]
	if post == new_post:
		continue
	post_vec = X_train.getrow(i)	#getting the corresponding vector
	d = dist_norm(post_vec, new_post_vec)
	print i, d, post
	if d<best_dist:
		best_dist = d
		best_i = i

#print vectorizer.get_feature_names()