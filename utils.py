from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
import scipy as sp
import itertools
from operator import itemgetter
import time
import subprocess
import os
import cPickle, re
import datetime
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import fastcluster
from sklearn.metrics.pairwise import cosine_similarity

###Helper Functions
def norm(a):
    return np.sqrt(np.sum(np.square(a)))


def cosine(a,b):
    return 1-np.dot(a,b)/np.sqrt(np.sum(a**2)*np.sum(b**2))

def l1(a,b):
    return abs(a-b).sum()

def l2(a,b):
    return np.sqrt(np.square(a-b).sum())


### Create a list of words to be clustered based on a model with some l2_threshold and can normalize the vectors and also repeat or no
def create_word_list(model,vocab,features,Texts,repeat=True,l2_threshold=0,normalized=True,min_count=100,min_length=0):
    data_d2v=[]
    word_d2v=[]
    words_text=[w for text in Texts for w in text]
    count=Counter(words_text)
    if repeat:
        for text in Texts:
            for w in text:
                if w in vocab and count[w]>min_count:
                    if len(w)>min_length and l2(model[w],np.zeros(features))>l2_threshold:
                        if normalized:
                            data_d2v.append(model[w]/l2(model[w],np.zeros(features)))
                        else:
                            data_d2v.append(model[w])
                        word_d2v.append(w)
    else:
        A=set(words_text)
        for w in vocab:
            if w in A and len(w)>min_length and l2(model[w],np.zeros(features))>l2_threshold and count[w]>min_count:
                if normalized:
                    data_d2v.append(model[w]/l2(model[w],np.zeros(features)))
                else:
                    data_d2v.append(model[w])
                word_d2v.append(w)

    return data_d2v, word_d2v



def calculate_depth(spcluster,words, num_points):
    cluster=[[] for w in xrange(2*num_points)]
    c=Counter()
    for i in xrange(num_points):
        cluster[i]=[i]

    for i in xrange(len(spcluster)):
        x=int(spcluster[i,0])
        y=int(spcluster[i,1])
        xval=[w for w in cluster[x]]
        yval=[w for w in cluster[y]]
        cluster[num_points+i]=xval+yval
        for w in cluster[num_points+i]:
            c[words[w]]+=1
        cluster[x][:]=[]
        cluster[y][:]=[]

    
    return c

