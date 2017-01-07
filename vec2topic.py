"""
The MIT License

Copyright (c) 2016 Ramandeep S. Randhawa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from collections import Counter
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import itertools
import argparse
import gensim, re
import nltk
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob, Word
import cPickle
import logging,time
import fastcluster
import shelve
import csv 

from utils import *


def main():
        LOG_FILENAME='vec2topic.log'
        logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s',"%b-%d-%Y %H:%M:%S")
        logger.handlers[0].setFormatter(formatter)



        parser = argparse.ArgumentParser(description='Run Vec2Topic on a text file')
        time1=time.time()
        parser.add_argument('-i', action="store", dest='inp',  help='Text file to be analyzed', required=True)
        parser.add_argument('-g','--globalvecs', action="store", dest='global_vecs_path',  help='Path to global vectors', required=True)
        parser.add_argument('-s','--stopwords_file', action="store", dest='stopwords_path',  help='[Optional] Path to stopwords file, useful for small corpus', required=False)
        parser.add_argument('-K', action="store", dest='K',type=int,  help='[Optional] Number of topics, default=10', required=False)
        parsed_stuff=parser.parse_args()
        filer = parsed_stuff.inp
        outfile_topics=filer.split('.')[0]+'_topics.csv'
        outfile_score=filer.split('.')[0]+'_score.csv'
        outfile_depth=filer.split('.')[0]+'_depth.csv'
        if parsed_stuff.K:
            K=parsed_stuff.K
        else:
            K=10
        wiki_path=parsed_stuff.global_vecs_path
        stopwords_path=parsed_stuff.stopwords_path

        logger.info('*'*50)
        logger.info('Running Vec2Topic on %s' %filer)
        logger.info('*'*50)
        text=open(filer,'rb').readlines()
        text=[' '.join(text)]
        text=cleanupContent(' '.join(text),logger,stopwords_path)
        V2T=vec2topic(text,wiki_path=wiki_path,logger=logger,NUM_TOPICS=K)
        if len(V2T)>1:
            Topics,score_words,deep_words=V2T
            
            b = open(outfile_topics, 'wb')
            a = csv.writer(b)
            a.writerows(Topics)
            b = open(outfile_score, 'wb')
            a = csv.writer(b)
            a.writerows([[w] for w in score_words])
            b = open(outfile_depth, 'wb')
            a = csv.writer(b)
            a.writerows([[w] for w in deep_words])
        else:
            print "Error! Most likely there was not enough data to run Vec2Topic; check vec2topic.log for details"
        logger.info('Total time: %.2f seconds' %(time.time()-time1))
        logger.info('*'*50)


def cleanupContent(inputString,logger,stopwords_path=[]):
        email_pattern = re.compile("[\w\.-]+@[\w\.-]+")
        try:
                inputString = unicode(inputString, errors='ignore')
        except:
                logger.info('unicode error')
        outputString = email_pattern.sub('', inputString)
        outputString=re.sub('[^\x00-\x7F]+',' ', outputString) #remove non-ascii
        outputString=re.sub('\\r',' ',outputString)
        outputString=re.sub('\\n','. ',outputString)
        outputString=re.sub('\\t',' ',outputString)
        outputString = re.sub(r"(?:\@|https?\://)\S+", " ", outputString)
#remove URLS and @xyz
        outputString=re.sub('\'','',outputString)

        outputString = re.sub('\d*.\d+',' ',outputString)
        outputString = re.sub('[.]{2,}', '.', outputString) #replace consecutive . with single .
        outputString = re.sub('\d+.\d+', ' ', outputString) #replace consecutive x.xx with space
        outputString= re.sub('[~/<>()_=-]', ' ', outputString)
        outputString=re.sub('\"',' ',outputString)
        outputString=re.sub('[,:\*!#%/$+\^]', '', outputString) #strip punct
        outputString=re.sub('[`\[\]\{\}\|]', ' ', outputString) #strip punct
        outputString=re.sub(r'\\',' ',outputString)
        outputString=re.sub(r'\b\d+\b',' ',outputString)  #remove numbers that are words
        outputString=re.sub(' +',' ',outputString)
        outputSentences=nltk.sent_tokenize(outputString)


        if stopwords_path is not None:
            logger.info("Using stop words")
            with open(stopwords_path,'rb') as f:
                stop=f.readlines()
            f.close()
            stop=set([re.sub('[^\w]','',w) for w in stop])
            outputSentences=[' '.join([x for x in w.split() if x not in stop]) for w in outputSentences if len(w)>1]
        else:
            outputSentences=[' '.join(w.split()) for w in outputSentences if len(w)>1]

        logger.info("Sentences: %d" %(len(outputSentences)))
        return outputSentences


def vec2topic(inpContent,logger,wiki_path='/data/wikimodel',NUM_TOPICS=10):
    try:
        local_vec_threshold=10000
        inpContent=[re.sub(r'[^\w]',' ',temp) for temp in inpContent]

        #lemmative mails
        wordnet_lemmatizer=WordNetLemmatizer()
        sentences_lem=[[wordnet_lemmatizer.lemmatize(w) for w in X.split()] for X in inpContent]
        sentences_lem=[[w.lower() for w in _] for _ in sentences_lem]

        flat_sent=[w for _ in sentences_lem for w in _]
        word_freq=Counter(flat_sent)
        num_words=len(flat_sent)
        logger.info('Num of words: %d' %num_words)
        if num_words<100:
            return ['Error, too few words!']
        min_count=min(5,max(2,np.percentile(word_freq.values(),q=50))) #10percentile freq

        logger.info("Min count= %d" %min_count)

        logger.info("Reading wiki vecs")

        model_wiki_vec=shelve.open(wiki_path+'wiki.shelve',flag='r')


        model_wiki_vocab_lowercase, wiki_bigram_word,wiki_exist=cPickle.load(open(wiki_path+'wiki.pkl','rb'))


        logger.info('Running Bigrams')

        #bigrams
        bigram=gensim.models.phrases.Phrases(sentences_lem,min_count=1,threshold=1)

        sentences_bigrams=list(bigram[sentences_lem])
        bigram_list=list(set([w for temp in sentences_bigrams for w in temp if '_' in w]))
        bigram_joined=[re.sub('_','',w) for w in bigram_list]

        #join bigrams that are also used as unigrams
        words=[w for _ in sentences_bigrams for w in _]
        bigram_freq=Counter(words)
        to_join=[bigram_list[w] for w in xrange(len(bigram_list)) if bigram_freq[bigram_joined[w]]>0]
        wiki_bigram_word_common=set([w for w in set(words) if wiki_exist[w]])
        # the wiki_set is large but the above set is only relevant and make the in set comparison fast.
        sentences_bigrammed_temp=[]

        #split bigrams not in wiki

        for sent in sentences_bigrams:
                new_sent=[]
                for w in sent:

                        if '_' not in w:
                                new_sent.append(w)
                        #elif w in to_join:
                        #        new_sent.append(re.sub('_','',w))
                        elif w in wiki_bigram_word_common:
                                new_sent.append(w)
                        else:
                                new_w=re.sub('_',' ',w)
                                new_w1=new_w.partition(' ')[0]
                                new_w2=new_w.partition(' ')[2]
                                new_sent.append(new_w1)
                                new_sent.append(new_w2)
                sentences_bigrammed_temp.append(new_sent)
        sentences_bigrammed=sentences_bigrammed_temp



        logger.info("Extracting Nouns")

        sentences_nouns=[]

        for sent in sentences_bigrammed:
                nouns=[]
                blob=TextBlob(' '.join(sent))
                for word,tag in blob.tags:
                        if tag in ['NN','NNP','NNS','NNPS']:
                                nouns.append(word)
                sentences_nouns.append(nouns)


        if num_words<local_vec_threshold:
                logger.info('Using Wiki Vecs Only')
                local_vec=False
                iterations=1 #for gensim.word2vec, building vocabulary
        else:
                logger.info('Using Local and Wiki Vecs')
                local_vec=True
                iterations=50 #for gensim.word2vec training

        logger.info('Word2Vec training starting...')

        #Word2vec training
        dim_wiki = 300    # Word vector dimensionality
        dim_data = 25

        model_w=gensim.models.word2vec.Word2Vec(sentences_bigrammed,workers=1,size=dim_data,iter=iterations,min_count=min_count)
        logger.info('Word2Vec training complete...')




        logger.info('Creating word vecs')


        words=[w for text in sentences_nouns for w in text]
        Vocab=set(words)


        model_comb={}
        model_comb_vocab=[]


        if local_vec:
                common_vocab=set(model_wiki_vocab_lowercase).intersection(model_w.vocab).intersection(Vocab)
        else:
                common_vocab=set(model_wiki_vocab_lowercase).intersection(model_w.vocab).intersection(Vocab)


        for w in common_vocab:
                if len(w)>2:
                        if local_vec:
                                model_comb[w]=np.array(np.concatenate((model_wiki_vec[str(w)],model_w[w])))
                        else:
                                model_comb[w]=model_wiki_vec[str(w)]
                        model_comb_vocab.append(w)


        sentences=sentences_bigrammed

        ##Create a frequency count of words in email
        words=[w for text in sentences_nouns for w in text]
        Vocab=set(words)

        #Run Agglomerative clustering
        logger.info('Clustering for depth...')

        data_d2v,word_d2v=create_word_list(model_comb,model_comb_vocab,25*local_vec+300,sentences_nouns,repeat=False,normalized=True,min_count=0,l2_threshold=0)
        spcluster=fastcluster.linkage(data_d2v,method='average',metric='cosine')


        ##Calculate depth of words
        num_points=len(data_d2v)
        depth=calculate_depth(spcluster,word_d2v,num_points)

        logger.info('Computing co-occurence graph')

        T=[' '.join(w) for w in sentences_nouns]

        ##Co-occurence matrix
        cv=CountVectorizer(token_pattern=u'(?u)\\b([^\\s]+)')
        bow_matrix = cv.fit_transform(T)
        id2word={}
        for key, value in cv.vocabulary_.items():
            id2word[value]=key

        ids=[]
        for key,value in cv.vocabulary_.iteritems():
            if key in model_comb_vocab:
                ids.append(value)


        sort_ids=sorted(ids)
        bow_reduced=bow_matrix[:,sort_ids]
        normalized = TfidfTransformer().fit_transform(bow_reduced)
        similarity_graph_reduced=bow_reduced.T * bow_reduced
        ##Depth-rank weighting of edges, weight of edge i,j=cosine of angle between them
        logger.info('Computing degree')
        m,n=similarity_graph_reduced.shape

        cx=similarity_graph_reduced.tocoo()
        keyz=[id2word[sort_ids[w]] for w in xrange(len(sort_ids))]
        data=[]
        ro=[]
        co=[]
        for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
                if v>0 and i!=j:
                    value=1
                    if value>0:
                        ro.append(i)
                        co.append(j)
                        data.append(value)

        SS=sp.sparse.coo_matrix((data, (ro, co)), shape=(m,n))
        SP_full=SS.tocsc()
        id_word={w:id2word[sort_ids[w]] for w in xrange(len(sort_ids))}
        word_id={value:key for key,value in id_word.items()}

        logger.info('Computing metrics')
        #compute metrics
        degsum=SP_full.sum(axis=1)
        deg={}
        for x in xrange(len(sort_ids)):
            deg[id2word[sort_ids[x]]]=int(degsum[x])

        max_deg=max(deg.values())
        max_depth=max(depth.values())

        temp_deg_mod={w:np.log(1+deg[w])/np.log(1+max_deg) for w in deg.iterkeys()}
        alpha=np.log(0.5)/np.log(np.median(temp_deg_mod.values()))
        deg_mod={key:value**alpha for key, value in temp_deg_mod.iteritems()}

        temp={key:value*1./max_depth for key, value in depth.iteritems()}
        alpha=np.log(0.5)/np.log(np.median(temp.values()))
        depth_mod={key:value**alpha for key, value in temp.iteritems()}

        temp={key:deg_mod[key]*depth_mod[key] for key in depth_mod.iterkeys()}
        max_metric=np.max(temp.values())
        metric={key:value*1./max_metric for key,value in temp.iteritems()}

        logger.info('Running K-means')

        ##Kmeans
        K=NUM_TOPICS
        kmeans=KMeans(n_clusters=K)
        kmeans.fit([w for w in data_d2v])
        kmeans_label={word_d2v[x]:kmeans.labels_[x] for x in xrange(len(word_d2v))}


        kmeans_label_ranked={}

        topic=[[] for i in xrange(K)]
        clust_depth=[[] for i in xrange(K)]
        for i in xrange(K):
                topic[i]=[word_d2v[x] for x in xrange(len(word_d2v)) if kmeans.labels_[x]==i]
                temp_score=[metric[w] for w in topic[i]]
                clust_depth[i]=-np.mean(sorted(temp_score,reverse=True)[:])#int(np.sqrt(len(topic[i])))])
        index=np.argsort(clust_depth)
        for num,i in enumerate(xrange(K)):
            for w in topic[index[i]]:
                    kmeans_label_ranked[w]=i
        logger.info('Done...Generating output')
        lister=[]
        to_show=K
        to_show_words=50 #the maximum number of words of each type to display
        for i in xrange(to_show):
                top=topic[index[i]]
                sort_top=[w[0] for w in sorted([[w,metric[w]] for w in top],key=itemgetter(1),reverse=True)]
                lister.append(['Topic %d' %(i+1)]+sort_top[:to_show_words])

        max_len=max([len(w) for w in lister])
        new_list=[]
        for list_el in lister:
                new_list.append(list_el + [''] * (max_len - len(list_el)))
        Topics=list(itertools.izip_longest(*new_list))
        #X.insert(len(X),[-int(clust_depth[index[w]]*100)*1./100 for w in xrange(K)])
        sorted_words=[w[0] for w in sorted(metric.items(),key=itemgetter(1),reverse=True)][:to_show_words]

        model_wiki_vec.close()
        return Topics,sorted_words,[w[0] for w in depth.most_common(to_show_words)]
    except Exception as e:
        logger.info('Exception: '+str(e))
        return ['Error']


if __name__ == "__main__":
    main()
