
from sklearn.cluster import KMeans
import metrics
import numpy as np
from multiEmbedding import metaembedding_load_search_snippet2,metaembedding_load_stackoverflow,load_tweet89,load_20ngnews

# from Tfidf import tf,tfidf

filename = 'data/20ngnews/20ngnews.txt'
x,y = load_20ngnews()

# x_tf = tf(filename)
# x_tfidf = tfidf(filename)


# print("x.shape: ",x.shape)
# print("x_tf.shape: ",x_tfidf.shape)
# print("x_tfidf.shape: ",x_tfidf.shape)
clusternum = len(set(y))
print("clusternum:",clusternum)
kmeans = KMeans(n_clusters= clusternum, n_init= 100)
y_pred = kmeans.fit_predict(x)
acc = np.round(metrics.acc(y, y_pred), 5)
# nmi值
nmi = np.round(metrics.nmi(y, y_pred), 5)

print('acc = %.5f, nmi = %.5f' % ( acc, nmi))







