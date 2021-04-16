from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer




def tf(filename):
    corpus = []
    with open(filename,'r') as f:
        info = f.readlines()
        for i in info:
            s = i.replace('\n','')
            corpus.append(s)

    # print(corpus[:10])
    tf = CountVectorizer()
    tfre = tf.fit_transform(corpus)
    x = tfre.toarray()

    return x


def tfidf(filename):
    corpus = []
    with open(filename,'r') as f:
        info = f.readlines()
        for i in info:
            s = i.replace('\n','')
            corpus.append(s)

    # print(corpus[:10])
    tfidf = TfidfVectorizer()
    tfidfre = tfidf.fit_transform(corpus)

    x = tfidfre.toarray()

    return x


# tf('data/webKB/webKB.txt')

