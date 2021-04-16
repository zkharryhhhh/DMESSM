# -*- coding: utf-8 -*-

import os
from collections import Counter
import nltk
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from gensim.models import KeyedVectors


def metaembedding_load_stackoverflow(data_path='data/stackoverflow/'):

    # load w2v embedding
    with open(data_path + 'vocab_withIdx.dic', 'r') as inp_indx, \
            open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
            open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        emb_index = inp_dic.readlines()
        emb_vec = inp_vec.readlines()
        word_vectors = {}

        for index, vec in zip(emb_index, emb_vec):
            word = index_word[index.replace('\n', '')]
            word_vectors[word] = np.array(list((map(float, vec.split()))))

    with open(data_path + 'title_StackOverflow.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(20000, 48))
        original_all_vector_repre = np.zeros(shape=(20000, 48))

        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)
            sent_rep = np.zeros(shape=[48, ])
            original_sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
                original_sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
                original_all_vector_repre[i] = original_sent_rep / j
            else:
                all_vector_representation[i] = sent_rep
                original_all_vector_repre[i] = original_sent_rep
    pca = PCA(n_components=1)
    pca.fit(all_vector_representation)
    pca = pca.components_
    XX1 = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca

    XX = XX1
    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    with open(data_path + 'label_StackOverflow.txt') as label_file:
        y = np.array(list((map(int, label_file.readlines()))))
    x_1 = XX
    y_1 = y

    del scaler

    #############################################################
    #  GOOGLENEWS WORD2VECTOR  it can be used in some datasets
    googlew2vmodel = KeyedVectors.load_word2vec_format('w2vmodel/GoogleNews-vectors-negative300.bin', binary=True)
    sentvectors2 = np.zeros(shape=(20000, 300))
    for i, line in enumerate(all_lines):
        word_sentence = nltk.word_tokenize(line)
        sent_rep = np.zeros(shape=[300, ])
        j = 0
        for word in word_sentence:
            try:
                wv = googlew2vmodel[word]
                j = j + 1
            except KeyError:
                continue
            sent_rep += wv
        if j != 0:
            sentvectors2[i] = sent_rep / j
        else:
            sentvectors2[i] = sent_rep
    
    scaler2 = MinMaxScaler()
    XX2 = scaler2.fit_transform(sentvectors2)
    x_2 = XX2

    #############################################################
    #  SBERT  load sbert embedding, the sbert embedding is obtained in other program and saved to npy
    
    stackoverflow_sbert = np.load(data_path+'stackoverflow.npy')
    scaler4 = MinMaxScaler()
    XX4 = scaler4.fit_transform(stackoverflow_sbert)
    x_4 = XX4
    del scaler4

    #############################################################
    # combine different embeddings

    XXconcat = np.concatenate([x_1,stackoverflow_sbert],axis=1)
    x_1_pad = np.pad(x_1,((0,0),(0,720)),'constant',constant_values=(0,0))
    XXavg = (x_1_pad + stackoverflow_sbert)/2
    XXsvd = PCA(n_components=500).fit_transform(XXconcat)
    return XXconcat , y


def load_data(dataset_name):
    print('load data')
    if dataset_name == 'stackoverflow':
        return metaembedding_load_stackoverflow()

    # elif dataset_name == 'search_snippets':
    #     # return load_search_snippet2()
    #     return metaembedding_load_search_snippet2()
    #
    # elif dataset_name == 'tweet89':
    #     return load_tweet89()
    #
    # elif dataset_name == '20ngnews':
    #     return load_20ngnews()
    else:
        raise Exception('dataset not found...')