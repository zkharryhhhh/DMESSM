## DMESSM:  Short Text Clustering with A Deep Multi-Embedded Self-Supervised Model 

This repository is an implementation of **"Short Text Clustering with A Deep Multi-Embedded Self-Supervised Model "**. The implementation is based [DEC-keras](https://github.com/XifengGuo/DEC-keras)  and [SIFAuto](https://github.com/hadifar/stc_clustering).



#### Install requirements

```
conda install --yes --file requirements.txt
```



#### Data

We release  the data of stackoverflow now.  The word2vec embedding is from [STCC](https://github.com/jacoxu/STC2) . The Sbert embedding is calculated by us , shown in stackoverflow.npy. 

We use four datasets, which are stackoverflow, SerchSnippets, Tweet89 and 20ngnewsshort. Our data including different embeddings will be released.



#### Run an example

```
python DMESSM.py --dataset stackoverflow -- maxiter 2600 --ae_weights data/stackoverflow/results/ae_weights.hs --save_dir data/stackoverflow/results 
```



#### Important notes

We release the complete code!

