# CleverRec: An Open-source Toolkit for Recommendation System.

In this repository, we provide an open-source toolkit for recommender system. It contains 20+ recommendation models which can be classified into two groups: ranking-based and rating-based. Meanwhile, this framework covers classic Collaborative Filtering models, DNN-based models, Metric learning-based models and Social-based models. It's quite flexible to include new models and releases you from repeating work.

Feedbacks and advices are always welcome!

## Algorithms

### Ranking-based Models

+ LFM: Latent Factor Model. 2006.
+ BPR: Bayesian personalized ranking from implicit feedback. UAI 2009.
+ SBPR: Leveraging Social Connections to Improve Personalized Ranking for Collaborative Filtering. CIKM 2014.
+ TBPR: Social recommendation with strong and weak ties. CIKM 2016.
+ FISM: Factored Item Similarity Models for Top-N Recommender Systems. KDD 2013.
+ NAIS: Neural Attentive Item Similarity Model for Recommendation. TKDE 2018.
+ GMF, MLP, NeuMF: Neural Collaborative Filtering. WWW 2017.
+ CML: Collaborative Metric Learning. WWW 2017.
+ LRML: Latent relational metric learning via memory-based attention for collaborative ranking. WWW 2018.
+ TransCF: Collaborative Translational Metric Learning. ICDM 2018.
+ CUNE-BPR: Collaborative User Network Embedding for Social Recommender Systems. SDM 2017.
+ SAMN: Social Attentional Memory Network_Modeling Aspect- and Friend-Level Differences in Recommendation. WSDM 2019.
+ DiffNet: A Neural Influence Diffusion Model for Social Recommendation. SIGIR 2019.
+ NGCF: Neural Graph Collaborative Filtering. SIGIR 2019.

### Rating-based Models

+ FM: Factorization Machines. ICDM 2010.
+ SLIM: Sparse linear methods for top-n recommender systems. 2011
+ SVD++: Factorization meets the neighborhood: a multifaceted collaborative filtering model. KDD 2008.
+ TrustSVD: Collaborative Filtering with Both the Explicit and Implicit Influence of User Trust and of Item Ratings. AAAI 2015.
+ FFM: Field-aware Factorization Machines for CTR Prediction. RecSys 2016.

## How to use it

+ For models in CleverRec: specify configurations (e.g. model, dataset, split ratio, ...) in 'CleverRec.properties' and then run 'python main.py'.
+ For new models: implement you model based on the interfaces in CleverRec and run as above.
+ Model tuning: run 'python main_tuning.py' to do parameters tuning.

## Evaluation & Metrics

We provides three different evaluation strateties: Leave-one-out, Random split and Random split with specified negative samples (typically 1000). The third one was designed for efficiency consideration since it may be too time-consuming to rank all unobserved items for each user. In addition, we adopt the following metrics:

+ For ranking-based models: HR@K, MRR@K, NDCG@K.
+ For rating-based models: RMSE, MAE

## Requirements

+ Python 3.5+, Tensorflow 1.13+, Pandas, Numpy, Scipy, Scikit-learn

























































