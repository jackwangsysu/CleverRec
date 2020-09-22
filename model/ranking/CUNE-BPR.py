# coding: utf-8

" CUNE-BPR: Collaborative User Network Embedding for Social Recommender Systems (2017). "

import numpy as np, tensorflow as tf
import time, os
from Recommender import Recommender

class CUNE_BPR(Recommender):
    def __init__(self, dataset='Ciao', split_ratio=[0.8,0.0,0.2], batch_size=128, embed_size=64, lr=0.001, reg=0.001, max_epoches=30, filter_num=2, n_negative=10, \
            walk_count=20, walk_length=10, walk_dim=20, window_size=5, topK=50, is_gpu=False):
        super(CUNE_BPR, self).__init__(dataset, split_ratio, batch_size, max_epoches, filter_num, n_negative, walk_count, walk_length, walk_dim, window_size, topK, is_gpu)
        self.embed_size = embed_size
        self.lr, self.reg = lr, reg
        self.model_name = 'CUNE-BPR'

    
    # Build the model
    def build_model(self):
        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        optim = tf.train.AdamOptimizer(self.lr, name='Adam')

        # Inputs of the model
        with tf.name_scope('inputs'):
            # Batch features
            self.u_idx = tf.placeholder(tf.int32, shape=[None], name='u_idx')
            self.i_idx = tf.placeholder(tf.int32, shape=[None], name='i_idx') # positive items
            self.i_s_idx = tf.placeholder(tf.int32, shape=[None], name='i_s_idx') # social items
            self.i_neg_idx = tf.placeholder(tf.int32, shape=[None], name='i_neg_idx') # negative items
            # self.suk = tf.placeholder(tf.float32, shape=[None], name='suk') # social coefficient to indicate the preference degree of u's friends

        # Embedding parameters
        with tf.variable_scope('model_parameters'):
            self.P = tf.get_variable(initializer=initializer([self.user_nums, self.embed_size]), regularizer=regularizer, name='P')
            self.Q = tf.get_variable(initializer=initializer([self.item_nums, self.embed_size]), regularizer=regularizer, name='Q')
            self.bias = tf.Variable(tf.zeros(self.item_nums), name='bias')
            self.s = tf.Variable(0.0, name='s')


        def create_inference(user_embed, item_input):
            item_embed = tf.nn.embedding_lookup(self.Q, item_input)
            item_bias = tf.nn.embedding_lookup(self.bias, item_input)
            # Predict the preference scores
            u_i_scores = tf.reduce_sum(tf.multiply(user_embed, item_embed), 1) + item_bias
            return item_embed, item_bias, u_i_scores

        # Dense embedding vectors and the preference scores
        with tf.name_scope('dense_vectors'):
            self.u_embed = tf.nn.embedding_lookup(self.P, self.u_idx)
            self.i_embed, i_bias, self.ui_scores = create_inference(self.u_embed, self.i_idx)
            self.i_s_embed, i_s_bias, self.uk_scores = create_inference(self.u_embed, self.i_s_idx)
            self.i_neg_embed, i_neg_bias, self.uj_scores = create_inference(self.u_embed, self.i_neg_idx)

        # Compute loss and optimize
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(-tf.log_sigmoid(self.ui_scores-self.uk_scores)) + tf.reduce_mean(-tf.log_sigmoid((self.uk_scores-self.uj_scores)/(self.s+1.0))) + \
                self.reg*(tf.nn.l2_loss(self.u_embed) + tf.nn.l2_loss(self.i_embed) + tf.nn.l2_loss(self.i_s_embed) + tf.nn.l2_loss(self.i_neg_embed) + \
                    tf.nn.l2_loss(i_bias) + tf.nn.l2_loss(i_s_bias) + tf.nn.l2_loss(i_neg_bias))

        with tf.name_scope('optimization'):
            self.train = optim.minimize(self.loss)

        # Prediction
        with tf.name_scope('predict'):
            self.pre_scores = tf.matmul(self.u_embed, self.Q, transpose_b=True)



if __name__ == '__main__':
    t1 = time.time()

    cune_bpr = CUNE_BPR(dataset='Ciao', split_ratio=[0.7,0.2,0.1], batch_size=2048, reg=0.001, max_epoches=50, filter_num=5, n_negative=10)
    cune_bpr.run_model()

    print("total time: ", time.strftime("%H: %M: %S", time.gmtime(time.time()-t1)))