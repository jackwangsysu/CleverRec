# coding: utf-8

" NAIS: Neural Attentive Item Similarity Model (2018)."

import tensorflow as tf
from model.Recommender import RankingRecommender
import os

class NAIS(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(NAIS, self).__init__(sess, data, configs, logger)
        self.embed_size, self.atten_size, self.reg = int(configs['embed_size']), int(configs['atten_size']), float(configs['reg'])
        self.beta = float(configs['beta']) # The smoothing coefficient of Softmax
        self.atten_type = configs['atten_type'] # concat/prod
        logger.info(' model_params: embed_size=%d, atten_size=%d, atten_type=%s, reg=%s, beta=%s' % (self.embed_size, self.atten_size, \
            self.atten_type, self.reg, self.beta) + ', ' + self.model_params)

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('nais_inputs'):
            self.u_idx = __create_p(tf.int32, [None, None], 'u_idx') # u's neighboring items
            self.i_idx = __create_p(tf.int32, [None], 'i_idx')
            self.y = __create_p(tf.float32, [None], 'y')
            self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_') # Number of testing users in current batch

    def _load_fism_params(self):
        dict_fism = {'FISM_paras/P': self.P, 'FISM_params/Q': self.Q, 'FISM_params/b': self.bias}
        saver_fism = tf.train.Saver(dict_fism)
        saver_fism.restore(self.sess, tf.train.latest_checkpoint(self.configs['fism_pretrain']))

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        def __create_b(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=tf.random_uniform(shape_, -0.1, 0.1))
        with tf.name_scope('NAIS_params'):
            # Embedding matrix
            self.P = __create_w([self.data.item_nums+1, self.embed_size], 'P')
            self.Q = __create_w([self.data.item_nums+1, self.embed_size], 'Q')
            # Item bias
            self.bias = __create_b([self.data.item_nums+1], 'bias')
            # Attention layer
            if self.atten_type == 'concat':
                self.W = __create_w([2*self.embed_size, self.atten_size], 'W') # Weight matrix
            else:
                self.W = __create_w([self.embed_size, self.atten_size], 'W')
            self.b = __create_b([self.atten_size], 'b') # bias
            self.h = __create_b([self.atten_size], 'h') # Linear transformation weight

    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            self.i_embed_p =tf.gather(self.P, self.u_idx) # [batch_size, max_nbr_nums, embed_size]
            self.i_embed_q = tf.gather(self.Q, self.i_idx)
            self.i_bias_embed = tf.gather(self.bias, self.i_idx)

    # Calculate user embedding
    def _get_u_embed(self, is_test=False):
        nbr_existed = tf.cast(tf.not_equal(self.u_idx, self.data.item_nums), tf.float32)
        self.i_embed_p = tf.einsum('ab,abc->abc', nbr_existed, self.i_embed_p) # [batch_size, max_nbr_nums, embed_size]

        # Calculate the attention weights
        if is_test:
            if self.atten_type == 'concat':
                joint_embed = tf.concat([tf.tile(tf.expand_dims(self.i_embed_p, 1), [1, self.data.item_nums+1, 1, 1]), \
                    tf.tile(tf.expand_dims(self.Q, 0), [self.batch_size_t_, 1, self.max_nbr_nums, 1])], 3)
            else:
                joint_embed = tf.einsum('acd,bd->abcd', self.i_embed_p, self.Q) # [batch_size, item_nums+1, max_nbr_nums, embed_size]
            atten_scores = tf.einsum('e,abce->abc', self.h, tf.nn.relu(tf.einsum('abcd,de->abce', joint_embed, self.W) + self.b)) # [batch_size, item_nums+1, embed_size]
        else:
            if self.atten_type == 'concat':
                joint_embed = tf.concat([self.i_embed_p, tf.tile(tf.expand_dims(self.i_embed_q, 1), [1, self.max_nbr_nums, 1])], 2)
            else:
                joint_embed = tf.einsum('abc,ac->abc', self.i_embed_p, self.i_embed_q)
            atten_scores = tf.einsum('c,abc->ab', self.h, tf.nn.relu(tf.einsum('abc,cd->abd', joint_embed, self.W) + self.b)) # [batch_size, max_nbr_nums]

        # Smoothed softmax
        k_dim = 2 if is_test else 1
        exp_atten_scores = tf.exp(atten_scores)
        d = tf.pow(tf.reduce_sum(exp_atten_scores, k_dim, keep_dims=True), self.beta) # dominator
        atten_weights = tf.div(exp_atten_scores, d)
        if is_test:
            u_embed = tf.einsum('abc,adc->abd', atten_weights, self.i_embed_p)
        else:
            u_embed = tf.einsum('ab,abc->ac', atten_weights, self.i_embed_p)
        return u_embed

    def _create_inference(self):
        with tf.name_scope('inference'):
            self.u_embed = self._get_u_embed()
            # Calculate u's preference score to i
            self.ui_scores = tf.einsum('ab,ab->a', self.u_embed, self.i_embed_q) + self.i_bias_embed
            # Optimize
            self.loss = tf.reduce_sum(self.loss_func(labels=self.y, logits=self.ui_scores)) + \
                self.reg*(tf.nn.l2_loss(self.u_embed) + tf.nn.l2_loss(self.i_embed_q) + tf.nn.l2_loss(self.i_bias_embed))
            self.train = self.optimizer.minimize(self.loss)

    def _predict(self):
        if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
            self.pre_scores = self.ui_scores
        else:
            u_embed_t = self._get_u_embed(is_test=True) # [batch_size, item_nums+1, embed_size]
            self.pre_scores = tf.einsum('abc,bc->ab', u_embed_t, self.Q) + self.i_bias

    def _save_model(self):
        pass

    def build_model(self):
        self._create_inputs()
        self._create_params()
        self._create_embeddings()
        self._create_inference()
        self._predict()
        self._save_model()