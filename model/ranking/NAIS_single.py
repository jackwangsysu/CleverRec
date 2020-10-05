# coding: utf-8

" NAIS: Neural Attentive Item Similarity Model (2018)."

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss
import os

# Form a mini-batch by single user (Much faster than pure NAIS)
class NAIS_single(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(NAIS_single, self).__init__(sess, data, configs, logger)
        self.embed_size, self.atten_size, self.reg = int(configs['embed_size']), int(configs['atten_size']), float(configs['reg'])
        self.beta = float(configs['beta']) # The smoothing coefficient of Softmax
        self.atten_type = configs['atten_type'] # concat/prod
        logger.info(' model_params: embed_size=%d, atten_size=%d, atten_type=%s, reg=%s, beta=%s' % (self.embed_size, self.atten_size, \
            self.atten_type, self.reg, self.beta) + ', ' + self.model_params)
        # Specify training and testing model
        self.train_model = self.train_model_nais
        self.test_model_rs, self.test_model_loo = self.test_model_rs_nais, self.test_model_loo_nais

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('nais_inputs'):
            # Form mini-batch by single user
            self.u_idx = __create_p(tf.int32, [None], 'u_idx') # u's neighboring items
            self.u_nbrs_num = __create_p(tf.int32, [], 'u_nbrs_num')
            self.i_idx = __create_p(tf.int32, [None], 'i_idx')
            self.i_nums = __create_p(tf.int32, [], 'i_nums')
            self.y = __create_p(tf.float32, [None], 'y') # label

    # Load pretrained FISM
    def _load_fism_params(self):
        dict_fism = {'FISM_paras/P': self.P, 'FISM_params/Q': self.Q, 'FISM_params/b': self.bias}
        saver_fism = tf.train.Saver(dict_fism)
        saver_fism.restore(self.sess, tf.train.latest_checkpoint(self.configs['fism_pretrain']))

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        def __create_b(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=tf.random_uniform(shape_, -0.1, 0.1))
        with tf.variable_scope('NAIS_params'):
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
            self.i_embed_p =tf.gather(self.P, self.u_idx) # [u_items_num, embed_size]
            self.i_embed_q = tf.gather(self.Q, self.i_idx) # [i_nums, embed_size]
            self.i_bias_embed = tf.gather(self.bias, self.i_idx)

    # Calculate user embedding (Train/Test)
    def _get_u_embed(self, i_embed_q_, i_nums_):
        # Calculate the attention weights
        if self.atten_type == 'concat':
            joint_embed = tf.concat([tf.tile(tf.expand_dims(self.i_embed_p, 0), [i_nums_, 1, 1]), \
                tf.tile(tf.expand_dims(i_embed_q_, 1), [1, self.u_nbrs_num, 1])], 2) # [i_nums/(item_nums+1), u_items_num, 2*embed_size]
        else:
            joint_embed = tf.einsum('ac,bc->abc', i_embed_q_, self.i_embed_p)
        atten_scores = tf.einsum('c,abc->ab', self.h, tf.nn.relu(tf.einsum('abc,cd->abd', joint_embed, self.W) + self.b)) # [i_nums/(item_nums+1), u_items_num]

        # Smoothed softmax
        exp_atten_scores = tf.exp(atten_scores)
        d = tf.pow(tf.reduce_sum(exp_atten_scores, 1, keep_dims=True), self.beta) # dominator
        atten_weights = tf.div(exp_atten_scores, d)
        u_embed = tf.einsum('ab,bc->ac', atten_weights, self.i_embed_p) # [i_nums/(item_nums+1), embed_size]
        return u_embed

    def _create_inference(self):
        with tf.name_scope('inference'):
            self.u_embed = self._get_u_embed(self.i_embed_q, self.i_nums)
            # Calculate u's preference score to i
            self.ui_scores = tf.einsum('ab,ab->a', self.u_embed, self.i_embed_q) + self.i_bias_embed
            # Optimize
            self.loss = tf.reduce_sum(self.loss_func(labels=self.y, logits=self.ui_scores)) + \
                self.reg*(tf.nn.l2_loss(self.u_embed) + tf.nn.l2_loss(self.i_embed_q) + tf.nn.l2_loss(self.i_bias_embed))
            self.train = self.optimizer.minimize(self.loss)

    def _predict(self):
        if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0: # loo/random split with 1000...
            self.pre_scores = self.ui_scores
        else: # random split with all
            u_embed_t = self._get_u_embed(self.Q, self.data.item_nums+1)
            self.pre_scores = tf.einsum('ab,ab->a', u_embed_t, self.Q) + self.bias

    def _save_model(self):
        var_list = {'NAIS_paras/P': self.P, 'NAIS_params/Q': self.Q, 'NAIS_params/bias': self.bias, 'NAIS_params/W': self.W, 'NAIS_params/b': self.b, \
            'NAIS_params/h': self.h}
        self.saver = tf.train.Saver(var_list=var_list)
        tmp_dir = os.path.join(self.saved_model_dir, self.model)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

    def build_model(self):
        self._create_inputs()
        self._create_params()
        self._create_embeddings()
        self._create_inference()
        self._predict()
        self._save_model()
        
        