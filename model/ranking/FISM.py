# coding: utf-8

" FISM: Factorized Item Similarity Model (2013). "

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss, get_ui_sp_mat
import os

class FISM(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(FISM, self).__init__(sess, data, configs, logger)
        self.embed_size, self.reg, self.reg_bias = int(configs['embed_size']), float(configs['reg']), float(configs['reg_bias'])
        self.alpha = float(configs['alpha']) # to control the number of neighborhood items that need to be similar for an item to get the high rating
        logger.info(' model_params: embed_size=%d, alpha=%s, reg=%s, reg_bias=%s' % (self.embed_size, self.alpha, self.reg, self.reg_bias) + ', ' + self.model_params)
        # Generate user-item sparse matrix
        self.ui_sp_mat = get_ui_sp_mat(data)

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('fism_inputs'):
            self.u_idx, self.i_idx = __create_p(tf.int32, [None], 'u_idx'), __create_p(tf.int32, [None], 'i_idx')
            self.u_neighbors_num = __create_p(tf.int32, [None], 'u_neighbors_num')
            self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_') # Number of testing users in current batch
            if self.is_pairwise == 'True':
                self.j_idx = __create_p(tf.int32, [None], 'j_idx')
            else:
                self.y = __create_p(tf.float16, [None], 'y')

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        with tf.variable_scope('FISM_params'):
            # Two item embedding matrix
            self.P = __create_w([self.data.item_nums+1, self.embed_size], 'P')
            self.Q = __create_w([self.data.item_nums+1, self.embed_size], 'Q')
            self.b = tf.Variable(tf.random_uniform([self.data.item_nums+1], -0.1, 0.1))

    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            self.u_neighbors_embed = tf.gather(tf.sparse_tensor_dense_matmul(self.ui_sp_mat, self.P), self.u_idx) # embeddings of u's neighbors
            self.i_embed = tf.nn.embedding_lookup(self.Q, self.i_idx)
            self.i_bias = tf.nn.embedding_lookup(self.b, self.i_idx)
            if self.is_pairwise == 'True':
                self.j_embed = tf.nn.embedding_lookup(self.Q, self.j_idx)
                self.j_bias = tf.nn.embedding_lookup(self.b, self.j_idx)

    def _create_inference(self):
        with tf.name_scope('inference'):
            self.coeff = tf.pow(tf.cast(self.u_neighbors_num, tf.float32), -self.alpha)
            # Calculate u's preference score to i
            self.ui_scores = tf.einsum('ab,ab->a', self.i_embed, tf.einsum('a,ab->ab', self.coeff, self.u_neighbors_embed)) + self.i_bias
            if self.is_pairwise == 'True':
                self.uj_scores = tf.einsum('ab,ab->a', self.j_embed, tf.einsum('a,ab->ab', self.coeff, self.u_neighbors_embed)) + self.j_bias
            # Calculate loss
            self.loss = (self.reg*(tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q)))/self.batch_size + self.reg_bias*tf.nn.l2_loss(self.b)
            if self.is_pairwise == 'True':
                self.loss += get_loss(self.loss_func, self.ui_scores - self.uj_scores)
            else:
                self.loss += get_loss(self.loss_func, y, logits=self.ui_scores)
            # Optimize
            self.train = self.optimizer.minimize(self.loss)

    def _predict(self):
        with tf.name_scope('predict'):
            if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
                self.pre_scores = self.ui_scores
            else:
                self.pre_scores = tf.matmul(tf.einsum('a,ab->ab', self.coeff, self.u_neighbors_embed), self.Q, transpose_b=True) + self.b

    def _save_model(self):
        var_list = {'FISM_paras/P': self.P, 'FISM_params/Q': self.Q, 'FISM_params/b': self.b}
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