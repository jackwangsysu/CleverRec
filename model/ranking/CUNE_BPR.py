# coding: utf-8

" Collaborative User Network Embedding for Social Recommender Systems (2017). "

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss, get_topK_friends_and_SPu
import os

class CUNE_BPR(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(CUNE_BPR, self).__init__(sess, data, configs, logger)
        self.walk_count, self.walk_length, self.walk_dim, self.window_size, self.topK_f = int(configs['walk_count']), int(configs['walk_length']), int(configs['walk_dim']), \
            int(configs['window_size']), int(configs['topk_f'])
        self.embed_size, self.reg = int(configs['embed_size']), float(configs['reg'])
        logger.info(' model_params: embed_size=%d, reg=%s, walk_count=%d, walk_length=%d, walk_dim=%d, window_size=%d, topK_f=%d' % (self.embed_size, self.reg, \
            self.walk_count, self.walk_length, self.walk_dim, self.window_size, self.topK_f) + ', ' + self.model_params)
        # Get the topK latent friends and SPu
        self.SPu = get_topK_friends_and_SPu(data, self.walk_count, self.walk_length, self.walk_dim, self.window_size, self.topK_f)
        # Specify training model
        self.train_model = self.train_model_sbpr(is_suk=False)

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('bpr_inputs'):
            self.u_idx, self.i_idx = __create_p(tf.int32, [None], 'u_idx'), __create_p(tf.int32, [None], 'i_idx')
            self.i_s_idx, self.i_neg_idx = __create_p(tf.int32, [None], 'i_s_idx'), __create_p(tf.int32, [None], 'i_neg_idx')
            self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_') # Number of testing users in current batch

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        with tf.variable_scope('sbpr_params'):
            # Embedding matrix
            self.P = __create_w([self.data.user_nums, self.embed_size], 'P')
            self.Q = __create_w([self.data.item_nums, self.embed_size], 'Q')
            self.bias = tf.Variable(tf.zeros(self.data.item_nums+1), name='bias')
            self.s = tf.Variable(0.0, name='s')

    def _create_embeddings(self):
        def __get_embeddings(user_embed, item_idx):
            item_embed = tf.gather(self.Q, item_idx)
            item_bias = tf.gather(self.bias, item_idx)
            # Calculate the preference scores
            ui_scores = tf.einsum('ab,ab->a', user_embed, item_embed) + item_bias
            return item_embed, item_bias, ui_scores
        with tf.name_scope('embeddings'):
            self.u_embed = tf.gather(self.P, self.u_idx)
            self.i_embed, self.i_bias, self.ui_scores = __get_embeddings(self.u_embed, self.i_idx)
            self.i_s_embed, self.i_s_bias, self.uk_scores = __get_embeddings(self.u_embed, self.i_s_idx)
            self.i_neg_embed, self.i_neg_bias, self.uj_scores = __get_embeddings(self.u_embed, self.i_neg_idx)

    def _create_inference(self):
        with tf.name_scope('inference'):
            # Optimize
            self.loss = get_loss(self.loss_func, self.ui_scores-self.uk_scores) + get_loss(self.loss_func, (self.uk_scores-self.uj_scores)/(self.s+1.0)) + \
                self.reg*(tf.nn.l2_loss(self.u_embed) + tf.nn.l2_loss(self.i_embed) + tf.nn.l2_loss(self.i_s_embed) + tf.nn.l2_loss(self.i_neg_embed) + \
                    tf.nn.l2_loss(i_bias) + tf.nn.l2_loss(i_s_bias) + tf.nn.l2_loss(i_neg_bias))
            self.train = self.optimizer.minimize(self.loss)

    def _predict(self):
        if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
            self.pre_scores = self.ui_scores
        else:
            self.pre_scores = tf.matmul(self.u_embed, self.Q, transpose_b=True)

    def _save_model(self):
        var_list = {'sbpr_params/P': self.P, 'sbpr_params/Q': self.Q, 'sbpr_params/bias': self.bias}
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