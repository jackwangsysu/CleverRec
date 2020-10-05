# coding: utf-8

" Generalized Matrix Factorization (2017 NCF). "

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
import os

class GMF(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(GMF, self).__init__(sess, data, configs, logger)
        self.embed_size, self.reg = int(configs['embed_size']), float(configs['reg'])
        logger.info(' model_params: embed_size=%d, reg=%s' % (self.embed_size, self.reg) + ', ' + self.model_params)

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('gmf_inputs'):
            self.u_idx, self.i_idx, self.y = __create_p(tf.int32, [None], 'u_idx'), __create_p(tf.int32, [None], 'i_idx'), __create_p(tf.float32, [None], 'y')
            self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_') # Number of testing users in current batch

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        def __create_b(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_))
        with tf.variable_scope('GMF_params'):
            # Embedding matrix
            self.P = __create_w([self.data.user_nums, self.embed_size], 'P')
            self.Q = __create_w([self.data.item_nums, self.embed_size], 'Q')
            self.h_gmf = __create_b([self.embed_size], 'h_gmf')
        
    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            self.u_embed, self.i_embed = tf.nn.embedding_lookup(self.P, self.u_idx), tf.nn.embedding_lookup(self.Q, self.i_idx)

    def _get_logits(self, is_test=False):
        # Different calculation way in train/test stage
        if is_test:
            logits = tf.einsum('abc,c->ab', tf.einsum('ac,bc->abc', self.u_embed, self.Q), self.h_gmf)
        else:
            logits = tf.einsum('ab,b->a', tf.einsum('ab,ab->ab', self.u_embed, self.i_embed), self.h_gmf)
        return logits

    def _create_inference(self):
        with tf.name_scope('inference'):
            self.logits = self._get_logits()
            self.loss = get_loss(self.loss_func, self.y, logits=self.logits) + self.reg*(tf.nn.l2_loss(self.u_embed) + tf.nn.l2_loss(self.i_embed))
            self.train = self.optimizer.minimize(self.loss)

    def _predict(self):
        with tf.name_scope('predict'):
            if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
                logits_t = self.logits
            else:
                logits_t = self._get_logits(is_test=True)
            self.pre_scores = tf.nn.sigmoid(logits_t)

    def _save_model(self):
        var_list = {'GMF_params/P': self.P, 'GMF_params/Q': self.Q, 'GMF_params/h_gmf': self.h_gmf}
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