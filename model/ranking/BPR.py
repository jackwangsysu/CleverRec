# coding: utf-8

" Bayesian Personalized Ranking (2009). "

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss
import os

class BPR(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(BPR, self).__init__(sess, data, configs, logger)
        self.embed_size, self.reg = int(configs['embed_size']), float(configs['reg'])
        logger.info(' model_params: embed_size=%d, reg=%s' % (self.embed_size, self.reg) + ', ' + self.model_params)

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('bpr_inputs'):
            self.u_idx, self.i_idx, self.j_idx = __create_p(tf.int32, [None], 'u_idx'), __create_p(tf.int32, [None], 'i_idx'), __create_p(tf.int32, [None], 'j_idx')
            self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_') # Number of testing users in current batch

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        with tf.variable_scope('BPR_params'):
            # Embedding matrix
            self.P = __create_w([self.data.user_nums, self.embed_size], 'P')
            self.Q = __create_w([self.data.item_nums, self.embed_size], 'Q')

    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            self.u_embed, self.i_embed, self.j_embed = tf.nn.embedding_lookup(self.P, self.u_idx), tf.nn.embedding_lookup(self.Q, self.i_idx), \
                tf.nn.embedding_lookup(self.Q, self.j_idx)

    def _create_inference(self):
        with tf.name_scope('inference'):
            # Calculate preference scores
            self.ui_scores = tf.einsum('ab,ab->a', self.u_embed, self.i_embed)
            self.uj_scores = tf.einsum('ab,ab->a', self.u_embed, self.j_embed)
            # Optimize
            self.loss = get_loss(self.loss_func, self.ui_scores - self.uj_scores) + self.reg*(tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q))/self.batch_size
            self.train = self.optimizer.minimize(self.loss)

    def _predict(self):
        with tf.name_scope('predict'):
            if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
                self.pre_scores = self.ui_scores
            else:
                self.pre_scores = tf.matmul(self.u_embed, self.Q, transpose_b=True)

    def _save_model(self):
        # var_list = {'BPR_params/P': self.P, 'BPR_params/Q': self.Q}
        # self.saver = tf.train.Saver(var_list=var_list)
        # tmp_dir = os.path.join(self.saved_model_dir, self.model)
        # if not os.path.exists(tmp_dir):
        #     os.makedirs(tmp_dir)

        # saved_model
        self.signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs = {'u_idx': tf.saved_model.utils.build_tensor_info(self.u_idx), 'i_idx': tf.saved_model.utils.build_tensor_info(self.i_idx), \
                'j_idx': tf.saved_model.utils.build_tensor_info(self.j_idx)},
            outputs = {'output': tf.saved_model.utils.build_tensor_info(self.ui_scores)},
            method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

    def build_model(self):
        self._create_inputs()
        self._create_params()
        self._create_embeddings()
        self._create_inference()
        self._predict()
        self._save_model()