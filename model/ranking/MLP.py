# coding: utf-8

" Multi-layer Perceptron (2017 NCF). "

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss
import os

class MLP(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(MLP, self).__init__(sess, data, configs, logger)
        self.layers = list(map(int, configs['layers'][1:-1].split(','))) # MLP layers size
        self.reg = float(configs['reg'])
        logger.info(' model_params: layers=%s, reg=%s' % (self.layers, self.reg) + ', ' + self.model_params)

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('mlp_inputs'):
            self.u_idx, self.i_idx, self.y = __create_p(tf.int32, [None], 'u_idx'), __create_p(tf.int32, [None], 'i_idx'), __create_p(tf.float32, [None], 'y')
            self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_') # Number of testing users in current batch

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        def __create_b(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_))
        with tf.variable_scope('MLP_params'):
            # Embedding matrix
            self.P = __create_w([self.data.user_nums, self.layers[0]//2], 'P') 
            self.Q = __create_w([self.data.item_nums, self.layers[0]//2], 'Q')
            # Layer parameters
            self.mlp_params = {}
            for id in range(len(self.layers)):
                self.mlp_params['W_'+str(id)] = __create_w([self.layers[id], self.layers[id]//2], 'W_'+str(id))
                self.mlp_params['b_'+str(id)] = __create_b([self.layers[id]//2], 'b_'+str(id))
            self.h_mlp = __create_b([self.layers[-1]//2], 'h_mlp')

    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            self.u_embed, self.i_embed = tf.nn.embedding_lookup(self.P, self.u_idx), tf.nn.embedding_lookup(self.Q, self.i_idx)

    def _get_y(self, is_test=False):
        # Different calculation way in train/test stage
        if is_test:
            y_ = tf.concat([tf.tile(tf.expand_dims(self.u_embed, 1), [1, self.data.item_nums, 1]), tf.tile(tf.expand_dims(self.Q, 0), [self.batch_size_t_, 1, 1])], 2)
        else:
            y_ = tf.concat([self.u_embed, self.i_embed], 1)
        # MLP
        for id in range(len(self.layers)):
            y_ = tf.nn.relu(tf.matmul(y_, self.mlp_params['W_'+str(id)]) + self.mlp_params['b_'+str(id)])
        return y_

    def _get_logits(self, y_, is_test=False):
        if is_test:
            logits = tf.einsum('abc,c->ab', y_, self.h_mlp)
        else:
            logits = tf.einsum('ab,b->a', y_, self.h_mlp)
        return logits

    def _create_inference(self):
        with tf.name_scope('inference'):
            y_tr = self._get_y()
            # Calculate logits
            self.logits = self._get_logits(y_tr)
            self.loss = get_loss(self.loss_func, self.y, logits=self.logits) + self.reg*(tf.nn.l2_loss(self.u_embed)+tf.nn.l2_loss(self.i_embed))
            self.train = self.optimizer.minimize(self.loss)

    def _predict(self):
        with tf.name_scope('predict'):
            if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
                logits_t = self.logits
            else:
                y_t = self._get_y(is_test=True)
                logits_t = self._get_logits(y_t, is_test=True)
            self.pre_scores = tf.nn.sigmoid(logits_t)

    def _save_model(self):
        var_list = {'MLP_params/P': self.P, 'MLP_params/Q': self.Q, 'MLP_params/h_mlp': self.h_mlp}
        for id in range(len(self.layers)):
            var_list.update({'MLP_params/W_%s' % id: self.mlp_params['W_'+str(id)]})
            var_list.update({'MLP_params/b_%s' % id: self.mlp_params['b_'+str(id)]})
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
    