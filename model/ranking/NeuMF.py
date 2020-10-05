# coding: utf-8

" Neural Collaborative Filtering (2017). "

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss
import os

class NeuMF(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(NeuMF, self).__init__(sess, data, configs, logger)
        self.embed_size = int(configs['embed_size'])
        self.layers = list(map(int, configs['layers'][1:-1].split(',')))
        self.reg1, self.reg2 = float(configs['reg1']), float(configs['reg2'])
        logger.info(' model_params: embed_size=%s, layers=%s, reg1=%s, reg2=%s' % (self.embed_size, self.layers, self.reg1, self.reg2) + \
            ', ' + self.model_params)

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('neumf_inputs'):
            self.u_idx, self.i_idx, self.y = __create_p(tf.int32, [None], 'u_idx'), __create_p(tf.int32, [None], 'i_idx'), __create_p(tf.float32, [None], 'y')
            self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_') # Number of testing users in current batch
            
    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        def __create_b(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_))
        with tf.variable_scope('NeuMF_params'):
            # GMF
            self.P_gmf = __create_w([self.data.user_nums, self.embed_size], 'P_gmf') 
            self.Q_gmf = __create_w([self.data.item_nums, self.embed_size], 'Q_gmf')
            self.h_gmf = __create_b([self.embed_size], 'h_gmf')

            # MLP
            self.P_mlp= __create_w([self.data.user_nums, self.layers[0]//2], 'P_mlp') 
            self.Q_mlp= __create_w([self.data.item_nums, self.layers[0]//2], 'Q_mlp')
            self.mlp_params = {}
            for id in range(len(self.layers)):
                self.mlp_params['W_'+str(id)] = __create_w([self.layers[id], self.layers[id]//2], 'W_'+str(id))
                self.mlp_params['b_'+str(id)] = __create_b([self.layers[id]//2], 'b_'+str(id))
            self.h_mlp = __create_b([self.layers[-1]//2], 'h_mlp')

            # Load pretrained parameters
            if 'gmf_pretrain' in self.configs and 'mlp_pretrain' in self.configs:
                self._load_pretrained_model()

            # NeuMF
            self.h_neumf = __create_b([self.embed_size + self.layers[-1]//2], 'h_neumf')

    def _load_pretrained_model(self):
        self._load_gmf_params()
        self._load_mlp_params()
        self.h_neumf = 0.5 * tf.concat([self.h_gmf, self.h_mlp], 0)

    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            self.u_embed_gmf, self.i_embed_gmf = tf.nn.embedding_lookup(self.P_gmf, self.u_idx), tf.nn.embedding_lookup(self.Q_gmf, self.i_idx)
            self.u_embed_mlp, self.i_embed_mlp = tf.nn.embedding_lookup(self.P_mlp, self.u_idx), tf.nn.embedding_lookup(self.Q_mlp, self.i_idx)

    def _get_y_gmf(self, is_test=False):
        # Different calculation way in train/test stage
        if is_test:
            y_gmf = tf.einsum('ac,bc->abc', self.u_embed_gmf, self.Q_gmf)
        else:
            y_gmf = tf.einsum('ab,ab->ab', self.u_embed_gmf, self.i_embed_gmf)
        return y_gmf

    def _get_y_mlp(self, is_test=False):
        if is_test:
            y_mlp = tf.concat([tf.tile(tf.expand_dims(self.u_embed_mlp, 1), [1, self.data.item_nums, 1]), tf.tile(tf.expand_dims(self.Q_mlp, 0), [self.batch_size_t_, 1, 1])], 2)
        else:
            y_mlp = tf.concat([self.u_embed_mlp, self.i_embed_mlp], 1)
        for id in range(len(self.layers)):
            y_mlp = tf.nn.relu(tf.matmul(y_mlp, self.mlp_params['W_'+str(id)]) + self.mlp_params['b_'+str(id)])
        return y_mlp

    def _get_logits(self, y_gmf_, y_mlp_, is_test=False):
        if is_test:
            logits = tf.einsum('abc,c->ab', tf.concat([y_gmf_, y_mlp_], 2), self.h_neumf)
        else:
            logits = tf.einsum('ab,b->a', tf.concat([y_gmf_, y_mlp_], 1), self.h_neumf)
        return logits

    def _create_inference(self):
        with tf.name_scope('inference'):
            y_gmf_tr = self._get_y_gmf()
            y_mlp_tr = self._get_y_mlp()
            # Fuse GMF and MLP
            self.logits = self._get_logits(y_gmf_tr, y_mlp_tr)
            self.loss = get_loss(self.loss_func, self.y, logits=self.logits) + self.reg1*(tf.nn.l2_loss(self.u_embed_gmf)+tf.nn.l2_loss(self.i_embed_gmf)) + \
                self.reg2*(tf.nn.l2_loss(self.u_embed_mlp)+tf.nn.l2_loss(self.i_embed_mlp))
            self.train = self.optimizer.minimize(self.loss)

    def _predict(self):
        with tf.name_scope('predict'):
            if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
                logits_t = self.logits
            else:
                y_gmf_t = self._get_y_gmf(is_test=True)
                y_mlp_t = self._get_y_mlp(is_test=True)
                logits_t = self._get_logits(y_gmf_t, y_mlp_t, is_test=True)
            self.pre_scores = tf.nn.sigmoid(logits_t)

    def _save_model(self):
        var_list = {'NeuMF_params/P_gmf': self.P_gmf, 'NeuMF_params/Q_gmf': self.Q_gmf, 'NeuMF_params/h_gmf': self.h_gmf, 'NeuMF_params/P_mlp': self.P_mlp, \
            'NeuMF_params/Q_mlp': self.Q_mlp, 'NeuMF_params/h_mlp': self.h_mlp, 'NeuMF_params/h_neumf': self.h_neumf}
        for id in range(len(self.layers)):
            var_list.update({'NeuMF_params/W_%s' % id: self.mlp_params['W_'+str(id)]})
            var_list.update({'NeuMF_params/b_%s' % id: self.mlp_params['b_'+str(id)]})
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

    # Load pretrained GMF.
    def _load_gmf_params(self):
        dict_gmf = {'GMF_params/P': self.P_gmf, 'GMF_params/Q': self.Q_gmf, 'GMF_params/h_gmf': self.h_gmf}
        saver_gmf = tf.train.Saver(dict_gmf)
        saver_gmf.restore(self.sess, tf.train.latest_checkpoint(self.configs['gmf_pretrain']))

    # Load pretrained MLP.
    def _load_mlp_params(self):
        dict_mlp = {'MLP_params/P':self.P_mlp, 'MLP_params/Q':self.Q_mlp, 'MLP_params/h_mlp': self.h_mlp}
        for id in range(len(self.layers)):
            dict_mlp.update({'MLP_params/W_%s' % id: self.mlp_params['W_'+str(id)]})
            dict_mlp.update({'MLP_params/b_%s' % id: self.mlp_params['b_'+str(id)]})
        saver_mlp = tf.train.Saver(dict_mlp)
        saver_mlp.restore(self.sess, tf.train.latest_checkpoint(self.configs['mlp_pretrain']))
    
    