# coding: utf-8

" SAMN: Social Attentional Memory Network (2019). "

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
from utils.tools import get_loss
import os

# Form a mini-batch by single user
class SAMN_single(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(SAMN_single, self).__init__(sess, data, configs, logger)
        self.embed_size, self.mem_size, self.atten_size, self.reg1, self.reg2 = int(configs['embed_size']), int(configs['mem_size']), int(configs['atten_size']), \
            float(configs['reg1']), float(configs['reg2'])
        logger.info(' model_params: embed_size=%d, mem_size=%d, atten_size=%d, reg1=%s, reg2=%s' % (self.embed_size, self.mem_size, self.atten_size, self.reg1, \
            self.reg2) + ', ' + self.model_params)
        # Specify training and testing model
        self.train_model = self.train_model_samn_single
        self.test_model_rs, self.test_model_loo = self.test_model_rs_samn_single, self.test_model_loo_samn_single

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('bpr_inputs'):
            self.u_idx = __create_p(tf.int32, [], 'u_idx') # For mini-batch by single user
            self.i_idx = __create_p(tf.int32, [None], 'i_idx') # u's consumed items
            self.i_nums = __create_p(tf.int32, [], 'i_nums')
            self.j_idx = __create_p(tf.int32, [None], 'j_idx')
            self.uf_idx = __create_p(tf.int32, [None], 'uf_idx') # u's friends
            self.uf_nums = __create_p(tf.int32, [], 'uf_nums')
            # self.batch_size_t_ = __create_p(tf.int32, [], 'batch_size_t_') # Number of testing users in current batch

    def _create_params(self):
        def __create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        def __create_b(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_))
        with tf.variable_scope('samn_params'):
            # Embedding matrix
            self.P = __create_w([self.data.user_nums+1, self.embed_size], 'P')
            self.Q = __create_w([self.data.item_nums, self.embed_size], 'Q')
            self.i_b = __create_b([self.data.item_nums], 'i_bias')
            # Memory network
            self.Key = __create_w([self.embed_size, self.mem_size], 'Key') # Key matrix
            self.Mem = __create_w([self.mem_size, self.embed_size], 'Mem') # Memory matrix
            # Friend-level attention
            self.W3 = __create_w([self.embed_size, self.atten_size], 'W3')
            self.b = __create_b([self.atten_size], 'b')
            self.h = __create_b([self.atten_size], 'h')

    def _create_embeddings(self):
        with tf.name_scope('embeddings'):
            self.u_embed = tf.gather(self.P, self.u_idx)
            self.i_embed, self.i_b_embed = tf.gather(self.Q, self.i_idx), tf.gather(self.i_b, self.i_idx)
            self.j_embed, self.j_b_embed = tf.gather(self.Q, self.j_idx), tf.gather(self.i_b, self.j_idx)
            self.uf_embed = tf.gather(self.P, self.uf_idx) # Friend embedding, [uf_nums, embed_size]

    # Attention-based memory module
    def _get_friend_vec(self):
        with tf.name_scope('memory_attention'):
            # Joint embedding
            u_embed_norm = tf.nn.l2_normalize(self.u_embed)
            uf_embed_norm = tf.nn.l2_normalize(self.uf_embed, 1)
            self.joint_embed = tf.einsum('ab,b->ab', uf_embed_norm, u_embed_norm)

            # Key addressing
            self.atten_key = tf.einsum('ab,bc->ac', self.joint_embed, self.Key) # [uf_nums, mem_size]
            self.atten_key = tf.nn.softmax(self.atten_key) # axis=-1

            # Generate friend vectors
            F = tf.einsum('ab,bc->ac', self.atten_key, self.Mem) # [uf_nums, embed_size]
            self.uf_vec = tf.multiply(F, self.uf_embed)

    # Friend-level attention module
    def _get_u_frien(self):
        with tf.name_scope('friend_attention'):
            self.atten_frien = tf.einsum('ac,c->a', tf.nn.relu(tf.einsum('ab,bc->ac', self.uf_vec, self.W3) + self.b), self.h) # [uf_nums]
            self.atten_frien = tf.nn.softmax(self.atten_frien)
            # Generate u's friend-part representation
            self.u_frien = tf.einsum('a,ab->b', self.atten_frien, self.uf_vec)

    def _create_inference(self):
        with tf.name_scope('inference'):
            self._get_friend_vec()
            self._get_u_frien()
            # u's final representation
            self.u_vec = self.u_embed + self.u_frien

            # Calculate preference scores
            self.ui_scores = tf.einsum('b,ab->a', self.u_vec, self.i_embed) + self.i_b_embed
            self.uj_scores = tf.einsum('b,ab->a', self.u_vec, self.j_embed) + self.j_b_embed
            
            # Loss
            l2_loss1 = tf.nn.l2_loss(self.u_vec) + tf.nn.l2_loss(self.i_embed) + tf.nn.l2_loss(self.j_embed) + tf.nn.l2_loss(self.i_b_embed) + \
                tf.nn.l2_loss(self.j_b_embed)
            l2_loss2 = tf.nn.l2_loss(self.W3) + tf.nn.l2_loss(self.b) + tf.nn.l2_loss(self.h)
            self.loss = get_loss(self.loss_func, self.ui_scores - self.uj_scores) + self.reg1 * l2_loss1 + self.reg2 * l2_loss2

            # Optimize
            self.train = self.optimizer.minimize(self.loss)

    def _predict(self):
        with tf.name_scope('predict'):
            if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0:
                self.pre_scores = self.ui_scores
            else:
                self.pre_scores = tf.einsum('b,ab->a', self.u_vec, self.Q)

    def _save_model(self):
        var_list = {'samn_params/P': self.P, 'samn_params/Q': self.Q, 'samn_params/i_b': self.i_b, 'samn_params/Key': self.Key, 'samn_params/Mem': self.Mem, \
            'samn_params/W3': self.W3, 'samn_params/b': self.b, 'samn_params/h': self.h}
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
        # self._save_model()
