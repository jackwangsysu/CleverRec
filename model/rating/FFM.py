# coding: utf-8

" Field-aware Factorization Machine. (2016)"

import tensorflow as tf
from model.RatingRecommender import RatingRecommender
from utils.tools import get_loss
import os

class FFM(RatingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(FFM, self).__init__(sess, data, configs, logger)
        self.embed_size, self.reg = int(configs['embed_size']), float(configs['reg'])
        logger.info(' model_params: embed_size=%d, field_nums=%d, reg=%s' % (self.embed_size, data.field_nums, self.reg) + ', ' + self.model_params)

    def _create_inputs(self):
        def __create_p(dtype_, shape_, name_):
            return tf.placeholder(dtype_, shape=shape_, name=name_)
        with tf.name_scope('fm_inputs'):
            self.x_idx = __create_p(tf.int32, [None, None], 'x_idx') # Input idx
            self.x_value = __create_p(tf.float32, [None, None], 'x_value') # Input value
            self.y = __create_p(tf.float32, [None], 'y') # Labels

    def _create_params(self):
        def _create_w(shape_, name_):
            return tf.get_variable(name_, dtype=tf.float32, initializer=self.initializer(shape_), regularizer=self.regularizer)
        with tf.variable_scope('ffm_params'):
            self.w0 = tf.Variable(tf.constant(0.0), name='w0') # bias
            self.wi = __create_w([self.data.feature_nums+1], 'wi') # Linear part (0 as the invalid id)
            self.vif = __create_w([self.data.feature_nums+1, self.field_nums, self.embed_size], 'vif') # Field-aware feature embeddings (2nd-order part)