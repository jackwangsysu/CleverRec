# coding: utf-8

" Base Recommender for Ranking/Rating Model. "

import tensorflow as tf
from utils.tools import *
import math

class Recommender(object):
    def __init__(self, sess, data, configs, logger):
        self.model = configs['recommender']
        self.sess, self.data, self.configs, self.logger = sess, data, configs, logger
        # Common parameters
        self._get_common_params()

    def _get_common_params(self):
        self.epoches, self.batch_size, self.batch_size_t, self.lr, self.neg_samples = int(self.configs['epoches']), int(self.configs['batch_size']), \
            int(self.configs['test.batch_size']), float(self.configs['lr']), int(self.configs['test.neg_samples'])
        self.fism_like, self.cml_like = True if 'fism_like' in self.configs else False, True if 'cml_like' in self.configs else False
        self.is_pairwise = self.configs['is_pairwise']
        self.initializer = get_initializer(self.configs['init_method'], float(self.configs['stddev']))
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        self.loss_func = self.configs['loss_func']
        self.optimizer = get_optimizer(self.configs['optimizer'], self.lr)
        self.saved_model_dir = self.configs['saved_model_dir']
        self.T = int(self.configs['test.interval']) # Test every T epoches
        self.topk = list(map(int, self.configs['topk'][1:-1].split(',')))
        self.model_params = 'lr=%s, loss_func=%s' % (self.lr, self.configs['loss_func'])

    def build_model(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def test_model(self):
        raise NotImplementedError

    def run_model(self):
        raise NotImplementedError