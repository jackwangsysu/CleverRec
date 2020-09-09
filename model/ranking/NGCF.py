# coding: utf-8

" CML: Collaborative Metric Learning (2017). "

import tensorflow as tf
from model.RankingRecommender import RankingRecommender
import os

class CML(RankingRecommender):
    def __init__(self, sess, data, configs, logger):
        super(CML, self).__init__(sess, data, configs, logger)
        self.embed_size, self.reg = int(configs['embed_size']), float(configs['reg'])
        logger.info(' model_params: embed_size=%d, reg=%s' % (self.embed_size, self.reg) + ', ' + self.model_params)

    def _create_inputs(self):
        pass

    def _create_params(self):
        pass

    def _create_embeddings(self):
        pass

    def _create_inference(self):
        pass

    def _predict(self):
        pass

    def _save_model(self):
        pass

    def build_model(self):
        self._create_inputs()
        self._create_params()
        self._create_embeddings()
        self._create_inference()
        self._predict()
        self._save_model()
