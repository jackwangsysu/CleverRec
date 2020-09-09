# coding: utf-8

" Recommender for Rating model. "

import numpy as np, tensorflow as tf
from model.Recommender import Recommender
from utils.tools import timer
from utils.metrics import cal_rmse_mae
import math, time

# For rating model.
class RatingRecommender(Recommender):
    def __init__(self, sess, data, configs, logger):
        super(RatingRecommender, self).__init__(sess, data, configs, logger)
        self.model = configs['recommender']
        if self.model == 'FM':
            self.is_real_valued = True if configs['is_real_valued'] == 'True' else False

    def train_model(self):
        pass

    def test_model(self):
        pass

    # For FM
    def train_fm(self):
        if self.is_real_valued:
            X_idx_tr, X_value_tr, y_tr = np.array(self.data.X_idx_tr), np.array(self.data.X_value_tr), np.array(self.data.y_tr)
        else:
            X_tr, y_tr = np.array(self.data.X_tr), np.array(self.data.y_tr)
        # Shuffle
        s_idx = np.random.permutation(y_tr.size)
        if self.is_real_valued:
            X_idx_tr, X_value_tr, y_tr = X_idx_tr[s_idx], X_value_tr[s_idx], y_tr[s_idx]
        else:
            X_tr, y_tr = X_tr[s_idx], y_tr[s_idx]

        # Train
        train_batches = math.ceil(y_tr.size/self.batch_size)
        total_loss = 0.0
        y_pre = []
        for id in range(train_batches):
            if self.is_real_valued:
                x_idx, x_value, y = X_idx_tr[id*self.batch_size:(id+1)*self.batch_size], X_value_tr[id*self.batch_size:(id+1)*self.batch_size], \
                    y_tr[id*self.batch_size:(id+1)*self.batch_size]
                train_dict = {self.x_idx: x_idx, self.x_value: x_value, self.y: y}
            else:
                x_idx, y = X_tr[id*self.batch_size:(id+1)*self.batch_size], y_tr[id*self.batch_size:(id+1)*self.batch_size]
                train_dict = {self.x_idx: x_idx, self.y: y}
            _, loss_val, y_pre_ = self.sess.run([self.train, self.loss, self.y_pre], train_dict)
            y_pre.extend(y_pre_)
            total_loss += loss_val
        # Evaluate
        rmse, mae = cal_rmse_mae(y_pre, y_tr)
        return rmse, mae, total_loss/train_batches

    # For FM
    def test_fm(self):
        if self.is_real_valued:
            X_idx_t, X_value_t, y_t = self.data.X_idx_t, self.data.X_value_t, self.data.y_t
        else:
            X_t, y_t = self.data.X_t, self.data.y_t
        y_pre = []
        test_batches = math.ceil(len(y_t)/self.batch_size_t)
        for t_id in range(test_batches):
            if self.is_real_valued:
                x_idx, x_value = X_idx_t[t_id*self.batch_size_t:(t_id+1)*self.batch_size_t], X_value_t[t_id*self.batch_size_t:(t_id+1)*self.batch_size_t]
                test_dict = {self.x_idx: x_idx, self.x_value: x_value}
            else:
                x_idx = X_t[t_id*self.batch_size_t:(t_id+1)*self.batch_size_t]
                test_dict = {self.x_idx: x_idx}
            # Predict
            y_pre_ = self.sess.run(self.y_pre, test_dict)
            y_pre.extend(y_pre_)
        # Evaluate
        rmse, mae = cal_rmse_mae(y_pre, y_t)
        return rmse, mae

    @timer('run_model')
    def run_model(self):
        self.build_model() # Build graph
        self.sess.run(tf.global_variables_initializer())

        best_rmse, best_epoch = float('inf'), 0 # Record best metrics w.r.t. RMSE
        best_metrics = []
        for epoch in range(self.epoches):
            # Train
            t1 = time.time()
            rmse_tr, mae_tr, avg_loss = self.train_fm()
            self.logger.info(' Training epoch %d\n time=%s, RMSE=%.4f, MAE=%.4f' % (epoch+1, time.strftime('%H:%M:%S', time.gmtime(time.time() - t1)), rmse_tr, mae_tr))

            # Test
            t2 = time.time()
            rmse_t, mae_t = self.test_fm()
            self.logger.info('  Testing time=%s, RMSE=%.4f, MAE=%.4f' % (time.strftime('%H:%M:%S', time.gmtime(time.time() - t2)), rmse_t, mae_t))

            # Record best performance
            if rmse_t < best_rmse:
                best_rmse = rmse_t
                best_metrics = [rmse_t, mae_t]
                best_epoch = epoch + 1
            self.sess.graph.finalize() # Lock the graph

        # Final results
        self.logger.info('best_epoch=%d, best_rmse=%.4f, best_mae=%.4f' % (best_epoch, best_metrics[0], best_metrics[1]))