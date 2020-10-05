" Recommender for Ranking/Rating Model. "

import numpy as np, scipy.sparse as sp, tensorflow as tf
from model.Recommender import Recommender
from collections import defaultdict
from utils.tools import timer, get_adj_mat_i, get_adj_mat_s
from utils.sampler import *
from utils.metrics import *
import os, time, math
import shutil

saved_model_dir = './saved_model'
version = 1

# For ranking model.
class RankingRecommender(Recommender):
    def __init__(self, sess, data, configs, logger):
        super(RankingRecommender, self).__init__(sess, data, configs, logger)
        self.neg_ratio = int(configs['neg_ratio'])
        self.model_params += ', neg_ratio=%d' % self.neg_ratio
        # Testing data
        self.test_users = list(data.ui_test.keys())
        self.test_batches = math.ceil(len(self.test_users)/self.batch_size_t)
        if self.model == 'SoHRML':
            max_i, max_s = int(configs['max_i']), int(configs['max_s'])
            self.adj_att_i, self.all_r_list_i, self.all_c_list_i = get_adj_mat_i(data, max_i)
            self.adj_att_s, self.all_r_list_s, self.all_c_list_s = get_adj_mat_s(data, max_s)
            self.adj_i_shape, self.adj_s_shape = (data.user_nums+data.item_nums, data.user_nums+data.item_nums), (data.user_nums, data.user_nums)
        self.saved_path = os.path.join(saved_model_dir, self.model, str(version))
        
    # Train the model (Single epoch)
    def train_model(self):
        total_loss = 0
        if self.is_pairwise == 'True':
            # Sample training instances
            train_ = pairwise_ranking_sampler(self.data, self.neg_ratio, self.batch_size, fism_like=self.fism_like)
            # Train
            for id in range(train_[0]):
                u_idx, i_idx, j_idx = train_[1][id*self.batch_size:(id+1)*self.batch_size], train_[2][id*self.batch_size:(id+1)*self.batch_size], \
                    train_[3][id*self.batch_size:(id+1)*self.batch_size]
                train_dict = {self.u_idx: u_idx, self.i_idx: i_idx, self.j_idx: j_idx}
                if self.fism_like:
                    u_neighbors_num = train_[4][id*self.batch_size:(id+1)*self.batch_size]
                    train_dict.update({self.u_neighbors_num: u_neighbors_num})
                _, loss_val = self.sess.run([self.train, self.loss], train_dict)
                total_loss += loss_val
        else:
            # Sample training instances
            train_ = pointwise_ranking_sampler(self.data, self.neg_ratio, self.batch_size, fism_like=self.fism_like)
            # Train
            for id in range(train_[0]):
                u_idx, i_idx, y = train_[1][id*self.batch_size:(id+1)*self.batch_size], train_[2][id*self.batch_size:(id+1)*self.batch_size], \
                    train_[3][id*self.batch_size:(id+1)*self.batch_size]
                train_dict = {self.u_idx: u_idx, self.i_idx: i_idx, self.y: y}
                if self.fism_like:
                    u_neighbors_num = train_[4][id*self.batch_size:(id+1)*self.batch_size]
                    train_dict.update({self.u_neighbors_num: u_neighbors_num})
                _, loss_val = self.sess.run([self.train, self.loss], train_dict)
                total_loss += loss_val
        return total_loss/train_[0]

    # Form mini-batch by user (NAIS)
    def train_model_nais(self):
        total_loss = 0.0
        for u, items in self.data.ui_train.items():
            u_idx = items
            i_idx, y = [], []
            seen_items = set(self.data.ui_train[u])
            for i in items:
                # Positive instances
                i_idx.append(i)
                y.append(1.0)
                # Negative instances
                random_j = set()
                for s in range(self.neg_ratio):
                    j = np.random.randint(self.data.item_nums)
                    while j in random_j or j in seen_items:
                        j = np.random.randint(self.data.item_nums)
                    random_j.add(j)
                    i_idx.append(j)
                    y.append(0.0)
            # Train
            train_dict = {self.u_idx: u_idx, self.u_nbrs_num: len(u_idx), self.i_idx: i_idx, self.i_nums: len(i_idx), self.y: y}
            _, loss_val = self.sess.run([self.train, self.loss], train_dict)
            total_loss += loss_val
        return total_loss/len(self.data.ui_train)

    # For CML
    def train_model_cml(self):
        total_loss = 0.0
        train_ = ranking_sampler_cml(self.data, self.neg_ratio, self.batch_size)
        # Train
        for id in range(train_[0]):
            u_idx, i_idx, neg_items = train_[1][id*self.batch_size:(id+1)*self.batch_size], train_[2][id*self.batch_size:(id+1)*self.batch_size], \
                train_[3][id*self.batch_size:(id+1)*self.batch_size]
            train_dict = {self.u_idx: u_idx, self.i_idx: i_idx, self.neg_items: neg_items}
            _, loss_val = self.sess.run([self.train, self.loss], train_dict)
            total_loss += loss_val
        return total_loss/train_[0]

    # For SBPR
    def train_model_sbpr(self, is_suk=True):
        total_loss = 0.0
        train_ = ranking_sampler_sbpr(self.data, self.SPu, self.neg_ratio, self.batch_size)
        # Train
        for id in range(train_[0]):
            u_idx, i_idx, i_s_idx, i_neg_idx, suk = train_[1][id*self.batch_size:(id+1)*self.batch_size], train_[2][id*self.batch_size:(id+1)*self.batch_size], \
                train_[3][id*self.batch_size:(id+1)*self.batch_size], train_[4][id*self.batch_size:(id+1)*self.batch_size], train_[5][id*self.batch_size:(id+1)*self.batch_size]
            train_dict = {self.u_idx: u_idx, self.i_idx: i_idx, self.i_s_idx: i_s_idx, self.i_neg_idx: i_neg_idx, self.suk: suk}
            _, loss_val = self.sess.run([self.train, self.loss], train_dict)
            total_loss += loss_val
        return total_loss/train_[0]

    # For SAMN
    def train_model_samn(self):
        total_loss = 0.0
        train_ = ranking_sampler_samn(self.data, self.neg_ratio, self.batch_size)
        # Train
        for id in range(train_[0]):
            u_idx, i_idx, j_idx, uf_idx = train_[1][id*self.batch_size:(id+1)*self.batch_size], train_[2][id*self.batch_size:(id+1)*self.batch_size], \
                train_[3][id*self.batch_size:(id+1)*self.batch_size], train_[4][id*self.batch_size:(id+1)*self.batch_size]
            train_dict = {self.u_idx: u_idx, self.i_idx: i_idx, self.j_idx: j_idx, self.uf_idx: uf_idx}
            _, loss_val = self.sess.run([self.train, self.loss], train_dict)
            total_loss += loss_val
        return total_loss/train_[0]

    # For SAMN_Single
    def train_model_samn_single(self):
        total_loss = 0.0
        for u, items in self.data.ui_train.items():
            if u not in self.data.user_friends:
                uf_idx = [self.data.user_nums]
            else:
                uf_idx = self.data.user_friends[u]
            seen_items = set(items)
            i_idx, j_idx = [], []
            for i in items:
                random_j = set()
                for s in range(self.neg_ratio):
                    i_idx.append(i)
                    j = np.random.randint(self.data.item_nums)
                    while j in random_j or j in seen_items:
                        j = np.random.randint(self.data.item_nums)
                    random_j.add(j)
                    j_idx.append(j)
            # Train
            train_dict = {self.u_idx: u, self.i_idx: i_idx, self.i_nums: len(i_idx), self.j_idx: j_idx, self.uf_idx: uf_idx, self.uf_nums: len(uf_idx)}
            _, loss_val = self.sess.run([self.train, self.loss], train_dict)
            total_loss += loss_val
        return total_loss/len(self.data.ui_train)

    # Update the sparse attentive adjacency matrix
    def _update_atten_mat(self):
        fold_len_i, fold_len_s = math.ceil(len(self.all_r_list_i)/self.adj_folds), math.ceil(len(self.all_r_list_s)/self.adj_folds) # Split into 100 folds
        att_scores_i, att_scores_s = [], []
        for id in range(self.adj_folds):
            start_i, end_i, start_s, end_s = id*fold_len_i, (id+1)*fold_len_i, id*fold_len_s, (id+1)*fold_len_s
            r_batch_i, c_batch_i = self.all_r_list_i[start_i:end_i], self.all_c_list_i[start_i:end_i]
            r_batch_s, c_batch_s = self.all_r_list_s[start_s:end_s], self.all_c_list_s[start_s:end_s]
            adj_dict = {self.r_batch_i: r_batch_i, self.c_batch_i: c_batch_i, self.r_batch_s: r_batch_s, self.c_batch_s: c_batch_s}
            # Calculate the attention scores
            temp_scores_i, temp_scores_s = self.sess.run([self.att_scores_i_batch, self.att_scores_s_batch], adj_dict)
            att_scores_i.append(temp_scores_i)
            att_scores_s.append(temp_scores_s)
        att_scores_i, att_scores_s = np.hstack(att_scores_i), np.hstack(att_scores_s)

        # Get the attention weights (Softmax)
        att_scores_dict = {self.att_scores_i: att_scores_i, self.att_scores_s: att_scores_s}
        new_adj_att_i, new_adj_att_s = self.sess.run([self.adj_att_i_out, self.adj_att_s_out], feed_dict=att_scores_dict)
        self.adj_att_i = sp.coo_matrix((new_adj_att_i.values, (new_adj_att_i.indices[:, 0], new_adj_att_i.indices[:, 1])), shape=self.adj_i_shape)
        self.adj_att_s = sp.coo_matrix((new_adj_att_s.values, (new_adj_att_s.indices[:, 0], new_adj_att_s.indices[:, 1])), shape=self.adj_s_shape)
        print("  Update the attentive adj_mat done!")

    # For RML-DGATs
    def train_model_sohrml(self):
        total_loss = 0.0
        # Sample training instances
        u_features_i, i_features, j_features, u_features_s, v_features, w_features, train_nums_i, train_nums_s = ranking_sampler_sohrml(self.data, self.neg_ratio)

        # Train
        batch_len_i, batch_len_s = math.ceil(train_nums_i/self.train_batches), math.ceil(train_nums_s/self.train_batches) # Split into 1000 batches
        for id in range(self.train_batches):
            u_idx_i, i_idx, j_idx = u_features_i[id*batch_len_i:(id+1)*batch_len_i], i_features[id*batch_len_i:(id+1)*batch_len_i], \
                j_features[id*batch_len_i:(id+1)*batch_len_i]
            u_idx_s, v_idx, w_idx = u_features_s[id*batch_len_s:(id+1)*batch_len_s], v_features[id*batch_len_s:(id+1)*batch_len_s], \
                w_features[id*batch_len_s:(id+1)*batch_len_s]
            train_dict = {self.u_idx: u_idx_i, self.i_idx: i_idx, self.j_idx: j_idx, self.u_idx_s: u_idx_s, self.v_idx: v_idx, self.w_idx: w_idx, self.is_train: 1}
            # Run
            _, loss_val = self.sess.run([self.train, self.loss], train_dict)
            total_loss += loss_val
        return total_loss/self.train_batches

    # Random split with all
    def test_model_rs(self):
        HR, MRR, NDCG = defaultdict(list), defaultdict(list), defaultdict(list) # evaluation metrics
        for t_id in range(self.test_batches):
            cur_users = self.test_users[t_id*self.batch_size_t:(t_id+1)*self.batch_size_t]
            u_idx = cur_users
            test_dict = {self.u_idx: u_idx, self.batch_size_t_: len(cur_users)}
            # For FISM
            if self.fism_like:
                u_neighbors_num = []
                for u in cur_users:
                    u_nbr_num = len(self.data.ui_train[u]) if u in self.data.ui_train else 0
                    u_neighbors_num.append(u_nbr_num)
                test_dict.update({self.u_neighbors_num: u_neighbors_num})
            # For RML-DGATs
            if self.model in ['RML_DGATs', 'SoHRML']:
                test_dict.update({self.is_train: 0})
            # For SAMN
            if self.model == 'SAMN':
                uf_idx = []
                for u in cur_users:
                    uf_idx.append(self.data.user_friends[u])
                test_dict.update({self.uf_idx: uf_idx})
            # Predict
            pre_scores = self.sess.run(self.pre_scores, test_dict)
            if self.cml_like:
                args = np.argsort(pre_scores, axis=1)
            else:
                args = np.argsort(-pre_scores, axis=1)
            # Evaluate
            for id in range(len(cur_users)):
                u = cur_users[id]
                args_u = args[id]
                topk_items = np.zeros(self.topk[-1])
                if u not in self.data.ui_train:
                    topk_items = args_u[:self.topk[-1]]
                else:
                    seen_items = set(self.data.ui_train[u])
                    count, j = 0, 0 
                    while count < self.topk[-1]:
                        if args_u[j] not in seen_items:
                            topk_items[count] = args_u[j]
                            count += 1
                        j += 1
                for kid in range(len(self.topk)):
                    rec_items = topk_items[:self.topk[kid]]
                    hr_u, mrr_u, ndcg_u = cal_ranking_metrics(self.data.ui_test[u], rec_items, self.topk[kid])
                    HR[kid].append(hr_u)
                    MRR[kid].append(mrr_u)
                    NDCG[kid].append(ndcg_u)
        return HR, MRR, NDCG

    # Leave-One-Out / Random split with 1000 negative items
    def test_model_loo(self):
        HR, MRR, NDCG = defaultdict(list), defaultdict(list), defaultdict(list) # evaluation metrics
        for t_id in range(self.test_batches):
            cur_users = self.test_users[t_id*self.batch_size_t:(t_id+1)*self.batch_size_t]
            u_idx, i_idx, i_nums = [], [], []
            if self.fism_like:
                u_neighbors_num = []
            for u in cur_users:
                tmp_items = self.data.ui_test[u]
                u_idx.extend([u]*len(tmp_items))
                i_idx.extend(tmp_items)
                i_nums.append(len(tmp_items))
                if self.fism_like:
                    u_nbr_num = len(self.data.ui_train[u]) if u in self.data.ui_train else 0
                    u_neighbors_num.extend([u_nbr_num]*len(tmp_items))
            test_dict = {self.u_idx: u_idx, self.i_idx: i_idx}
            if self.fism_like:
                test_dict.update({self.u_neighbors_num: u_neighbors_num})
            # For RML-DGATs
            if self.model in ['RML_DGATs', 'SoHRML']:
                test_dict.update({self.is_train: 0})
            # For SAMN
            if self.model == 'SAMN':
                uf_idx = []
                for u in cur_users:
                    uf_idx.append(self.data.user_friends[u])
                test_dict.update({self.uf_idx: uf_idx})
            # Predict
            pre_scores = self.sess.run(self.pre_scores, test_dict)
            # Evaluate
            s_id, e_id = 0, 0
            for id in range(len(cur_users)):
                u = cur_users[id]
                e_id += i_nums[id]
                pre_scores_u = pre_scores[s_id:e_id]
                if self.cml_like:
                    args_u = np.argsort(pre_scores_u)[:self.topk[-1]]
                else:
                    args_u = np.argsort(-pre_scores_u)[:self.topk[-1]]
                real_items = self.data.ui_test[u][self.neg_samples:]
                if not isinstance(real_items, list): # loo
                    real_items = [real_items]
                for kid in range(len(self.topk)):
                    rec_items = np.take(self.data.ui_test[u], args_u[:self.topk[kid]])
                    hr_u, mrr_u, ndcg_u = cal_ranking_metrics(real_items, rec_items, self.topk[kid])
                    HR[kid].append(hr_u)
                    MRR[kid].append(mrr_u)
                    NDCG[kid].append(ndcg_u)
                s_id = e_id
        return HR, MRR, NDCG

    def test_model_rs_nais(self):
        HR, MRR, NDCG = defaultdict(list), defaultdict(list), defaultdict(list) # evaluation metrics
        for u in self.test_users:
            if u in self.data.ui_train:
                u_idx = self.data.ui_train[u]
                seen_items = set(u_idx)
            else:
                u_idx = [self.data.item_nums]
                seen_items = set()
            test_dict = {self.u_idx: u_idx, self.u_nbrs_num: len(u_idx)}
            # Predict
            pre_scores = self.sess.run(self.pre_scores, test_dict)[:-1] # Exclude id=item_nums
            args_u = np.argsort(-pre_scores)
            # Evaluate
            topk_items = np.zeros(self.topk[-1])
            count, j = 0, 0
            while count < self.topk[-1]:
                if args_u[j] not in seen_items:
                    topk_items[count] = args_u[j]
                    count += 1
                j += 1
            for kid in range(len(self.topk)):
                rec_items = topk_items[:self.topk[kid]]
                hr_u, mrr_u, ndcg_u = cal_ranking_metrics(self.data.ui_test[u], rec_items, self.topk[kid])
                HR[kid].append(hr_u)
                MRR[kid].append(mrr_u)
                NDCG[kid].append(ndcg_u)
        return HR, MRR, NDCG

    def test_model_loo_nais(self):
        HR, MRR, NDCG = defaultdict(list), defaultdict(list), defaultdict(list) # evaluation metrics
        for u in self.test_users:
            u_idx = self.data.ui_train[u] if u in self.data.ui_train else [self.data.item_nums]
            i_idx = self.data.ui_test[u]
            test_dict = {self.u_idx: u_idx, self.u_nbrs_num: len(u_idx), self.i_idx: i_idx, self.i_nums: len(i_idx)}
            # Predict
            pre_scores = self.sess.run(self.pre_scores, test_dict)
            args_u = np.argsort(-pre_scores)[:self.topk[-1]]
            real_items = self.data.ui_test[u][self.neg_samples:]
            if not isinstance(real_items, list): # loo
                real_items = [real_items]
            for kid in range(len(self.topk)):
                rec_items = np.take(self.data.ui_test[u], args_u[:self.topk[kid]])
                hr_u, mrr_u, ndcg_u = cal_ranking_metrics(real_items, rec_items, self.topk[kid])
                HR[kid].append(hr_u)
                MRR[kid].append(mrr_u)
                NDCG[kid].append(ndcg_u)
        return HR, MRR, NDCG

    def test_model_rs_samn_single(self):
        HR, MRR, NDCG = defaultdict(list), defaultdict(list), defaultdict(list) # evaluation metrics
        for u in self.test_users:
            uf_idx = self.data.user_friends[u] if u in self.data.user_friends else [self.data.user_nums]
            seen_items = set(self.data.ui_train[u]) if u in self.data.ui_train else set()
            test_dict = {self.u_idx: u, self.uf_idx: uf_idx, self.uf_nums: len(uf_idx)}
            # Predict
            pre_scores = self.sess.run(self.pre_scores, test_dict)
            args_u = np.argsort(-pre_scores)
            # Evaluate
            topk_items = np.zeros(self.topk[-1])
            count, j = 0, 0
            while count < self.topk[-1]:
                if args_u[j] not in seen_items:
                    topk_items[count] = args_u[j]
                    count += 1
                j += 1
            for kid in range(len(self.topk)):
                rec_items = topk_items[:self.topk[kid]]
                hr_u, mrr_u, ndcg_u = cal_ranking_metrics(self.data.ui_test[u], rec_items, self.topk[kid])
                HR[kid].append(hr_u)
                MRR[kid].append(mrr_u)
                NDCG[kid].append(ndcg_u)
        return HR, MRR, NDCG

    def test_model_loo_samn_single(self):
        HR, MRR, NDCG = defaultdict(list), defaultdict(list), defaultdict(list) # evaluation metrics
        for u in self.test_users:
            i_idx = self.data.ui_test[u]
            uf_idx = self.data.user_friends[u] if u in self.data.user_friends else [self.data.user_nums]
            test_dict = {self.u_idx: u, self.i_idx: i_idx, self.i_nums: len(i_idx), self.uf_idx: uf_idx, self.uf_nums: len(uf_idx)}
            # Predict
            pre_scores = self.sess.run(self.pre_scores, test_dict)
            args_u = np.argsort(-pre_scores)[:self.topk[-1]]
            real_items = self.data.ui_test[u][self.neg_samples:]
            if not isinstance(real_items, list): # loo
                real_items = [real_items]
            for kid in range(len(self.topk)):
                rec_items = np.take(self.data.ui_test[u], args_u[:self.topk[kid]])
                hr_u, mrr_u, ndcg_u = cal_ranking_metrics(real_items, rec_items, self.topk[kid])
                HR[kid].append(hr_u)
                MRR[kid].append(mrr_u)
                NDCG[kid].append(ndcg_u)
        return HR, MRR, NDCG
    
    @timer('run_model')
    def run_model(self):
        self.build_model() # Build graph
        self.sess.run(tf.global_variables_initializer())

        best_ndcg10, best_epoch = 0, 0 # Record best metrics w.r.t. NDCG@10
        best_metrics = {}
        for epoch in range(self.epoches):
            if self.model == 'SoHRML':
                # Update the attentive adjacency matrix
                self._update_atten_mat()
            # Train
            t1 = time.time()
            avg_loss = self.train_model()
            self.logger.info(' epoch %d\n  Training loss: %.4f, time: %s' % (epoch+1, avg_loss, time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))))

            # Test
            t2 = time.time()
            if (epoch + 1) % self.T: # Test every T epoches
                continue
            if self.configs['data.split_way'] == 'loo' or self.neg_samples > 0: # loo/random sampling with 1000 ...
                HR, MRR, NDCG = self.test_model_loo()
            else: # random split
                HR, MRR, NDCG = self.test_model_rs()
            self.logger.info('  Testing time: %s' % time.strftime('%H:%M:%S', time.gmtime(time.time() - t2)))

            # Record best performance
            best_flag = False
            for id in range(len(self.topk)):
                hr, mrr, ndcg = np.mean(HR[id]), np.mean(MRR[id]), np.mean(NDCG[id])
                self.logger.info('  (k=%d) HR=%.4f, MRR=%.4f, NDCG=%.4f' % (self.topk[id], hr, mrr, ndcg))
                if id == 0 and ndcg > best_ndcg10:
                    best_flag = True
                    best_ndcg10 = ndcg
                if best_flag:
                    best_metrics[id] = (hr, mrr, ndcg)
                    best_epoch = epoch + 1
                    # if id == len(self.topk) - 1:
                    #     self.saver.save(self.sess, os.path.join(self.saved_model_dir, self.model, self.model)) # Save the model
            self.sess.graph.finalize() # Lock the graph

        # Final results
        self.logger.info('best_epoch: %d' % best_epoch)
        for id in range(len(self.topk)):
            hr, mrr, ndcg = best_metrics[id]
            self.logger.info('  (k=%d) HR=%.4f, MRR=%.4f, NDCG=%.4f' % (self.topk[id], hr, mrr, ndcg))