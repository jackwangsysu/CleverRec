# coding: utf-8

" Load and preprocess the dataset for ranking model. "

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import os, time, warnings
from utils.tools import timer, re_index

warnings.filterwarnings('ignore')

class RankingPreprocess(object):
    def __init__(self, configs, logger):
        self.configs, self.logger = configs, logger
        self.file_path = os.path.join(configs['data.root_dir'], configs['data.dataset'])
        ratings, item_set = self._load_data()
        self.ui_train, self.ui_test = self._split_data(ratings, item_set)
    
    @timer('Load data')
    def _load_data(self):
        def _read_csv(file_, header, names, usecols):
            return pd.read_csv(file_, sep=self.configs['data.sep'], header=header, names=names, usecols=usecols)

        rating_file = os.path.join(self.file_path, self.configs['data.file_name'])
        data_format = self.configs['data.format']
        if data_format == 'UI':
            ratings = _read_csv(rating_file, 0, ['u_id', 'i_id'], [0, 1])
        elif data_format == 'UIR':
            ratings = _read_csv(rating_file, 0, ['u_id', 'i_id', 'rating'], [0, 1, 2])
        elif data_format == 'UIRT':
            ratings = _read_csv(rating_file, 0, ['u_id', 'i_id', 'rating', 'time'], [0, 1, 2, 3])
            ratings.time = ratings.time.astype(int)

        # Filter users/items
        user_min, item_min = int(self.configs['data.user_min']), int(self.configs['data.item_min'])
        if user_min > 0:
            ratings = self._filter_users(ratings, user_min)
        if item_min > 0:
            ratings = self._filter_items(ratings, item_min)
        
        # Reindex
        user_set_, item_set_ = set(ratings.u_id.unique()), set(ratings.i_id.unique())
        self.user_nums, self.item_nums = len(user_set_), len(item_set_)
        user_map, item_map = re_index(user_set_), re_index(item_set_)
        ratings.u_id = ratings.u_id.map(lambda x : user_map[x])
        ratings.i_id = ratings.i_id.map(lambda x : item_map[x])
        item_set = set(ratings.i_id.unique())

        # Load social data
        if 'social_file' in self.configs:
            social_file = os.path.join(self.file_path, self.configs['social_file'])
            trusts = _read_csv(social_file, 0, ['u_id', 'v_id'], [0, 1])
            # Filter out invalid users
            trusts = trusts[trusts.u_id.isin(user_set_) & trusts.v_id.isin(user_set_)]
            # Reindex
            trusts.u_id = trusts.u_id.map(lambda x : user_map[x])
            trusts.v_id = trusts.v_id.map(lambda x : user_map[x])
            self.user_friends = trusts.groupby('u_id').v_id.apply(list).to_dict()
            # print('total_relations: ', trusts.shape[0])
        
        return ratings, item_set

    # Filter users
    def _filter_users(self, ratings, user_min):
        user_lens = ratings.groupby('u_id').size().to_dict()
        user_deleted = set()
        for u in user_lens:
            if user_lens[u] < user_min:
                user_deleted.add(u)
        # Remove corresponding rows in ratings
        ratings = ratings[~(ratings.u_id.isin(user_deleted))].reset_index(drop=True)
        return ratings

    # Filter items
    def _filter_items(self, ratings, item_min):
        item_lens = ratings.groupby('i_id').size().to_dict()
        item_deleted = set()
        for i in item_lens:
            if item_lens[i] < item_min:
                item_deleted.add(i)
        ratings = ratings[~(ratings.i_id.isin(item_deleted))].reset_index(drop=True)
        return ratings

    # Train/test split
    @timer('Split data')
    def _split_data(self, ratings, item_set=None):
        split_way = self.configs['data.split_way'] # Random split/Leave-one-out
        if self.configs['data.split_by_time'] == 'True':
            ratings.sort_values(['u_id', 'time'], inplace=True)
        if split_way == 'loo': # loo
            train_data, test_data = [], []
            user_items = ratings.groupby('u_id')
            for user, items in user_items:
                if len(items) <= 3:
                    train_data.append(items)
                else:
                    train_data.append(items.iloc[:-1])
                    test_data.append(items.iloc[-1:])
            train_data, test_data = pd.concat(train_data, ignore_index=True), pd.concat(test_data, ignore_index=True)
        else: # Random split
            r1, r2, r3 = map(float, self.configs['data.split_ratio'][1:-1].split(','))
            if r2 > 0:
                train_data, tmp_data = train_test_split(ratings, test_size=1.0-r1)
                _, test_data = train_test_split(tmp_data, test_size=r3/(r2+r3))
            else:
                train_data, test_data = train_test_split(ratings, test_size=r3)
            train_data, test_data = train_data.reset_index(drop=True), test_data.reset_index(drop=True)
        ui_train = train_data.groupby('u_id').i_id.apply(list).to_dict()
        ui_test = test_data.groupby('u_id').i_id.apply(list).to_dict()
        
        # Sample negative 99/1000 items for testing
        neg_samples = int(self.configs['test.neg_samples'])
        if split_way == 'loo' or neg_samples > 0:
            tmp_ui_test = {}
            for u in ui_test:
                seen_items = set() if u not in ui_train else set(ui_train[u])
                u_neg_items = np.random.choice(list(item_set - seen_items), size=neg_samples, replace=False).tolist()
                tmp_ui_test[u] = u_neg_items
                tmp_ui_test[u].extend(ui_test[u])
            ui_test = tmp_ui_test

        tmp_str = '' if split_way == 'loo' else ('split_ratio=%s, ' % self.configs['data.split_ratio'])
        self.logger.info(' Data: dataset=%s, split_way=%s, neg_samples=%d, %suser_nums=%d, item_nums=%d, ratings_num=%d' % (self.configs['data.dataset'], \
            split_way, neg_samples, tmp_str, self.user_nums, self.item_nums, ratings.shape[0]))
        return ui_train, ui_test
