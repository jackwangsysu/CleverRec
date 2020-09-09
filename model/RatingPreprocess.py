# coding: utf-8

" Load and preprocess the dataset for rating model. "

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import os, time, warnings
from datetime import datetime
from dateutil import rrule
from utils.tools import timer, re_index

warnings.filterwarnings('ignore')

class RatingPreprocess(object):
    def __init__(self, configs, logger):
        self.configs, self.logger = configs, logger
        self.model = configs['recommender']
        self.file_path = os.path.join(configs['data.root_dir'], configs['data.dataset'])
        if self.model == 'FM':
            self.is_real_valued = True if configs['is_real_valued'] == 'True' else False # Whether consider real-valued features
            if self.is_real_valued:
                self.X_idx_tr, self.X_value_tr, self.y_tr, self.X_idx_t, self.X_value_t, self.y_t= self._load_data()
            else:
                self.X_tr, self.y_tr, self.X_t, self.y_t = self._load_data()
            self.feature_nums = len(self.all_features)
            del self.all_features
        else:
            pass

    @timer('Load data')
    def _load_data(self):
        if self.model == 'FM':
            self.all_features = {}
            train_file = os.path.join(self.file_path, self.configs['data.dataset']+self.configs['train'])
            test_file = os.path.join(self.file_path, self.configs['data.dataset']+self.configs['test'])

            if self.is_real_valued:
                # Consider real values
                X_idx_tr, X_value_tr, y_tr = self._read_file(train_file)
                X_idx_t, X_value_t, y_t = self._read_file(test_file)
                # Map to new indices
                X_idx_tr = [list(map(lambda x : self.all_features[x], b)) for b in X_idx_tr]
                X_idx_t = [list(map(lambda x : self.all_features[x], b)) for b in X_idx_t]
                return X_idx_tr, X_value_tr, y_tr, X_idx_t, X_value_t, y_t
            else:
                X_tr, y_tr = self._read_file(train_file)
                X_t, y_t = self._read_file(test_file)
                # Map to new indices
                X_tr = [list(map(lambda x : self.all_features[x], b)) for b in X_tr]
                X_t = [list(map(lambda x : self.all_features[x], b)) for b in X_t]
                return X_tr, y_tr, X_t, y_t
        else:
            pass

    # For FM
    def _read_file(self, file_name):
        X_, X_idx, X_value, y_ = [], [], [], []
        f_count = len(self.all_features)
        with open(file_name, 'r') as fr:
            for line in fr.readlines():
                lst = line.strip().split(',')
                y_.append(float(lst[0]))
                if self.is_real_valued:
                    row_idx, row_value = [], []
                    for col in lst[1:]:
                        idx, value = col.split(':')
                        row_idx.append(idx)
                        row_value.append(float(value))
                        if idx not in self.all_features:
                            self.all_features[idx] = f_count
                            f_count += 1
                    X_idx.append(row_idx)
                    X_value.append(row_value)
                else:
                    row_x = []
                    for col in lst[1:]:
                        row_x.append(col)
                        if col not in self.all_features:
                            self.all_features[col] = f_count
                            f_count += 1
                    X_.append(row_x)
        if self.is_real_valued:
            return X_idx, X_value, y_
        else:
            return X_, y_