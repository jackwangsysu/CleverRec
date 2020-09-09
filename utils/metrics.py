# coding: utf-8

" Evaluation metrics for ranking/rating. "

import numpy as np
import math

# Calculate HR@K, MRR@K and NDCG@K
def cal_ranking_metrics(real_items, rec_items, K):
    hit, mrr, dcg, idcg=0, 0, 0, 0
    for id in range(len(real_items)):
        item = real_items[id]
        if item in rec_items:
            hit += 1
            idx = np.where(rec_items == item)[0][0] # item's rank in predicted_items
            mrr += 1.0/(idx+1)
            dcg += 1.0/(np.log2(idx+2))
        idcg += 1.0/(np.log2(id+2))
    return hit/min(K, len(real_items)), mrr, dcg/idcg

# Calculate RMSE, MAE
def cal_rmse_mae(y, y_pre):
    abs_sum, square_sum = 0, 0
    for id in range(len(y)):
        res = y[id] - y_pre[id]
        abs_sum += abs(res)
        square_sum += res ** 2
    rmse, mae = math.sqrt(square_sum/len(y)), abs_sum/len(y)
    return rmse, mae
