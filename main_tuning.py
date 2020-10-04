# coding: utf-8

" CleverRec: An Open-source Toolkit for Recommendation System. "

import numpy as np, tensorflow as tf
import configparser as cp
import os, time, importlib
from utils.tools import get_logger
from model.RankingPreprocess import RankingPreprocess
from model.RatingPreprocess import RatingPreprocess

os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == '__main__':
    # Read configures
    conf = cp.ConfigParser()
    # Project configures
    conf.read('./CleverRec.properties', encoding='utf-8')
    configs = dict(conf.items('default'))
    recommender = configs['recommender']
    # Model configures
    conf.read(os.path.join('./conf/', recommender+'.properties'), encoding='utf-8')
    configs.update(dict(conf.items('parameters')))

    # Get logger
    logger = get_logger(configs['log.dir'], recommender)
    logger.info('='*100)
    logger.info('Current model: %s' % recommender)

    # Read and preprocess data
    if configs['model_type'] == 'ranking':
        data = RankingPreprocess(configs, logger)
    else:
        data = RatingPreprocess(configs, logger)

    # Grid Search
    embed_sizes = map(int, configs['embed_size'][1:-1].split(','))
    regs = map(float, configs['reg'][1:-1].split(','))
    neg_ratios = map(int, configs['neg_ratio'][1:-1].split(','))
    for embed_size in embed_sizes:
        for reg in regs:
            for neg_ratio in neg_ratios:
                tmp = {'embed_size': embed_size, 'reg': reg, 'neg_ratio': neg_ratio}
                configs.update(tmp)

                # tf settings
                tf.reset_default_graph()
                if configs['gpu.is_gpu']:
                    os.environ['CUDA_VISIBLE_DEVICES'] = configs['gpu.id']
                    tf_conf = tf.ConfigProto()
                    tf_conf.gpu_options.per_process_gpu_memory_fraction = float(configs['gpu.mem_frac'])
                    sess = tf.Session(config=tf_conf)
                else:
                    sess = tf.Session()
                with sess.as_default():
                    module = 'model.' + configs['model_type'] + '.' + recommender
                    if importlib.util.find_spec(module) is not None:
                        model = importlib.import_module(module)
                    else:
                        raise Exception('Module %s not found.' % module)
                    myclass = getattr(model, recommender)
                    model = myclass(sess, data, configs, logger)
                    # Run model
                    model.run_model()