from typing import Tuple
import pandas as pd
from pathlib import Path
import json
import argparse

import pickle
import logging
import lightgbm as lgb
from evaluation import uAUC

logger = logging.getLogger('lgb_new')

from pandarallel import pandarallel

pandarallel.initialize()

ORIGINAL_ROOT = Path('../data/wechat_algo_data1')
TG_FEATURE_ROOT = Path('../tg_feature')
TG_FEATURE_CONFIG = TG_FEATURE_ROOT / 'feature_config.json'
TG_FEATURE_USER = TG_FEATURE_ROOT / 'userid_breakdown_scale.csv'
TG_FEATURE_FEED = TG_FEATURE_ROOT / 'feedid_breakdown_scale.csv'
LGB_FEATURE_ROOT = Path('../lgb_feature')
LGB_OUT_OFFLINE_ROOT = Path('../lgb_out_offline')
LGB_OUT_ONLINE_ROOT = Path('../lgb_out_online')
LGB_TRAIN_OFFLINE = LGB_FEATURE_ROOT / 'train_offline.csv'
LGB_TEST_OFFLINE = LGB_FEATURE_ROOT / 'test_offline.csv'
LGB_TRAIN_ONLINE = LGB_FEATURE_ROOT / 'train_online.csv'
LGB_TEST_ONLINE = LGB_FEATURE_ROOT / 'test_online.csv'
LGB_TRAIN_OFFLINE_PKL = LGB_FEATURE_ROOT / 'train_offline.pkl'
LGB_TEST_OFFLINE_PKL = LGB_FEATURE_ROOT / 'test_offline.pkl'
LGB_TRAIN_ONLINE_PKL = LGB_FEATURE_ROOT / 'train_online.pkl'
LGB_TEST_ONLINE_PKL = LGB_FEATURE_ROOT / 'test_online.pkl'
ACTION_LIST_ALL = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
ACTION_LIST_PRE = ["read_comment", "like", "click_avatar",  "forward"]
FEA_FEED = 'feedid,authorid,videoplayseconds,bgm_song_id,bgm_singer_id'.split(',')
ACTION_WEIGHT = { 'read_comment': 4, 'like': 3, 'click_avatar': 2, 'forward': 1 }


LGB_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': -1,
    'num_leaves': 31,
    'learning_rate': 0.25,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': 42,
    'n_jobs': -1,
    'force_col_wise': True,
    # 'two_round': True
}

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )

def get_original_data(with_feedinfo: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info('getting original data')
    user_action = pd.read_csv(ORIGINAL_ROOT / 'user_action.csv')
    test_a = pd.read_csv(ORIGINAL_ROOT / 'test_a.csv')

    with open(TG_FEATURE_ROOT / 'userid_map.pkl', 'rb') as f:
        userid_map = pickle.load(f)

    with open(TG_FEATURE_ROOT / 'feedid_map.pkl', 'rb') as f:
        feedid_map = pickle.load(f)

    # merge feed info
    feed_info = pd.read_csv(ORIGINAL_ROOT / 'feed_info.csv')[FEA_FEED]
    if with_feedinfo:
        user_action = user_action.merge(feed_info, how='left', on='feedid')
        test_a = test_a.merge(feed_info, how='left', on='feedid')

    user_action['userid'] = user_action['userid'].parallel_apply(lambda x: userid_map[x])
    user_action['feedid'] = user_action['feedid'].parallel_apply(lambda x: feedid_map[x])
    test_a['userid'] = test_a['userid'].parallel_apply(lambda x: userid_map[x])
    test_a['feedid'] = test_a['feedid'].parallel_apply(lambda x: feedid_map[x])

    return user_action, test_a

def prepare_tg_dense_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(TG_FEATURE_CONFIG, 'r') as f:
        feature_config = json.load(f)

    logger.info('preparing tg dense features')
    
    user_dense_feature_names = { **(feature_config['user_breakdown']), **(feature_config['duration_breakdown']) }
    user_dense_feature_names = user_dense_feature_names.keys()
    user_dense_feature_names = [x.replace('{}', '.*') for x in user_dense_feature_names] + ['userid', 'date_']

    feed_dense_feature_names = { **(feature_config['feed_breakdown']), **(feature_config['device_breakdown']) }
    feed_dense_feature_names = feed_dense_feature_names.keys()
    feed_dense_feature_names = [x.replace('{}', '.*') for x in feed_dense_feature_names] + ['feedid', 'date_']

    user_dense_features = pd.read_csv(TG_FEATURE_USER)
    user_dense_features = user_dense_features.filter(user_dense_features)
    
    feed_dense_features = pd.read_csv(TG_FEATURE_FEED)
    feed_dense_features = feed_dense_features.filter(feed_dense_features)   

    return user_dense_features, feed_dense_features

def prepare_train_test_data(user_action: pd.DataFrame, test_a: pd.DataFrame, user_dense_features: pd.DataFrame, feed_dense_features: pd.DataFrame):
    train_offline = user_action[(user_action['date_'] < 14) & (user_action['date_'] >= 7)]
    test_offline = user_action[user_action['date_'] == 14]
    # user_dense_features_14 = user_dense_features[user_dense_features['date_'] == 14]
    # feed_dense_features_14 = feed_dense_features[feed_dense_features['date_'] == 14]
    train_offline = train_offline.merge(user_dense_features, how='left', on=['userid', 'date_'])
    train_offline = train_offline.merge(feed_dense_features, how='left', on=['feedid', 'date_'])
    test_offline = test_offline.merge(user_dense_features, how='left', on=['userid', 'date_'])
    test_offline = test_offline.merge(feed_dense_features, how='left', on=['feedid', 'date_'])

    logger.info('writing train&test offline data, train shape: {}, test shape: {}'.format(train_offline.shape, test_offline.shape))    
    train_offline.to_pickle(LGB_TRAIN_OFFLINE_PKL)
    test_offline.to_pickle(LGB_TEST_OFFLINE_PKL)


    user_dense_features_15 = user_dense_features[user_dense_features['date_'] == 15]
    feed_dense_features_15 = feed_dense_features[feed_dense_features['date_'] == 15]
    # last week
    train_online = user_action[(user_action['date_'] <= 14) & (user_action['date_'] > 7)]  # all data
    train_online = train_online.merge(user_dense_features, how='left', on=['userid', 'date_'])
    train_online = train_online.merge(feed_dense_features, how='left', on=['feedid', 'date_'])
    
    test_online = test_a.merge(user_dense_features_15, how='left', on=['userid'])
    test_online = test_online.merge(feed_dense_features_15, how='left', on=['feedid'])

    logger.info('writing train&test online data, train shape: {}, test shape: {}'.format(train_online.shape, test_online.shape))
    train_online.to_pickle(LGB_TRAIN_ONLINE_PKL)
    test_online.to_pickle(LGB_TEST_ONLINE_PKL)


def offline_train(phases: str):
    # load train data
    # label = 'read_comment'
    # for label in ACTION_LIST_PRE:
    phases = phases.split(',')
    logger.info('loading offline train&test data')
    train_on = pd.read_pickle(LGB_TRAIN_OFFLINE_PKL)
    test_on = pd.read_pickle(LGB_TEST_OFFLINE_PKL)
    aucs = {}
    boost_round = {}
    for label in phases:
        logger.info('-------------------------')
        logger.info('PHASE: {}'.format(label))
        logger.info('-------------------------')
        # only read_comment for now
        ACTION_LIST = ACTION_LIST_ALL.copy()
        ACTION_LIST.remove(label)
        train_on_phase = train_on.drop(['date_', 'play', 'stay'] + ACTION_LIST, axis=1)
        y_train_on = train_on_phase[label]
        x_train_on = train_on_phase.drop(label, axis=1)

        test_on_phase = test_on.drop(['date_', 'play', 'stay'] + ACTION_LIST, axis=1)
        y_test_on = test_on_phase[label]
        x_test_on = test_on_phase.drop(label, axis=1)

        logger.info('getting train & val dataset')

        ul = x_test_on['userid'].tolist()
        dtrain = lgb.Dataset(x_train_on, label=y_train_on)
        dval = lgb.Dataset(x_test_on, label=y_test_on)

        logger.info('going to train')
        lgb_model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dval],
            early_stopping_rounds=50,
            verbose_eval=50,
        )


        pred = lgb_model.predict(x_test_on, num_iteration=lgb_model.best_iteration)
        logger.info('best iteration: {}, best score: {}'.format(lgb_model.best_iteration, lgb_model.best_score))
        v = uAUC(y_test_on.tolist(), pred.tolist(), ul)
        logger.info('uAUC: {}'.format(v))
        aucs[label] = v
        boost_round[label] = lgb_model.best_iteration

        logger.debug('features: {}'.format(x_train_on.columns.tolist()))
        importance_split = lgb_model.feature_importance('split')
        logger.debug('feature importance split: {}'.format(importance_split))
        importance_gain = lgb_model.feature_importance('gain')
        logger.debug('feature importance gain: {}'.format(importance_gain))

        logger.info('save them to file')
        str = 'features: {}\nfeature importance (split): {}\nfeature importance (gain): {}\n\n'.format(
            x_train_on.columns.tolist(), importance_split, importance_gain
        )

        xobj = { x[0]: {'split': x[1], 'gain': x[2] } for x in list(zip(x_train_on.columns.tolist(), importance_split, importance_gain)) }

        with open(LGB_OUT_OFFLINE_ROOT / './feature_importance_{}.txt'.format(label), 'w', encoding='utf8') as f:
            f.write(str)
            f.flush()

        with open(LGB_OUT_OFFLINE_ROOT / 'feature_importance_{}.json'.format(label), 'w', encoding='utf8') as f:
            f.write("{}".format(xobj).replace("'", '"'))
            f.flush()
            
        
        logger.info('saving model: {}'.format(label))
        lgb_model.save_model(LGB_OUT_OFFLINE_ROOT / './lgb_{}.lgb_model'.format(label))
        lgb.plot_importance(lgb_model, figsize=(30, 180)).figure.savefig(LGB_OUT_OFFLINE_ROOT / './lgb_importance_{}.png'.format(label))

    # calculate uAUC
    numerator = 0
    denominator = 0
    for label in phases:
        numerator += ACTION_WEIGHT[label] * aucs[label]
        denominator += aucs[label]
    
    logger.info('offline uAUC calculated: {}'.format(numerator / denominator))

    # save boost_round
    with open(LGB_OUT_OFFLINE_ROOT / './boost_round.json', 'w', encoding='utf8') as f:
        json.dump(boost_round, f)

    pass

def online_train(runid: str):
    logger.info('loading online train&test data')
    train_on = pd.read_pickle(LGB_TRAIN_ONLINE_PKL)
    test_on = pd.read_pickle(LGB_TEST_ONLINE_PKL)

    submit = pd.read_csv(ORIGINAL_ROOT / 'test_a.csv')
    for label in ACTION_LIST_PRE:
        logger.info('-------------------------')
        logger.info('PHASE: {}'.format(label))
        logger.info('-------------------------')
        # only read_comment for now
        ACTION_LIST = ACTION_LIST_ALL.copy()
        ACTION_LIST.remove(label)
        train_on_phase = train_on.drop(['date_', 'play', 'stay'] + ACTION_LIST, axis=1)
        y_train_on = train_on_phase[label]
        x_train_on = train_on_phase.drop(label, axis=1)

        # no label in test
        # test_on_phase = test_on.drop(['date_', 'play', 'stay'] + ACTION_LIST, axis=1)
        # y_test_on = test_on_phase[label]
        # x_test_on = test_on_phase.drop(label, axis=1)
        x_test_on = test_on

        logger.info('getting train & val dataset')

        ul = x_test_on['userid'].tolist()
        dtrain = lgb.Dataset(x_train_on, label=y_train_on)
        # dval = dtrain

        logger.info('loading best boost round when offline train')
        with open(LGB_OUT_OFFLINE_ROOT / 'boost_round.json', 'r', encoding='utf8') as f:
            boost_round = json.load(f)

        logger.info('going to train')
        lgb_model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=boost_round[label],
            valid_sets=[dtrain],
            early_stopping_rounds=50,
            verbose_eval=50,
        )


        pred = lgb_model.predict(x_test_on, num_iteration=lgb_model.best_iteration)
        submit[label] = pred
        logger.info('best iteration: {}, best score: {}'.format(lgb_model.best_iteration, lgb_model.best_score))
        # there's no label for online predict
        # v = uAUC(y_test_on.tolist(), pred.tolist(), ul)
        # logger.info('uAUC: {}'.format(v))

        logger.debug('features: {}'.format(x_train_on.columns.tolist()))
        importance_split = lgb_model.feature_importance('split')
        logger.debug('feature importance split: {}'.format(importance_split))
        importance_gain = lgb_model.feature_importance('gain')
        logger.debug('feature importance gain: {}'.format(importance_gain))

        logger.info('save them to file')
        str = 'features: {}\nfeature importance (split): {}\nfeature importance (gain): {}\n\n'.format(
            x_train_on.columns.tolist(), importance_split, importance_gain
        )

        xobj = { x[0]: {'split': x[1], 'gain': x[2] } for x in list(zip(x_train_on.columns.tolist(), importance_split, importance_gain)) }

        with open(LGB_OUT_ONLINE_ROOT / './feature_importance_{}_{}.txt'.format(label, runid), 'w', encoding='utf8') as f:
            f.write(str)
            f.flush()

        with open(LGB_OUT_ONLINE_ROOT / 'feature_importance_{}_{}.json'.format(label, runid), 'w', encoding='utf8') as f:
            f.write("{}".format(xobj).replace("'", '"'))
            f.flush()
            
        
        logger.info('saving model: {}'.format(label))
        lgb_model.save_model(LGB_OUT_ONLINE_ROOT / './lgb_{}_{}.lgb_model'.format(label, runid))
        lgb.plot_importance(lgb_model, figsize=(30, 180)).figure.savefig(LGB_OUT_ONLINE_ROOT / './lgb_importance_{}_{}.png'.format(label, runid))

    # save submit
    submit.to_csv(LGB_OUT_ONLINE_ROOT / './submit_lgb_{}.csv'.format(runid))
    pass

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, choices=['process', 'online_train', 'online_train'])
    p.add_argument('--phase', type=str)
    p.add_argument('--with_feedinfo', action='store_true')
    p.add_argument('--runid', type=str)
    return p.parse_args()

def main():
    init_logging()
    args = vars(parse_args())
    if args['mode'] == 'process':
        user_action, test_a = get_original_data(args['with_feedinfo'])
        user_dense_features, feed_dense_features = prepare_tg_dense_features()
        prepare_train_test_data(user_action, test_a, user_dense_features, feed_dense_features)
    elif args['mode'] == 'offline_train':
        offline_train(args['phase'])
    elif args['mode'] == 'online_train':
        online_train(args['runid'])
    pass

if __name__ == '__main__':
    main()

