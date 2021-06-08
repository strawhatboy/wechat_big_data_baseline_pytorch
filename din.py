

# -*- coding: utf-8 -*-


# sparse: userid, feedid, bgm_song_id, bgm_singer_id, authorid, 
# dense: play, stay, videoplayseconds(video_length)
# varlen_sparse: hist_feedid, hist_bgm_song_id, hist_bgm_singer_id, hist_authorid
# dense: hist_videoplayseconds

# behavior_fea: feedid, bgm_song_id, bgm_singer_id, authorid, videoplayseconds

# behavior_length: history length (count), in hist_xxx need padding




import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import random
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from prepare_data import process_embed
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.basemodel import *
import gc

# 存储数据的根目录
ROOT_PATH = "../data"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_a.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 4, "forward": 8, "comment": 10, "follow": 10,
                      "favorite": 10}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runid', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # parser.add_argument('--')
    return parser.parse_args()

if __name__ == "__main__":
    args = vars(parse_args())
    random.seed(datetime.now())
    feed_embed = pd.read_csv(ROOT_PATH + '/feed_embeddings_new.csv')
    submit = pd.read_csv(ROOT_PATH + '/test_data.csv')[['userid', 'feedid']]
    for action in ACTION_LIST:
        print('now in phase: {}'.format(action))
        USE_FEAT = ['userid', 'feedid', action] + FEA_FEED_LIST[1:]
        train = pd.read_csv(ROOT_PATH + f'/train_data_for_{action}.csv')[USE_FEAT]
        train = train.sample(frac=1, random_state=42).reset_index(drop=True)
        
    
        print("posi prop:")
        print(sum((train[action]==1)*1)/train.shape[0])
        test = pd.read_csv(ROOT_PATH + '/test_data.csv')[[i for i in USE_FEAT if i != action]]
        
        # USE_FEAT = USE_FEAT
        target = [action]
        test[target[0]] = 0
        test = test[USE_FEAT]
        data = pd.concat((train, test)).reset_index(drop=True)
        dense_features = ['videoplayseconds'] + [f"embed{i}" for i in range(512)] + [f"user_embed{i}" for i in range(512)]
        sparse_features = [i for i in USE_FEAT if i not in dense_features and i not in target]

        dense_featuresx = ['videoplayseconds']
        data[sparse_features] = data[sparse_features].fillna(0)
        data[dense_featuresx] = data[dense_featuresx].fillna(0)

        print('features ok?')

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))

        # only transform videoplayseconds
        data[dense_featuresx] = mms.fit_transform(data[dense_featuresx])

        # 2.count #unique features for each sparse field,and record dense feature field name
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                  for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                  for feat in dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)

        print('train shape: {}, test shape: {}, data shape: {}'.format(train.shape, test.shape, data.shape))

        # data = pd.merge(data, feed_embed, on='feedid', how='left')
        # # data = process_embed(data)
        # data = data[USE_FEAT + [f"embed{i}" for i in range(512)]]
        # data[dense_features] = data[dense_features].fillna(0)
        # data = data.drop_duplicates(['feedid'], keep='last')
        # print('train shape: {}, test shape: {}, data shape: {}'.format(train.shape, test.shape, data.shape))


    

        # print(feature_names)

        print('generating input data for model')
        # 3.generate input data for model
        train, test = data.iloc[:train.shape[0]].reset_index(drop=True), data.iloc[train.shape[0]:].reset_index(drop=True)
        
        user_embed = pd.read_csv(ROOT_PATH + '/user_embeddings_{}.csv'.format(action))
        train = pd.merge(train, feed_embed, on='feedid', how='left')
        train = pd.merge(train, user_embed, on='userid', how='left')
        # data = process_embed(data)
        train = train[USE_FEAT + [f"embed{i}" for i in range(512)] + [f"user_embed{i}" for i in range(512)]]
        train[dense_features] = train[dense_features].fillna(0)
        test = pd.merge(test, feed_embed, on='feedid', how='left')
        test = pd.merge(test, user_embed, on='userid', how='left')
        # data = process_embed(data)
        test = test[USE_FEAT + [f"embed{i}" for i in range(512)] + [f"user_embed{i}" for i in range(512)]]
        test[dense_features] = test[dense_features].fillna(0)
        # print('train shape: {}, test shape: {}, data shape: {}'.format(train.shape, test.shape, data.shape))

        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}


        del data
        del user_embed
        print('train_data: ')
        print(train.head())
        print('test_data: ')
        print(test.head())

                
        gc.collect()
        print('train shape: {}, test shape: {}'.format(train.shape, test.shape))

        
        # 4.Define Model,train,predict and evaluate
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:{}'.format(args['gpu'])

        model = MyDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                       task='binary', dnn_dropout=args['dropout'],
                       l2_reg_embedding=1e-1, device=device, seed=int(random.random() * 10240))

        model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])

        best_model_path = './out/saved_model_{}_{}.dict'.format(action, args['runid'])
        history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=1,
                            validation_split=0.2, model_path=best_model_path)
        # load best model
        model.load_state_dict(torch.load(best_model_path))
        pred_ans = model.predict(test_model_input, 128)
        submit[action] = pred_ans
        torch.cuda.empty_cache()
        
        del train
        del test
        gc.collect()
    # 保存提交文件
    submit.to_csv("./submit/submit_base_deepfm_{}.csv".format(args['runid']), index=False)
