# -*- coding: utf-8 -*-
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

class MyBaseModel(BaseModel):

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, early_stop=50, model_path='saved.dict'):

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        max_auc = -999999
        early_stop_count = 0
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for i, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()


                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                try:
                                    temp = metric_fun(
                                        y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64"))
                                except Exception:
                                    temp = 0
                                finally:
                                    train_result[name].append(temp)
                            
                            # if i % 1000 == 0 and do_validation:
                                
                            #     eval_result = self.evaluate(val_x, val_y, batch_size)
                            #     for name, result in eval_result.items():
                            #         epoch_logs["val_" + name] = result
                                
                            #     print('validating in the middle, auc: {}, max_auc: {}'.format(epoch_logs['val_auc'], max_auc))
                            #     if epoch_logs['val_auc'] >= max_auc:
                            #         torch.save(self.__dict__, model_path)
                            #         max_auc = epoch_logs['val_auc']
                            #         early_stop_count = 0
                            #     else:
                            #         early_stop_count += 1
                            #         if early_stop_count >= early_stop:
                            #             print('early stopped.')
                            #             print('best auc: {}'.format(max_auc))
                            #             self.stop_training = True
                
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
                if epoch_logs['val_auc'] >= max_auc:
                    torch.save(self.state_dict(), model_path)
                    max_auc = epoch_logs['val_auc']
                    early_stop_count = 0
                else:
                    # early_stop_count += 1
                    # if early_stop_count >= early_stop:
                    print('early stopped.')
                    print('best auc: {}'.format(max_auc))
                    self.stop_training = True
                    
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            try:
                temp = metric_fun(y, pred_ans)
            except Exception:
                temp = 0
            finally:
                eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

class MyDeepFM(MyBaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0.2,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(MyDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred

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
    tg_features = 'feed_week_impression,feed_week_engage,feed_week_read_comment,feed_week_like,feed_week_click_avatar,feed_week_forward,feed_week_comment,feed_week_follow,feed_week_favorite,feed_week_videocomplete,feed_week_device1_impression,feed_week_device1_engage,feed_week_device1_read_comment,feed_week_device1_like,feed_week_device1_click_avatar,feed_week_device1_forward,feed_week_device1_comment,feed_week_device1_follow,feed_week_device1_favorite,feed_week_device1_videocomplete,feed_week_device2_impression,feed_week_device2_engage,feed_week_device2_read_comment,feed_week_device2_like,feed_week_device2_click_avatar,feed_week_device2_forward,feed_week_device2_comment,feed_week_device2_follow,feed_week_device2_favorite,feed_week_device2_videocomplete,feed_week_read_comment_ratio,feed_week_like_ratio,feed_week_click_avatar_ratio,feed_week_forward_ratio,feed_week_comment_ratio,feed_week_follow_ratio,feed_week_favorite_ratio,feed_week_engage_ratio,feed_week_videocomplete_ratio,feed_week_device1_read_comment_ratio,feed_week_device1_like_ratio,feed_week_device1_click_avatar_ratio,feed_week_device1_forward_ratio,feed_week_device1_comment_ratio,feed_week_device1_follow_ratio,feed_week_device1_favorite_ratio,feed_week_device1_engage_ratio,feed_week_device1_videocomplete_ratio,feed_week_device2_read_comment_ratio,feed_week_device2_like_ratio,feed_week_device2_click_avatar_ratio,feed_week_device2_forward_ratio,feed_week_device2_comment_ratio,feed_week_device2_follow_ratio,feed_week_device2_favorite_ratio,feed_week_device2_engage_ratio,feed_week_device2_videocomplete_ratio,author_week_impression,author_week_engage,author_week_read_comment,author_week_like,author_week_click_avatar,author_week_forward,author_week_comment,author_week_follow,author_week_favorite,author_week_videocomplete,author_week_device1_impression,author_week_device1_engage,author_week_device1_read_comment,author_week_device1_like,author_week_device1_click_avatar,author_week_device1_forward,author_week_device1_comment,author_week_device1_follow,author_week_device1_favorite,author_week_device1_videocomplete,author_week_device2_impression,author_week_device2_engage,author_week_device2_read_comment,author_week_device2_like,author_week_device2_click_avatar,author_week_device2_forward,author_week_device2_comment,author_week_device2_follow,author_week_device2_favorite,author_week_device2_videocomplete,author_week_read_comment_ratio,author_week_like_ratio,author_week_click_avatar_ratio,author_week_forward_ratio,author_week_comment_ratio,author_week_follow_ratio,author_week_favorite_ratio,author_week_engage_ratio,author_week_videocomplete_ratio,author_week_device1_read_comment_ratio,author_week_device1_like_ratio,author_week_device1_click_avatar_ratio,author_week_device1_forward_ratio,author_week_device1_comment_ratio,author_week_device1_follow_ratio,author_week_device1_favorite_ratio,author_week_device1_engage_ratio,author_week_device1_videocomplete_ratio,author_week_device2_read_comment_ratio,author_week_device2_like_ratio,author_week_device2_click_avatar_ratio,author_week_device2_forward_ratio,author_week_device2_comment_ratio,author_week_device2_follow_ratio,author_week_device2_favorite_ratio,author_week_device2_engage_ratio,author_week_device2_videocomplete_ratio,song_week_impression,song_week_engage,song_week_read_comment,song_week_like,song_week_click_avatar,song_week_forward,song_week_comment,song_week_follow,song_week_favorite,song_week_videocomplete,song_week_device1_impression,song_week_device1_engage,song_week_device1_read_comment,song_week_device1_like,song_week_device1_click_avatar,song_week_device1_forward,song_week_device1_comment,song_week_device1_follow,song_week_device1_favorite,song_week_device1_videocomplete,song_week_device2_impression,song_week_device2_engage,song_week_device2_read_comment,song_week_device2_like,song_week_device2_click_avatar,song_week_device2_forward,song_week_device2_comment,song_week_device2_follow,song_week_device2_favorite,song_week_device2_videocomplete,song_week_read_comment_ratio,song_week_like_ratio,song_week_click_avatar_ratio,song_week_forward_ratio,song_week_comment_ratio,song_week_follow_ratio,song_week_favorite_ratio,song_week_engage_ratio,song_week_videocomplete_ratio,song_week_device1_read_comment_ratio,song_week_device1_like_ratio,song_week_device1_click_avatar_ratio,song_week_device1_forward_ratio,song_week_device1_comment_ratio,song_week_device1_follow_ratio,song_week_device1_favorite_ratio,song_week_device1_engage_ratio,song_week_device1_videocomplete_ratio,song_week_device2_read_comment_ratio,song_week_device2_like_ratio,song_week_device2_click_avatar_ratio,song_week_device2_forward_ratio,song_week_device2_comment_ratio,song_week_device2_follow_ratio,song_week_device2_favorite_ratio,song_week_device2_engage_ratio,song_week_device2_videocomplete_ratio'.split(',')
    feed_tg_feature = pd.read_csv('/home/wzh/Extra/workspace/zsx/WeChat_Big_Data_Challenge/WX_Changllenge_2021/data/feature/feedid_breakdown.csv')
    # feed_tg_feature = feed_tg_feature.groupby('feedid').mean()
    for action in ACTION_LIST:
        print('now in phase: {}'.format(action))
        USE_FEAT = ['userid', 'feedid', action] + FEA_FEED_LIST[1:]
        train = pd.read_csv(ROOT_PATH + f'/train_data_for_{action}.csv')[USE_FEAT + ['date_']]
        train = train.sample(frac=1, random_state=42).reset_index(drop=True)
        
    
        print("posi prop:")
        print(sum((train[action]==1)*1)/train.shape[0])
        test = pd.read_csv(ROOT_PATH + '/test_data.csv')[[i for i in USE_FEAT if i != action]]
        
        # USE_FEAT = USE_FEAT
        target = [action]
        test[target[0]] = 0
        test = test[USE_FEAT]
        data = pd.concat((train, test)).reset_index(drop=True)
        dense_features = ['videoplayseconds'] + [f"embed{i}" for i in range(512)] # + tg_features # + [f"user_embed{i}" for i in range(512)]
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
        train = pd.merge(train, feed_tg_feature, on=['feedid', 'date_'], how='left')
        # train = pd.merge(train, user_embed, on='userid', how='left')
        # data = process_embed(data)
        train = train[USE_FEAT + [f"embed{i}" for i in range(512)] + tg_features] # + [f"user_embed{i}" for i in range(512)]]
        train[dense_features] = train[dense_features].fillna(0)
        test = pd.merge(test, feed_embed, on='feedid', how='left')
        test = pd.merge(test, feed_tg_feature[feed_tg_feature['date_'] == 15], on=['feedid'], how='left')
        # test = pd.merge(test, user_embed, on='userid', how='left')
        # data = process_embed(data)
        test = test[USE_FEAT + [f"embed{i}" for i in range(512)] + tg_features] # + [f"user_embed{i}" for i in range(512)]]
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
