# -*- coding: utf-8 -*-


# sparse: userid, feedid, bgm_song_id, bgm_singer_id, authorid,
# dense: play, stay, videoplayseconds(video_length)
# varlen_sparse: hist_feedid, hist_bgm_song_id, hist_bgm_singer_id, hist_authorid
# dense: hist_videoplayseconds

# behavior_fea: feedid, bgm_song_id, bgm_singer_id, authorid, videoplayseconds

# behavior_length: history length (count), in hist_xxx need padding

import pretty_errors
import argparse
from datetime import datetime
from operator import length_hint
import pickle
import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
from deepctr_torch.inputs import (
    SparseFeat,
    DenseFeat,
    VarLenSparseFeat,
    get_feature_names,
)
from deepctr_torch.models.din import *
from deepctr_torch.callbacks import EarlyStopping
import gc
from din_prepare import (
    ACTION_LIST,
    DIN_DATA_PATH,
    MAX_HISTORY_SIZE,
    ORIGINAL_DATA_PATH,
    init_logging,
)
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger("din")

EMBEDDING_DIM = 10
SPARSE_FEAT = ["userid", "feedid", "authorid", "bgm_singer_id", "bgm_song_id"]
VAC_SIZE = {
    "userid": 20000,
    "feedid": 106444,
    "authorid": 18789,
    "bgm_singer_id": 17500,
    "bgm_song_id": 25159,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    # parser.add_argument('--')
    return parser.parse_args()


if __name__ == "__main__":
    init_logging()
    args = vars(parse_args())
    random.seed(datetime.now())
    device = "cuda:{}".format(args["gpu"])
    # device = "cpu"
    test = pd.read_csv(ORIGINAL_DATA_PATH / "test_din.csv")
    test_a = pd.read_csv(ORIGINAL_DATA_PATH / "test_a.csv")
    submit = test_a[["userid", "feedid"]]

    for action in ACTION_LIST:
        logger.info("------------------------------------")
        logger.info("now training on {}".format(action))

        data_path = DIN_DATA_PATH / action

        train = pd.read_csv(data_path / "train.csv")
        test = test[['userid', 'feedid', 'authorid', 'bgm_singer_id', 'bgm_song_id', 'videoplayseconds']]
        test[action] = 0
        
        logger.info("train shape: {}, test shape: {}".format(train.shape, test.shape))
        logger.info("train columns: {}".format(train.columns))
        logger.info("test columns: {}".format(test.columns))

        all_data = pd.concat((train, test)).reset_index(drop=True)

        logger.info("concatenated all data shape: {}".format(all_data.shape))
        # for feat in SPARSE_FEAT:
        #     lbe = LabelEncoder()
        #     all_data[feat] = lbe.fit_transform(all_data[feat])
        #     logger.info('feature {} max: {}, nunique: {}'.format(feat, all_data[feat].max(), all_data[feat].nunique()))

        feature_columns = [
            SparseFeat("userid", VAC_SIZE["userid"], EMBEDDING_DIM),
            SparseFeat("feedid", VAC_SIZE["feedid"], EMBEDDING_DIM),
            SparseFeat("authorid", VAC_SIZE["authorid"], EMBEDDING_DIM),
            SparseFeat(
                "bgm_singer_id", VAC_SIZE["bgm_singer_id"], EMBEDDING_DIM
            ),
            SparseFeat(
                "bgm_song_id", VAC_SIZE["bgm_song_id"], EMBEDDING_DIM
            ),
            DenseFeat("videoplayseconds", 1),
        ]

        # TODO: hist_authorid should be label encodered...
        feature_columns += [
            VarLenSparseFeat(
                SparseFeat(
                    "hist_authorid", VAC_SIZE["authorid"], EMBEDDING_DIM
                ),
                MAX_HISTORY_SIZE,
                length_name="seq_length",
            ),
            VarLenSparseFeat(
                SparseFeat(
                    "hist_bgm_singer_id",
                    VAC_SIZE["bgm_singer_id"],
                    EMBEDDING_DIM,
                ),
                MAX_HISTORY_SIZE,
                length_name="seq_length",
            ),
            VarLenSparseFeat(
                SparseFeat(
                    "hist_bgm_song_id",
                    VAC_SIZE["bgm_song_id"],
                    EMBEDDING_DIM,
                ),
                MAX_HISTORY_SIZE,
                length_name="seq_length",
            ),
        ]

        train, test = (
            all_data.iloc[: train.shape[0]].reset_index(drop=True),
            all_data.iloc[train.shape[0] :].reset_index(drop=True),
        )
        uids = train["userid"].to_numpy().astype(int)
        fids = train["feedid"].to_numpy().astype(int)
        aids = train["authorid"].to_numpy().astype(int)
        bgm_singer_ids = train["bgm_singer_id"].to_numpy().astype(int)
        bgm_song_ids = train["bgm_song_id"].to_numpy().astype(int)
        videoplayseconds = np.log(train["videoplayseconds"] + 1).to_numpy()

        feature_dict = {
            "userid": uids,
            "feedid": fids,
            "authorid": aids,
            "bgm_singer_id": bgm_singer_ids,
            "bgm_song_id": bgm_song_ids,
            "videoplayseconds": videoplayseconds,
        }

        for k, v in feature_dict.items():
            logger.debug("feature {} shape: {}".format(k, v.shape))

        feature_names = get_feature_names(feature_columns)
        x = {name: feature_dict[name] for name in feature_names}
        y = train[action].to_numpy()
        logger.info('shape of y: {}'.format(y.shape))

        behavior_feature_list = ["authorid", "bgm_singer_id", "bgm_song_id"]

        model = DIN(
            feature_columns,
            behavior_feature_list,
            device=device,
            att_weight_normalization=True,
            l2_reg_embedding=1e-1,
            task='binary',
            dnn_dropout=args['dropout']
        )
        model.compile(
            "adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"]
        )

        early_stop = EarlyStopping(monitor="val_auc", verbose=1, patience=1)
        history = model.fit(
            x,
            y,
            batch_size=512,
            epochs=10,
            verbose=1,
            validation_split=0.2,
            callbacks=[early_stop],
            shuffle=True,
        )

        torch.save(model.state_dict(), "./out/deepfm_{}.dict")

        # prepare test data
        logger.info('preparing test data')
        # with open(data_path / "hist_test.pkl", "rb") as f:
        #     hist_test = pickle.load(f)

        feature_dict = {
            "userid": test["userid"].to_numpy().astype(int),
            "feedid": test["feedid"].to_numpy().astype(int),
            "authorid": test["authorid"].to_numpy().astype(int),
            "bgm_singer_id": test["bgm_singer_id"].to_numpy().astype(int),
            "bgm_song_id": test["bgm_song_id"].to_numpy().astype(int),
            "videoplayseconds": np.log(test["videoplayseconds"] + 1).to_numpy(),
            # "hist_authorid": hist_test["authorids"],
            # "hist_bgm_singer_id": hist_test["bgm_singer_ids"],
            # "hist_bgm_song_id": hist_test["bgm_song_ids"],
            # "seq_length": hist_test["behavior_length"],
        }

        x_test = {name: feature_dict[name] for name in feature_names}

        logger.info("predicting...")
        y_pred = model.predict(x_test, batch_size=512)
        submit[action] = y_pred
        gc.collect()

    submit.to_csv("./submit/submit_din_{}.csv".format(args["runid"]), index=False)

    # save submit
    # submit.to_csv("./submit/submit_base_din_{}.csv".format(args['runid']), index=False)
