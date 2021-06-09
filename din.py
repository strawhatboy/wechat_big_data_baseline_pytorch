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
    # device = 'cuda:{}'.format(args['gpu'])
    device = "cpu"
    test = pd.read_csv(ORIGINAL_DATA_PATH / "test_din.csv")

    for action in ACTION_LIST:
        logger.info("now training on {}".format(action))

        data_path = DIN_DATA_PATH / action

        train = pd.read_csv(data_path / "train.csv")
        test[action] = 0

        logger.info('train shape: {}, test shape: {}'.format(train.shape, test.shape))
        logger.info('train columns: {}'.format(train.columns))
        logger.info('test columns: {}'.format(test.columns))

        all_data = pd.concat((train, test)).reset_index(drop=True)

        logger.info('concatenated all data shape: {}'.format(all_data.shape))
        for feat in SPARSE_FEAT:
            lbe = LabelEncoder()
            all_data[feat] = lbe.fit_transform(all_data[feat])
            logger.info('feature {} max: {}, nunique: {}'.format(feat, all_data[feat].max(), all_data[feat].nunique()))

        feature_columns = [
            SparseFeat("userid", all_data["userid"].nunique(), EMBEDDING_DIM),
            SparseFeat("feedid", all_data["feedid"].nunique(), EMBEDDING_DIM),
            SparseFeat("authorid", all_data["authorid"].nunique(), EMBEDDING_DIM),
            SparseFeat(
                "bgm_singer_id", all_data["bgm_singer_id"].nunique(), EMBEDDING_DIM
            ),
            SparseFeat("bgm_song_id", all_data["bgm_song_id"].nunique(), EMBEDDING_DIM),
            DenseFeat("videoplayseconds", 1),
        ]

        #TODO: hist_authorid should be label encodered...
        feature_columns += [
            VarLenSparseFeat(
                SparseFeat(
                    "hist_authorid", all_data["authorid"].nunique(), EMBEDDING_DIM
                ),
                MAX_HISTORY_SIZE,
                length_name="seq_length",
            ),
            VarLenSparseFeat(
                SparseFeat(
                    "hist_bgm_singer_id",
                    all_data["bgm_singer_id"].nunique(),
                    EMBEDDING_DIM,
                ),
                MAX_HISTORY_SIZE,
                length_name="seq_length",
            ),
            VarLenSparseFeat(
                SparseFeat(
                    "hist_bgm_song_id", all_data["bgm_song_id"].nunique(), EMBEDDING_DIM
                ),
                MAX_HISTORY_SIZE,
                length_name="seq_length",
            ),
        ]

        train, test = (
            all_data.iloc[: train.shape[0]].reset_index(drop=True),
            all_data.iloc[train.shape[0] :].reset_index(drop=True),
        )
        uids = train["userid"].to_numpy()
        fids = train["feedid"].to_numpy()
        aids = train["authorid"].to_numpy()
        bgm_singer_ids = train["bgm_singer_id"].to_numpy()
        bgm_song_ids = train["bgm_song_id"].to_numpy()
        videoplayseconds = train["videoplayseconds"].to_numpy()

        with open(data_path / "hist_train.pkl", "rb") as f:
            hist_train = pickle.load(f)

        feature_dict = {
            "userid": uids,
            "feedid": fids,
            "authorid": aids,
            "bgm_singer_id": bgm_singer_ids,
            "bgm_song_id": bgm_song_ids,
            "videoplayseconds": videoplayseconds,
            "hist_authorid": hist_train["authorids"],
            "hist_bgm_singer_id": hist_train["bgm_singer_ids"],
            "hist_bgm_song_id": hist_train["bgm_song_ids"],
            "seq_length": hist_train["behavior_length"],
        }

        for k, v in feature_dict.items():
            logger.info("feature {} shape: {}".format(k, v.shape))

        x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
        y = train[action].to_numpy()

        behavior_feature_list = ["authorid", "bgm_singer_id", "bgm_song_id"]

        model = DIN(
            feature_columns,
            behavior_feature_list,
            device=device,
            att_weight_normalization=True,
        )
        model.compile(
            "adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"]
        )

        early_stop = EarlyStopping(monitor="val_auc")
        history = model.fit(
            x,
            y,
            batch_size=256,
            epochs=10,
            verbose=1,
            validation_split=0.2,
            callbacks=[early_stop],
        )

    # save submit
    # submit.to_csv("./submit/submit_base_din_{}.csv".format(args['runid']), index=False)
