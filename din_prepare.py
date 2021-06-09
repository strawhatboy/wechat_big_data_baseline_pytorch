# prepare data for DIN model

import argparse
from operator import index
from pathlib import Path
import pathlib
import pickle
from typing import Tuple
import numpy as np
from tqdm import tqdm

import pandas as pd
import logging

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )

logger = logging.getLogger("din_prepare")

MAX_HISTORY_SIZE = 1024
ORIGINAL_DATA_PATH = Path("../data/wechat_algo_data1")
DIN_DATA_PATH = Path("../data/din")

ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
ACTION_SAMPLE_RATE = {
    "read_comment": 5,
    "like": 5,
    "click_avatar": 4,
    "forward": 8,
    "comment": 10,
    "follow": 10,
    "favorite": 10,
}

VAL_RATIO = 0.2


def pad_max_history_size(arr):
    return np.pad(
        arr, (0, MAX_HISTORY_SIZE - len(arr)), "constant", constant_values=(0)
    )


def gen_user_info(df: pd.DataFrame, action: str, save_path: Path):
    user_action_13_x = df[df[action] == 1][["userid", "feedid", action]]
    user_action_13_x = user_action_13_x.drop_duplicates(
        subset=["userid", "feedid", action], keep="last"
    )

    userids = user_action["userid"].unique()
    user_info = {}

    logger.info("getting user history info")
    for uid in tqdm(userids):
        hist_interation = user_action_13_x[user_action_13_x["userid"] == uid]["feedid"]
        hist_feed: pd.DataFrame = feed_info.loc[hist_interation]
        # logger.info('BOOM: {}'.format(hist_feed))
        if hist_feed.empty:
            user_info[uid] = {
                "hist_records_count": 0,
                "bgm_song_ids": np.zeros(MAX_HISTORY_SIZE),
                "bgm_singer_ids": np.zeros(MAX_HISTORY_SIZE),
                "authorids": np.zeros(MAX_HISTORY_SIZE),
                "videoplayseconds": np.zeros(MAX_HISTORY_SIZE),
            }
        else:
            user_info[uid] = {
                "hist_records_count": len(hist_feed.index),
                "bgm_song_ids": pad_max_history_size(
                    hist_feed["bgm_song_id"].fillna(0).to_numpy()
                ),
                "bgm_singer_ids": pad_max_history_size(
                    hist_feed["bgm_singer_id"].fillna(0).to_numpy()
                ),
                "authorids": pad_max_history_size(
                    hist_feed["authorid"].fillna(0).to_numpy()
                ),
                "videoplayseconds": pad_max_history_size(
                    hist_feed["videoplayseconds"].fillna(0).to_numpy()
                ),
            }

    # save user_info
    with open(save_path, "wb") as f:
        pickle.dump(user_info, f)

    logger.info('user info saved to {}.'.format(save_path))

    return user_info

    pass


def gen_train_data(
    df: pd.DataFrame, feed_info: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    user_action_14_x = df[["userid", "feedid", action]]
    user_action_14_x = user_action_14_x.drop_duplicates(
        subset=["userid", "feedid", action], keep="last"
    )
    pos = user_action_14_x[user_action_14_x[action] == 1]
    neg = user_action_14_x[user_action_14_x[action] == 0]
    logger.info("pos ratio: {}".format(len(pos.index) / len(neg.index)))
    neg = neg.sample(
        frac=1.0 / ACTION_SAMPLE_RATE[action], random_state=42, replace=False
    )
    all = pd.concat([pos, neg])
    all = all.merge(feed_info, how="left", on=["feedid"])[
        [
            "userid",
            "feedid",
            action,
            "bgm_song_id",
            "bgm_singer_id",
            "authorid",
            "videoplayseconds",
        ]
    ]
    all = all.sample(frac=1, random_state=42, replace=False).reset_index(drop=True)

    all.to_csv(data_path / "train.csv", index=False)
    logger.info("data file saved.")

    # no need for train/val splitting, DIN support it...

    # split_point = int(len(all.index) * (1 - VAL_RATIO))
    # train = all.iloc[0:split_point]
    # val = all.iloc[split_point:]

    # train.to_csv(data_path / "train.csv", index=False)
    # val.to_csv(data_path / "val.csv", index=False)
    # logger.info("train & val data file saved.")
    return all


def gen_hist_columns(df: pd.DataFrame, user_info: dict, save_path: Path):
    uids = df["userid"]
    bgm_song_ids = []
    bgm_singer_ids = []
    authorids = []
    videoplayseconds = []
    behavior_length = []

    for uid in uids:
        bgm_song_ids.append(user_info[uid]["bgm_song_ids"])
        bgm_singer_ids.append(user_info[uid]["bgm_singer_ids"])
        authorids.append(user_info[uid]["authorids"])
        videoplayseconds.append(user_info[uid]["videoplayseconds"])
        behavior_length.append(user_info[uid]["hist_records_count"])

    res = {
        "bgm_song_ids": np.array(bgm_singer_ids),
        "bgm_singer_ids": np.array(bgm_singer_ids),
        "authorids": np.array(authorids),
        "videoplayseconds": np.array(videoplayseconds),
        "behavior_length": np.array(behavior_length),
    }

    with open(save_path, "wb") as f:
        pickle.dump(res, f)

    logger.info("hist data saved to {}.".format(save_path))

    pass


def parse_args():
    p = argparse.ArgumentParser()

    return p.parse_args()
    # p.add_argument('')


if __name__ == "__main__":
    init_logging()
    DIN_DATA_PATH.mkdir(exist_ok=True)
    # args = vars(parse_args())
    feed_info = pd.read_csv(ORIGINAL_DATA_PATH / "feed_info.csv", index_col="feedid")
    user_action = pd.read_csv(ORIGINAL_DATA_PATH / "user_action.csv")
    test_a = pd.read_csv(ORIGINAL_DATA_PATH / "test_a.csv")

    # filter other features to reduce memory
    user_action = user_action[["userid", "feedid", "date_"] + ACTION_LIST]

    # 13 days as history, 14th day as label, when prediction, use 2-14 days as history
    user_action_13 = user_action[user_action["date_"] != 14]
    user_action_14 = user_action[user_action["date_"] == 14]
    user_action_test = user_action[user_action["date_"] != 1]
    test_a = test_a.merge(feed_info, how="left", on=["feedid"])[
        [
            "userid",
            "feedid",
            "bgm_song_id",
            "bgm_singer_id",
            "authorid",
            "videoplayseconds",
        ]
    ]

    test_a.to_csv(ORIGINAL_DATA_PATH / 'test_din.csv', index=False)

    for action in ACTION_LIST:
        logger.info("-------------------------")
        logger.info("now in phase: {}".format(action))
        data_path = DIN_DATA_PATH / action
        data_path.mkdir(exist_ok=True)

        user_info = gen_user_info(
            user_action_13, action, data_path / "user_info_train.pkl"
        )
        train = gen_train_data(user_action_14, feed_info)

        user_info_test = gen_user_info(
            user_action_test, action, data_path / "user_info_test.pkl"
        )

        # gen corresponding hist columns
        gen_hist_columns(train, user_info, data_path / "hist_train.pkl")
        # gen_hist_columns(val, user_info, data_path / "hist_val.pkl")
        gen_hist_columns(test_a, user_info_test, data_path / 'hist_test.pkl')
