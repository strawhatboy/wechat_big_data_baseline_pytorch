# prepare data for DIN model

import argparse
from operator import index
from pathlib import Path
import pathlib
import pickle
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

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


def gen_user_info(df: pd.DataFrame, feed_info: pd.DataFrame, action: str, userids: List[int], save_path: Path):
    user_action_13_x = df[df[action] == 1][["userid", "feedid", action]]
    user_action_13_x = user_action_13_x.drop_duplicates(
        subset=["userid", "feedid", action], keep="last"
    )

    # userids = user_action_13_x["userid"].unique()
    logger.info('user count: {}'.format(len(userids)))
    user_info = {}

    logger.info("getting user history info")
    # line below is using LabelEncoder ids
    for uid in tqdm(range(len(userids))):
    # for uid in tqdm(userids):
        hist_interation = user_action_13_x[user_action_13_x["userid"] == uid]["feedid"]
        hist_feed: pd.DataFrame = feed_info.loc[hist_interation]
        # logger.info('BOOM: {}'.format(hist_feed))
        if hist_feed.empty:
            user_info[uid] = {
                "hist_records_count": 0,
                "bgm_song_ids": np.zeros(MAX_HISTORY_SIZE, dtype=int),
                "bgm_singer_ids": np.zeros(MAX_HISTORY_SIZE, dtype=int),
                "authorids": np.zeros(MAX_HISTORY_SIZE, dtype=int),
                "videoplayseconds": np.zeros(MAX_HISTORY_SIZE, dtype=float),
            }
        else:
            user_info[uid] = {
                "hist_records_count": len(hist_feed.index),
                "bgm_song_ids": pad_max_history_size(
                    hist_feed["bgm_song_id"].to_numpy(dtype=int)
                ),
                "bgm_singer_ids": pad_max_history_size(
                    hist_feed["bgm_singer_id"].to_numpy(dtype=int)
                ),
                "authorids": pad_max_history_size(
                    hist_feed["authorid"].to_numpy(dtype=int)
                ),
                "videoplayseconds": pad_max_history_size(
                    hist_feed["videoplayseconds"].to_numpy()
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
        # if uid not in user_info:
        #     logger.warning('user id: {} not in recorded user_info'.format(uid))
        # else:
        bgm_song_ids.append(user_info[uid]["bgm_song_ids"])
        bgm_singer_ids.append(user_info[uid]["bgm_singer_ids"])
        authorids.append(user_info[uid]["authorids"])
        videoplayseconds.append(user_info[uid]["videoplayseconds"])
        behavior_length.append(user_info[uid]["hist_records_count"])

    res = {
        "bgm_song_ids": np.array(bgm_singer_ids, dtype=int),
        "bgm_singer_ids": np.array(bgm_singer_ids, dtype=int),
        "authorids": np.array(authorids, dtype=int),
        "videoplayseconds": np.array(videoplayseconds),
        "behavior_length": np.array(behavior_length, dtype=int),
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
    feed_info = pd.read_csv(ORIGINAL_DATA_PATH / "feed_info.csv")
    feed_info['bgm_song_id'] = feed_info['bgm_song_id'].fillna(0)
    feed_info['authorid'] = feed_info['authorid'].fillna(0)
    feed_info['bgm_singer_id'] = feed_info['bgm_singer_id'].fillna(0)

    user_action = pd.read_csv(ORIGINAL_DATA_PATH / "user_action.csv")
    test_a = pd.read_csv(ORIGINAL_DATA_PATH / "test_a.csv")

    # filter other features to reduce memory
    user_action = user_action[["userid", "feedid", "date_"] + ACTION_LIST]

    # encoders
    logger.info('init encoders')
    user_lbe = LabelEncoder()
    user_lbe.fit(user_action['userid'])

    feed_lbe = LabelEncoder()
    feed_lbe.fit(feed_info['feedid'])

    author_lbe = LabelEncoder()
    author_lbe.fit(feed_info['authorid'])

    bgm_song_lbe = LabelEncoder()
    bgm_song_lbe.fit(feed_info['bgm_song_id'])

    bgm_singer_lbe = LabelEncoder()
    bgm_singer_lbe.fit(feed_info['bgm_singer_id'])

    user_action['userid'] = user_lbe.transform(user_action['userid'])
    user_action['feedid'] = feed_lbe.transform(user_action['feedid'])
    feed_info['feedid'] = feed_lbe.transform(feed_info['feedid'])
    feed_info['authorid'] = author_lbe.transform(feed_info['authorid'])
    feed_info['bgm_song_id'] = bgm_song_lbe.transform(feed_info['bgm_song_id'])
    feed_info['bgm_singer_id'] = bgm_singer_lbe.transform(feed_info['bgm_singer_id'])

    test_a['userid'] = user_lbe.transform(test_a['userid'])
    test_a['feedid'] = feed_lbe.transform(test_a['feedid'])

    logger.info('total users: {}'.format(len(user_lbe.classes_)))
    logger.info('total feeds: {}'.format(len(feed_lbe.classes_)))
    logger.info('total authorids: {}'.format(len(author_lbe.classes_)))
    logger.info('total bgm_song_ids: {}'.format(len(bgm_song_lbe.classes_)))
    logger.info('total bgm_singer_ids: {}'.format(len(bgm_singer_lbe.classes_)))

    feed_info.set_index(['feedid'])

    # 13 days as history, 14th day as label, when prediction, use 2-14 days as history
    user_action_13 = user_action[user_action["date_"] != 14]
    logger.info('user count in previous 13 days: {}'.format(user_action_13['userid'].nunique()))
    user_action_14 = user_action[user_action["date_"] == 14]
    logger.info('user count in last 14th day: {}'.format(user_action_14['userid'].nunique()))
    user_action_test = user_action[user_action["date_"] != 1]
    logger.info('user count in last 13 days: {}'.format(user_action_test['userid'].nunique()))
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
            # user_action_13, feed_info, action, user_action['userid'].unique(), data_path / "user_info_train.pkl"
            user_action_13, feed_info, action, user_lbe.classes_, data_path / "user_info_train.pkl"
        )
        train = gen_train_data(user_action_14, feed_info)

        user_info_test = gen_user_info(
            # user_action_test, feed_info, action, user_action['userid'].unique(), data_path / "user_info_test.pkl"
            user_action_test, feed_info, action, user_lbe.classes_, data_path / "user_info_test.pkl"
        )

        # gen corresponding hist columns
        gen_hist_columns(train, user_info, data_path / "hist_train.pkl")
        # gen_hist_columns(val, user_info, data_path / "hist_val.pkl")
        gen_hist_columns(test_a, user_info_test, data_path / 'hist_test.pkl')
