import argparse
from functools import reduce
import numbers
import pandas as pd
import torch

from baseline import MyDeepFM

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default='mean')
    p.add_argument('--run_count', type=int, required=True)
    p.add_argument('--run_prefix', type=str)
    return p.parse_args()


if __name__ == '__main__':
    args = vars(parse_args())
    if (args['mode'] == 'mean'):
        # simply load those csv and mean
        outputs = [pd.read_csv('./submit/submit_base_deepfm_{}{}.csv'.format(args['run_prefix'], x)) for x in range(args['run_count'])]
        # for i, x in enumerate(outputs):
        #     x.columns = ['userid', 
        #     'feedid', 
        #     'read_comment_{}'.format(i),
        #     'like_{}'.format(i),
        #     'click_avatar_{}'.format(i),
        #     'forward_{}'.format(i)
        #     ]
        df: pd.DataFrame = reduce(lambda left,right: pd.merge(left, right, on=['userid', 'feedid']), outputs)
        print(df.head())

        df = df.groupby(by=df.columns, axis=1).apply(lambda g: g.mean(axis=1) if isinstance(g.iloc[0,0], numbers.Number) else g.iloc[:,0])
        df['read_comment'] = (df['read_comment_x'] + df['read_comment_y']) / 2
        df['like'] = (df['like_x'] + df['like_y']) / 2
        df['click_avatar'] = (df['click_avatar_x'] + df['click_avatar_y']) / 2
        df['forward'] = (df['forward_x'] + df['forward_y']) / 2
        df = df.drop(['read_comment_x', 'read_comment_y', 'like_x', 'like_y', 'click_avatar_x', 'click_avatar_y', 'forward_x', 'forward_y'], axis=1)
        df['userid'] = df['userid'].astype(int)
        df['feedid'] = df['feedid'].astype(int)
        df.to_csv('./submit/submit_base_deepfm.csv', index=False)


