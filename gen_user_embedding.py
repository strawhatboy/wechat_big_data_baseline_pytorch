import gc
from baseline import ACTION_LIST, FEED_EMBEDDINGS, ROOT_PATH, USER_ACTION
from prepare_data import process_embed
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=24)

USER_EMBEDDINGS = ROOT_PATH + '/user_embeddings'
if __name__ == '__main__':
    # feed_embeddings = pd.read_csv(FEED_EMBEDDINGS)
    # feed_embeddings = process_embed(feed_embeddings)
    # feed_embeddings = feed_embeddings.drop(['feed_embedding'], axis=1)
    # feed_embeddings.to_csv(ROOT_PATH + '/feed_embeddings_new.csv', index=False)
    feed_embeddings = pd.read_csv(ROOT_PATH + '/feed_embeddings_new.csv')
    user_action = pd.read_csv(USER_ACTION)
    for action in ACTION_LIST:
        user_actionx = user_action[['userid', 'feedid', action]]
        user_actionx = user_actionx.drop_duplicates(['userid', 'feedid', action], keep='last')
        user_actionx = user_actionx.merge(feed_embeddings, how='left', on=['feedid'])
        # user_action = process_embed(user_action)
        user_actionx = user_actionx.drop(['feedid', action], axis=1)
        user_embeddings = user_actionx.groupby(by=['userid']).mean()
        user_embeddings.columns = [f"user_embed{i}" for i in range(512)]

        output_file = ROOT_PATH + '/user_embeddings_{}.csv'.format(action)
        print('writing to file: {}'.format(output_file))

        user_embeddings.to_csv(output_file)
        del user_actionx
        del user_embeddings
        gc.collect()
        


