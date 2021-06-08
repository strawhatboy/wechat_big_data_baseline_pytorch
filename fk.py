import pandas as pd

col0 = pd.read_csv('./submit/submit_base_deepfm_col0.csv')
col1 = pd.read_csv('./submit/submit_base_deepfm_col1,3.csv')
col2 = pd.read_csv('./submit/submit_base_deepfm_col2.csv')
col3 = col1

res = col0
res['click_avatar'] = col2['click_avatar']
res['like'] = col1['like']
res['forward'] = col3['forward']

res.to_csv('shit.csv', index=False)
