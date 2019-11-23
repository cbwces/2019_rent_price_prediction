#!/usr/bin/env python
# coding: utf-8
############################  
#File Name: house_price_lgb1.py  
#Author: 零落  
#Mail: sknyqbcbw@gmail.com  
#Created Time: 2019-06-03 20:16:33  
############################  

import warnings
import time
import gc
import os
import re
import datetime
import numpy as np 
import pandas as pd 
import lightgbm as lgb
import tqdm
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


# [1170]:


def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)

def adjR2_error(preds, train_data):
    labels = train_data.get_label()
    value = adjustedR2(r2_score(labels, preds), X_train.shape[0], X_train.shape[1])
    return 'R2', value, True

# [1171]:


def single_group_map(df, group, transform, agg, money=False):
    if money == False:
        for i in agg:
            df['{0}_{1}_{2}'.format(group, transform, i)] = df[group].map(dict(df.loc[df.tradeMoney >= 0].groupby(group)[transform].agg(i)))
    else:
        for i in agg:
            df['{0}_{1}_{2}'.format(group, transform, i)] = df[group].map(dict(train.groupby(group)[transform].agg(i)))
            
def double_group_map(df, group, transform, agg):
    for i in agg:
        df['{0}_{1}_{2}'.format('_'.join(group), transform, i)] = pd.MultiIndex.from_frame(df_all[group]).map(dict(df_all.loc[df.tradeMoney >= 0].groupby(group)[transform].agg(i)))


# [1172]:


def get_houseType_num(x, mode_num):
    return_list = []
    for i in x.values:
        if type(i) != type(np.nan):
            tmp = re.findall(r'(\d+)', i)
            return_list.append(int(tmp[mode_num]))
        else:
            return_list.append(np.nan)
    return return_list

# [1174]:

print('Loading data...')
train_path = os.environ['DATASET_DIR']
test_path = os.environ['TESTSET_DIR']
predict_path = os.environ['PREDICTION_FILE']
model_path = os.environ['MODEL_DIR']

train = pd.read_csv(train_path + '//train_data.csv', parse_dates=['tradeTime'])
# test = pd.read_csv(path + '//test_a.csv', parse_dates=['tradeTime'])
testb = pd.read_csv(test_path + '//test_b.csv', parse_dates=['tradeTime'])
print('Load complete!')

# [1175]:


print('Fixing data...')
train.loc[16224, 'totalFloor'] = 37
train.loc[21847, 'totalFloor'] = 37
train.loc[38128, 'totalFloor'] = 11
train.loc[35478, 'totalFloor'] = 8
train.loc[5434, 'totalFloor'] = 8

train.loc[16232, 'totalFloor'] = 11
train.loc[21855, 'totalFloor'] = 11
train.loc[38137, 'totalFloor'] = 11

train.loc[train.area == 15055, 'area'] = 150.55
train.loc[train.area == 3000, 'area'] = 30

# train.loc[(train.communityName == 'XQ00876') & (train.area == 14.8), 'houseToward'] = '南'
# train.loc[(train.communityName == 'XQ02234') & (train.area == 18), 'houseToward'] = '南'

train.loc[(train.communityName == 'XQ01137') & (train.tradeMoney == 99999999.99), 'tradeMoney'] = 15000
train.loc[(train.communityName == 'XQ01941') & (train.tradeMoney == 50000000), 'tradeMoney'] = 5000
train.loc[(train.communityName == 'XQ00209') & (train.tradeMoney == 10000000), 'tradeMoney'] = 1000
train.loc[(train.communityName == 'XQ01834') & (train.tradeMoney == 450000), 'tradeMoney'] = 4500
train.loc[(train.communityName == 'XQ02612') & (train.tradeMoney == 450000), 'tradeMoney'] = 4500
train.loc[(train.communityName == 'XQ03076') & (train.tradeMoney == 450000), 'tradeMoney'] = 4500
train.loc[(train.communityName == 'XQ03908') & (train.tradeMoney == 450000), 'tradeMoney'] = 4500
train.loc[(train.communityName == 'XQ01349') & (train.tradeMoney == 430000), 'tradeMoney'] = 4300
train.loc[(train.communityName == 'XQ01348') & (train.tradeMoney == 380000), 'tradeMoney'] = 3800
train.loc[(train.communityName == 'XQ00092') & (train.tradeMoney == 370000), 'tradeMoney'] = 3700
train.loc[(train.communityName == 'XQ03077') & (train.tradeMoney == 360000), 'tradeMoney'] = 3600
train.loc[(train.communityName == 'XQ00629') & (train.tradeMoney == 360000), 'tradeMoney'] = 3600
train.loc[(train.communityName == 'XQ01293') & (train.tradeMoney == 360000), 'tradeMoney'] = 3600
train.loc[(train.communityName == 'XQ01715') & (train.tradeMoney == 350000), 'tradeMoney'] = 3500
train.loc[(train.communityName == 'XQ02813') & (train.tradeMoney == 320000), 'tradeMoney'] = 3200
train.loc[(train.communityName == 'XQ00669') & (train.tradeMoney == 220000), 'tradeMoney'] = 2200
train.loc[(train.communityName == 'XQ00552') & (train.tradeMoney == 95000), 'tradeMoney'] = 9500

train.loc[(train.communityName == 'XQ01339') & (train.tradeMoney == 140), 'tradeMoney'] = 1400
train.loc[(train.communityName == 'XQ02343') & (train.tradeMoney == 140), 'tradeMoney'] = 1400
train.loc[(train.communityName == 'XQ02337') & (train.tradeMoney == 140), 'tradeMoney'] = 1400
train.loc[(train.communityName == 'XQ02379') & (train.tradeMoney == 150), 'tradeMoney'] = 1500
train.loc[(train.communityName == 'XQ03412') & (train.tradeMoney == 160), 'tradeMoney'] = 1600
train.loc[(train.communityName == 'XQ02444') & (train.tradeMoney == 160), 'tradeMoney'] = 1600
train.loc[(train.communityName == 'XQ01631') & (train.tradeMoney == 200), 'tradeMoney'] = 2000
train.loc[(train.communityName == 'XQ00200') & (train.tradeMoney == 210), 'tradeMoney'] = 2100
train.loc[(train.communityName == 'XQ01320') & (train.tradeMoney == 220), 'tradeMoney'] = 2200
train.loc[(train.communityName == 'XQ01316') & (train.tradeMoney == 250), 'tradeMoney'] = 2500
train.loc[(train.communityName == 'XQ00021') & (train.tradeMoney == 300), 'tradeMoney'] = 3000
train.loc[(train.communityName == 'XQ01834') & (train.tradeMoney == 300), 'tradeMoney'] = 3000
train.loc[(train.communityName == 'XQ01316') & (train.tradeMoney == 300), 'tradeMoney'] = 3000
train.loc[(train.communityName == 'XQ00312') & (train.tradeMoney == 300), 'tradeMoney'] = 3000
train.loc[(train.communityName == 'XQ01626') & (train.tradeMoney == 300), 'tradeMoney'] = 3000
train.loc[(train.communityName == 'XQ00092') & (train.tradeMoney == 300), 'tradeMoney'] = 3000
train.loc[(train.communityName == 'XQ03077') & (train.tradeMoney == 320), 'tradeMoney'] = 3200
train.loc[(train.communityName == 'XQ01040') & (train.tradeMoney == 350), 'tradeMoney'] = 3500
train.loc[(train.communityName == 'XQ01639') & (train.tradeMoney == 400), 'tradeMoney'] = 4000

train.reset_index(inplace=True, drop=True)


# [1176]:


train.loc[train.houseType == '0室0厅1卫', 'houseType'] = '1室0厅0卫'
train.loc[train.houseType == '1室0厅0卫', 'houseType'] = np.nan

tmp_features = [c for c in train.columns if c not in ['ID', 'tradeMoney']]
tmp = train[tmp_features].drop_duplicates().index
train = train.iloc[tmp]
train.reset_index(drop=True, inplace=True)

testb['tradeMoney'] = -200


# [1177]:


def fix_strange(left_range=2, right_range=5, small_inrange=1500, large_inrange=45000, 
                small_outrange=3000, large_outrange=25000, 
                small_scale_inrange=8, large_scale_inrange=3, 
                small_scale_outrange=5, large_scale_outrange=2.5):
    token = 0
    for idx in tqdm.tqdm(train.groupby(['communityName'])['tradeMoney'].agg(lambda x: np.ptp(x)).sort_values(ascending=False).index):
        if train.loc[(train.communityName == idx), 'tradeMoney'].sort_values().min() == 0:
            continue
        if len(train.loc[(train.communityName == idx), 'tradeMoney'].sort_values()) > left_range:
            tmp = train.loc[(train.communityName == idx), 'tradeMoney'].sort_values().copy()
            small_value = tmp.iloc[0]
            second_small_value = tmp.iloc[1]
            large_value = tmp.iloc[-1]
            second_large_value = tmp.iloc[-2]
            current_min = train.loc[(train.communityName == idx), 'tradeMoney'].sort_values().min()
            current_max = train.loc[(train.communityName == idx), 'tradeMoney'].sort_values().max()
        else:
            continue
        if len(train.loc[(train.communityName == idx), 'tradeMoney'].sort_values()) < right_range:
            if small_value <= small_inrange:
                while ((second_small_value / small_value) >= small_scale_inrange):
                    small_value = small_value * 10
            if large_value >= large_inrange:
                while ((large_value / second_large_value) >= large_scale_inrange):
                    large_value = large_value / 10
        else:
            if small_value <= small_outrange:
                while ((second_small_value / small_value) >= small_scale_outrange):
                    small_value = small_value * 10
            if large_value >= large_outrange:
                while ((large_value / second_large_value) >= large_scale_outrange):
                    large_value = large_value / 10
        if small_value != current_min:
            if (small_value % np.floor(small_value) == 0) & (small_value < train.loc[(train.communityName == idx), 'tradeMoney'].sort_values().max()):
                train.loc[(train.communityName == idx) & (train.tradeMoney == current_min), 'tradeMoney'] = small_value
                token += 1
        if large_value != current_max:
            if (large_value % np.floor(large_value) == 0) & (large_value > train.loc[(train.communityName == idx), 'tradeMoney'].sort_values().min()):
                train.loc[(train.communityName == idx) & (train.tradeMoney == current_max), 'tradeMoney'] = large_value
                token += 1
    return token

count = fix_strange(small_scale_inrange=10, large_scale_inrange=5, small_scale_outrange=8, large_scale_outrange=4.5)
print("Done with {} values fixed!".format(count))


# [1178]:

print('Preprocessing...')
df_all = pd.concat([train, testb], axis=0)
df_all.reset_index(drop=True, inplace=True)
# df_all = pd.concat([df_all, test], axis=0)
# df_all.reset_index(drop=True, inplace=True)
del df_all['city'], df_all['ID']


# [1179]:


df_all['buildYear'].replace('暂无信息', np.nan, inplace=True)
df_all['buildYear'] = df_all['buildYear'].astype('float')


# [1180]:


df_all['rentType'].replace('--', '未知方式', inplace=True)
df_all['rentType'].replace('未知方式', np.nan, inplace=True)


# [1181]:


df_all['houseDecoration'].replace('其他', np.nan, inplace=True)
df_all.loc[df_all.houseDecoration == '毛坯', 'houseDecoration'] = 1
df_all.loc[df_all.houseDecoration == '简装', 'houseDecoration'] = 2
df_all.loc[df_all.houseDecoration == '精装', 'houseDecoration'] = 3
df_all['houseDecoration'] = df_all['houseDecoration'].astype(np.float16)

df_all.loc[df_all.houseFloor == '低', 'houseFloor'] = 1
df_all.loc[df_all.houseFloor == '中', 'houseFloor'] = 2
df_all.loc[df_all.houseFloor == '高', 'houseFloor'] = 3
df_all['houseFloor'] = df_all['houseFloor'].astype(np.float16)

df_all['houseToward'].replace('暂无数据', np.nan, inplace=True)


# [1182]:


df_all['tradeTime_month'] = df_all['tradeTime'].dt.month
df_all['tradeTime_week'] = df_all['tradeTime'].dt.weekday
df_all['tradeTime_day'] = df_all['tradeTime'].dt.day
df_all['tradeTime_dayofyear'] = df_all['tradeTime'].dt.dayofyear
df_all['tradeTime_weekofyear'] = df_all['tradeTime'].dt.weekofyear

train['tradeTime_month'] = train['tradeTime'].dt.month
del df_all['tradeTime'], train['tradeTime']

# [1188]:

df_all['uv'].replace(0, np.nan, inplace=True)
df_all['pv'].replace(0, np.nan, inplace=True)

df_all['pv'] = df_all['pv'].fillna(df_all.groupby('communityName')['pv'].transform('median'))
df_all['uv'] = df_all['uv'].fillna(df_all.groupby('communityName')['uv'].transform('median'))

df_all['pv/uv'] = df_all['pv'] / df_all['uv']


# [1189]:


df_all = df_all.merge(df_all.groupby(['houseFloor', 'totalFloor'])['totalFloor'].size().reset_index(name='floor_count'), on=['houseFloor', 'totalFloor'], how='left')
df_all = df_all.merge(df_all.groupby(['communityName', 'totalFloor'])['totalFloor'].size().reset_index(name='community_floor_count'), on=['communityName', 'totalFloor'], how='left')
df_all = df_all.merge(df_all.groupby(['communityName', 'houseFloor'])['houseFloor'].size().reset_index(name='community_house_count'), on=['communityName', 'houseFloor'], how='left')


# [1190]:


age = 2018 - df_all['buildYear']
df_all['age'] = age

# age = np.log1p(-age) + np.log1p(age.max())

single_group_map(df_all, 'age', 'plate', ['nunique'])

df_all['floor/age'] = df_all['totalFloor'] / age
df_all['communitycount/age'] = df_all['communityName'].map(dict(df_all['communityName'].value_counts())) / age


# [1191]:


single_group_map(df_all, 'region', 'tradeMoney', ['mean', 'median', 'max', 'min', 'std', 'sum'], money=True)
single_group_map(df_all, 'plate', 'tradeMoney', ['mean', 'median', 'max', 'min', 'std', 'sum'], money=True)
single_group_map(df_all, 'totalFloor', 'tradeMoney', ['mean', 'median', 'max', 'min', 'sum', 'std'], money=True)

# single_group_map(df_all, 'houseDecoration', 'tradeMoney', ['median', 'max', 'min', 'std'], money=True)

single_group_map(df_all, 'region', 'area', ['mean', 'median', 'max', 'min', 'std', 'sum'])
single_group_map(df_all, 'plate', 'area', ['mean', 'median', 'max', 'min', 'std', 'sum'])
single_group_map(df_all, 'totalFloor', 'area', ['mean', 'median', 'max', 'min', 'std', 'sum'])
single_group_map(df_all, 'houseFloor', 'area', ['mean', 'median', 'max', 'min', 'std', 'sum'])
# single_group_map(df_all, 'houseDecoration', 'area', ['median', 'max', 'min', 'std'])
# double_group_map(df_all, ['communityName', 'totalFloor'], 'area', ['mean', 'std'])

single_group_map(df_all, 'plate', 'totalFloor', ['mean', 'median', 'max', 'min', 'std', 'sum'])
single_group_map(df_all, 'region', 'totalFloor', ['mean', 'median', 'max', 'min', 'std', 'sum'])

single_group_map(df_all, 'region', 'age', ['median', 'std'])
single_group_map(df_all, 'plate', 'age', ['median', 'std'])
# single_group_map(df_all, 'totalFloor', 'age', ['median', 'std'])

# df_all['totalFloor_buildYear_ptp'] = df_all['totalFloor'].map(dict(df_all.groupby('totalFloor')['buildYear'].agg(lambda x: np.ptp(x))))
# single_group_map(df_all, 'totalFloor', 'buildYear', ['mean', 'max', 'min', 'std'])

single_group_map(df_all, 'communityName', 'totalFloor', ['mean', 'median', 'max', 'min', 'std', 'sum'])
single_group_map(df_all, 'communityName', 'area', ['mean', 'median', 'max', 'min', 'std', 'sum'])

#第一批添加的ptp
df_all['communityName_totalFloor_ptp'] = df_all['communityName_totalFloor_max'] - df_all['communityName_totalFloor_min']
df_all['communityName_area_ptp'] = df_all['communityName_area_max'] - df_all['communityName_area_min']

age_area = df_all.area / np.log1p(age)
age_totalFloor = df_all.totalFloor / np.log1p(age)
tmp_df = df_all.copy()
tmp_df['area'] = age_area
tmp_df['totalFloor'] = age_totalFloor

for i in ['totalFloor', 'area']:
    for j in ['median', 'std']:
        df_all[f'communtyName_{i}_{j}/age'] = df_all['communityName'].map(dict(tmp_df.groupby('communityName')[i].agg(j)))
        df_all[f'plate_{i}_{j}/age'] = df_all['plate'].map(dict(tmp_df.groupby('plate')[i].agg(j)))
        df_all[f'region_{i}_{j}/age'] = df_all['region'].map(dict(tmp_df.groupby('region')[i].agg(j)))
        
single_group_map(df_all, 'communityName', 'uv', ['mean'])
single_group_map(df_all, 'communityName', 'pv', ['mean'])
# single_group_map(df_all, 'communityName', 'pv/uv', ['mean'])

single_group_map(df_all, 'plate', 'communityName', ['nunique'])
single_group_map(df_all, 'region', 'communityName', ['nunique'])

df_all['plate_tradeTime_month_tradeMeanPrice_mean'] =df_all['plate'].map(dict(df_all.groupby(['plate', 'tradeTime_month'])['tradeMeanPrice'].                          apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').                          groupby('plate')['tradeMeanPrice'].mean()))
df_all['plate_tradeTime_month_tradeMeanPrice_std'] =df_all['plate'].map(dict(df_all.groupby(['plate', 'tradeTime_month'])['tradeMeanPrice'].                          apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').                          groupby('plate')['tradeMeanPrice'].std()))

# df_all['region_tradeTime_month_tradeMeanPrice_mean'] = df_all.groupby(['region', 'plate'])['plate_tradeTime_month_tradeMeanPrice_mean'].\
# apply(lambda x: x.iloc[0]).reset_index('plate').groupby('region')['plate_tradeTime_month_tradeMeanPrice_mean'].mean()
# df_all['region_tradeTime_month_tradeMeanPrice_std'] = df_all.groupby(['region', 'plate'])['plate_tradeTime_month_tradeMeanPrice_mean'].\
# apply(lambda x: x.iloc[0]).reset_index('plate').groupby('region')['plate_tradeTime_month_tradeMeanPrice_mean'].std()

# df_all['plate_tradeTime_month_tradeMeanPrice_mean'] =\
# df_all['plate'].map(dict(df_all.groupby(['plate', 'tradeTime_month'])['tradeMeanPrice'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('plate')['tradeMeanPrice'].mean()))
# df_all['plate_tradeTime_month_tradeMeanPrice_std'] =\
# df_all['plate'].map(dict(df_all.groupby(['plate', 'tradeTime_month'])['tradeMeanPrice'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('plate')['tradeMeanPrice'].std()))

# df_all['plate_tradeTime_month_tradeSecNum_mean'] =\
# df_all['plate'].map(dict(df_all.groupby(['plate', 'tradeTime_month'])['tradeSecNum'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('plate')['tradeSecNum'].mean()))
# df_all['plate_tradeTime_month_tradeSecNum_std'] =\
# df_all['plate'].map(dict(df_all.groupby(['plate', 'tradeTime_month'])['tradeSecNum'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('plate')['tradeSecNum'].std()))

# df_all['plate_tradeTime_month_tradeNewMeanPrice_mean'] =\
# df_all['plate'].map(dict(df_all.groupby(['plate', 'tradeTime_month'])['tradeNewMeanPrice'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('plate')['tradeNewMeanPrice'].mean()))
# df_all['plate_tradeTime_month_tradeNewMeanPrice_std'] =\
# df_all['plate'].map(dict(df_all.groupby(['plate', 'tradeTime_month'])['tradeNewMeanPrice'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('plate')['tradeNewMeanPrice'].std()))
# df_all['plate_tradeTime_month_tradeNewMeanPrice_max'] =\
# df_all['plate'].map(dict(df_all.groupby(['plate', 'tradeTime_month'])['tradeNewMeanPrice'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('plate')['tradeNewMeanPrice'].max()))
# df_all['plate_tradeTime_month_tradeNewMeanPrice_min'] =\
# df_all['plate'].map(dict(df_all.groupby(['plate', 'tradeTime_month'])['tradeNewMeanPrice'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('plate')['tradeNewMeanPrice'].min()))

# df_all['plate_tradeTime_month_tradeMoney_mean'] =\
# df_all['plate'].map(dict(train.groupby(['plate', 'tradeTime_month'])['tradeMoney'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('plate')['tradeMoney'].mean()))
# df_all['plate_tradeTime_month_tradeMoney_std'] =\
# df_all['plate'].map(dict(train.groupby(['plate', 'tradeTime_month'])['tradeMoney'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('plate')['tradeMoney'].std()))

# df_all['region_tradeTime_month_tradeMoney_mean'] =\
# df_all['region'].map(dict(train.groupby(['region', 'tradeTime_month'])['tradeMoney'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('region')['tradeMoney'].mean()))
# df_all['region_tradeTime_month_tradeMoney_std'] =\
# df_all['region'].map(dict(train.groupby(['region', 'tradeTime_month'])['tradeMoney'].\
#                           apply(lambda x: x.iloc[0]).reset_index('tradeTime_month').\
#                           groupby('region')['tradeMoney'].std()))

df_all['community_totalFloor/whole_totalFloor_sum'] = df_all['communityName_totalFloor_sum'] / df_all.groupby('communityName')['communityName_totalFloor_sum'].head(1).sum()
df_all['community_totalFloor/whole_totalFloor_median'] = df_all['communityName_totalFloor_median'] / df_all.groupby('communityName')['communityName_totalFloor_median'].head(1).sum()
df_all['community_area/whole_area_sum'] = df_all['communityName_area_sum'] / df_all.groupby('communityName')['communityName_area_sum'].head(1).sum()
df_all['community_area/whole_area_median'] = df_all['communityName_area_median'] / df_all.groupby('communityName')['communityName_area_median'].head(1).sum()

df_all['totalFloor_area/whole_area_sum'] = df_all['totalFloor_area_sum'] / df_all.groupby('totalFloor')['totalFloor_area_sum'].head(1).sum()
df_all['totalFloor_area/whole_area_median'] = df_all['totalFloor_area_median'] / df_all.groupby('totalFloor')['totalFloor_area_median'].head(1).sum()

# '''测试'''
# df_all['plate_money/whole_money_sum'] = \
# df_all['plate_tradeMoney_sum'] / df_all.groupby('plate')['plate_tradeMoney_sum'].head(1).sum()
# df_all['plate_money/whole_money_median'] = \
# df_all['plate_tradeMoney_median'] / df_all.groupby('plate')['plate_tradeMoney_median'].head(1).sum()

# df_all['region_money/whole_money_sum'] = \
# df_all['region_tradeMoney_sum'] / df_all.groupby('region')['region_tradeMoney_sum'].head(1).sum()
# df_all['region_money/whole_money_median'] = \
# df_all['region_tradeMoney_median'] / df_all.groupby('region')['region_tradeMoney_median'].head(1).sum()
# '''封闭线'''

df_all['community_area/plate_area_mean'] = df_all['communityName_area_mean'] / df_all['plate_area_mean']
df_all['community_area/plate_area_max'] = df_all['communityName_area_max'] / df_all['plate_area_max']
df_all['community_area/plate_area_sum'] = df_all['communityName_area_sum'] / df_all['plate_area_sum']

df_all['community_buildYear/plate_buildYear_median'] = df_all['age'] / df_all['plate_age_median']
df_all['community_buildYear/region_buildYear_median'] = df_all['age'] / df_all['region_age_median']
# df_all['plate_buildYear_median/region_buildYear_median'] = df_all['plate_buildYear_median'] / df_all['plate_buildYear_median']


# [1192]:


'''renkouliudong'''
df_all['new_worker_rate'] = df_all['newWorkers'] / df_all['totalWorkers']
df_all['worker_rate'] = df_all['totalWorkers'] / (df_all['residentPopulation'] + df_all['totalWorkers'])
df_all['new_worker_rate'].replace(np.inf, 0, inplace=True)
df_all['worker_rate'].replace(np.inf, 0, inplace=True)


# [1193]:


df_all.loc[df_all.houseToward == '南', 'houseToward'] = 2
df_all.loc[df_all.houseToward == '西南', 'houseToward'] = 2
df_all.loc[df_all.houseToward == '南北', 'houseToward'] = 2
df_all.loc[df_all.houseToward == '东南', 'houseToward'] = 2
df_all.loc[df_all.houseToward == '西北', 'houseToward'] = 1
df_all.loc[df_all.houseToward == '西', 'houseToward'] = 1
df_all.loc[df_all.houseToward == '东西', 'houseToward'] = 1
df_all.loc[df_all.houseToward == '东', 'houseToward'] = 1
df_all.loc[df_all.houseToward == '北', 'houseToward'] = 1

df_all['houseToward'] = df_all['houseToward'].astype(np.float16)


# [1194]:


# double_group_map(df_all, ['houseFloor', 'totalFloor'], 'communityName', ['count', 'nunique'])
# double_group_map(df_all, ['communityName', 'houseFloor'], 'totalFloor', ['count', 'nunique', 'mean', 'max', 'min'])
# double_group_map(df_all, ['communityName', 'totalFloor'], 'houseFloor', ['count', 'nunique'])


# [1195]:


# community_month_count = pd.MultiIndex.from_frame(df_all[['communityName', 'tradeTime_month']]).\
#                         map(dict(df_all.groupby(['communityName', 'tradeTime_month']).size()))
# community_count = df_all['communityName'].map(dict(df_all.groupby(['communityName']).size()))
# df_all['communityName_month_ratio'] = community_month_count / community_count


# [1196]:


'''5.15pm新增'''
# double_group_map(df_all, ['communityName', 'tradeTime_month'], 'totalFloor', ['mean', 'max', 'min', 'count', 'nunique'])
# double_group_map(df_all, ['communityName', 'tradeTime_month'], 'area', ['mean', 'max', 'min', 'nunique'])
# double_group_map(df_all, ['communityName', 'tradeTime_month'], 'uv', ['mean', 'max', 'min'])
# double_group_map(df_all, ['communityName', 'tradeTime_month'], 'pv', ['mean', 'max', 'min'])

# double_group_map(df_all, ['communityName', 'tradeTime_month'], 'lookNum', ['mean'])


# [1197]:


df_all['region_count'] = df_all['region'].map(df_all['region'].value_counts())
df_all['communityName_count'] = df_all['communityName'].map(df_all['communityName'].value_counts())
df_all['plate_count'] = df_all['plate'].map(df_all['plate'].value_counts())

# df_all['totalFloor_nunique_count_rate'] = df_all['communityName_totalFloor_nunique'] / df_all['communityName_count']
# df_all['area_nunique_count_rate'] = df_all['communityName_area_nunique'] / df_all['communityName_count']
# df_all['houseToward_nunique_count_rate'] = df_all['communityName_houseToward_nunique'] / df_all['communityName_count']
# df_all['houseDecoration_nunique_count_rate'] = df_all['communityName_houseDecoration_nunique'] / df_all['communityName_count']
# df_all['houseType_nunique_count_rate'] = df_all['communityName_houseType_nunique'] / df_all['communityName_count']


# [1198]:


df_all['tradeTime_week'] += 1

# single_group_map(df_all, 'plate', 'tradeTime_weekofyear', ['mean', 'std'])
single_group_map(df_all, 'communityName', 'tradeTime_weekofyear', ['mean', 'std'])
# df_all['week_dummies'] = 0
# df_all.loc[df_all.tradeTime_week.isin([1, 2, 3, 4, 5]), 'week_dummies'] = 1
# df_all['day_dummies'] = 0
# df_all.loc[df_all.tradeTime_day.isin(range(11)), 'day_dummies'] = 1
# df_all.loc[df_all.tradeTime_day.isin(range(11, 21)), 'day_dummies'] = 2
# df_all.loc[df_all.tradeTime_day.isin(range(21, 32)), 'day_dummies'] = 3
# df_all = pd.concat([df_all, pd.get_dummies(df_all['week_dummies'], prefix='week')], axis=1)
# df_all = pd.concat([df_all, pd.get_dummies(df_all['day_dummies'], prefix='day')], axis=1)
# df_all = pd.concat([df_all, pd.get_dummies(df_all['tradeTime_month'], prefix='month')], axis=1)
# df_all = pd.concat([df_all, pd.get_dummies(df_all['tradeTime_weekofyear'], prefix='weekofyear')], axis=1)


# [1199]:


area = df_all['area'].astype(int).copy()


# [1200]:


'''houseType信息提取'''
for i in range(3):
    df_all[f'houseType_{i}'] = get_houseType_num(df_all['houseType'], i)
del df_all['houseType']

for i in range(3):
    train[f'houseType_{i}'] = get_houseType_num(train['houseType'], i)
del train['houseType']


# [1201]:


df_all['room_rate'] = (df_all['houseType_0']) / (3000 * area)
df_all['living_rate'] = (df_all['houseType_1']) / (2600 * area) 
df_all['wc_rate'] = (df_all['houseType_2']) / ((10000 / 14) * area)

df_all['wc_living_area'] = df_all['houseType_2'] * df_all['wc_rate']
df_all['total_living_area'] = df_all['houseType_1'] * df_all['living_rate']

df_all['rest_area'] = (10000 * area) - df_all['wc_living_area']
df_all['room_living_num'] = df_all['houseType_0'] + df_all['houseType_1']

# df_all['living/room'] = df_all['houseType_1'] / df_all['houseType_0']
# df_all['room/age'] = df_all['houseType_0'] / age

# df_all['total_houseType'] = df_all['houseType_0'] + df_all['houseType_1'] + df_all['houseType_2']


# [1202]:


df_all['community_sum_room_rate'] = df_all['communityName'].map(df_all.groupby('communityName')['houseType_0'].sum()) / (3000 * df_all['communityName_area_sum'].astype(int))
df_all['community_sum_living_rate'] = df_all['communityName'].map(df_all.groupby('communityName')['houseType_1'].sum()) / (2600 * df_all['communityName_area_sum'].astype(int))
df_all['community_sum_wc_rate'] = df_all['communityName'].map(df_all.groupby('communityName')['houseType_2'].sum()) / ((10000 / 14) * df_all['communityName_area_sum'].astype(int))

df_all['community_sum_wc_living_area'] = df_all['communityName'].map(df_all.groupby('communityName')['houseType_2'].sum()) * df_all['community_sum_wc_rate']
df_all['community_sum_total_living_area'] = df_all['communityName'].map(df_all.groupby('communityName')['houseType_1'].sum()) * df_all['community_sum_living_rate']

df_all['community_sum_rest_area'] = (10000 * df_all['communityName_area_sum'].astype(int)) - df_all['community_sum_wc_living_area']


# [1203]:


df_all['community_mean_room_rate'] = df_all['communityName'].map(df_all.groupby('communityName')['houseType_0'].mean()) / (3000 * df_all['communityName_area_mean'].astype(int))
df_all['community_mean_living_rate'] = df_all['communityName'].map(df_all.groupby('communityName')['houseType_1'].mean()) / (2600 * df_all['communityName_area_mean'].astype(int))
df_all['community_mean_wc_rate'] = df_all['communityName'].map(df_all.groupby('communityName')['houseType_2'].mean()) / ((10000 / 14) * df_all['communityName_area_mean'].astype(int))

df_all['community_mean_wc_living_area'] = df_all['communityName'].map(df_all.groupby('communityName')['houseType_2'].mean()) * df_all['community_mean_wc_rate']
df_all['community_mean_total_living_area'] = df_all['communityName'].map(df_all.groupby('communityName')['houseType_1'].mean()) * df_all['community_mean_living_rate']

df_all['community_mean_rest_area'] = (10000 * df_all['communityName_area_sum'].astype(int)) - df_all['community_mean_wc_living_area']


# [1204]:


df_all['houseFloor/totalFloor'] = df_all['houseFloor'] / df_all['totalFloor']
df_all['area/totalFloor'] = df_all['area'] / df_all['totalFloor']
# df_all['year_old'] = 2019 - df_all['buildYear']
df_all['area/room'] = df_all['area'] / df_all['houseType_0']
df_all['busStationNum/totalFloor'] = df_all['busStationNum'] / df_all['totalFloor']
df_all['interSchoolNum/school'] = df_all['interSchoolNum'] / (df_all['privateSchoolNum'] + df_all['schoolNum'])

# df_all['area/houseFloor'] = df_all['area'] / df_all['houseFloor']
# df_all['decoration/area'] = df_all['houseDecoration'] / df_all['area']
# df_all['room/area'] = df_all['houseType_0'] / df_all['area']
# df_all['room/totalFloor'] = df_all['houseType_0'] / df_all['totalFloor']


# [1205]:


'''上分'''
single_group_map(df_all, 'communityName', 'houseType_0', ['mean', 'std', 'max'])
single_group_map(df_all, 'communityName', 'houseType_1', ['mean', 'std', 'max'])
single_group_map(df_all, 'totalFloor', 'houseType_0', ['mean', 'std', 'max', 'sum'])
single_group_map(df_all, 'totalFloor', 'houseType_1', ['mean', 'std', 'max', 'sum'])
single_group_map(df_all, 'houseFloor', 'houseType_0', ['mean', 'std', 'max'])
single_group_map(df_all, 'houseFloor', 'houseType_1', ['mean', 'std', 'max'])
# single_group_map(df_all, 'houseFloor', 'houseType_0', ['mean'])
# single_group_map(df_all, 'houseFloor', 'houseType_1', ['mean'])


# [1206]:


del df_all['region'],df_all['plate'], df_all['rentType'], df_all['communityName']


# [1207]:


fe_col = [i for i in df_all.columns if i not in ['tradeMoney', 
                                                'rentType', 'floor_level']]


# [1208]:


df_all.drop(df_all.loc[df_all.tradeMoney == 0].index, inplace=True)
# df_all.drop(df_all['tradeMoney'].sort_values()[-17:].index, inplace=True)
df_all.drop(df_all.loc[df_all.totalFloor == 88].index, inplace=True)
# df_all.drop(df_all['area'].sort_values()[-3:].index, inplace=True)
df_all.reset_index(drop=True, inplace=True)

train = df_all.loc[df_all.tradeMoney > 0]
train.reset_index(drop=True, inplace=True)
test = df_all.loc[df_all.tradeMoney == -200]
test.reset_index(drop=True, inplace=True)

# train.drop(np.log1p((train.tradeMoney / train.area)).sort_values()[:4].index, inplace=True)
# train.drop(np.log1p((train.tradeMoney / train.area)).sort_values()[-3:].index, inplace=True)
# train.drop(train.area.sort_values().index[-1], inplace=True)
# train.drop(train.tradeMoney.sort_values().index[-9:], inplace=True)
# train.reset_index(drop=True, inplace=True)
# test.reset_index(drop=True, inplace=True)

train.drop(np.log1p((train.tradeMoney / train.area)).sort_values()[:10].index, inplace=True)
train.drop(np.log1p((train.tradeMoney / train.area)).sort_values()[-5:].index, inplace=True)
train.drop(train.area.sort_values().index[-1], inplace=True)
train.drop(train.tradeMoney.sort_values().index[-1], inplace=True)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
print('Preprocessing complete!')


# [1211]:

train['tradeMoney'] = train['tradeMoney'] / train['area']
# train['tradeMoney'] = np.log1p(train['area'] * 100 / train['tradeMoney'])
train['area'] = train['area'].astype(int)
test['area'] = test['area'].astype(int)

y_train = train.pop('tradeMoney')
X_train = train
X_test = test[X_train.columns]

X_train.replace(np.inf, np.nan, inplace=True)
X_test.replace(np.inf, np.nan, inplace=True)

# [1210]:

print(X_train.shape)
print(X_test.shape)

# [1165]

used_features = [c for c in X_train.columns if c not in {'area/totalFloor',
#  'bankNum',
 'communityName_area_sum',
 'communityName_count',
 'community_area/whole_area_sum',
 'community_house_count',
 'community_mean_living_rate',
 'community_mean_rest_area',
 'community_mean_wc_rate',
 'community_sum_rest_area',
 'community_sum_total_living_area',
 'drugStoreNum',
#  'houseFloor_houseType_0_max',
 'houseFloor_houseType_0_std',
 'houseFloor_houseType_1_max',
 'landMeanPrice',
 'landTotalPrice',
 'living_rate',
 'newWorkers',
 'new_worker_rate',
 'plate_area_max',
 'plate_area_mean',
 'plate_area_std',
 'plate_area_sum',
 'plate_communityName_nunique',
 'plate_count',
#  'plate_totalFloor_std',
 'plate_tradeMoney_min',
 'plate_tradeMoney_sum',
 'pv',
 'pv/uv',
 'region_area_max',
 'region_area_sum',
 'region_communityName_nunique',
 'region_count',
 'region_totalFloor_mean',
 'region_totalFloor_std',
 'region_totalFloor_sum',
 'remainNewNum',
 'residentPopulation',
 'rest_area',
 'room_rate',
 'shopNum',
 'supplyLandArea',
 'supplyLandNum',
 'supplyNewNum',
 'totalNewTradeArea',
 'totalNewTradeMoney',
 'totalTradeArea',
 'totalTradeMoney',
 'total_living_area',
 'tradeLandArea',
 'tradeLandNum',
 'tradeNewMeanPrice',
 'tradeNewNum',
 'tradeSecNum',
 'tradeTime_dayofyear',
 'uv',
 'wc_living_area',
 'wc_rate'}]

used_features = [c for c in used_features if c not in  ['communityName', 'houseType', 'houseToward']]

print('Start training...')
mae = []
feature_importance_df = pd.DataFrame(columns=['Feature', 'importance', 'fold'])
param = {
#          'min_data_in_leaf': 20, 
         'objective':'regression_l1',
         'max_depth': -1, #7
         'learning_rate': 0.005, #0.01
         'num_leaves': 1100,
         'max_bin': 255, 
#          "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.72, #0.72
         "bagging_freq": 15,#10
         "bagging_fraction": 0.79, #0.79
#          "bagging_seed": 11,
         "metric": 'mae',
         "lambda_l1": 2.333,#2.333
         'lambda_l2': 5,#5
         "verbosity": -1}
kf = KFold(n_splits=5, shuffle=True, random_state=2019)
oof = np.zeros(len(X_train))
predictions2 = np.zeros(len(X_test))

for i, (train_index, val_index) in enumerate(kf.split(X_train[used_features], y_train)):
    print("fold {}".format(i))
#     X_tr, X_val = X_train[train_index], X_train[val_index]
#     y_tr, y_val = y_train[train_index], y_train[val_index]
    X_tr, X_val = X_train.loc[train_index,used_features], X_train.loc[val_index, used_features]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    lgb_train = lgb.Dataset(X_tr,y_tr)
    lgb_val = lgb.Dataset(X_val,y_val)
    num_round = 50000
    clf = lgb.train(param, lgb_train, num_round, valid_sets=[lgb_train, lgb_val],# feval=mae_cum,
                    verbose_eval=300,
                    early_stopping_rounds=100, 
#                     categorical_feature=['houseType'], 
#                     feval=adjR2_error
                   )
    with open(model_path + f'//lgb1_model{i}.pkl', 'wb') as file:
        pickle.dump(clf, file)
    oof[val_index] = clf.predict(X_val, num_iteration=clf.best_iteration)
    mae.append(clf.best_score['valid_1']['l1'])
    
    predictions2 += clf.predict(X_test[used_features], num_iteration=clf.best_iteration) / kf.n_splits
print('Training complete!')

pd.Series(np.around(predictions2 * test.area)).to_csv(predict_path + '//prediction1.csv', index=None, header=None)
