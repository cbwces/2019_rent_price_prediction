#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd 
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold,train_test_split,StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,MinMaxScaler
from sklearn.decomposition import PCA,TruncatedSVD
pd.set_option('display.max_columns', 500)
import warnings
import time
import math
import sys
import os
import re
import gc
import pickle
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.metrics import log_loss
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


def square(y_true, y_pred):
    y = y_pred.get_label()
    score =r2_score(y_true,y)
    return 'socre', score, True

def get_train():
    # train_pred.to_csv('./train_pred.csv', index=False)
    train_pred = pd.read_csv('./train_pred.csv')
    houchuli = train.merge(train_pred, on='ID', how='left')
    houchuli['diff'] = houchuli['tradeMoney'] - houchuli['pred']
    houchuli['diff'] = (houchuli['diff'] / houchuli['tradeMoney']).values
    houchuli.loc[(houchuli['diff'].abs()>1)&(houchuli['tradeMoney']<2000), 'tradeMoney'] = (houchuli['pred']/100).round()*100
    money_map = houchuli[['ID', 'tradeMoney']].copy()
    money_map = money_map.set_index('ID')
    money_map = money_map['tradeMoney'].to_dict()
    train['tradeMoney'] = train['ID'].map(money_map)
    return train

def single_group_map(df, group, transform, agg, money=False):
    if money == False:
        for i in agg:
            df['{0}_{1}_{2}'.format(group, transform, i)] = df[group].map(dict(df.groupby(group)[transform].agg(i)))
    else:
        for i in agg:
            df['{0}_{1}_{2}'.format(group, transform, i)] = df[group].map(dict(train.groupby(group)[transform].agg(i)))


# In[4]:

train_path = os.environ['DATASET_DIR']
test_path = os.environ['TESTSET_DIR']
model_path = os.environ['MODEL_DIR']
prediction_path = os.environ['PREDICTION_FILE']

train = pd.read_csv(train_path + '//train_data.csv', parse_dates=['tradeTime'])
test = pd.read_csv(test_path + '//test_b.csv', parse_dates=['tradeTime'])
print(train.shape, test.shape, set(train.columns)-set(test.columns))
test['tradeMoney'] = np.nan


# In[5]:


"""检查test是否重复"""
tmp_features = [c for c in train.columns if c not in ['ID']]
train = train.drop_duplicates(subset=tmp_features)


# In[6]:


drop_ID = [100306895, 100048559, 100020516, 100306582, 100007093, 100089204, 100084542, 100136489,
          100103100, 100081082, 100085546, 100083675, 100193743, 100023095, 100314175, 100313791, 
          100310820, 100042658, 100040161, 100307082, 100020705, 100047710, 100028451, 100022073,
          100012093, 100262956]
drop_area = [100121332, 100092253, 100044554]
train = train[~train['ID'].isin(drop_ID+drop_area)]


# In[7]:


train = train[(train['tradeMoney']>300)]#300
train = train[(train['area']<1000)&(train['area']>4)]
train.loc[train['ID'].isin([100127642, 100089959]), 'totalFloor'] = 37
train = train[(train['totalFloor']<88)&(train['totalFloor']>0)]
train = train.reset_index(drop=True)
train.loc[train['ID'] == 100308971, 'tradeMoney'] = 10200
train.loc[train['ID'] == 100091529, 'tradeMoney'] = 4500
# train['tradeMoney'] = (train['tradeMoney'] / train['area']).values
# train['tradeMoney'] = train['tradeMoney'].apply(np.log1p)
data = train.append(test).reset_index(drop=True)
del data['city']


# In[67]:


# plt.figure(figsize=(15, 5))
# plt.hist(pd.factorize(train['communityName'])[0], normed=True, label='train')
# plt.hist(pd.factorize(test['communityName'])[0], normed=True, alpha=0.7, label='test')
# plt.legend()
# plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
# plt.xlabel('word_match_share', fontsize=15)
# train[train['tradeMoney']<1.5]
# train['tradeMoney'].hist()


# In[238]:


# train[train['communityName'] == 'XQ01429'][col]


# In[239]:


# for i in train['plate'].unique():
#     print(i)
#     cc = train[train['plate'] == i].sort_values(['area', 'tradeTime'])
#     plt.plot(cc['tradeMoney'].values)
#     plt.show()


# In[11]:


col = ['ID', 'area', 'rentType', 'houseType', 'houseFloor', 'totalFloor','tradeMoney','tradeTime',
       'houseToward', 'houseDecoration', 'communityName','pv', 'uv', 'lookNum']


# In[130]:


# cc = train[train['communityName'] == 'XQ01019'].sort_values(['area','tradeTime'])[col+['plate']]


# In[240]:


# drop_floor_id = [100004805, 100310758, 100314060, 100024132,100131119,
#                 ]
# plt.plot(cc['tradeMoney'].values)


# In[241]:


# cc
# train.loc[train['ID'] == 100149996, 'tradeMoney'] = 4700#?
# train[train['plate'] == 'BK00031'].sort_values(['totalFloor','tradeTime'])[col+['plate']]


# In[59]:


# train['area_money'] = train['tradeMoney'] / train['area']


# In[18]:


# train.groupby('totalFloor')['area_money'].mean().plot.bar()


# In[88]:


# train['houseFloor'].value_counts()


# In[93]:


# for i in ['低', '中', '高']:
#     plt.figure(figsize=(12,8))
#     train[train['houseFloor'] == i].groupby('totalFloor')['tradeMoney'].median().plot.bar()
#     plt.show()


# In[412]:


# """25000,500"""
# cc = train[train['tradeMoney'] < 500]['communityName'].unique()
# dd = data[data['communityName'].isin(cc)][col]
# dd[dd['tradeMoney'].isnull()]['communityName'].unique()


# In[475]:


# """25000阈值, 500"""
# # drop_ID = [100306895, 100048559, 100020516, 100306582, 100007093, 100089204, 100084542, 100136489,
# #           100103100, 100081082, 100085546, 100083675, 100193743, 100023095, 100314175, 100313791, 
# #           100310820, 100042658, 100040161, 100307082, 100020705, 100047710, 100028451, 100022073,
# #           100012093, 100262956]
# drop_IDlow = [100004805, 100020818, 100304665,100304683, 100042939, 100094209,100074445,
#              100109584, 100305713, 100312203, 100305713, 100086375, 100117915, 100073193,
#              100090432, 100281933, 100279227, 100276327, 100311951]
# ss = data[data['communityName'] == 'XQ01834'].sort_values(['area', 'tradeTime'])[col+['plate']]
# ss[50:]


# In[8]:


data['buildYear'] = data['buildYear'].replace('暂无信息', np.nan).astype('float64')
single_group_map(data, 'region', 'tradeMoney', ['mean', 'median', 'max', 'min', 'std', 'sum'], money=True)
single_group_map(data, 'plate', 'tradeMoney', ['mean', 'median', 'max', 'min', 'std', 'sum'], money=True)
single_group_map(data, 'totalFloor', 'tradeMoney', ['mean', 'median', 'max', 'min', 'sum', 'std'], money=True)

single_group_map(data, 'region', 'area', ['mean', 'median', 'max', 'std', 'sum'])
single_group_map(data, 'plate', 'area', ['median', 'sum'])
single_group_map(data, 'totalFloor', 'area', ['mean', 'median', 'max', 'min', 'std', 'sum'])
# single_group_map(data, 'houseFloor', 'area', ['mean', 'median', 'max', 'min', 'std', 'sum'])

single_group_map(data, 'plate', 'totalFloor', ['mean', 'median', 'max', 'min', 'std', 'sum'])
single_group_map(data, 'region', 'totalFloor', ['mean', 'median', 'max', 'min', 'std', 'sum'])

single_group_map(data, 'region', 'buildYear', ['median', 'std'])
single_group_map(data, 'communityName', 'totalFloor', ['median', 'sum'])
single_group_map(data, 'communityName', 'area', ['median', 'mean', 'sum'])


# In[9]:


data['community_totalFloor/whole_totalFloor_sum'] = data['communityName_totalFloor_sum'] / data.groupby('communityName')['communityName_totalFloor_sum'].head(1).sum()
data['community_totalFloor/whole_totalFloor_median'] = data['communityName_totalFloor_median'] / data.groupby('communityName')['communityName_totalFloor_median'].head(1).sum()
data['community_area/whole_area_sum'] = data['communityName_area_sum'] / data.groupby('communityName')['communityName_area_sum'].head(1).sum()
data['community_area/whole_area_median'] = data['communityName_area_median'] / data.groupby('communityName')['communityName_area_median'].head(1).sum()

data['totalFloor_area/whole_area_sum'] = data['totalFloor_area_sum'] / data.groupby('totalFloor')['totalFloor_area_sum'].head(1).sum()
data['totalFloor_area/whole_area_median'] = data['totalFloor_area_median'] / data.groupby('totalFloor')['totalFloor_area_median'].head(1).sum()
data['community_area/plate_area_sum'] = data['communityName_area_sum'] / data['plate_area_sum']


# In[10]:


# data['rentType'] = data['rentType'].replace('--', '未知方式')


# In[57]:


# group = data.groupby('buildYear')['totalFloor'].agg(['min', 'max', 'mean','std']).reset_index()
# group.columns = ['buildYear', 'b_t_min', 'b_t_max', 'b_t_median', 'b_t_std']
# data = data.merge(group, on='buildYear', how='left')


# In[58]:


# group = data.groupby('buildYear')['area'].agg(['min', 'max', 'mean', 'skew', 'std']).reset_index()
# group.columns = ['buildYear', 'b_a_min', 'b_a_max', 'b_a_median', 'b_a_skew', 'b_a_std']
# data = data.merge(group, on='buildYear', how='left')


# In[9]:


# group = data.groupby('houseType')['area'].agg(['min', 'max', 'mean', 'skew', 'std']).reset_index()
# group.columns = ['houseType', 'h_a_min', 'h_a_max', 'h_a_median', 'h_a_skew', 'h_a_std']
# data = data.merge(group, on='houseType', how='left')


# In[11]:


# data['buildYear'] = data['buildYear'].replace('暂无信息', np.nan).astype('float64')
data['houseType1'] = data['houseType'].values
data.loc[data.houseType1 == '0室0厅1卫', 'houseType'] = '1室0厅0卫'
data.loc[data.houseType1 == '1室0厅0卫', 'houseType'] = np.nan


# In[12]:


def get_houseType_num(x, mode_num):
    return_list = []
    for i in x.values:
        if type(i) != type(np.nan):
            tmp = re.findall(r'(\d+)', i)
            return_list.append(int(tmp[mode_num]))
        else:
            return_list.append(np.nan)
    return return_list
'''houseType信息提取'''
for i in range(3):
    data[f'houseType_{i}'] = get_houseType_num(data['houseType'], i)
del data['houseType1']


# In[13]:


area = data['area'].astype(int).copy()
data['room_rate'] = (data['houseType_0']) / (3000 * area)
data['living_rate'] = (data['houseType_1']) / (2600 * area) 
data['wc_rate'] = (data['houseType_2']) / ((10000 / 14) * area)

data['wc_living_area'] = data['houseType_2'] * data['wc_rate']
data['total_living_area'] = data['houseType_1'] * data['living_rate']

data['rest_area'] = (10000 * area) - data['wc_living_area']
data['room_living_num'] = data['houseType_0'] + data['houseType_1']


# In[14]:


data['community_sum_room_rate'] = data['communityName'].map(data.groupby('communityName')['houseType_0'].sum()) / (3000 * data['communityName_area_sum'].astype(int))
data['community_sum_living_rate'] = data['communityName'].map(data.groupby('communityName')['houseType_1'].sum()) / (2600 * data['communityName_area_sum'].astype(int))
data['community_sum_wc_rate'] = data['communityName'].map(data.groupby('communityName')['houseType_2'].sum()) / ((10000 / 14) * data['communityName_area_sum'].astype(int))

data['community_sum_wc_living_area'] = data['communityName'].map(data.groupby('communityName')['houseType_2'].sum()) * data['community_sum_wc_rate']
data['community_sum_total_living_area'] = data['communityName'].map(data.groupby('communityName')['houseType_1'].sum()) * data['community_sum_living_rate']

data['community_sum_rest_area'] = (10000 * data['communityName_area_sum'].astype(int)) - data['community_sum_wc_living_area']


# In[15]:


data['community_mean_room_rate'] = data['communityName'].map(data.groupby('communityName')['houseType_0'].mean()) / (3000 * data['communityName_area_mean'].astype(int))
data['community_mean_living_rate'] = data['communityName'].map(data.groupby('communityName')['houseType_1'].mean()) / (2600 * data['communityName_area_mean'].astype(int))
data['community_mean_wc_rate'] = data['communityName'].map(data.groupby('communityName')['houseType_2'].mean()) / ((10000 / 14) * data['communityName_area_mean'].astype(int))

data['community_mean_wc_living_area'] = data['communityName'].map(data.groupby('communityName')['houseType_2'].mean()) * data['community_mean_wc_rate']
data['community_mean_total_living_area'] = data['communityName'].map(data.groupby('communityName')['houseType_1'].mean()) * data['community_mean_living_rate']

data['community_mean_rest_area'] = (10000 * data['communityName_area_sum'].astype(int)) - data['community_mean_wc_living_area']


# In[16]:


single_group_map(data, 'communityName', 'houseType_0', ['mean', 'std', 'max'])
single_group_map(data, 'communityName', 'houseType_1', ['mean', 'std', 'max'])
single_group_map(data, 'totalFloor', 'houseType_0', ['mean', 'std', 'max', 'sum'])
single_group_map(data, 'totalFloor', 'houseType_1', ['mean', 'std', 'max', 'sum'])
single_group_map(data, 'houseFloor', 'houseType_0', ['mean', 'std', 'max'])
single_group_map(data, 'houseFloor', 'houseType_1', ['mean', 'std', 'max'])


# In[17]:


df_numeric = data.select_dtypes(exclude=['object'])
df_obj = data.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]
data_all = pd.concat([df_numeric, df_obj], axis=1)


# In[18]:


data_all['rate'] = data_all['area'].apply(lambda x: float('%.3f' % (x / math.ceil(x))) if int(x) != 0 else 0)


# In[19]:


data_all['tradeTime_month'] = data_all['tradeTime'].dt.month


# In[20]:


def get_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
agg_func = {
            'communityName': ['count'],
            'area': ['min', 'max'],
            'totalFloor': ['min', 'max', 'mean', 'nunique'],
            'tradeTime_month': ['min', 'max', 'nunique'],
            'rentType': ['mean', 'std', 'nunique'],
            'houseType': ['min', 'max', 'mean', 'std', 'nunique'],
            'houseFloor': ['mean', 'std', 'nunique'],
            'houseToward': ['mean', 'std', 'nunique'],
            'houseDecoration': ['mean', 'std', 'nunique'],
            'pv': ['median'],
            'uv': ['median'],
            'lookNum': ['mean', 'std', 'nunique'],
            'saleSecHouseNum':['mean', 'std', 'nunique'],
            'supplyNewNum':['mean', 'std', 'nunique'],
            'newWorkers': ['mean', 'std'],  
    }
new_columns = get_new_columns('com',agg_func)
df_group = data_all.groupby('communityName').agg(agg_func)
df_group.columns = new_columns
df_group.reset_index(drop=False,inplace=True)
data_all = data_all.merge(df_group, on='communityName', how='left')
del df_group


# In[21]:


fenmu = data_all['com_communityName_count']
data_all['com_rentType_nunique'] = fenmu / data_all['com_rentType_nunique']
data_all['com_houseType_nunique'] = fenmu / data_all['com_houseType_nunique']
data_all['com_houseFloor_nunique'] = fenmu / data_all['com_houseFloor_nunique']
data_all['com_houseDecoration_nunique'] = fenmu / data_all['com_houseDecoration_nunique']
data_all['com_houseToward_nunique'] = fenmu / data_all['com_houseToward_nunique']
data_all['com_totalFloor_nunique'] = fenmu / data_all['com_totalFloor_nunique']
data_all['com_tradeTime_month_nunique'] = fenmu / data_all['com_tradeTime_month_nunique']
data_all['com_tradeTime_month_ptp'] = data_all['com_tradeTime_month_max'] - data_all['com_tradeTime_month_min']


# In[22]:


agg_func = {
            'plate':['count'],
            'buildYear': ['mean', 'std', 'nunique'],
            'area':['min', 'mean', 'max', 'std', ],#skew
            'totalFloor': ['min', 'max', 'mean', 'std', ],#skew
    }
new_columns = get_new_columns('p',agg_func)
df_group = data_all.groupby('plate').agg(agg_func)
df_group.columns = new_columns
df_group.reset_index(drop=False,inplace=True)
data_all = data_all.merge(df_group, on='plate', how='left')
del df_group


# In[54]:


# data_all[['plate', 'totalWorkers', 'tradeTime_month']].drop_duplicates()['plate'].value_counts()


# In[23]:


data_all['community_area/plate_area_mean'] = data_all['communityName_area_mean'] / data_all['p_area_mean']
data_all['community_area/plate_area_max'] = data_all['com_area_max'] / data_all['p_area_max']


# In[24]:


for i in ['totalFloor', 'rentType', 'houseType', 'houseFloor', 'buildYear', 'houseToward','houseDecoration', 'region']:
    data_all[i + '_count'] = data_all[i].map(data_all[i].value_counts())


# In[365]:


# group = data_all.groupby('region')['communityName'].apply(lambda x:x.value_counts().index[0]).reset_index(name='region_mode')
# data_all = data_all.merge(group, on='region', how='left')
# group = data_all.groupby('plate')['buildYear'].apply(lambda x:x.value_counts().index[0]).reset_index(name='region_mode')
# data_all = data_all.merge(group, on='plate', how='left')


# In[25]:


data_all['p/u_month'] = data_all['pv'] / data_all['uv']
data_all['p_buildYear_nunique'] = data_all['p_plate_count'] / data_all['p_buildYear_nunique']


# In[26]:


peitao = ['subwayStationNum', 'busStationNum', 'interSchoolNum',
          'schoolNum', 'privateSchoolNum', 'hospitalNum', 'drugStoreNum', 
          'gymNum', 'bankNum', 'shopNum','parkNum', 'mallNum', 'superMarketNum','saleSecHouseNum']
for i in peitao:
    data_all[i+'/count'] = (data_all[i] / data_all['com_communityName_count']).values
#     region_sum = data_all[['region', 'busStationNum']].drop_duplicates().groupby('region')['busStationNum'].sum()
#     data_all[i+'_region'] = data_all['region'].map(region_sum)
# data_all['saleSecHouseNum/count'] = (data_all['saleSecHouseNum'] / data_all['com_communityName_count']).values
# data_all['saleSecHouseNum_c'] = data_all['saleSecHouseNum'].map(data_all['saleSecHouseNum'].value_counts())


# In[268]:


# data_all['worker_rate'] = data_all['totalWorkers'] / data_all['residentPopulation']


# In[27]:


data_all['tradeTime_day'] = data_all['tradeTime'].dt.day
data_all['tradeTime_week'] = data_all['tradeTime'].dt.weekday


# In[28]:


age = np.log1p(2018 - data_all['buildYear'])
age = -age + age.max()

data_all['buildYear_communityName_nunique'] = data_all['buildYear'].map(dict(data_all.groupby('buildYear')['communityName'].nunique()))
data_all['totalFloor_communityName_nunique'] = data_all['totalFloor'].map(dict(data_all.groupby('totalFloor')['communityName'].nunique()))
data_all['floor_count/age'] = data_all['totalFloor_communityName_nunique'] / age
data_all['area/age'] = data_all['area'] / age
data_all['floor/age'] = data_all['totalFloor'] / age
data_all['communitycount/age'] = data_all['communityName'].map(dict(data_all['communityName'].value_counts())) / age


# In[29]:


data_all=data_all.drop(['totalTradeMoney','totalTradeArea', 'tradeMeanPrice', 'tradeSecNum', 'totalNewTradeMoney',
                'totalNewTradeArea', 'tradeNewMeanPrice', 'tradeNewNum', 'remainNewNum', 'supplyNewNum', 'supplyLandNum',
 'supplyLandArea', 'tradeLandNum', 'tradeLandArea', 'landTotalPrice', 'landMeanPrice','saleSecHouseNum'],axis = 1) 
print(data_all.shape)
#220


# In[30]:


train = data_all[:len(train)].copy()
test = data_all[len(train):].copy()
train['tradeMoney'] = (train['tradeMoney'] / train['area']).values
train['tradeMoney'] = train['tradeMoney'].apply(np.log1p)


# In[31]:


fe_col = [i for i in train.columns if i not in ['ID', 'tradeMoney', 'tradeTime']]
X_train = train[fe_col].copy()
y_train = train['tradeMoney'].copy()
X_test = test[fe_col].copy()


# In[32]:


param = {
         'num_leaves': 2**8,
         'objective':'regression_l1',#regression_l1
         'max_depth': -1,
#          'learning_rate': 0.2,
         'boosting': 'gbdt',
         'feature_fraction': 0.7,
         'bagging_fraction': 0.8,
         'metric': 'huber',
#          'reg_sqrt':False,
#          'lambda_l1': 1,    
         'lambda_l2': 10,
         'nthread': -1,
         'verbosity': -1}


# In[42]:


kf = KFold(n_splits=5, shuffle=True, random_state=1234)
oof = np.zeros(len(X_train))
predictions1 = np.zeros(len(X_test))
score = []
for i, (train_index, val_index) in enumerate(kf.split(X_train,y_train)):
    print("fold {}".format(i))
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    lgb_train = lgb.Dataset(X_tr,y_tr)
    lgb_val = lgb.Dataset(X_val,y_val)
    num_round = 10000
    clf = lgb.train(param, lgb_train, num_round, valid_sets = [lgb_train, lgb_val], feval=square,
                    verbose_eval=1000, early_stopping_rounds = 101,
                   )
    with open(model_path + f'//lgb2_model{i}.pkl', 'wb') as file:
        pickle.dump(clf, file)
 
    oof[val_index] = clf.predict(X_val, num_iteration=clf.best_iteration)
    score.append(clf.best_score['valid_1']['socre'])
    
    predictions1 += clf.predict(X_test, num_iteration=clf.best_iteration) / kf.n_splits

pd.Series(np.expm1(predictions1)*test['area']).round().to_csv(prediction_path + '//prediction2.csv', index=None, header=None)
