import datetime
import os
import warnings

import numpy as np
import pandas  as pd


from generate_feature import get_beta_target, add_last_next_time4fault, get_feature, \
    get_duration_minutes_fea, get_nearest_msg_fea, get_server_model_sn_fea_2, \
    get_server_model_fea, get_msg_text_fea_all, get_key_word_cross_fea, get_server_model_time_interval_stat_fea, \
    get_w2v_feats, get_key_for_top_fea,get_time_diff_feats_v2
from model import run_cbt
from utils import RESULT_DIR, TRAIN_DIR, \
    TEST_A_DIR, KEY_WORDS, get_word_counter, search_weight, macro_f1, TIME_INTERVAL,PSEUDO_FALG,GENERATION_DIR

warnings.filterwarnings('ignore')


def get_label(PSEUDO_FALG):
    preliminary_train_label_dataset = pd.read_csv(preliminary_train_label_dataset_path)
    preliminary_train_label_dataset_s = pd.read_csv(preliminary_train_label_dataset_s_path)

    if PSEUDO_FALG:
        print('获取伪标签LABEL')
        pseudo_labels = pd.read_csv(os.path.join(TRAIN_DIR, 'pseudo_labels.csv'))
        label = pd.concat([preliminary_train_label_dataset,
                           pseudo_labels,
                           preliminary_train_label_dataset_s],
                          ignore_index=True,
                          axis=0).sort_values(
            ['sn', 'fault_time']).reset_index(drop=True)
    else:
        print('不使用伪标签数据')
        label = pd.concat([preliminary_train_label_dataset,
                           preliminary_train_label_dataset_s],
                          ignore_index=True,
                          axis=0).sort_values(
            ['sn', 'fault_time']).reset_index(drop=True)
    label['fault_time'] = label['fault_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    label['fault_time'] = label['fault_time'].apply(lambda x: str(x))
    return label


def get_log_dateset(PSEUDO_FALG):
    preliminary_sel_log_dataset = pd.read_csv(preliminary_sel_log_dataset_path)
    preliminary_sel_log_dataset_a = pd.read_csv(preliminary_sel_log_dataset_a_path)
    if PSEUDO_FALG:
        print('获取伪标签日志数据')
        pseudo_sel_log_dataset = pd.read_csv(os.path.join(TRAIN_DIR, 'pseudo_sel_log_dataset.csv'))
        log_dataset = pd.concat([preliminary_sel_log_dataset,
                                 pseudo_sel_log_dataset,
                                 preliminary_sel_log_dataset_a],
                                ignore_index=True,
                                axis=0).sort_values(
            ['sn', 'time', 'server_model']).reset_index(drop=True)
    else:
        print('不使用伪标签数据')
        log_dataset = pd.concat([preliminary_sel_log_dataset,
                                 preliminary_sel_log_dataset_a],
                                ignore_index=True,
                                axis=0).sort_values(
            ['sn', 'time', 'server_model']).reset_index(drop=True)
    log_dataset['time'] = log_dataset['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    return log_dataset

def get_fea_distribute(feature_df, feature_importances, dataset_type, top=30):
    print('根据特征重要性，获取数据集的分布情况，用于验证训练集和测试集是否分布一致')
    fea_distribute_list = []
    for i in feature_importances[:top]['fea'].to_list():
        fea_distribute_tmp = (feature_df[i].value_counts() / len(feature_df)).reset_index().rename(
            columns={'index': 'value'})
        fea_distribute_list.append(fea_distribute_tmp)

    fea_distribute = fea_distribute_list[-1]
    for i in fea_distribute_list[:-1]:
        fea_distribute = fea_distribute.merge(i, on='value', how='left')
    fea_distribute['value'] = fea_distribute['value'].apply(lambda x: f'{dataset_type}_{int(x)}')
    return fea_distribute



def get_train_test(label, preliminary_submit_dataset_a, log_dataset):
    print('获取训练集数据与测试集数据')
    train = label.merge(log_dataset, on='sn', how='left')
    test = preliminary_submit_dataset_a.merge(log_dataset, on='sn', how='left')
    #     train['time_interval']  = (pd.to_datetime( train['fault_time'])-train['time']  ).apply(lambda x:x.total_seconds())
    #     test['time_interval']  = (pd.to_datetime( test['fault_time'])- test['time']  ).apply(lambda x:x.total_seconds())
    #     train = train.query('time_interval > 0')
    #     test = test.query('time_interval > 0')
    print(f'训练集维度:{train.shape},测试集维度:{test.shape}')
    train = train.drop_duplicates().reset_index(drop=True)
    test = test.drop_duplicates().reset_index(drop=True)
    train['time'] = pd.to_datetime(train['time'])
    test['time'] = pd.to_datetime(test['time'])
    return train, test
start_time = datetime.datetime.now()

additional_sel_log_dataset_path = os.path.join(TRAIN_DIR, 'additional_sel_log_dataset.csv')
preliminary_train_label_dataset_path = os.path.join(TRAIN_DIR, 'preliminary_train_label_dataset.csv')
preliminary_train_label_dataset_s_path = os.path.join(TRAIN_DIR, 'preliminary_train_label_dataset_s.csv')
preliminary_sel_log_dataset_path = os.path.join(TRAIN_DIR, 'preliminary_sel_log_dataset.csv')

preliminary_submit_dataset_a_path = os.path.join(TEST_A_DIR, 'final_submit_dataset_b.csv')
preliminary_sel_log_dataset_a_path = os.path.join(TEST_A_DIR, 'final_sel_log_dataset_b.csv')

print(preliminary_submit_dataset_a_path, preliminary_sel_log_dataset_a_path)

preliminary_submit_dataset_a = pd.read_csv(preliminary_submit_dataset_a_path)
preliminary_submit_dataset_a.head()

log_dataset = get_log_dateset(PSEUDO_FALG)
label = get_label(PSEUDO_FALG)


next_time_list = [i / TIME_INTERVAL for i in [3, 5, 10, 15, 30, 45, 60, 90, 120, 240, 360, 480, 540, 600]] + [1000000]

label, preliminary_submit_dataset_a = add_last_next_time4fault(label, preliminary_submit_dataset_a, TIME_INTERVAL,
                                                               next_time_list)
train, test = get_train_test(label, preliminary_submit_dataset_a, log_dataset)
train = train.drop_duplicates(['sn', 'fault_time', 'time', 'msg', 'server_model']).reset_index(drop=True)

train['time_interval'] = (pd.to_datetime(train['fault_time']) - pd.to_datetime(train['time'])).apply(
    lambda x: x.total_seconds())
test['time_interval'] = (pd.to_datetime(test['fault_time']) - pd.to_datetime(test['time'])).apply(
    lambda x: x.total_seconds())

all_data = pd.concat([train, test], axis=0, ignore_index=True)
all_data = all_data.sort_values(['sn','server_model', 'fault_time', 'time'])
w2v_feats = get_w2v_feats(all_data,
                          f1_list = ['sn'],
                          f2_list = ['msg_list', 'msg_0', 'msg_1', 'msg_2'])
# 获取 time_diff_feats_v2
time_diff_feats_v2 = get_time_diff_feats_v2(all_data)
# 获取 server_model_time_interval_stat_fea
server_model_time_interval_stat_fea = get_server_model_time_interval_stat_fea(all_data)

msg_text_fea = get_msg_text_fea_all(all_data)
# 获取时间差特征
duration_minutes_fea = get_duration_minutes_fea(train, test)

# 获取时间server_model特征
server_model_fea = get_server_model_fea(train, test)
counter = get_word_counter(train)

# 获取时间 nearest_msg 特征
nearest_msg_fea = get_nearest_msg_fea(train, test)
# 获取时间 server_model beta_target 特征
beta_target_fea = get_beta_target(train, test)

key = ['sn', 'fault_time', 'label', 'server_model']

fea_num = len(KEY_WORDS)
time_list = [i * TIME_INTERVAL for i in next_time_list]
train = get_feature(train, time_list, KEY_WORDS, fea_num, key=['sn', 'fault_time', 'label', 'server_model'])
test = get_feature(test, time_list, KEY_WORDS, fea_num, key=['sn', 'fault_time', 'server_model'])

print('添加 时间差 特征')
train = train.merge(duration_minutes_fea, on=['sn', 'fault_time', 'server_model'])
test = test.merge(duration_minutes_fea, on=['sn', 'fault_time', 'server_model'])

print('添加 server_model特征')
train = train.merge(server_model_fea, on=['sn', 'server_model'])
test = test.merge(server_model_fea, on=['sn', 'server_model'])

print('添加 w2v_feats')
train = train.merge(w2v_feats, on=['sn' ])
test = test.merge(w2v_feats, on=['sn', ])

print('添加 nearest_msg 特征')
train = train.merge(nearest_msg_fea, on=['sn', 'server_model', 'fault_time'])
test = test.merge(nearest_msg_fea, on=['sn', 'server_model', 'fault_time'])

print('添加 beta_target 特征')
train = train.merge(beta_target_fea, on=['sn', 'server_model', 'fault_time'])
test = test.merge(beta_target_fea, on=['sn', 'server_model', 'fault_time'])

server_model_sn_fea_2 = get_server_model_sn_fea_2(train, test)
print('添加 server_model_sn_fea_2 特征')
train = train.merge(server_model_sn_fea_2, on=['sn', 'server_model'])
test = test.merge(server_model_sn_fea_2, on=['sn', 'server_model'])

print('添加 time_diff_feats_v2 特征')
train = train.merge(time_diff_feats_v2, on=['sn', 'server_model', 'fault_time'])
test = test.merge(time_diff_feats_v2, on=['sn', 'server_model', 'fault_time'])

# test.to_csv(os.path.join(GENERATION_DIR,'test.csv'),index =False)
# train.to_csv(os.path.join(GENERATION_DIR,'train.csv'),index =False)

# crashdump_venus_fea = pd.read_csv(os.path.join(GENERATION_DIR,'crashdump_venus_fea.csv') )
# print('添加 crashdump_venus_fea 特征')
# print(train.shape,test.shape,crashdump_venus_fea.shape)
# train = train.merge(crashdump_venus_fea, on=['sn' , 'fault_time'],how = 'left')
# test = test.merge(crashdump_venus_fea, on=['sn', 'fault_time' ],how = 'left')
# print(train.shape,test.shape )

# crashdump_venus_fea = pd.read_csv(os.path.join(GENERATION_DIR,'crashdump_venus_fea_v1.csv') )
# print('添加 crashdump_venus_fea 特征')
# print(train.shape,test.shape,crashdump_venus_fea.shape)
# train = train.merge(crashdump_venus_fea, on=['sn' , 'fault_time'],how = 'left')
# test = test.merge(crashdump_venus_fea, on=['sn', 'fault_time' ],how = 'left')
# print(train.shape,test.shape )
# test.to_csv(os.path.join(GENERATION_DIR,'test.csv'),index =False)
# train.to_csv(os.path.join(GENERATION_DIR,'train.csv'),index =False)
# print('添加 key_for_top_fea 特征')
# train,test = get_key_for_top_fea(train,test)

# print('添加 w2v_tfidf_doc2v_fea 特征')
# w2v_tfidf_fea = pd.read_csv(os.path.join(GENERATION_DIR,'w2v_tfidf_fea.csv'))
# drop_cols = [i for i in w2v_tfidf_fea if 'doc2vec' in i ]+[i for i in w2v_tfidf_fea if 'tfidf' in i ]
# for col in drop_cols:
#     del w2v_tfidf_fea[col]
#
# train = train.merge(w2v_tfidf_fea, on=['sn'  ], how='left')
# test = test.merge(w2v_tfidf_fea, on=['sn' ], how='left')

# print('添加 关键词交叉特征  ')
# train,test = get_key_word_cross_fea(train,test)

# print('添加 server_model_time_interval_stat_fea 特征')
# train = train.merge(server_model_time_interval_stat_fea, on=['server_model' ],how ='left')
# test = test.merge(server_model_time_interval_stat_fea, on=['server_model'  ],how ='left')


use_less_cols_1 = ['last_last_msg_cnt', 'last_first_msg_cnt','time_diff_1_min',
       'last_msg_list_unique_LabelEnc', 'last_msg_0_unique_LabelEnc',
       'last_msg_1_unique_LabelEnc', 'last_msg_2_unique_LabelEnc',
       'last_msg_list_list_LabelEnc', 'last_msg_0_list_LabelEnc',
       'last_msg_1_list_LabelEnc', 'last_msg_2_list_LabelEnc',
       'last_msg_0_first_LabelEnc', 'last_msg_1_first_LabelEnc',
       'last_msg_2_first_LabelEnc', 'last_msg_0_last_LabelEnc',
       'last_msg_1_last_LabelEnc', 'last_msg_2_last_LabelEnc',
       'last_msg_last_LabelEnc', 'last_msg_first_LabelEnc']

use_less_col = [i for i in train.columns if train[i].nunique() < 2] + use_less_cols_1


print(f'use_less_col:{len(use_less_col)}')
use_cols = [i for i in train.columns if i not in ['sn', 'fault_time', 'label', 'server_model'] + use_less_col]

cat_cols = ['server_model_LabelEnc', 'msg_LabelEnc', 'msg_0_LabelEnc', 'msg_1_LabelEnc', 'msg_2_LabelEnc',]
use_cols = sorted(use_cols)

cat_cols = []
for i in use_cols:
    if '_LabelEnc' in i:
        cat_cols.append(i)
print('使用的特征维度:',len(use_cols),'类别特征维度:',len(cat_cols))
# fs = FeatureSelector(data=train[use_cols], labels=train['label'])
#
# # 选择出missing value 百分比大于60%的特征
# fs.identify_missing(missing_threshold=0.9)
#
# # # 查看选择出的特征
# # fs.ops['missing']
# # 不对feature进行one-hot encoding（默认为False）, 然后选择出相关性大于98%的feature,
# fs.identify_collinear(correlation_threshold=0.99, one_hot=False)
#
# # # 查看选择的feature
# # fs.ops['collinear']
#
# # 选择出只有单个值的feature
# fs.identify_single_unique()
#
# # # 查看选择出的feature
# # fs.ops['single_unique']
#
# train_removed = fs.remove(methods = ['missing', 'single_unique', 'collinear',], keep_one_hot=False)
# use_cols = train_removed.columns
# print('特征选择之后，使用的特征维度:',len(use_cols))


oof_prob = np.zeros((train.shape[0], 4))
test_prob = np.zeros((test.shape[0], 4))
# seeds = [42,4242,40424,1024,2048]
seeds = [42 ]
for seed in seeds:
    oof_prob, test_prob, fea_imp_df, model_list = run_cbt(train[use_cols] , train[['label']] , test[use_cols], k=5,
                                              seed=seed, cat_cols=cat_cols)
    oof_prob +=oof_prob/len(seeds)
    test_prob +=test_prob/len(seeds)


weight = search_weight(train, train[['label']], oof_prob, init_weight=[1.0], class_num=4, step=0.001)
oof_prob = oof_prob * np.array(weight)
test_prob = test_prob * np.array(weight)

target_df = train[['sn', 'fault_time', 'label']]
submit_df = train[['sn', 'fault_time']]
submit_df['label'] = oof_prob.argmax(axis=1)

score = macro_f1(target_df=target_df, submit_df=submit_df)
print(f'********************** BEST MACRO_F1 : {score} **********************')
score = round(score, 5)

y_pred = test_prob.argmax(axis=1)
result = test[['sn', 'fault_time']]
result['label'] = y_pred
result = preliminary_submit_dataset_a.merge(result, on=['sn', 'fault_time'], how='left')[['sn', 'fault_time', 'label']]
result['label'] = result['label'].fillna(0).astype(int)

result.to_csv(os.path.join(RESULT_DIR,f'catboost_result.csv'), index=False)
print(result['label'].value_counts())
fea_imp_df = fea_imp_df.reset_index(drop = True)
fea_imp_df.to_csv(os.path.join(RESULT_DIR,f'./cat_fea_imp_{int(score*100000)}.csv'),index = False)

train_result_prob = pd.DataFrame(oof_prob).add_prefix('cat_class_')
test_result_prob = pd.DataFrame(test_prob).add_prefix('cat_class_')
train_result_prob['label'] = train['label']
train_result_prob['sn'] = train['sn']
train_result_prob['fault_time'] = train['fault_time']
test_result_prob['sn'] = test['sn']
test_result_prob['fault_time'] = test['fault_time']

result_prob = pd.concat([train_result_prob,test_result_prob],ignore_index = True)
result_prob.to_csv(os.path.join(RESULT_DIR,f'cat_prob_result.csv'),index = False)

end_time = datetime.datetime.now()
cost_time = end_time - start_time
print('****************** CATBOOST COST TIME : ',str(cost_time),' ******************')

'''
 
v7: 最优 线下 0.7303
v8: v7 添加 关键词交叉特征 并作为类别变量输入模型 0.73114

'''