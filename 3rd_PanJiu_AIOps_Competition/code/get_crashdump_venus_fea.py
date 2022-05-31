import datetime
import os
import gc
import warnings
import pandas as pd
import pickle
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from generate_feature import add_w2v_feats, cat2num
from generate_feature import get_key

from generate_feature import get_beta_target, add_last_next_time4fault, get_feature, \
    get_duration_minutes_fea, get_nearest_msg_fea, get_server_model_sn_fea_2, \
    get_server_model_fea, get_msg_text_fea_all, get_key_word_cross_fea, get_server_model_time_interval_stat_fea, \
    get_w2v_feats, get_key, get_class_key_words_nunique
from model import run_cbt, run_lgb
from utils import RESULT_DIR, TRAIN_DIR, \
    TEST_A_DIR, KEY_WORDS, TOP_KEY_WORDS, get_word_counter, search_weight, macro_f1, TIME_INTERVAL, PSEUDO_FALG, \
    GENERATION_DIR

warnings.filterwarnings('ignore')


def get_fault_code_list(x):
    try:
        x = x.replace('.', ',').split(',')
    except:
        x = []
    return x


def get_module_cause_list(x):
    try:
        x = x.replace(',', '_').replace('，', '_')
        x = list(set(x.split('_')))
    except:
        x = []
    return x


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


def get_module_cause_code(x, code_name):
    code_list = []
    for i in x:
        if code_name in i:
            code_list.append(i)
    return code_list


def get_alertname_code(x, alertname):
    x = x.split(',')

    try:
        alertname_code = x[x.index(alertname) + 1]
    except:
        alertname_code = np.nan
    return alertname_code


def get_alertname_code_2(x, alertname):
    # x =x.split(',')

    try:
        alertname_code = x[x.index(alertname) + 1]
    except:
        alertname_code = ' '
    return alertname_code


def get_last_msg_cnt(x):
    last_msg = x[-1]
    cnt = x.count(last_msg)
    return cnt


def get_first_msg_cnt(x):
    first_msg = x[0]
    cnt = x.count(first_msg)
    return cnt


def get_crashdump_venus_data():
    final_venus_dataset = pd.read_csv(os.path.join(TEST_A_DIR, 'final_venus_dataset_b.csv'))
    final_crashdump_dataset = pd.read_csv(os.path.join(TEST_A_DIR, 'final_crashdump_dataset_b.csv'))
    final_crashdump_venus = final_crashdump_dataset.merge(final_venus_dataset, on=['sn', 'fault_time'],
                                                          how='outer')

    preliminary_venus_dataset = pd.read_csv(os.path.join(TRAIN_DIR, 'preliminary_venus_dataset.csv'))
    preliminary_crashdump_dataset = pd.read_csv(os.path.join(TRAIN_DIR, 'preliminary_crashdump_dataset.csv'))
    preliminary_crashdump_venus = preliminary_crashdump_dataset.merge(preliminary_venus_dataset,
                                                                      on=['sn', 'fault_time'],
                                                                      how='outer')

    crashdump_venus = pd.concat([final_crashdump_venus, preliminary_crashdump_venus],
                                ignore_index=True).drop_duplicates()
    crashdump_venus = crashdump_venus.sort_values(['sn', 'fault_time']).reset_index(drop=True)
    return crashdump_venus


def get_crashdump_venus_fea(crashdump_venus):
    print('生成 crashdump_venus 特征')
    crashdump_venus['module_cause_list'] = crashdump_venus['module_cause'].apply(lambda x: get_module_cause_list(x))
    crashdump_venus['fault_code_list'] = crashdump_venus['fault_code'].apply(lambda x: get_fault_code_list(x))

    code_name_list = ['module', 'cod1', 'cod2', 'addr', 'port']
    for code_name in code_name_list:
        crashdump_venus[f'module_cause_{code_name}'] = crashdump_venus['module_cause_list'].apply(
            lambda x: get_module_cause_code(x, code_name))
        crashdump_venus[f'module_cause_{code_name}_len'] = crashdump_venus[f'module_cause_{code_name}'].apply(
            lambda x: len(x))
        crashdump_venus[f'module_cause_{code_name}'] = crashdump_venus[f'module_cause_{code_name}'].apply(
            lambda x: '_'.join(set(x)))
    code_name_list = ['cha', '0x', 'cod', 'core', 'cpu', 'm2m', 'pcu']
    for code_name in code_name_list:
        crashdump_venus[f'fault_{code_name}'] = crashdump_venus['fault_code_list'].apply(
            lambda x: get_module_cause_code(x, code_name))
        crashdump_venus[f'fault_{code_name}_len'] = crashdump_venus[f'fault_{code_name}'].apply(lambda x: len(x))
        crashdump_venus[f'fault_{code_name}'] = crashdump_venus[f'fault_{code_name}'].apply(lambda x: '_'.join(set(x)))

    cols_tmp = ['module_cause', 'fault_code', 'module_cause_module',
                'module_cause_cod1', 'module_cause_cod2', 'module_cause_addr',
                'module_cause_port', 'fault_cha', 'fault_0x', 'fault_cod', 'fault_core',
                'fault_cpu', 'fault_m2m', 'fault_pcu', ]
    new_cat_cols = []
    crashdump_venus = cat2num(crashdump_venus, cols_tmp)
    for name in cols_tmp:
        # le = LabelEncoder()
        # crashdump_venus[f'{name}_LabelEnc'] = le.fit_transform(crashdump_venus[name])
        new_cat_cols.append(f'{name}_LabelEnc')

    num_cols = ['fault_pcu_len', 'fault_m2m_len',
                'fault_cpu_len', 'fault_0x_len', 'fault_cod_len',
                'module_cause_module_len', 'module_cause_cod1_len',
                'module_cause_cod2_len', 'module_cause_addr_len',
                'module_cause_port_len', 'fault_cha_len', 'fault_core_len', ]

    crashdump_venus = crashdump_venus[['sn', 'fault_time'] + new_cat_cols + num_cols]
    crashdump_venus = crashdump_venus.rename(columns={'fault_time': 'crashdump_fault_time'})

    crashdump_venus['crashdump_fault_time'] = pd.to_datetime(crashdump_venus['crashdump_fault_time'])
    del crashdump_venus['crashdump_fault_time']
    print(f'生成 crashdump_venus 特征完毕,特征维度 {crashdump_venus.shape}')
    return crashdump_venus


def get_location_word(x, num):
    try:
        return x[num]
    except:
        return


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


module_list = ['module0','module1','module2','module3','module4','module5','module7','module8','module9',
             'module10','module11','module12','module13','module14','module17','module18','module19',
             'in traffic control',
             'irpp0','irpp1',
             'pcie rootport 0:0.0','pcie rootport a2:0.0','pcie rootport 2b:3.0',
             'port a','port c']
module_list2 = ['module0','module1','module2','module3','module4','module5','module7','module8','module9',
'module10','module11','module12','module13','module14','module17','module18','module19']
other_module_list = ['in traffic control', 'irpp0', 'irpp1', 'pcie rootport 0:0.0',
       'pcie rootport a2:0.0', 'pcie rootport 2b:3.0', 'port a', 'port c']
module_content_list = ['module0_cod1', 'module0_cod2', 'module0_addr',
       'module1_cod1', 'module1_cod2', 'module1_addr', 'module2_cod1',
       'module2_cod2', 'module2_addr', 'module3_cod1', 'module3_cod2',
       'module3_addr', 'module4_cod1', 'module4_cod2', 'module4_addr',
       'module5_cod1', 'module5_cod2', 'module5_addr', 'module7_cod1',
       'module7_cod2', 'module7_addr', 'module8_cod1', 'module8_cod2',
       'module8_addr', 'module9_cod1', 'module9_cod2', 'module9_addr',
       'module10_cod1', 'module10_cod2', 'module10_addr', 'module11_cod1',
       'module11_cod2', 'module11_addr', 'module12_cod1', 'module12_cod2',
       'module12_addr', 'module13_cod1', 'module13_cod2', 'module13_addr',
       'module14_cod1', 'module14_cod2', 'module14_addr', 'module17_cod1',
       'module17_cod2', 'module17_addr', 'module18_cod1', 'module18_cod2',
       'module18_addr', 'module19_cod1', 'module19_cod2', 'module19_addr']
fault_code_content_list = ['fault_code_cod1', 'fault_code_cod2',
       'fault_code_cpu0', 'fault_code_cpu1']


crashdump_venus = get_crashdump_venus_data()
crashdump_venus['module_cause_list'] = crashdump_venus['module_cause'].fillna('_').apply(lambda x:x.split(','))
crashdump_venus['module_cause'] = crashdump_venus['module_cause'].fillna('_').apply(lambda x:x.replace(':','_').replace(',','_'))
for module in module_list:
    crashdump_venus['module_cause'] = crashdump_venus['module_cause'].fillna('_').apply(
    lambda x:x.replace(f'{module}_',f'{module}:').replace(f'_{module}',f',{module}'))
crashdump_venus['module_cause'] = crashdump_venus['module_cause'].apply(lambda x:x.replace(':',','))

for module in module_list:
    crashdump_venus[module] = crashdump_venus['module_cause'].apply(lambda x:get_alertname_code(x,module))
    crashdump_venus[module] = crashdump_venus.loc[:,module].fillna(' ').apply(lambda x:x.replace('_',' '))
    crashdump_venus[module] = crashdump_venus[module].apply(lambda x:x.split(' '))
crashdump_venus['module_cause_new'] = crashdump_venus.loc[:,module_list].sum(1)


for module in module_list2:
    crashdump_venus[f'{module}_cod1'] = crashdump_venus[module].apply(lambda x:[get_alertname_code_2(x,'cod1')])
    crashdump_venus[f'{module}_cod2'] = crashdump_venus[module].apply(lambda x:[get_alertname_code_2(x,'cod2')])
    crashdump_venus[f'{module}_addr'] = crashdump_venus[module].apply(lambda x:[get_alertname_code_2(x,'addr')])
    del crashdump_venus[module]
    gc.collect()

crashdump_venus['fault_code_list'] = crashdump_venus['fault_code'].fillna(' ').apply(lambda x:x.split('.'))
for i in ['cod1','cod2','cpu0','cpu1']:
    crashdump_venus[f'fault_code_{i}'] = crashdump_venus['fault_code_list'].apply(lambda x:[get_alertname_code_2(x,i)])


crashdump_venus['other_module_list'] = crashdump_venus.loc[:,other_module_list].sum(1)
crashdump_venus['module_content_list'] = crashdump_venus.loc[:,module_content_list].sum(1)
crashdump_venus['module_cause_new'] = crashdump_venus.loc[:,other_module_list+module_content_list].sum(1)
crashdump_venus['fault_code_content_list'] = crashdump_venus.loc[:,fault_code_content_list].sum(1)
crashdump_venus['all_crashdump_venus'] = crashdump_venus.loc[:,other_module_list+module_content_list+fault_code_content_list].sum(1)

f1_list = ['sn']
f2_list = ['other_module_list','module_content_list','module_cause_new','fault_code_content_list','all_crashdump_venus']
w2v_feats_df = crashdump_venus[f1_list].drop_duplicates()
w2v_feats_df_list = []
for f1 in f1_list:
    for f2 in f2_list:
        w2v_fea_tmp = add_w2v_feats(crashdump_venus,w2v_feats_df,f1,f2,emb_size = 10,window = 5,min_count  =5,)
        w2v_feats_df_list.append(w2v_fea_tmp)
w2v_feats_df = w2v_feats_df_list[0]
for i in w2v_feats_df_list[1:]:
    w2v_feats_df = w2v_feats_df.merge(i,on = 'sn',how = 'left')

for i in other_module_list+module_content_list+fault_code_content_list:
    crashdump_venus[i] = crashdump_venus[i].astype(str)

crashdump_venus = cat2num(crashdump_venus,other_module_list+module_content_list+fault_code_content_list)
for i in other_module_list+module_content_list+fault_code_content_list:
    del crashdump_venus[i]
gc.collect()
crashdump_venus = crashdump_venus.merge(w2v_feats_df,on ='sn',how ='left').rename(columns ={'fault_time':'crashdump_venus_fault_time'} )

preliminary_train_label_dataset_path = os.path.join(TRAIN_DIR, 'preliminary_train_label_dataset.csv')
preliminary_train_label_dataset_s_path = os.path.join(TRAIN_DIR, 'preliminary_train_label_dataset_s.csv')
test = pd.read_csv(os.path.join(TEST_A_DIR, 'final_submit_dataset_b.csv'))[['sn', 'fault_time' ]]
train = get_label(False)[['sn', 'fault_time', 'label',]]

test_tmp = test[['sn', 'fault_time']]
test_tmp = test_tmp.merge(crashdump_venus, on='sn').drop_duplicates(['sn', 'fault_time']).reset_index(drop=True)
train_tmp = train[['sn', 'fault_time', 'label', ]]
train_tmp = train_tmp.merge(crashdump_venus, on='sn').drop_duplicates(['sn', 'fault_time']).reset_index(drop=True)


train_tmp['duration_fault_time'] = pd.to_datetime(train_tmp['fault_time']) - pd.to_datetime(train_tmp['crashdump_venus_fault_time'])
test_tmp['duration_fault_time'] = pd.to_datetime(test_tmp['fault_time']) - pd.to_datetime(test_tmp['crashdump_venus_fault_time'])

train_tmp['duration_fault_time'] = train_tmp['duration_fault_time'].apply(lambda x:x.total_seconds())
test_tmp['duration_fault_time']  = test_tmp['duration_fault_time'].apply(lambda x:x.total_seconds())


drop_cols = ['sn', 'fault_time', 'fault_code', 'module_cause', 'module','crashdump_venus_fault_time',
       'module_cause_list', 'module_cause_new', 'fault_code_list','label','duration_fault_time',
       'other_module_list', 'module_content_list', 'fault_code_content_list',
       'all_crashdump_venus',]
use_cols = [i for i in train_tmp.columns if i not in drop_cols]

cat_cols = [f'{i}_LabelEnc' for i in other_module_list+module_content_list+fault_code_content_list]

oof_prob = np.zeros((train.shape[0], 4))

test_prob = np.zeros((test.shape[0], 4))
# seeds = [42,4242,40424,1024,2048]
seeds = [42 ]
for seed in seeds:
    oof_prob, test_prob, fea_imp_df, model_list = run_cbt(train_tmp[use_cols], train_tmp[['label']], test_tmp[use_cols], k=5,
                                              seed=seed, cat_cols=cat_cols)
    oof_prob +=oof_prob/len(seeds)
    test_prob +=test_prob/len(seeds)


weight = search_weight(train_tmp, train_tmp[['label']], oof_prob, init_weight=[1.0], class_num=4, step=0.001)
oof_prob = oof_prob * np.array(weight)
test_prob = test_prob * np.array(weight)


target_df = train_tmp[['sn', 'fault_time', 'label']].drop_duplicates(['sn', 'fault_time'])
submit_df = train_tmp[['sn', 'fault_time']]
submit_df['label'] = oof_prob.argmax(axis=1)
submit_df = submit_df.drop_duplicates(['sn', 'fault_time'])
# submit_df = pd.read_csv(os.path.join(GENERATION_DIR,'crashdump_venus_fea1.csv')).rename(columns = {'crashdump_venus_label':'label'})


score = macro_f1(target_df=target_df, submit_df=submit_df)
print(f'********************** BEST MACRO_F1 : {score} **********************')
score = round(score, 5)

print(fea_imp_df[:20])
y_pred = test_prob.argmax(axis=1)
result = test_tmp[['sn', 'fault_time']]
result['label'] = y_pred
result = result.drop_duplicates(['sn', 'fault_time'])

crashdump_venus_fea = pd.concat([submit_df,result],ignore_index = False,axis = 0)
crashdump_venus_fea = crashdump_venus_fea.rename(columns = {'label':'crashdump_venus_label_v1'})
crashdump_venus_fea.to_csv(os.path.join(GENERATION_DIR,'crashdump_venus_fea_v1.csv'),index= False)
print(crashdump_venus_fea['crashdump_venus_label_v1'].value_counts())

