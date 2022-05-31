import datetime
import os
import pickle
from collections import Counter
from utils import get_new_cols
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from utils import GENERATION_DIR
from utils import KEY_1, KEY_2, KEY_3, KEY_4
from tqdm import tqdm
from scipy import stats

def cat2num(df, cat_cols, Transfer2num=True):
    '''

    :param df:
    :param cat_cols: 类别特征列表
    :param Transfer2num: 类别特征转换为数值特征
    :return:
    '''
    if Transfer2num:

        print('Transfer category feature to  num feature ')
        for col in cat_cols:

            if not os.path.exists(os.path.join(GENERATION_DIR, f'{col}_map.pkl')):
                print(f'Transfer : {col}')
                tmp_map = dict(zip(df[col].unique(), range(df[col].nunique())))
                with open(os.path.join(GENERATION_DIR, f'{col}_map.pkl'), 'wb') as f:
                    pickle.dump(tmp_map, f)
            else:
                with open(os.path.join(GENERATION_DIR, f'{col}_map.pkl'), 'rb') as f:
                    tmp_map = pickle.load(f)
            df[f'{col}_LabelEnc'] = df[col].map(tmp_map).fillna(-1).astype(int)
    else:
        print('Transfer category feature to  category feature ')
        for col in cat_cols:
            df[col] = df[col].astype('category')
    print('Transfer category feature to  num feature  Down...')
    return df

def add_minutes(x, minutes=5):
    dt = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    out_date = (dt + datetime.timedelta(minutes=minutes)
                ).strftime('%Y-%m-%d %H:%M:%S')
    return out_date


def time_process(df, time_cols, minutes_):
    df[f'time_{minutes_}'] = df[time_cols].apply(
        lambda x: add_minutes(str(x), minutes_))
    return df


def get_fea(x, fea):
    if fea in x:
        return 1
    else:
        return 0


def get_last_msg_cnt(x):
    last_msg = x[-1]
    cnt = x.count(last_msg)
    return cnt


def get_first_msg_cnt(x):
    first_msg = x[0]
    cnt = x.count(first_msg)
    return cnt


def add_last_next_time4fault(label, preliminary_submit_dataset_a,
                             time_interval, next_time_list):
    print(f'添加自定义异常出现的时间间隔{time_interval}的前后的时间点')
    for i in tqdm([-i for i in next_time_list] + next_time_list):
        label = time_process(label, 'fault_time', i * time_interval)
        preliminary_submit_dataset_a = time_process(
            preliminary_submit_dataset_a, 'fault_time', i * time_interval)

    return label, preliminary_submit_dataset_a


def get_msg_text_fea(df, msg_type='last'):
    print(f'获取 msg text {msg_type}特征')

    df_fea = df.groupby(['sn', 'fault_time']).agg(
        {'msg_list': 'sum', 'msg_0': 'sum', 'msg_1': 'sum', 'msg_2': 'sum'}).reset_index()
    df_fea['msg_list_unique'] = df_fea['msg_list'].apply(lambda x: str(set(x)))
    df_fea['msg_0_unique'] = df_fea['msg_0'].apply(lambda x: str(set(x)))
    df_fea['msg_1_unique'] = df_fea['msg_1'].apply(lambda x: str(set(x)))
    df_fea['msg_2_unique'] = df_fea['msg_2'].apply(lambda x: str(set(x)))

    df_fea['msg_list_list'] = df_fea['msg_list'].apply(lambda x: str(x))
    df_fea['msg_0_list'] = df_fea['msg_0'].apply(lambda x: str(x))
    df_fea['msg_1_list'] = df_fea['msg_1'].apply(lambda x: str(x))
    df_fea['msg_2_list'] = df_fea['msg_2'].apply(lambda x: str(x))

    df_fea['msg_0_first'] = df_fea['msg_0'].apply(lambda x: x[0])
    df_fea['msg_1_first'] = df_fea['msg_1'].apply(lambda x: x[0])
    df_fea['msg_2_first'] = df_fea['msg_2'].apply(lambda x: x[0])

    df_fea['msg_0_last'] = df_fea['msg_0'].apply(lambda x: x[-1])
    df_fea['msg_1_last'] = df_fea['msg_1'].apply(lambda x: x[-1])
    df_fea['msg_2_last'] = df_fea['msg_2'].apply(lambda x: x[-1])

    df_fea['msg_last'] = df.groupby(['sn', 'fault_time']).apply(
        lambda x: x['msg'].to_list()[-1]).values
    df_fea['msg_first'] = df.groupby(['sn', 'fault_time']).apply(
        lambda x: x['msg'].to_list()[0]).values

    df_fea['last_msg_cnt'] = df_fea['msg_list'].apply(
        lambda x: get_last_msg_cnt(x))
    df_fea['first_msg_cnt'] = df_fea['msg_list'].apply(
        lambda x: get_first_msg_cnt(x))
    cat_cols = ['msg_list', 'msg_0', 'msg_1', 'msg_2',
                'msg_list_unique', 'msg_0_unique', 'msg_1_unique', 'msg_2_unique',
                'msg_list_list', 'msg_0_list', 'msg_1_list', 'msg_2_list',
                'msg_0_first', 'msg_1_first', 'msg_2_first', 'msg_0_last', 'msg_1_last',
                'msg_2_last', 'msg_last', 'msg_first']
    num_cols = ['last_msg_cnt', 'first_msg_cnt']
    id_cols = ['sn', 'fault_time']

    df_fea = df_fea.rename(
        columns={
            i: f'{msg_type}_{i}' for i in (cat_cols + num_cols)})
    cat_cols = [f'{msg_type}_{i}' for i in cat_cols]
    for cat_col in cat_cols:
        df_fea[cat_col] = df_fea[cat_col].astype(str)
    df_fea = cat2num(df_fea, cat_cols, Transfer2num=True)
    for i in cat_cols:
        del df_fea[i]
    return df_fea

def add_w2v_feats(all_data,w2v_feats_df,f1,f2,emb_size = 32,window = 5,min_count  =5,):
    print(f'生成 {f1}_{f2}_w2v 特征')

    df_fea = all_data.groupby(f1).agg({f2:'sum'}).reset_index()
    df_emb = df_fea[[f1 ]]
    sencences = df_fea[f2].to_list()
    if not os.path.exists(os.path.join(GENERATION_DIR, f'{f1}_{f2}_w2v_model.pkl')):
        print(f'{f1}_{f2}_w2v_model 不存在，开始训练......')
        model = Word2Vec(sencences, vector_size=emb_size, window=window,
                         min_count=min_count, sg=0, hs=1, seed=42)
        with open(os.path.join(GENERATION_DIR, f'{f1}_{f2}_w2v_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
    else:
        print(f'{f1}_{f2}_w2v_model 已存在，开始读取......')
        with open(os.path.join(GENERATION_DIR, f'{f1}_{f2}_w2v_model.pkl'), 'rb') as f:
            model = pickle.load(f)

    emb_matrix_mean = []
    for sent in sencences:
        vec = []
        for w in sent:
            if w in model.wv:
                vec.append(model.wv[w])
        if len(vec) >0:
            emb_matrix_mean.append(np.mean(vec,axis = 0))
        else:
            emb_matrix_mean.append([0]*emb_size)
    df_emb_mean = pd.DataFrame(emb_matrix_mean).add_prefix(f'{f1}_{f2}_w2v_')

    df_emb = pd.concat([df_emb,df_emb_mean],axis = 1)
    w2v_feats_df = w2v_feats_df.merge(df_emb,on = f1,how ='left')
    return w2v_feats_df
def get_w2v_feats(all_data,f1_list,f2_list):
    all_data['msg_list'] = all_data['msg'].apply(lambda x: [i.strip() for i in x.split(' | ')])
    all_data['msg_0'] = all_data['msg'].apply(lambda x: [get_msg_location(x.split(' | '), 0)])
    all_data['msg_1'] = all_data['msg'].apply(lambda x: [get_msg_location(x.split(' | '), 1)])
    all_data['msg_2'] = all_data['msg'].apply(lambda x: [get_msg_location(x.split(' | '), 2)])
    w2v_feats_df = all_data[f1_list].drop_duplicates()
    for f1 in f1_list:
        for f2 in f2_list:
            w2v_feats_df = add_w2v_feats(all_data,w2v_feats_df,f1,f2,emb_size = 10,window = 5,min_count  =5,)
    print(f'w2v_feats 的特征维度: {w2v_feats_df.shape}')
    return w2v_feats_df



def get_time_diff_feats_v2(all_data):
    print('生成时间差特征 time_diff_feats_v2')
    all_data['duration_seconds'] = all_data['time_interval']
    all_data['duration_minutes'] = all_data['time_interval'] / 60
    df_merge_log = all_data[['sn', 'fault_time', 'label', 'time', 'msg',
                             'server_model', 'time_interval', 'duration_seconds',
                             'duration_minutes']]
    df_merge_log['fault_id'] = df_merge_log['sn'] + '_' + df_merge_log['fault_time'] + '_' + df_merge_log[
        'server_model']
    f1_list = ['fault_id', 'sn', 'server_model']
    f2_list = ['duration_minutes', 'duration_seconds']
    time_diff_feats_v2 = df_merge_log[['sn', 'fault_time', 'fault_id', 'server_model']].drop_duplicates().reset_index(
        drop=True)

    for f1 in f1_list:
        for f2 in f2_list:
            func_opt = ['count', 'nunique', 'min', 'max', 'median', 'sum']
            for opt in func_opt:
                tmp = df_merge_log.groupby([f1])[f2].agg([(f'{f2}_in_{f1}_' + opt, opt)]).reset_index()
                # print(f'{f1}_in_{f2}_{opt}:{tmp.shape}' )
                time_diff_feats_v2 = time_diff_feats_v2.merge(tmp, on=f1, how='left')

            temp = df_merge_log.groupby([f1])[f2].apply(lambda x: stats.mode(x)[0][0])
            time_diff_feats_v2[f'{f2}_in_{f1}_mode'] = time_diff_feats_v2[f1].map(temp).fillna(np.nan)
            secs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for sec in secs:
                temp = df_merge_log.groupby([f1])[f2].quantile(sec).reset_index(
                    name=f'log_{f2}_in_{f1}_quantile_' + str(sec * 100))
                # print(f'log_{f1}_in_{f2}_quantile_{str(sec * 100)}:{tmp.shape}' )
                time_diff_feats_v2 = pd.merge(time_diff_feats_v2, temp, on=f1, how='left')
    del time_diff_feats_v2['fault_id']
    return time_diff_feats_v2

def get_feature(data, time_list, log_fea, fea_num, key):
    print(f'当前特征维度{data.shape}')
    fea_df_list = []
    fea_cnt_list = ['OEM record c2', 'Processor CPU_Core_Error', '001c4c', 'System Event Sys_Event','OEM CPU0 MCERR',
                    'OEM CPU0 CATERR', 'Reading 0 &lt; Threshold 2 degrees C', '0203c0a80101',
                    'Unknown CPU0 MCERR', 'Unknown CPU0 CATERR','Memory', 'Correctable ECC logging limit reached',
                    'Memory MEM_CHE0_Status', 'Memory Memory_Status',  'Memory #0x87', 'Memory CPU0F0_DIMM_Stat',
                    'Drive Fault', 'NMI/Diag Interrupt', 'Failure detected',  'Power Supply AC lost', ]
    for time_tmp in tqdm(time_list):
        print(f'获取异常前后 {time_tmp} min的数据进行聚合')
        tmp1 = data[(pd.to_datetime(data['time']) < pd.to_datetime(data[f'time_{time_tmp}'])) & (pd.to_datetime(data['time']) > pd.to_datetime(data[f'time_-{time_tmp}']))].sort_values(
            ['sn', 'fault_time'])
        tmp1 = tmp1.groupby(key).apply(
            lambda x: ' | '.join(x['msg'].to_list())).reset_index().rename(columns={0: 'msg'})
        tmp1[f'msg_len'] = tmp1['msg'].apply(lambda x: len(x.split(' | ')))
        #         tmp1[f'msg_len_two'] = tmp1['msg'].apply(lambda x: len(x))
        # 添加数字个数
        # tmp1[f'msg_num_two'] = tmp1['msg'].apply(
        #     lambda x: len([int(s) for s in re.findall(r'\b\d+\b', x)]))
        print(f'根据异常前后 {time_tmp} min的数据的日志数据提取 {fea_num} 个稀疏特征')
        feature = log_fea + ['msg_len']
        for fea in feature:
            tmp1[fea] = tmp1['msg'].apply(lambda x: get_fea(x, fea))
            # 添加计数特征
            if fea in fea_cnt_list:
                tmp1[f'{fea}_cnt'] = tmp1['msg'].apply(lambda x:x.replace('|',' ').replace('_',' ').split(' ').count(fea))
                feature.append(f'{fea}_cnt')
        tmp1_new_col_map = {i: i + '_' + str(int(time_tmp)) for i in feature}
        tmp1 = tmp1.rename(columns=tmp1_new_col_map)
        del tmp1['msg']
        fea_df_list.append(tmp1)
    fea_df = fea_df_list[-1]
    print(fea_df.shape)
    for i in fea_df_list[:-1]:
        fea_df = fea_df.merge(i, on=key, how='left')
        print(fea_df.shape)
    return fea_df


def get_msg_location(x, num):
    try:
        return x[num]
    except BaseException:
        return '其它'


def get_nearest_msg_fea(train, test):
    print('生成 nearest_msg 特征')
    df = pd.concat([train, test], axis=0, ignore_index=True)
    df['duration_minutes'] = (pd.to_datetime(df['fault_time']) - pd.to_datetime(df['time'])).apply(
        lambda x: x.total_seconds())
    df = df.sort_values(
        ['sn', 'server_model', 'fault_time', 'time']).reset_index(drop=True)
    df['duration_minutes_abs'] = np.abs(df['duration_minutes'])

    df['duration_minutes_abs_rank'] = df.groupby(['sn', 'server_model', 'fault_time'])['duration_minutes_abs'].rank(
        method='first', ascending=False)

    key = ['sn', 'server_model', 'fault_time', 'duration_minutes_abs']
    df = df.sort_values(key, ascending=False)
    df = df.drop_duplicates(
        ['sn', 'server_model', 'fault_time', ], keep='first')

    df.loc[df['duration_minutes'] ==
           df['duration_minutes_abs'], 'last_or_next'] = 1
    df.loc[df['duration_minutes'] !=
           df['duration_minutes_abs'], 'last_or_next'] = 0
    df['msg_cnt'] = df['msg'].map(df['msg'].value_counts())
    df['msg_0'] = df['msg'].apply(
        lambda x: get_msg_location(
            x.split(' | '), 0))
    df['msg_0_cnt'] = df['msg_0'].map(df['msg_0'].value_counts())
    df['msg_1'] = df['msg'].apply(
        lambda x: get_msg_location(
            x.split(' | '), 1))
    df['msg_1_cnt'] = df['msg_1'].map(df['msg_1'].value_counts())
    df['msg_2'] = df['msg'].apply(
        lambda x: get_msg_location(
            x.split(' | '), 2))
    df['msg_2_cnt'] = df['msg_2'].map(df['msg_2'].value_counts())
    cat_feats = ['msg', 'msg_0', 'msg_1',
                 'msg_2']  # ,'server_model_day_date','server_model_dayofmonth','server_model_dayofweek','server_model_hour']
    # for name in cat_feats:
    #     le = LabelEncoder()
    #     df[f'{name}_LabelEnc'] = le.fit_transform(df[name])
    df = cat2num(df,cat_feats)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[['sn', 'server_model', 'fault_time', 'msg_cnt',
             'msg_0_cnt', 'msg_1_cnt', 'msg_2_cnt',
             #              'duration_minutes_abs','duration_minutes', 'duration_minutes_abs_rank',
             'last_or_next', 'msg_LabelEnc', 'msg_0_LabelEnc', 'msg_1_LabelEnc', 'msg_2_LabelEnc']]
    print(f'生成 nearest_msg 特征完毕，特征维度{df.shape}')
    return df

def get_server_model_time_interval_stat_fea(all_data):
    server_model_time_interval_stat_fea = all_data.groupby('server_model').agg({'time_interval':['min','max','mean','median']}).reset_index()
    server_model_time_interval_stat_fea = get_new_cols(server_model_time_interval_stat_fea,key = ['server_model' ])

    server_model_time_interval_stat_fea.columns  = ['server_model', 'sm_time_interval_min', 'sm_ttime_interval_max',
           'sm_ttime_interval_mean', 'sm_ttime_interval_median']
    return server_model_time_interval_stat_fea

def get_server_model_sn_fea_2(train, test):
    df = pd.concat([train[['sn', 'server_model']],
                   test[['sn', 'server_model']]], ignore_index=True)
    df['server_model_count_sn_2'] = df.groupby(
        ['server_model'])['sn'].transform('count')
    df['server_model_nunique_sn_2'] = df.groupby(
        ['server_model'])['sn'].transform('nunique')
    df['sn_cnt_2'] = df['sn'].map(df['sn'].value_counts())
    return df.drop_duplicates().reset_index(drop=True)


def get_4_time_stat_fea(df):
    print('     生成时间统计特征')
    time_stat_fea_df = df.groupby(['sn', 'fault_time', 'server_model']).agg(
        {'duration_minutes': ['min', 'max', 'mean', 'median', 'skew', 'sum', 'std', 'count'],
         'log_duration_minutes': ['min', 'max', 'mean', 'median', 'skew', 'sum', 'std'],
         'time_diff_1': ['min', 'max', 'mean', 'median', 'skew', 'sum', 'std'],
         'log_time_diff_1': ['min', 'max', 'median'],
         }).reset_index()
    new_time_stat_cols = []
    for i in time_stat_fea_df.columns:
        if i[0] in ['sn', 'fault_time', 'server_model']:
            new_time_stat_cols.append(i[0])
        else:
            new_time_stat_cols.append(f'{i[0]}_{i[1]}')
            #             print(f'{i[0]}_{i[1]}')
            time_stat_fea_df.loc[time_stat_fea_df[i[0]]
                                 [i[1]] == -np.inf, (i[0], i[1])] = -20
            time_stat_fea_df.loc[time_stat_fea_df[i[0]]
                                 [i[1]] == np.inf, (i[0], i[1])] = 30
    time_stat_fea_df.columns = new_time_stat_cols
    time_stat_fea_df['duration_minutes_range'] = time_stat_fea_df['duration_minutes_max'] - time_stat_fea_df[
        'duration_minutes_min']
    time_stat_fea_df['log_duration_minutes_range'] = time_stat_fea_df['log_duration_minutes_max'] - time_stat_fea_df[
        'log_duration_minutes_min']
    time_stat_fea_df['time_diff_1_range'] = time_stat_fea_df['time_diff_1_max'] - \
        time_stat_fea_df['time_diff_1_min']
    time_stat_fea_df['log_time_diff_1_range'] = time_stat_fea_df['log_time_diff_1_max'] - time_stat_fea_df[
        'log_time_diff_1_min']
    time_stat_fea_df['duration_minutes_freq'] = time_stat_fea_df['duration_minutes_range'] / time_stat_fea_df[
        'duration_minutes_count']
    print(f'    生成时间统计特征完毕，特征维度:{time_stat_fea_df.shape}')
    return time_stat_fea_df


def get_time_std_fea(train, test):
    print('生成 server_model 特征')
    df = pd.concat([train, test], axis=0, ignore_index=True)
    # df['year'] = df['time'].dt.year
    # df['month'] = df['time'].dt.month
    df['hour'] = df['time'].dt.hour
    # df['week'] = df['time'].dt.week
    df['minute'] = df['time'].dt.minute
    time_std = df.groupby(['sn', 'server_model']).agg(
        {'hour': 'std', 'minute': 'std'}).reset_index()
    time_std = time_std.rename(
        columns={
            'hour': 'hour_std',
            'minute': 'minute_std'})
    return time_std


def get_key(all_data):
    all_data['msg_list'] = all_data['msg'].apply(lambda x: [i.strip() for i in x.split(' | ')])
    class_fea_cnt_list = []
    for label in [0,1,2,3]:
        class_df = all_data.query(f'label =={label}')
        counter = Counter()
        for i in class_df['msg_list']:
            counter.update(i)
        class_fea_cnt = pd.DataFrame({i[0]:i[1] for i in counter.most_common()},index = [f'fea_cnt_{label}']).T.reset_index().rename(columns = {'index':'fea'})
        class_fea_cnt_list.append(class_fea_cnt)

    fea_cnt_df = class_fea_cnt_list[0]
    for tmp in class_fea_cnt_list[1:]:
        fea_cnt_df = fea_cnt_df.merge(tmp,on = 'fea')

    fea_cnt_df['fea_cnt_sum'] = fea_cnt_df.loc[:,['fea_cnt_0', 'fea_cnt_1', 'fea_cnt_2', 'fea_cnt_3']].sum(1)

    all_fea_cnt = fea_cnt_df['fea_cnt_sum'].sum()

    for i in ['fea_cnt_0', 'fea_cnt_1', 'fea_cnt_2', 'fea_cnt_3']:
        fea_cnt_df[f'{i}_ratio'] = fea_cnt_df[i]/fea_cnt_df['fea_cnt_sum']
        fea_cnt_df[f'{i}_all_ratio'] = fea_cnt_df[i]/all_fea_cnt

    fea_cnt_df['fea_cnt_ratio_std'] = fea_cnt_df.loc[:,['fea_cnt_0_ratio','fea_cnt_1_ratio','fea_cnt_2_ratio','fea_cnt_3_ratio', ]].std(1)
    fea_cnt_df['fea_cnt_std'] = fea_cnt_df.loc[:,['fea_cnt_0', 'fea_cnt_1','fea_cnt_2','fea_cnt_3',]].std(1)

    fea_cnt_df['fea_cnt_all_ratio_std'] = fea_cnt_df.loc[:,['fea_cnt_0_all_ratio','fea_cnt_1_all_ratio',
           'fea_cnt_2_all_ratio','fea_cnt_3_all_ratio',]].std(1)

    fea_cnt_df = fea_cnt_df[~fea_cnt_df['fea_cnt_ratio_std'].isnull()].sort_values('fea_cnt_ratio_std',ascending = False)

    fea_cnt_df['fea_max'] = np.argmax(fea_cnt_df.loc[:,['fea_cnt_0', 'fea_cnt_1', 'fea_cnt_2', 'fea_cnt_3',]].values,axis = 1)
    key_0 = fea_cnt_df.query('fea_max ==0 ')['fea'].to_list()
    key_1 = fea_cnt_df.query('fea_max ==1 ')['fea'].to_list()
    key_2 = fea_cnt_df.query('fea_max ==2 ')['fea'].to_list()
    key_3 = fea_cnt_df.query('fea_max ==3 ')['fea'].to_list()
    # key_1 = ['OEM record c2','Processor CPU_Core_Error','001c4c','System Event Sys_Event','Power Supply PS0_Status','Temperature CPU0_Margin_Temp','Reading 51 &gt; Threshold 85 degrees C','Lower Non-critical going low','Temperature CPU1_Margin_Temp','System ACPI Power State #0x7d','Lower Critical going low']
    # key_2 = ['OEM CPU0 MCERR','OEM CPU0 CATERR','Reading 0 &lt; Threshold 2 degrees C','0203c0a80101','Unknown CPU0 MCERR','Unknown CPU0 CATERR','Microcontroller #0x3b','System Boot Initiated','Processor #0xfa','Power Unit Pwr Unit Status','Hard reset','Power off/down','System Event #0xff','Memory CPU1A1_DIMM_Stat','000000','Power cycle','OEM record c3','Memory CPU1C0_DIMM_Stat','Reading 0 &lt; Threshold 1 degrees C','IERR']
    # key_3 = ['Memory','Correctable ECC logging limit reached','Memory MEM_CHE0_Status','Memory Memory_Status','Memory #0x87','Memory CPU0F0_DIMM_Stat','Memory Device Disabled','Memory #0xe2','OS Stop/Shutdown OS Status','System Boot Initiated System Restart','OS Boot BIOS_Boot_Up','System Boot Initiated BIOS_Boot_UP','Memory DIMM101','OS graceful shutdown','OS Critical Stop OS Status','Memory #0xf9','Memory CPU0C0_DIMM_Stat','Memory DIMM111','Memory DIMM021',]
    # key_4 = ['Drive Fault','NMI/Diag Interrupt','Failure detected','Power Supply AC lost','Power Supply PSU0_Supply','AC out-of-range, but present','Predictive failure','Drive Present','Temperature Temp_DIMM_KLM','Temperature Temp_DIMM_DEF','Power Supply PS1_Status','Identify Status','Power Supply PS2_Status','Temperature DIMMG1_Temp','Upper Non-critical going high','Temperature DIMMG0_Temp','Upper Critical going high','Power Button pressed','System Boot Initiated #0xb8','Deasserted']
    return key_0,key_1,key_2,key_3

def get_class_key_words_nunique(all_data):
    print('获取 class_key_words_nunique 特征')

    key_0,key_1,key_2,key_3 = get_key(all_data)

    df = all_data[['sn', 'fault_time', 'msg_list']]
    df_tmp = df.groupby(['sn' ]).agg({'msg_list':'sum'}).reset_index()
    df_tmp['class_0_key_words_nunique'] = df_tmp['msg_list'].apply(lambda x:len(set(x)&set(key_0)))
    df_tmp['class_1_key_words_nunique'] = df_tmp['msg_list'].apply(lambda x:len(set(x)&set(key_1)))
    df_tmp['class_2_key_words_nunique'] = df_tmp['msg_list'].apply(lambda x:len(set(x)&set(key_2)))
    df_tmp['class_3_key_words_nunique'] = df_tmp['msg_list'].apply(lambda x:len(set(x)&set(key_3)))
    del df_tmp['msg_list']
    return df_tmp
def get_key_for_top_fea(train,test):
    KEY_FOR_TOP_COLS = []
    print('添加 key_for_top_fea 特征')
    for TIME in [3, 5, 10, 15, 30, 45, 60, 90, 120, 240, 360, 480, 540, 600,60000000]:
        for i in range(10):
            train[f'KEY_FOR_TOP_{i}_{TIME}'] = train[f'{KEY_1[i]}_{TIME}'].astype(str)+'_'+train[f'{KEY_2[i]}_{TIME}'].astype(str)+'_'+train[f'{KEY_3[i]}_{TIME}'].astype(str)+'_'+train[f'{KEY_4[i]}_{TIME}'].astype(str)
            test[f'KEY_FOR_TOP_{i}_{TIME}'] = test[f'{KEY_1[i]}_{TIME}'].astype(str)+'_'+test[f'{KEY_2[i]}_{TIME}'].astype(str)+'_'+test[f'{KEY_3[i]}_{TIME}'].astype(str)+'_'+test[f'{KEY_4[i]}_{TIME}'].astype(str)
            KEY_FOR_TOP_COLS.append(f'KEY_FOR_TOP_{i}_{TIME}')
    train = cat2num(train,KEY_FOR_TOP_COLS)
    test = cat2num(test,KEY_FOR_TOP_COLS)
    for KEY_FOR_TOP_COL in KEY_FOR_TOP_COLS:
        del train[KEY_FOR_TOP_COL]
        del test[KEY_FOR_TOP_COL]
    return train,test

def get_key_word_cross_fea(train,test):
    print('获取关键词交叉特征......')
    KEY_WORDS_MAP  = {'CPU0':KEY_1,'CPU1':KEY_2,'CPU2':KEY_3,'CPU3':KEY_4}
    KEY_WORDS_CROSS_COLS =[]
    for KEY_WORDS in KEY_WORDS_MAP:
        for i in [3, 5, 10, 15, 30, 45, 60, 90, 120, 240, 360, 480, 540, 600,60000000]:
            KEY_WORDS_COLS = [f'{col}_{i}' for col in KEY_WORDS_MAP[KEY_WORDS]]
            train[f'{KEY_WORDS}_WORDS_{i}'] = train[KEY_WORDS_COLS].astype(str).sum(1)
            test[f'{KEY_WORDS}_WORDS_{i}'] = test[KEY_WORDS_COLS].astype(str).sum(1)
            KEY_WORDS_CROSS_COLS.append(f'{KEY_WORDS}_WORDS_{i}')
    train = cat2num(train,KEY_WORDS_CROSS_COLS)
    test = cat2num(test,KEY_WORDS_CROSS_COLS)

    for COLS in KEY_WORDS_CROSS_COLS:
        del train[COLS]
        del test[COLS]
    print('获取关键词交叉特征完毕......')
    return train,test
def get_time_quantile_fea(df):
    print('    生成时间分位数特征')
    secs = [0.2, 0.4, 0.6, 0.8]
    time_fea_list = []
    for sec in tqdm(secs):
        for time_fea_type in [
                'duration_minutes', 'log_duration_minutes', 'time_diff_1', 'log_time_diff_1']:
            temp = df.groupby(['sn', 'server_model', 'fault_time'])[time_fea_type].quantile(sec).reset_index(
                name=f'{time_fea_type}_' + str(sec * 100))

            time_fea_list.append(temp)
    time_fea_df = time_fea_list[0]
    for time_fea in time_fea_list[1:]:
        time_fea_df = time_fea_df.merge(
            time_fea, how='left', on=[
                'sn', 'server_model', 'fault_time'])
    print(f'    生成时间分位数特征完毕，特征维度:{time_fea_df.shape}')
    return time_fea_df


def get_server_model_fea(train, test):
    print('生成 server_model 特征')
    df = pd.concat([train, test], axis=0, ignore_index=True)
    df['server_model_count_sn'] = df.groupby(
        ['server_model'])['sn'].transform('count')
    df['server_model_nunique_sn'] = df.groupby(
        ['server_model'])['sn'].transform('nunique')
    #     df['server_model_count'] = df.groupby('server_model')['server_model'].transform('count')
    #     df['server_model_cnt_quantile'] = df['server_model'].map(
    #         df['server_model'].value_counts().rank() / len(df['server_model'].unique()))
    #     df['server_model_cnt_rank'] = df[f'server_model_cnt_quantile'].rank(method='min')

    df['sn_cnt'] = df['sn'].map(df['sn'].value_counts())
    df['sn_freq'] = df['sn'].map(df['sn'].value_counts() / len(df))
    df['server_model_cnt'] = df['server_model'].map(
        df['server_model'].value_counts())
    df['server_model_freq'] = df['server_model'].map(
        df['server_model'].value_counts() / len(df))
    select_cols = ['sn', 'server_model',
                   'server_model_count_sn', 'server_model_nunique_sn',
                   'sn_cnt', 'sn_freq', 'server_model_cnt', 'server_model_freq'
                   #                    'server_model_count','server_model_cnt_quantile', 'server_model_cnt_rank'
                   ]
    server_model_fea = df[select_cols]

    cat_feats = [
        'server_model']  # ,'server_model_day_date','server_model_dayofmonth','server_model_dayofweek','server_model_hour']
    # for name in cat_feats:
    #     le = LabelEncoder()
    #     server_model_fea[f'{name}_LabelEnc'] = le.fit_transform(
    #         server_model_fea[name])
    server_model_fea = cat2num(server_model_fea, cat_feats, Transfer2num=True)
    server_model_fea = server_model_fea.drop_duplicates().reset_index(drop=True)
    print(f'生成 server_model 特征完毕，特征维度:{server_model_fea.shape}')

    return server_model_fea


def get_time_type_msg_unique_fea(df):
    df['msg_list'] = df['msg'].apply(
        lambda x: [i.strip() for i in x.split(' | ')])

    df['msg_0'] = df['msg'].apply(
        lambda x: [
            get_msg_location(
                x.split(' | '),
                0)])
    df['msg_1'] = df['msg'].apply(
        lambda x: [
            get_msg_location(
                x.split(' | '),
                1)])
    df['msg_2'] = df['msg'].apply(
        lambda x: [
            get_msg_location(
                x.split(' | '),
                2)])

    df = df.groupby(['sn', 'fault_time']).agg(
        {'msg_list': 'sum', 'msg_0': 'sum', 'msg_1': 'sum', 'msg_2': 'sum'}).reset_index()

    df['msg_set'] = df['msg_list'].apply(lambda x: '|'.join(list(set(x))))

    df['msg_0_set'] = df['msg_0'].apply(lambda x: '|'.join(list(set(x))))
    df['msg_1_set'] = df['msg_1'].apply(lambda x: '|'.join(list(set(x))))
    df['msg_2_set'] = df['msg_2'].apply(lambda x: '|'.join(list(set(x))))
    df = df[['sn', 'fault_time', 'msg_set',
             'msg_0_set', 'msg_1_set', 'msg_2_set']]
    return df


def get_msg_unique_fea(train, test, time_type='last'):
    print('生成msg_unique_ fea')
    common_cols = ['msg_set', 'msg_0_set', 'msg_1_set', 'msg_2_set']
    df = pd.concat([train, test], axis=0, ignore_index=True)
    df['time_interval'] = (
        pd.to_datetime(
            df['fault_time']) -
        df['time']).apply(
            lambda x: x.total_seconds())

    last_fea = get_time_type_msg_unique_fea(df.query('time_interval >0'))
    last_fea = last_fea.rename(columns={i: f'last_{i}' for i in common_cols})
    next_fea = get_time_type_msg_unique_fea(df.query('time_interval <0'))
    next_fea = next_fea.rename(columns={i: f'next_{i}' for i in common_cols})
    all_fea = get_time_type_msg_unique_fea(df)
    all_fea = all_fea.rename(columns={i: f'all_{i}' for i in common_cols})
    msg_unique_fea = all_fea.merge(
        last_fea, on=['sn', 'fault_time'], how='outer')
    msg_unique_fea = msg_unique_fea.merge(
        next_fea, on=['sn', 'fault_time'], how='outer')
    return msg_unique_fea


def get_duration_minutes_fea(train, test):
    print('生成 duration_minutes 特征')
    df = pd.concat([train, test], axis=0, ignore_index=True)
    df['duration_minutes'] = (pd.to_datetime(df['fault_time']) - pd.to_datetime(df['time'])).apply(
        lambda x: x.total_seconds())
    df['log_duration_minutes'] = np.log(df['duration_minutes'])

    df = df.sort_values(['sn', 'label', 'server_model',
                        'fault_time', 'time']).reset_index(drop=True)
    df['time_diff_1'] = (df.groupby(['sn', 'server_model', 'fault_time'])['time'].diff(1)).apply(
        lambda x: x.total_seconds())
    df['time_diff_1'] = df['time_diff_1'].fillna(0)
    df['log_time_diff_1'] = np.log(df['time_diff_1'])

    # time_quantile_fea_df = get_time_quantile_fea(df)
    # time_stat_fea_df = get_4_time_stat_fea(df)
    # df_tmp = time_quantile_fea_df.merge(time_stat_fea_df, on= ['sn',   'server_model','fault_time'],how = 'left')
    time_stat_fea_df = get_4_time_stat_fea(df)
    df_tmp = time_stat_fea_df
    print(f'生成 duration_minutes 特征完毕，特征维度{df_tmp.shape}')
    return df_tmp


def get_msg_text_fea_all(all_data):
    all_data['label'] = all_data['label'].fillna(-1)
    all_data['msg_list'] = all_data['msg'].apply(lambda x: [i.strip() for i in x.split(' | ')])
    all_data['msg_0'] = all_data['msg'].apply(lambda x: [get_msg_location(x.split(' | '), 0)])
    all_data['msg_1'] = all_data['msg'].apply(lambda x: [get_msg_location(x.split(' | '), 1)])
    all_data['msg_2'] = all_data['msg'].apply(lambda x: [get_msg_location(x.split(' | '), 2)])

    all_data = all_data.sort_values(['sn', 'fault_time', 'time']).reset_index(drop=True)
    del all_data['label']
    last_data = all_data.query('time_interval >0')
    next_data = all_data.query('time_interval <=0')

    # id_cols = ['sn', 'fault_time', 'label']

    # all_msg_text_fea = get_msg_text_fea(all_data, msg_type='all')
    last_msg_text_fea = get_msg_text_fea(last_data, msg_type='last')
    # next_msg_text_fea = get_msg_text_fea(next_data, msg_type='next')
    msg_text_fea = last_msg_text_fea
    return msg_text_fea

def get_test_key_words(train,test):

    df = pd.concat([train[['sn', 'fault_time', 'label','msg']],test[['sn', 'fault_time',  'msg']]],ignore_index = True).drop_duplicates(['sn', 'fault_time',  'msg'])
    df['label'] = df['label'].fillna(5)
    df['msg_list'] = df['msg'].apply(lambda x:[i.strip() for i in x.split(' | ')])
    words_cnt_df_list = []
    for label in df['label'].unique():
        label = int(label)
        df_tmp = df.query(f'label == {label}')
        counter = Counter()
        for words in df_tmp['msg_list']:
            words = [i.replace('_',' ') for i in words]
            # word_list = []
            # for i in words:
            #     word_list+=i.split(' ')
            # words = word_list
            counter.update(words)
        words_cnt_df = pd.DataFrame(counter,index = [0]).T.reset_index().rename(columns = {'index':'word',0:f'cnt_{label}'})
        words_cnt_df_list.append(words_cnt_df)
    words_cnt_df = words_cnt_df_list[0]
    for i in words_cnt_df_list[1:]:
        words_cnt_df = words_cnt_df.merge(i,on = 'word',how = 'outer' )

    words_cnt_df = words_cnt_df.fillna(-1)
    words_cnt_df1 = words_cnt_df.query('cnt_0 >10 and cnt_2 >10 and cnt_1 >10 and cnt_3>10 and cnt_5>10 ')
    cnt_class = ['cnt_0','cnt_1','cnt_2','cnt_3','cnt_5']
    words_cnt_df1['word_cnt_sum'] = words_cnt_df1.loc[:,cnt_class].sum(1)
    for i in cnt_class:
        words_cnt_df1[f'{i}_ratio'] = words_cnt_df1[i]/words_cnt_df1['word_cnt_sum']
    words_cnt_df1['word_cnt_ratio_std'] = words_cnt_df1.loc[:,['cnt_0_ratio','cnt_1_ratio', 'cnt_2_ratio', 'cnt_3_ratio']].std(1)
    words_cnt_df1['cnt_1_0_diff'] = (words_cnt_df1['cnt_1_ratio'] - words_cnt_df1['cnt_0_ratio'])
    test_key_words = words_cnt_df1.sort_values('cnt_5',ascending = False)['word'].to_list()[5:40]
    return test_key_words

def get_w2v_mean(w2v_model,sentences):
    emb_matrix = list()
    vec = list()
    for w in sentences.split():
        if w in w2v_model.wv:
            vec.append(w2v_model.wv[w])
    if len(vec) > 0:
        emb_matrix.append(np.mean(vec, axis=0))
    else:
        emb_matrix.append([0] * w2v_model.vector_size)
    return emb_matrix
def get_tfidf_svd(tfv,svd,sentences, n_components=16):
    X_tfidf = tfv.transform(sentences)
    X_svd = svd.transform(X_tfidf)
    return np.mean(X_svd, axis=0)
def get_w2v_tfidf_fea(all_data):
    print('w2v编码')
    df = all_data
    df['msg_list'] = df['msg'].apply(lambda x: [i.strip().lower().replace(' ','_') for i in x.split(" | ")])
    df = df.groupby(['sn']).agg({'msg_list': 'sum'}).reset_index()
    df['text'] = df['msg_list'].apply(lambda x: ' '.join(x))

    sentences_list = df['text'].values.tolist()
    sentences = []
    for s in sentences_list:
        sentences.append([w for w in s.split()])
    w2v_model = Word2Vec(sentences, vector_size=10, window=3, min_count=5, sg=0, hs=1, seed=2022)
    df['text_w2v'] = df['text'].apply(lambda x: get_w2v_mean(w2v_model, x)[0])

    print('tfidf编码')
    X = df['text'].to_list()
    tfv = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_features=50000)
    tfv.fit(X)
    X_tfidf = tfv.transform(X)
    svd = TruncatedSVD(n_components=16)  # 降维
    svd.fit(X_tfidf)
    df['text_tfidf'] = df['text'].apply(lambda x: get_tfidf_svd(tfv, svd, x.split()))

    print("doc2vec编码")
    texts = df['text'].tolist()
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(documents, window=5, min_count=3, workers=4)
    docvecs = model.docvecs
    df['doc2vec'] = [docvecs[i] for i in range(len(docvecs))]

    for i in range(32):
        df[f'msg_w2v_{i}'] = df['text_w2v'].apply(lambda x: x[i])
    for i in range(16):
        df[f'msg_tfv_{i}'] = df['text_tfidf'].apply(lambda x: x[i])
    for i in range(100):
        df[f'msg_doc2vec_{i}'] = df['doc2vec'].apply(lambda x: x[i])

    save_cols = [i for i in df.columns if i not in ['msg_list', 'text', 'text_w2v', 'text_tfidf', 'doc2vec']]
    return df[save_cols]

# w2v_tfidf_fea = get_w2v_tfidf_fea(all_data)
class BetaEncoder(object):

    def __init__(self, group):

        self.group = group
        self.stats = None

    # get counts from df
    def fit(self, df, target_col):
        # 先验均值
        self.prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        # count和sum
        stats = stats.agg(['sum', 'count'])[target_col]
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)
        self.stats = stats

    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):

        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n = df_stats['n'].copy()
        N = df_stats['N'].copy()

        # fill in missing
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0

        # prior parameters
        N_prior = np.maximum(N_min - N, 0)
        alpha_prior = self.prior_mean * N_prior
        beta_prior = (1 - self.prior_mean) * N_prior

        # posterior parameters
        alpha = alpha_prior + n
        beta = beta_prior + N - n

        # calculate statistics
        if stat_type == 'mean':
            num = alpha
            dem = alpha + beta

        elif stat_type == 'mode':
            num = alpha - 1
            dem = alpha + beta - 2

        elif stat_type == 'median':
            num = alpha - 1 / 3
            dem = alpha + beta - 2 / 3

        elif stat_type == 'var':
            num = alpha * beta
            dem = (alpha + beta) ** 2 * (alpha + beta + 1)

        elif stat_type == 'skewness':
            num = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1)
            dem = (alpha + beta + 2) * np.sqrt(alpha * beta)

        elif stat_type == 'kurtosis':
            num = 6 * (alpha - beta) ** 2 * (alpha + beta + 1) - \
                alpha * beta * (alpha + beta + 2)
            dem = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)

        # replace missing
        value = num / dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value


def get_beta_target(train, test):
    N_min = 1000
    feature_cols = []

    # encode variables
    for c in ['server_model']:
        # fit encoder
        be = BetaEncoder(c)
        be.fit(train, 'label')

        # mean
        feature_name = f'{c}_mean'
        train[feature_name] = be.transform(train, 'mean', N_min)
        test[feature_name] = be.transform(test, 'mean', N_min)
        feature_cols.append(feature_name)

        # mode
        feature_name = f'{c}_mode'
        train[feature_name] = be.transform(train, 'mode', N_min)
        test[feature_name] = be.transform(test, 'mode', N_min)
        feature_cols.append(feature_name)

        # median
        feature_name = f'{c}_median'
        train[feature_name] = be.transform(train, 'median', N_min)
        test[feature_name] = be.transform(test, 'median', N_min)
        feature_cols.append(feature_name)

        # var
        feature_name = f'{c}_var'
        train[feature_name] = be.transform(train, 'var', N_min)
        test[feature_name] = be.transform(test, 'var', N_min)
        feature_cols.append(feature_name)

        #     # skewness
        #     feature_name = f'{c}_skewness'
        #     train[feature_name] = be.transform(train, 'skewness', N_min)
        #     test[feature_name]  = be.transform(test,  'skewness', N_min)
        #     feature_cols.append(feature_name)

        # kurtosis
        feature_name = f'{c}_kurtosis'
        train[feature_name] = be.transform(train, 'kurtosis', N_min)
        test[feature_name] = be.transform(test, 'kurtosis', N_min)
        feature_cols.append(feature_name)
    df = train.append(test).reset_index(drop=True)
    df = df[['sn', 'fault_time', 'server_model', 'server_model_mean',
             'server_model_mode', 'server_model_median', 'server_model_var',
             'server_model_kurtosis']].drop_duplicates().reset_index(drop=True)
    return df
