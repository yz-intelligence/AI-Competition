import pandas as pd
import numpy as np
import os
from utils import TRAIN_DIR ,TEST_A_DIR,TEST_B_DIR,RESULT_DIR,DATA_DIR

log_dataset_a = pd.read_csv(os.path.join(DATA_DIR,'preliminary_a_test/preliminary_sel_log_dataset_a.csv'))
log_dataset_b = pd.read_csv(os.path.join(DATA_DIR,'preliminary_b_test/preliminary_sel_log_dataset_b.csv'))
submit_dataset_a = pd.read_csv(os.path.join(DATA_DIR,'preliminary_a_test/preliminary_submit_dataset_a.csv'))
submit_dataset_b = pd.read_csv(os.path.join(DATA_DIR,'preliminary_b_test/preliminary_submit_dataset_b.csv'))

log_dataset_c = pd.concat([log_dataset_a,log_dataset_b],ignore_index = True,axis = 0)
submit_dataset_c = pd.concat([submit_dataset_a,submit_dataset_b],ignore_index = True,axis = 0)

log_dataset_c.to_csv(os.path.join(TEST_A_DIR,'final_log_dataset_c.csv'),index =False)
submit_dataset_c.to_csv(os.path.join(TEST_A_DIR,'final_submit_dataset_c.csv'),index =False)


#
# cat_prob = pd.read_csv(os.path.join(RESULT_DIR,'../../../TianchiAIOps_bert_model/cat_prob_result.csv'))
# lgb_prob = pd.read_csv(os.path.join(RESULT_DIR,'../../../TianchiAIOps_bert_model/lgb_prob_result.csv'))

cat_prob = pd.read_csv(os.path.join(RESULT_DIR,'B_prob_7511.csv'))
lgb_prob = pd.read_csv(os.path.join(RESULT_DIR,'baseline_prob_7495.csv'))
cat_prob.columns  = ['cat_class_0', 'cat_class_1', 'cat_class_2', 'cat_class_3', 'label', 'sn',
       'fault_time']
lgb_prob.columns = ['lgb_class_0', 'lgb_class_1', 'lgb_class_2', 'lgb_class_3', 'label', 'sn',
       'fault_time']

lgb_prob = lgb_prob[lgb_prob['label'].isnull()]
cat_prob = cat_prob[cat_prob['label'].isnull()]

cat_prob['cat_prob'] = cat_prob.loc[:,['cat_class_0', 'cat_class_1', 'cat_class_2', 'cat_class_3']].max(1)
cat_prob['cat_label'] = np.argmax(cat_prob.loc[:,['cat_class_0', 'cat_class_1', 'cat_class_2', 'cat_class_3']].values,axis = 1)

lgb_prob['lgb_prob'] = lgb_prob.loc[:,['lgb_class_0', 'lgb_class_1', 'lgb_class_2', 'lgb_class_3']].max(1)
lgb_prob['lgb_label'] = np.argmax(lgb_prob.loc[:,['lgb_class_0', 'lgb_class_1', 'lgb_class_2', 'lgb_class_3']].values,axis = 1)

lgb_prob = lgb_prob[['sn','fault_time','lgb_label','lgb_prob']]
cat_prob = cat_prob[['sn','fault_time','cat_label','cat_prob']]

# prob = cat_prob.merge(lgb_prob,on =['sn','fault_time'],
#                how = 'left' )

prob = pd.concat([cat_prob,lgb_prob],ignore_index = True)
prob['cat_prob']=prob['cat_prob'].fillna(1)
prob['lgb_prob']=prob['lgb_prob'].fillna(1)
prob.loc[prob['cat_label'].isnull(),'cat_label'] = prob.loc[prob['cat_label'].isnull(),'lgb_label']
prob.loc[prob['lgb_label'].isnull(),'lgb_label'] = prob.loc[prob['lgb_label'].isnull(),'cat_label']


pseudo_labels = prob.query('cat_prob >0.85 and lgb_prob >0.85 and lgb_label == cat_label  ')

pseudo_labels = pseudo_labels[['sn','fault_time','cat_label']].rename(columns = {'cat_label':'label'}).reset_index(drop = True)
pseudo_labels.to_csv(os.path.join(TRAIN_DIR,'pseudo_labels.csv'),index= False)
print(f'生成伪标签的数据维度:{pseudo_labels.shape}')

pseudo_sel_log_dataset = pd.read_csv(os.path.join(TEST_A_DIR,'final_sel_log_dataset_c.csv'))
pseudo_sel_log_dataset = pseudo_sel_log_dataset[pseudo_sel_log_dataset['sn'].isin(pseudo_labels['sn'].to_list())]
pseudo_sel_log_dataset.to_csv(os.path.join(TRAIN_DIR,'pseudo_sel_log_dataset.csv'),index = False)
print(f'生成伪标签的日志数据维度:{pseudo_sel_log_dataset.shape}')

# 制作新的测试集
final_submit_dataset_d= prob.merge(pseudo_labels,on =['sn','fault_time'],how = 'left' )
final_submit_dataset_d = final_submit_dataset_d[final_submit_dataset_d['label'].isnull()][['sn','fault_time' ]].reset_index(drop = True)
final_submit_dataset_d.to_csv(os.path.join(TEST_A_DIR,'final_submit_dataset_d.csv'),index= False)
print(f'生成新的测试集维度:{final_submit_dataset_d.shape}')

final_sel_log_dataset_d = pd.read_csv(os.path.join(TEST_A_DIR,'final_sel_log_dataset_c.csv'))
final_sel_log_dataset_d = final_sel_log_dataset_d[final_sel_log_dataset_d['sn'].isin(final_submit_dataset_d['sn'].to_list())]

final_sel_log_dataset_d.to_csv(
    os.path.join(TEST_A_DIR,'final_sel_log_dataset_d.csv'),index = False)
print(f'生成新的测试集日志数据维度:{final_sel_log_dataset_d.shape}')