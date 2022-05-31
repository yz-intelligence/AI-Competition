import os
import numpy as np
import pandas as pd
from  utils import RESULT_DIR
lgb_result = pd.read_csv(os.path.join(RESULT_DIR,'lgb_prob_result.csv'))
lgb_result = lgb_result[lgb_result['label'].isnull()]
print(lgb_result.columns)
del lgb_result['label']

cat_result = pd.read_csv(os.path.join(RESULT_DIR,'cat_prob_result.csv'))
cat_result = cat_result[cat_result['label'].isnull()]
del cat_result['label']

# bert_result = pd.read_csv(os.path.join(RESULT_DIR,'bert_prob_result.csv'))

model_weight = {'lgb':0.2,'cat':0.8,'bert':0.2}
print(f'MODEL WEIGHT: {model_weight}')
# for i in ['bert_class_0', 'bert_class_1', 'bert_class_2','bert_class_3']:
#     bert_result[i] = bert_result[i]*model_weight['bert']

for i in  ['cat_class_0', 'cat_class_1', 'cat_class_2', 'cat_class_3']:
    cat_result[i] = cat_result[i]*model_weight['cat']

for i in  ['lgb_class_0', 'lgb_class_1', 'lgb_class_2', 'lgb_class_3']:
    lgb_result[i] = lgb_result[i]*model_weight['lgb']

result= lgb_result.merge(cat_result,on =['sn', 'fault_time'],how ='left' )

# result= bert_result.merge(cat_result,on =['sn', 'fault_time'],how ='left' )
#
# result['class_0'] =result.loc[:,['cat_class_0','bert_class_0']].sum(1)
# result['class_1'] =result.loc[:,['cat_class_1','bert_class_0']].sum(1)
# result['class_2'] =result.loc[:,['cat_class_2','bert_class_0']].sum(1)
# result['class_3'] =result.loc[:,['cat_class_3','bert_class_0']].sum(1)

result['class_0'] =result.loc[:,['lgb_class_0','cat_class_0']].sum(1)
result['class_1'] =result.loc[:,['lgb_class_1','cat_class_1']].sum(1)
result['class_2'] =result.loc[:,['lgb_class_2','cat_class_2']].sum(1)
result['class_3'] =result.loc[:,['lgb_class_3','cat_class_3']].sum(1)

result['label'] = np.argmax(result.loc[:,['class_0', 'class_1', 'class_2', 'class_3']].values,axis = 1)
result = result[['sn', 'fault_time','label']]
result.to_csv(os.path.join(RESULT_DIR,'stacking_result.csv'),index = False)
