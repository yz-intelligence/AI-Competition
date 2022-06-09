import warnings
import datetime
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from utils import N_ROUNDS
import pickle
import os
warnings.filterwarnings('ignore')


def get_model_feature_importances(model):
    feature_importances = pd.DataFrame()
    feature_importances['fea'] = model.feature_names_
    feature_importances['importances'] = model.feature_importances_
    feature_importances = feature_importances.sort_values('importances', ascending=False).reset_index(drop=True)

    return feature_importances


def run_cbt(train, target, test, k, seed, NUM_CLASS=4, cat_cols=[]):
    print('********************** RUN CATBOOST MODEL **********************')
    print(f'******************  当前的 SEED {seed} ********************** ')
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof_prob = np.zeros((train.shape[0], NUM_CLASS))
    test_prob = np.zeros((test.shape[0], NUM_CLASS))
    feature_importance_df = []
    offline_score = []
    model_list = []

    ## K-Fold
    for fold, (trn_idx, val_idx) in enumerate(folds.split(train, target)):
        print("FOLD {} IS RUNNING...".format(fold + 1))
        trn_x, trn_y = train.loc[trn_idx], target.loc[trn_idx]
        val_x, val_y = train.loc[val_idx], target.loc[val_idx]
        catboost_model = CatBoostClassifier(
            iterations=N_ROUNDS,
            od_type='Iter',
            od_wait=120,
            max_depth=8,
            learning_rate=0.05,
            l2_leaf_reg=9,
            random_seed=seed,
            fold_len_multiplier=1.1,
            loss_function='MultiClass',
            logging_level='Verbose',
            # task_type="GPU"

        )

        start_time = datetime.datetime.now()

        catboost_model.fit(trn_x,
                           trn_y,
                           eval_set=(val_x, val_y),
                           use_best_model=True,
                           verbose=800,
                           early_stopping_rounds=100,
                           cat_features=cat_cols,
                           )
        end_time = datetime.datetime.now()
        model_train_cost_time = end_time - start_time
        print('****************** 模型训练 COST TIME : ',str(model_train_cost_time),' ******************')

        start_time = datetime.datetime.now()
        oof_prob[val_idx] = catboost_model.predict_proba(train.loc[val_idx])
        end_time = datetime.datetime.now()
        model_pred_cost_time = end_time - start_time
        print('****************** 模型预测 COST TIME : ', str(model_pred_cost_time), ' ******************')
        #         catboost_model = catboost_model.get_best_iteration()
        test_prob += catboost_model.predict_proba(test) / folds.n_splits
        print(catboost_model.get_best_score())
        offline_score.append(catboost_model.get_best_score()['validation']['MultiClass'])

        feature_importance_df.append(get_model_feature_importances(catboost_model))
        model_list.append(catboost_model)
        with open(os.path.join('../model', f'cat_model_flod_{fold}.pkl'), 'wb') as f:
            pickle.dump(catboost_model, f)
    print('\nOOF-MEAN-ERROR score:%.6f, OOF-STD:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    fea_imp_df = pd.concat(feature_importance_df, ignore_index=True).groupby('fea').agg(
        {'importances': 'mean'}).reset_index().sort_values('importances', ascending=False).reset_index(drop=True)

    return oof_prob, test_prob, fea_imp_df, model_list


def run_lgb(train, target, test, k, seed=42, NUM_CLASS=4, cat_cols=[]):
    # feats = [f for f in train.columns if f not in ['cust_no', 'label', 'I7', 'I9', 'B6']]
    #     print('Current num of features:', len(feats))
    print(f'********************** RUN LGBM MODEL **********************')
    print(f'******************  当前的 SEED {seed} ********************** ')
    cols_map = {j: i for i, j in enumerate(train.columns)}
    cat_cols = [cols_map[i] for i in cat_cols]
    train = train.rename(columns=cols_map)
    test = test.rename(columns=cols_map)
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof_prob = np.zeros((train.shape[0], NUM_CLASS))
    test_prob = np.zeros((test.shape[0], NUM_CLASS))
    fea_imp_df_list = []
    offline_score = []
    model_list = []
    ## K-Fold
    for fold, (trn_idx, val_idx) in enumerate(folds.split(train, target)):
        params = {
            "objective": "multiclass",
            "num_class": NUM_CLASS,
            "learning_rate": 0.01,
            "max_depth": -1,
            "num_leaves": 32,
            "verbose": -1,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
            "seed": seed,
            'metric': 'multi_error'

        }
        print("FOLD {} IS RUNNING...".format(fold + 1))
        trn_data = lgb.Dataset(train.loc[trn_idx], label=target.loc[trn_idx])
        val_data = lgb.Dataset(train.loc[val_idx], label=target.loc[val_idx])

        # train
        params['seed'] = seed
        lgb_model = lgb.train(
            params,
            trn_data,
            num_boost_round=N_ROUNDS,
            valid_sets=[trn_data, val_data],
            early_stopping_rounds=100,
            verbose_eval=200,
            categorical_feature=cat_cols,

        )
        # predict
        oof_prob[val_idx] = lgb_model.predict(train.loc[val_idx], num_iteration=lgb_model.best_iteration)
        test_prob += lgb_model.predict(test, num_iteration=lgb_model.best_iteration) / folds.n_splits
        offline_score.append(lgb_model.best_score['valid_1']['multi_error'])
        fea_imp = pd.DataFrame()
        fea_imp['feature_name'] = lgb_model.feature_name()
        fea_imp['importance'] = lgb_model.feature_importance()
        fea_imp['feature_name'] = fea_imp['feature_name'].map({str(cols_map[i]): i for i in cols_map})
        fea_imp = fea_imp.sort_values('importance', ascending=False)
        fea_imp_df_list.append(fea_imp)

        model_list.append(lgb_model)
    print('\nOOF-MEAN-ERROR score:%.6f, OOF-STD:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    fea_imp_df = pd.concat(fea_imp_df_list, ignore_index=True).groupby('feature_name').agg(
        {'importance': 'mean'}).reset_index().sort_values('importance', ascending=False)
    return oof_prob, test_prob, fea_imp_df, model_list
