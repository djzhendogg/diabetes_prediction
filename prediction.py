import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from lazypredict.Supervised import LazyRegressor
from lazypredict.Supervised import LazyClassifier
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
import json
import os

def LGBMClassifier_grid_cv(features, target,
                                random_st = 0,
                                n_cv = 5,
                                shuffle_on = True,
                                objective_st='binary',
                                learning_rat=0.1,
                                n_boost_round=500
                                ):
        
        param_gr = {
            'num_leaves': [8, 16, 31],
            'n_estimators': np.array([50, 100, 200], dtype = int),
            'learning_rate': np.array([0.08, 0.1, 0.34]),
            'max_depth': np.array([4, 8, 12, -1], dtype = int),
            'class_weight': ["balanced", None],
            'reg_alpha': np.array([0, 0.1, 1]),
            'reg_lambda': np.array([0, 0.1, 1])
            }
        
        gkf = KFold(n_splits=n_cv, shuffle=shuffle_on, random_state=random_st).split(features, target)
        
        
        lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective=objective_st, num_boost_round=n_boost_round, learning_rate=learning_rat)

        gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_gr, cv=gkf)
        lgb_model = gsearch.fit(features, target)

        return lgb_model.best_params_, lgb_model.best_score_


def LGBMClass_Learning(features, target, target_name, best_grid,
                            random_st = 0,
                            size_of_test=0.2,
                            filename_for_wght = '.pkl'
                            ):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = size_of_test, random_state = random_st)

        lgb_model = lgb.LGBMClassifier(**best_grid)
        lgb_model.fit(X_train, y_train)
        print("Train Accuracy:",lgb_model.score(X_train, y_train))
        print("TEST ACCURACY: ",lgb_model.score(X_test, y_test))
        joblib.dump(lgb_model, filename_for_wght)


        lgb_model = joblib.load(filename_for_wght)
        y_pred = lgb_model.predict(X_test)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred, digits=3))
        y_test_proba = lgb_model.predict_proba(X_test)
        # print(y_test_proba)
        varience_df = pd.DataFrame(y_test_proba, columns = ['healthy', target_name])
        yyy = y_test.to_numpy()
        varience_df['real_target'] = yyy
        print("VARIANCE DATA")
        print(varience_df)

        x_train_names = pd.DataFrame(X_train, columns=X_train.columns)
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(x_train_names)
        shap.summary_plot(shap_values, x_train_names)
        return


df = pd.read_csv(r'C:\Users\1665865\Downloads\bonus_track\diabetes_cleaned_dataset.csv', index_col = 0)
cat_list = ["gender", "smoking_history", 'heart_disease', 'hypertension']

target = df['diabetes']
df.drop(['diabetes'], axis = 1, inplace = True)
df_num = df.drop(cat_list, axis=1)
df_cat = df[cat_list]

scaler = MinMaxScaler()
scaler.fit(df_num)
df_num_sc = scaler.transform(df_num)

features = pd.concat([pd.DataFrame(df_num_sc, index = df_num.index, columns = df_num.columns), df_cat], axis = 1)
params, score = LGBMClassifier_grid_cv(features, target)

print(params, score)

with open("params_neiro_grid.json", 'w', encoding="utf8") as fp:
    json.dump(params, fp, ensure_ascii=False, indent=4)


k_str = LGBMClass_Learning(features = features, target = target, target_name = 'diabetes', best_grid = params, filename_for_wght = '/content/drive/MyDrive/bonustrack/diabetes.pkl')