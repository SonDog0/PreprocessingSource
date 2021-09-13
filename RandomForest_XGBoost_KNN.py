
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, f1_score ,recall_score , precision_score , confusion_matrix

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier

import xgboost as xgb


def mk_training_dataset():

    # concat df
    df1 = pd.read_csv('data/0903_ECG_PPG_학습데이터1_Null_삭제_중복컬럼제거.csv' , encoding='CP949')
    df1.rename(columns = {'Diagnosis' : 'type'} , inplace = True)
    df2 = pd.read_csv('data/0906_ECG_PPG_학습데이터2_Null_삭제_중복컬럼제거.csv' , encoding='CP949')
    df = pd.concat([df1,df2])

    # replace
    df['type'] = df['type'].replace('뇌졸중', 1)
    df['type'] = df['type'].replace('고령자', 0)

    # under sampling
    count_class_0, count_class_1 = df.type.value_counts()

    # divide by class
    df_class_0 = df[df['type'] == 0]
    df_class_1 = df[df['type'] == 1]

    df_class_0_under = df_class_0.sample(count_class_1, random_state=42)
    df = pd.concat([df_class_0_under, df_class_1], axis=0)

    # split train/test
    x = df.drop('type', axis=1)
    y = df['type']

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # scailing
    std_scaler = StandardScaler()
    std_scaler.fit(train_x)
    train_x_scaling = std_scaler.transform(train_x)
    test_x_scaling = std_scaler.transform(test_x)

    return train_x, test_x, train_y, test_y, train_x_scaling, test_x_scaling




def modeling_randomforest():

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]



    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(train_x, train_y)

    return rf_random


def modeling_XGBoost():

    # Set parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 4

    d_train = xgb.DMatrix(train_x, label=train_y)
    d_valid = xgb.DMatrix(test_x, label=test_y)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
    d_test = xgb.DMatrix(test_x)
    p_test = bst.predict(d_test)

    y_preds = [1 if x > 0.5 else 0 for x in p_test]

    xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
    xgb_wrapper.fit(train_x, train_y)

    return xgb_wrapper


def modeling_KNN():
    
    params = {'n_neighbors': [5, 6, 7, 8, 9, 10],
              'leaf_size': [1, 2, 3, 5],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'n_jobs': [-1]}


    knn = KNeighborsClassifier(n_jobs=-1)

    knn_random = RandomizedSearchCV(knn, param_distributions=params, n_jobs= -1 , random_state=42, n_iter=100, cv=3)

    knn_random.fit(train_x_scaling, train_y)

    return knn_random



def predict(model, df, answer_file_location):

    prediction = model.predict(df)

    df['Prediction'] = prediction

    df['Prediction'] = df['Prediction'].replace(1,'뇌졸중')
    df['Prediction'] = df['Prediction'].replace(0,'고령자')

    df_answer = pd.read_csv(answer_file_location , encoding='CP949')
    df_answer['type'] = df_answer['type'].replace('뇌졸중', 1)
    df_answer['type'] = df_answer['type'].replace('고령자', 0)
    answer = df_answer['type'].values

    print('Confusion Matrix')
    print(confusion_matrix(answer,prediction))

    print('======================================')
    print('Precision Score : {}'.format(precision_score(answer,prediction)))
    print('Recall Score : {}'.format(recall_score(answer,prediction)))
    print('Accuracy Score : {}'.format(accuracy_score(answer,prediction)))
    print('F1 Score : {}'.format(f1_score(answer,prediction)))


    return df



if __name__ == '__main__':

    df_val1 = pd.read_csv('data/ECG_PPG_시험데이터_1_Null_삭제_중복컬럼제거_컬럼공백제거_라벨제거_한글라벨제거.csv', encoding='CP949')
    df_val2 = pd.read_csv('data/ECG_PPG_시험데이터_2_Null_삭제_중복컬럼제거_컬럼공백제거_라벨제거_한글라벨제거.csv', encoding='CP949')

    train_x, test_x, train_y, test_y, train_x_scaling, test_x_scaling = mk_training_dataset()

    # model = modeling_randomforest()
    # model = modeling_XGBoost()
    model = modeling_KNN()

    result = predict(model, df_val1, 'data/ECG_PPG_시험 데이터1_블라인드 결과_000001.csv')
    #
    # print(result)
    result.to_csv('210913_ECG_PPG_시험데이터1_KNN.csv', encoding='CP949', index=False)




