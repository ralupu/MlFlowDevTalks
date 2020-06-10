# importing libraries for data handling and analysis

import mlflow
import mlflow.sklearn
import warnings
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split  # import 'train_test_split'
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from azureml.core import Workspace


# ws = Workspace.from_config('azure_config.json')
# mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())


def fit_model(model, x, y):
    model.fit ( x, y )
    return model


if __name__ == "__main__":
    # experiment_name = 'Employee_Attrition'
    # mlflow.set_experiment ( experiment_name )
    # warnings.filterwarnings ( "ignore" )

    # LOAD THE DATA
    df_sourcefile = pd.read_excel (
        'Data/WA_Fn-UseC_-HR-Employee-Attrition.xlsx', sheet_name=0 )
    print ( "Shape of dataframe is: {}".format ( df_sourcefile.shape ) )
    # Make a copy of the original sourcefile
    df_HR = df_sourcefile.copy ()

    # DATA PREPARATION
    # Encoding - Create a label encoder object
    le = LabelEncoder ()

    # Label Encoding will be used for columns with 2 or less unique values
    le_count = 0
    for col in df_HR.columns[1:]:
        if df_HR[col].dtype == 'object':
            if len ( list ( df_HR[col].unique () ) ) <= 2:
                le.fit ( df_HR[col] )
                df_HR[col] = le.transform ( df_HR[col] )
                le_count += 1
    print ( '{} columns were label encoded.'.format ( le_count ) )

    # convert rest of categorical variable into dummy
    df_HR = pd.get_dummies ( df_HR, drop_first=True )

    # Feature Scaling
    scaler = MinMaxScaler ( feature_range=(0, 5) )
    HR_col = list ( df_HR.columns )
    HR_col.remove ( 'Attrition' )
    for col in HR_col:
        df_HR[col] = df_HR[col].astype ( float )
        df_HR[[col]] = scaler.fit_transform ( df_HR[[col]] )
    df_HR['Attrition'] = pd.to_numeric ( df_HR['Attrition'], downcast='float' )
    df_HR.head ()

    # SPLIT DATA INTO TRAIN AND TEST
    # assign the target to a new dataframe and convert it to a numerical feature
    target = df_HR['Attrition'].copy ()

    # remove the target feature and redundant features from the dataset
    df_HR.drop ( ['Attrition', 'EmployeeCount', 'EmployeeNumber',
                  'StandardHours', 'Over18'], axis=1, inplace=True )
    print ( 'Size of Full dataset is: {}'.format ( df_HR.shape ) )

    X_train, X_test, y_train, y_test = train_test_split ( df_HR,
                                                          target,
                                                          test_size=0.25,
                                                          random_state=7,
                                                          stratify=target )


    # START MLFLOW
    with mlflow.start_run():
        # for n_est in [75, 100, 125]:
        for n_est in [75]:
            # for learn_rate in [0.05, 0.08, 0.1, 0.12, 0.15]:
            for learn_rate in [0.05]:
                for max_d in [1, 2, 3]:
                    xgb_mod = xgb.XGBClassifier ( objective="binary:logistic", random_state=42, n_estimators=n_est, learning_rate=learn_rate,max_depth=max_d)

                    xgb_fit = fit_model ( xgb_mod, X_train, y_train )

                    pred = xgb_fit.predict ( X_test )
                    accuracy = accuracy_score ( y_test, pred )
                    precision, recall, fscore, support = precision_recall_fscore_support ( y_test, pred )
                    mlflow.log_metric ( 'accuracy', accuracy )
                    mlflow.log_metric ( 'precision', precision[1] )
                    mlflow.log_metric ( 'recall', recall[1] )
                    mlflow.log_metric ( 'fscore', fscore[1] )

                    # tracking_url_type_store = urlparse ( mlflow.get_tracking_uri () ).scheme
                    mlflow.sklearn.log_model ( xgb_fit, "model")



