import os
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from category_encoders import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# from catboost import CatBoostClassifier
# from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)

user_data_df = pd.read_sql(
    """SELECT * FROM user_data """,
    con=engine
)

post_text_df = pd.read_sql(
    """SELECT * FROM post_text_df """,
    con=engine
)

feed_data_df = pd.read_sql(
    """SELECT * FROM feed_data LIMIT 1000 """,
    con=engine
)

df = feed_data_df.merge(user_data_df, how='left')
df = df.merge(post_text_df, how='left')

df['ts'] = df[['timestamp']].apply(lambda x: x[0].timestamp(), axis=1).astype(int)

df = df.drop('timestamp', axis = 1)
df.rename(columns = ({'ts':'timestamp'}), inplace = True)
df.sort_values(by='timestamp', inplace=True)

df = df.reset_index()
df = df.drop('index', axis=1)

train = df.loc[:799].drop(['text'],  axis = 1)
test = df.loc[800:].drop(['text'],  axis = 1)

X_train = train.drop(['target'],  axis = 1)
X_test = test.drop(['target'],  axis = 1)

y_train = train['target']
y_test = test['target']

obj_columns = X_train.loc[:, X_train.dtypes == object].columns

ohe_columns = [col for col in obj_columns if X_train[col].nunique() < 5]
mte_columns = [col for col in obj_columns if X_train[col].nunique() > 5]
num_columns = list(X_train.select_dtypes(exclude='object').columns)

ohe_columns_idx = [list(X_train.columns).index(col) for col in ohe_columns]
mte_columns_idx = [list(X_train.columns).index(col) for col in mte_columns]
num_columns_idx = [list(X_train.columns).index(col) for col in num_columns]

tranformer = [('ohe', OneHotEncoder(), ohe_columns_idx),
             ('mte', TargetEncoder(), mte_columns_idx),
             ('scaler', StandardScaler(), num_columns_idx)]

col_transform = ColumnTransformer(transformers=tranformer)

col_transform.fit(X_train, y_train)

pipe = Pipeline([('transform', col_transform), ('random_forest', RandomForestClassifier())])

param_grid = {
    "random_forest__max_depth": [10, 15, 20],
    "random_forest__min_samples_split": [2, 5, 10],
    "random_forest__min_samples_leaf": [1, 3, 5]
}

grid = GridSearchCV(pipe, param_grid)

grid.fit(X_train, y_train)

filename = 'model.pkl'
pickle.dump(grid, open(filename, 'wb'))