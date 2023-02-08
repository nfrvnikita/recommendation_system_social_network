
import pandas as pd

from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, auc, roc_curve, RocCurveDisplay
# from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

SEED = 42

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
df = df.drop('text', axis=1)

df['ts'] = df[['timestamp']].apply(lambda x: x[0].timestamp(), axis=1).astype(int)

df = df.drop('timestamp', axis = 1)
df.rename(columns = ({'ts':'timestamp'}), inplace = True)
df.sort_values(by='timestamp', inplace=True)

df = df.reset_index()
df = df.drop('index', axis=1)


train = df.iloc[:int(df.shape[0] * 4/5)].copy()
test = df.iloc[int(df.shape[0] * 4/5):].copy()

X_train = train.drop(['target'],  axis = 1)
X_test = test.drop(['target'],  axis = 1)

y_train = train['target']
y_test = test['target']

for col in ['topic', 'country', 'exp_group', 'os', 'source', 'action']:
    one_hot_train = pd.get_dummies(X_train[col], prefix=col, drop_first=True)
    X_train = pd.concat((X_train.drop(col, axis=1), one_hot_train), axis=1)
    one_hot_test = pd.get_dummies(X_test[col], prefix=col, drop_first=True)
    X_test = pd.concat((X_test.drop(col, axis=1), one_hot_test), axis=1)

X_train['city'] = LabelEncoder().fit_transform(X_train['city'])
X_test['city'] = LabelEncoder().fit_transform(X_test['city'])

def func_predictive(y_test, X_test, model):
    roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    fpr, tpr, threshold = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    auc(fpr, tpr)

catboost = CatBoostClassifier(verbose=100, random_seed=SEED, scale_pos_weight=round((y_test == 0).sum()/(y_test == 1).sum(), 3))

catboost.fit(X_train, y_train)

func_predictive(y_test=y_test, X_test=X_test, model=catboost)

catboost.save_model('catboost', format="cbm")