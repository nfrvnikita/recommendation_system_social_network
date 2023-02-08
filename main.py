import os
import pickle
from typing import List
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
import pandas as pd
from catboost import CatBoostClassifier
from sqlalchemy import create_engine

engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml")


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("/my/super/path")
    from_file = CatBoostClassifier()
    model = from_file.load_model(model_path)
    return model


def load_models():
    model_path = get_model_path("/my/super/path")
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model


def batch_load_sql(query: str) -> pd.DataFrame:
    chunksize = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=chunksize):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features() -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT * FROM nfrv_nikita_features_lesson_22", 
        con=engine
        )
    return df


app = FastAPI()

model = load_models()
load_features = load_features()
post_text_df = pd.read_sql(
        "SELECT * FROM public.post_text_df",
        con=engine
        )


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    df_predict = pd.DataFrame({
        'post_id': post_text_df['post_id'], 
        'user_id': id
        })

    df_with_user_features = load_features.drop(['post_id', 'topic_covid', 'topic_entertainment', 'topic_movie', 'topic_politics',
                                                'topic_sport', 'topic_tech'], axis=1)
    df_with_post_features = post_text_df[['topic', 'post_id']]
    
    for col in ['topic']:
        one_hot = pd.get_dummies(df_with_post_features[col], prefix=col, drop_first=True)
        df_with_post_features = pd.concat((df_with_post_features.drop(col, axis=1), one_hot), axis=1)

    df_predict = df_predict.merge(df_with_user_features, on='user_id', how='left')
    df_predict = df_predict.merge(df_with_post_features, on='post_id', how='left')

    prediction = model.predict_proba(df_predict)[:, 1]

    df_predict['prediction'] = prediction
    df_predict = df_predict.merge(post_text_df, on='post_id', how='left')

    df_predict = df_predict.sort_values(by='prediction', ascending=False).reset_index(drop=True)

    df_predict = df_predict.loc[:limit - 1]

    return [PostGet(**{
        'id': post_id,
        'text': df_predict[df_predict['post_id'] == post_id]['text'].values[0],
        'topic': df_predict[df_predict['post_id'] == post_id]['topic'].values[0]
    }) for post_id in df_predict['post_id']]