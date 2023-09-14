from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import ast as ast
import pickle
import time
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# 데이터 로딩
df = pd.read_csv('akbonara_musicSheet_info_small.csv')

# 필요한 열만 선택
data = df[["cde_id", "title", "summary", "artist", "album", "genres", "hit", "price", "orderCnt", "part"]]

# genres 안의 데이터를 string으로 변환
data["genres"] = data["genres"].apply(ast.literal_eval)
data["genres"] = data["genres"].apply(lambda x: [d['name'] for d in x] if isinstance(x, list) else ['Unknown']).apply(lambda x: " ".join(x))


@app.get("/")
def health_check():
    return {"message": "fastAPI running!"}

@app.get("/training")
def training():
    
    # 추천 시작
    start = time.time()
    print(start)

    # 장르를 기준으로 필터링 - 장르 벡터화
    vectorizer = CountVectorizer()
    features_matrix = vectorizer.fit_transform(data["genres"])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(features_matrix, features_matrix).argsort()[:, ::-1]
    # print(cosine_sim)

    # 모델을 pickle 파일로 저장 (파일 이름에 현재 시간 포함)
    filename = f'cosine_similarity.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(cosine_sim, f)

    # 추천 완료
    end = time.time()
    print("학습 소요시간", (end - start))

    return {"status_code": 200, "message": "코사인 유사도 pickle 파일로 저장 완료", "data": filename}

# 이전에 한 번만 코사인 유사도 행렬 로드 - 파일이 존재하는 경우에
if os.path.exists('cosine_similarity.pkl'):
    with open('cosine_similarity.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
        print("학습완료된 코사인 유사도 :", cosine_sim)


@app.get("/recommend")
def get_recommendations(
    music_title: str = Query(..., description="추천받을 악보 제목을 입력해주세요."),
    target_part: str = Query(..., description="추천받을 악보의 파트를 입력해주세요.")
):
    # 추천 시작
    start = time.time()
    
    # 타겟 음악의 index 추출
    target_music_title = data[data['title'] == music_title].index.values

    # 타겟 음악과 비슷한 코사인 유사도 값
    similar_index = cosine_sim[target_music_title, :500].reshape(-1)

    # 자기 자신 제외
    similar_index = similar_index[~np.isin(similar_index, target_music_title)]

    # 추천 결과 샘플링
    num_samples = min(10, len(similar_index))
    random_indices = np.random.choice(similar_index, num_samples, replace=False)

    # 추천 완료
    end = time.time()
    print("추천 소요시간", (end - start))

    # 결과 생성
    result = data.iloc[random_indices]
    filtered_result = result[result['part'] == target_part]
    
    return {"status_code": 200, "datas": filtered_result.to_dict(orient='records')}
