# Hugging Face 모델을 사용한 임베딩 생성 코드 (텍스트 데이터를 벡터 데이터로 변환)

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import pandas as pd
import json

# Hugging Face 모델 불러오기 (컴퓨터가 텍스트(문자)를 이해할 수 있도록 숫자(벡터 데이터)로 바꾸는 역할)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# DialoGPT 토크나이저 불러오기 (토큰 수 계산을 위해 사용)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# 파일 경로 지정 (불러오기 및 저장 경로)
input_csv_path = "C:/text_embedding_DialoGPT/scraped.csv"
output_csv_path = "C:/text_embedding_DialoGPT/embeddings.csv"

# 'scraped.csv' 파일을 불러와서 읽은 이후, 칼럼 이름을 'title(제목)'과 'text(본문)'로 변경
df = pd.read_csv(input_csv_path)
df.columns = ['title', 'text']

# 텍스트에 대해 임베딩 생성 (=텍스트를 숫자로 변환하는 함수)
def get_embedding(text):
    try:
        embedding = model.encode(text)  # 문자열을 벡터로 변환
        return embedding.tolist()  # 벡터화된 데이터를 리스트로 변환하여 JSON 형식으로 저장 가능
    except Exception as e:
        print(f"Error embedding text: {e}")  # 문제가 생기면 "Error embedding text"라는 메시지 출력
        return None

# 각 텍스트의 토큰 수를 계산하는 함수
def count_tokens(text):
    if isinstance(text, str):
        return len(tokenizer.tokenize(text))  # 문자열일 경우 토큰 수 계산
    return 0  # 문자열이 아니면 토큰 수 0으로 설정

# 텍스트를 숫자로 변환하고 데이터에 추가하기 ('text' 열의 모든 데이터에 대해 embedding을 수행하여 벡터 데이터를 CSV 파일로 저장)
df["embeddings"] = df['text'].apply(lambda x: get_embedding(x) if isinstance(x, str) else None)

# 각 텍스트의 토큰 수를 계산하고 'n_tokens' 열에 추가
df["n_tokens"] = df["text"].apply(lambda x: count_tokens(x))

# 변환되지 않은 값(빈 값) 제거 (숫자로 변환되지 않거나 텍스트가 없어서 변환이 되지 않은 값들은 제거)
df = df[df['embeddings'].notnull()]

# Embedding한 결과를 JSON 문자열로 변환 (JSON은 데이터를 파일에 쉽게 저장할 수 있도록 도와주는 형식)
df["embeddings"] = df["embeddings"].apply(lambda x: json.dumps(x))

# CSV 파일로 저장
df.to_csv(output_csv_path, index=False)




