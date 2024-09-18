from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import torch
from sentence_transformers import SentenceTransformer

# DialoGPT 모델 로드
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# SentenceTransformer 모델 로드 (임베딩 생성용)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def distances_from_embeddings(query_embedding, embeddings, distance_metric='cosine'):
    """
    주어진 질문 임베딩과 데이터셋 임베딩들 간의 거리를 계산하는 함수
    distance_metric이 'cosine'일 때 코사인 유사도를 사용하여 계산합니다.
    """
    distance_metrics = {
        'cosine': cosine,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances

def create_context(question, df, max_len=1800):
    """
    질문과 학습 데이터를 비교해 컨텍스트를 만드는 함수
    """
    # 질문 임베딩 생성 (SentenceTransformer 사용)
    q_embeddings = embedding_model.encode(question)

    # 질문과 학습 데이터 비교하여 코사인 유사도 계산
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].apply(eval).apply(np.array).values, distance_metric='cosine')

    # 유사도 높은 데이터를 순서대로 문맥으로 추출
    returns = []
    cur_len = 0

    for _, row in df.sort_values('distances', ascending=True).iterrows():
        cur_len += row['n_tokens'] + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])

    # 문맥 결합하여 반환
    context = "\n\n###\n\n".join(returns)
    return context

def answer_question(question, conversation_history):
    """
    문맥에 따라 질문에 답하는 기능 (DialoGPT 사용)
    """
    df = pd.read_csv('C:/text_embedding_DialoGPT/embeddings.csv')
    
    # 임베딩 데이터를 기반으로 문맥 생성
    context = create_context(question, df, max_len=200)

    # DialoGPT를 사용하여 답변 생성
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    # 입력된 대화 이력과 새로운 질문을 결합
    max_length = 200  # 입력 텍스트의 최대 길이를 200 토큰으로 제한
    new_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    new_input_ids = new_input_ids[:, -max_length:]  # 초과된 토큰은 잘라냄

    # 패딩 토큰 처리 (없을 경우 eos_token 사용)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attention_mask = new_input_ids.ne(pad_token_id).long()

    # 대화 이력을 포함하여 입력 크기 맞추기
    if conversation_history is not None:
        # 대화 이력과 새로운 입력 크기를 맞추기 위해 패딩 적용
        conversation_history = torch.nn.functional.pad(conversation_history, (0, new_input_ids.shape[-1] - conversation_history.shape[-1]), 'constant', pad_token_id)
        bot_input_ids = torch.cat([conversation_history, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # 모델에 입력을 전달할 때 attention_mask도 함께 전달
    chat_history_ids = dialogpt_model.generate(
        bot_input_ids, 
        max_new_tokens=300,  # 새롭게 생성할 토큰 수를 300으로 늘림
        pad_token_id=tokenizer.eos_token_id, 
        attention_mask=attention_mask, 
        do_sample=True,  
        top_p=0.9,      # 샘플링 범위를 좀 더 좁게 설정
        temperature=0.5  
    )

    # 모델 응답을 디코딩
    answer = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # 대화 이력 갱신
    conversation_history = chat_history_ids

    return answer, conversation_history, context




