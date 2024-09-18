# text_embedding_DialoGPT 🐾

## Description
가상으로 생성한 고양이 카페 운영 메뉴얼의 텍스트를 임베딩하여 벡터 데이터로 변환하고, 이를 바탕으로 터미널에서 임베딩한 데이터를 바탕으로 답변하는 챗봇 프로그램입니다. 

<br>

## Installation and Execution

### Requirements
프로그램을 실행하기 위해 필요한 라이브러리들이 `requirements.txt` 파일에 명시되어 있습니다. 다음 명령어를 사용하여 필요한 라이브러리를 설치하세요:

```bash
pip install -r requirements.txt
```

`requirements.txt` 파일에는 다음 라이브러리들이 포함되어 있습니다:
- transformers
- pandas
- numpy
- scipy
- sentence-transformers
- torch

### Text to CSV Conversion
먼저 텍스트 파일을 CSV 파일로 변환해야 합니다. 텍스트 파일을 `text_To_CSV_Converter.py`를 실행하여 CSV 파일로 변환하세요:

```bash
python text_To_CSV_Converter.py
```

이 스크립트는 텍스트 파일을 읽어들여 데이터를 CSV 파일로 변환합니다.

### Embedding Generation
변환된 텍스트 데이터를 벡터로 임베딩하려면 `generate_VectorData.py`를 실행합니다:

```bash
python generate_VectorData.py
```

이 스크립트는 텍스트 데이터를 벡터로 변환하여 임베딩 데이터를 생성합니다. 생성된 벡터 데이터는 `embeddings.csv` 파일에 저장됩니다.

### Run the Chatbot
챗봇 프로그램을 실행하려면 `app.py`를 실행하세요:

```bash
python app.py
```

이 스크립트는 터미널에서 사용자가 입력한 질문에 대해 임베딩된 데이터를 바탕으로 답변을 제공합니다. 프로그램은 `search.py`에서 임베딩된 데이터를 활용하여 답변을 생성합니다.

<br>

## 주요 함수 설명

#### 임베딩 생성 함수: `generate_VectorData.py`
```python
def get_embedding(text):
    try:
        embedding = model.encode(text)  # Hugging Face의 SentenceTransformer 모델을 사용해 텍스트를 벡터로 변환
        return embedding.tolist()  # 변환된 벡터 데이터를 리스트 형식으로 변환하여 저장
    except Exception as e:
        print(f"Error embedding text: {e}")  # 오류 발생 시 메시지 출력
        return None
```
이 함수는 `SentenceTransformer` 모델을 사용하여 입력된 텍스트 데이터를 임베딩(벡터화)합니다. 이 과정에서 텍스트는 컴퓨터가 이해할 수 있는 숫자 데이터로 변환됩니다. `encode()` 함수는 입력된 텍스트를 임베딩으로 변환하며, 그 결과는 리스트 형식으로 변환됩니다. 변환된 벡터 데이터는 차후에 텍스트 간의 유사도를 비교하거나 검색하는 데 사용됩니다.

#### 질문에 답변하는 함수: `search.py`
```python
def answer_question(question, conversation_history):
    df = pd.read_csv('C:/text_embedding_DialoGPT/embeddings.csv')  # 임베딩된 CSV 파일 불러오기
    context = create_context(question, df)  # 질문과 관련된 문맥을 생성

    # 문맥과 질문을 결합하여 DialoGPT 모델에 전달할 프롬프트 작성
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    
    # 사용자 질문과 대화 이력을 기반으로 답변 생성
    new_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = dialogpt_model.generate(
        new_input_ids,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.5
    )

    # 생성된 답변을 디코딩하여 반환
    answer = tokenizer.decode(chat_history_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return answer, conversation_history, context
```
이 함수는 사용자로부터 질문을 입력받아 임베딩된 데이터를 바탕으로 관련 문맥을 추출한 후, `DialoGPT` 모델을 사용하여 질문에 대한 답변을 생성합니다. 

1. **임베딩 데이터 로드**: CSV 파일에서 임베딩된 데이터를 불러옵니다. 이는 미리 생성된 텍스트의 벡터화 데이터입니다.
2. **문맥 생성**: 사용자의 질문과 관련된 텍스트를 찾아 문맥을 형성합니다. 이 문맥은 `create_context()` 함수를 통해 질문과 가장 유사한 임베딩을 찾아 생성됩니다.
3. **DialoGPT를 사용한 답변 생성**: 생성된 문맥과 질문을 합쳐서 모델에 입력으로 전달합니다. 모델은 대화 형식으로 답변을 생성하며, 이 답변은 사용자가 터미널에서 확인할 수 있습니다.
4. **대화 이력 관리**: 대화의 연속성을 유지하기 위해, 이전 대화 이력을 저장하고, 새로운 입력을 통해 계속해서 대화를 이어나갑니다.

#### 텍스트에서 줄 바꿈을 제거하고 데이터프레임으로 변환하는 함수: `text_To_CSV_Converter.py`
```python
def remove_newlines(text):
    text = re.sub(r'\n', ' ', text)  # 텍스트에서 줄 바꿈 기호 제거
    text = re.sub(r' +', ' ', text)  # 연속된 공백을 하나로 축소
    return text

def text_to_df(data_file):
    texts = []
    with open(data_file, 'r', encoding="utf-8") as file:
        text = file.read()
        sections = text.split('\n\n')  # 각 섹션을 두 줄의 공백을 기준으로 분리
        for section in sections:
            lines = section.split('\n')  # 섹션 내에서 줄바꿈 기준으로 텍스트를 나눔
            fname = lines[0]  # 첫 줄을 제목으로 사용
            content = ' '.join(lines[1:])  # 나머지 줄들을 본문으로 결합
            texts.append([fname, content])
    df = pd.DataFrame(texts, columns=['fname', 'text'])
    df['text'] = df['text'].apply(remove_newlines)  # 줄바꿈 기호 제거
    return df
```
이 함수는 텍스트 파일을 읽어들여, 각 섹션을 제목과 본문으로 분리한 후 데이터프레임으로 변환합니다. 이후 줄바꿈 및 공백을 제거하여 CSV 파일로 저장할 수 있도록 정리합니다.

<br>

## Note
**DialoGPT**는 Microsoft에서 개발한 대화형 언어 모델로, GPT-2 모델을 기반으로 구축되었습니다. 주로 대화 시나리오에 적합하도록 학습되었으며, Reddit에서 수집한 대규모 대화 데이터를 학습하여 자연스러운 대화 생성을 목표로 합니다. 
이 모델은 사용자의 질문이나 입력에 대해 일관성 있고 인간적인 답변을 생성하는 데 특화되어 있으며, 챗봇이나 대화 에이전트 구축에 유용하게 사용됩니다. 다양한 대화 상황에서 유연하게 대처할 수 있도록 훈련되어, 문맥에 맞는 답변을 생성할 수 있습니다.
DialoGPT는 Hugging Face의 `transformers` 라이브러리를 통해 쉽게 사용할 수 있으며, `AutoModelForCausalLM` 및 `AutoTokenizer`와 함께 사용하여 대화 기반 애플리케이션을 구축하는 데 활용됩니다.

## Contributor
- kks0507

## License
This project is licensed under the MIT License.

## Repository
코드 및 프로젝트의 최신 업데이트는 [여기](https://github.com/kks0507/text_embedding_DialoGPT.git)에서 확인할 수 있습니다.
