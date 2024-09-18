import pandas as pd  
import re  # re 라이브러리는 텍스트 안에서 줄 바꿈 같은 특수한 기호를 처리하는데 사용한다.


def remove_newlines(text):  # 문자열의 줄 바꿈과 연속된 공백을 삭제하는 함수
    text = re.sub(r'\n', ' ', text)  # "안녕\n하세요"라는 글자가 있다면, 이 함수는 \n(줄 바꿈 기호)를 없애서 "안녕하세요"로 만들어준다.
    text = re.sub(r' +', ' ', text)  # 공백(빈칸)이 두 개 이상 있다면, 하나로 줄여준다.
    return text

def text_to_df(data_file):  # 텍스트 파일을 처리하여 DataFrame을 반환하는 함수
    texts = []  # 텍스트를 저장할 빈 목록 만들기

    with open(data_file, 'r', encoding="utf-8") as file:  # 지정된 파일(data_file)을 읽어들여 변수 'file'에 저장
        text = file.read()  # 파일 내용을 문자열로 불러오기
        sections = text.split('\n\n')  # 줄 바꿈을 기준으로 섹션 나누기

        for section in sections:  # 각 섹션에 대해 처리하기
            lines = section.split('\n')  # 섹션을 줄 바꿈으로 나누기
            fname = lines[0]  # 첫 줄은 제목으로 가져오기
            content = ' '.join(lines[1:])  # 그 다음 줄들은 본문으로 가져오기
            texts.append([fname, content])  # 목록에 제목과 본문 추가하기

    # 목록을 표 형태로 바꾸기 (데이터프레임이라고 부름)
    df = pd.DataFrame(texts, columns=['fname', 'text'])

    # 'text' 열에 줄 바꿈을 없애기 위해 위에서 만든 함수 사용
    df['text'] = df['text'].apply(remove_newlines)

    return df  # 정리된 표를 반환

# 'catCafe.txt'의 데이터를 처리 (data_file의 위치에 catCafe.txt를 넣어준다.)
df = text_to_df('C:/text_embedding_DialoGPT/catCafe.txt') 
# 변환된 CSV 파일을 특정 경로에 저장
df.to_csv('C:/text_embedding_DialoGPT/scraped.csv', index=False, encoding='utf-8')

