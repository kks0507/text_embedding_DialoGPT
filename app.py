from search import answer_question  # 수정한 search.py에서 answer_question 함수 불러오기

# 먼저 메시지 표시하기
print("질문을 입력하세요")

conversation_history = None  # 초기 대화 이력을 None으로 설정

while True:
    # 사용자가 입력한 문자를 'user_input' 변수에 저장
    user_input = input()

    # 사용자가 입력한 문자가 'exit'인 경우 루프에서 빠져나옴
    if user_input.lower() == "exit":
        break
    
    # 질문에 대해 답변을 생성하는 함수 호출
    answer, conversation_history, context = answer_question(user_input, conversation_history)
    
    # 생성된 컨텍스트와 답변을 출력
    print(f"Generated context: {context}")






