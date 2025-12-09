from langchain_core.chat_history import InMemoryChatMessageHistory
# 새 메모리 객체 생성
history = InMemoryChatMessageHistory()
history.add_user_message("안녕하세요, 제 이름은 철수입니다.")
history.add_ai_message("안녕하세요 철수님, 무엇을 도와드릴까요?")

# 현재까지의 대화 내용 확인
for msg in history.messages:
    # print(msg)
    print(f"{msg.type}: {msg.content}    ")    