# 반드시 docker desktop을 실행해서 dokcer 가 실행중이여야 한다.
# docker desktop이 실행안되면
# 1. powershell 을 관리자 권한으로 실행
# taskkill /F /IM "Docker Desktop.exe"
# taskkill /F /IM "com.docker.backend.exe"
# taskkill /F /IM "com.docker.proxy.exe"

# 2. 재시작
# net stop com.docker.service
# net start com.docker.service

# docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest



import os
from langchain_redis import RedisChatMessageHistory

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
session_id = "user_123"
session_id2 = "user_456"

# 세션별 Redis 기반 채팅 기록 가져오기/생성
history = RedisChatMessageHistory(session_id=session_id, redis_url=REDIS_URL)
history.add_user_message("Redis에 이 대화 기록을 저장하고 있나요?")
history.add_ai_message("네, 이 대화는 Redis에 저장됩니다. 세션이 유지되는 한 기록을 복원할 수 있어요.")

history2 = RedisChatMessageHistory(session_id=session_id2, redis_url=REDIS_URL)
history2.add_user_message("Redis에 이 대화 기록을 저장하고 있나요?")
history2.add_ai_message("네, 이 대화는 Redis에 저장됩니다. 세션이 유지되는 한 기록을 복원할 수 있어요.")

print(f"현재 Redis에 저장된 '{history.session_id}' 대화 수: {len(history.messages)}")
print(f"현재 Redis에 저장된 '{history2.session_id}' 대화 수: {len(history2.messages)}")