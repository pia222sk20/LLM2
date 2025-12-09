from langchain_redis import RedisChatMessageHistory
def get_history(session_id:str, redis_url : str = 'redis://localhost:6379'):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=redis_url
    )