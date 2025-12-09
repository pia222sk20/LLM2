import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REDIS_URL = 'redis://localhost:6379'
INDEX_NAME = 'rag_index'