from langsmith import Client
import os
from dotenv import load_dotenv
load_dotenv()

# 연결 테스트
client = Client()
print(" LangSmith 연결 성공!")
print(f"현재 프로젝트: {os.getenv('LANGCHAIN_PROJECT', 'default')}")

# 자동추적
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model='gpt-4o-mini')
prompt = ChatPromptTemplate.from_template('''질문:{question}''')

from langchain_core.output_parsers import StrOutputParser
chain = prompt | llm | StrOutputParser()
result = chain.invoke({'question' : 'RAG란?'})
print(result)

# 커스텀 추적
from langsmith.run_helpers import traceable
@traceable(name='custom_rag_pipeline')
def my_rag_duction(question:str) ->str:
    result = chain.invoke({'question' : question})
    return result

result = my_rag_duction('langsmith란?')
print(result)
