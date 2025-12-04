# LangSmith API를 이용한 LLM 모니터링
import os
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangSmith 임포트
from langsmith import Client
from langsmith.run_helpers import traceable

# 환경설정
load_dotenv()

# langsmith 추적 활성화
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'langsmith_prj_001'
print(f"프로젝트 : {os.getenv('LANGCHAIN_PROJECT')}")

llm = ChatOpenAI(model = 'gpt-4o-mini')
prompt = ChatPromptTemplate.from_template('{question}에대해서 간단하게 설명해주세요')
chain = prompt | llm | StrOutputParser()

test_questions = [
    'AI','LangSmith','대한민국'
]
for q in test_questions:
    response = chain.invoke({'question':q})
    print(f'질문 : {q}  답변 : {response}')


# @traceable 로 설정된 함수는 자동으로 추적
@traceable(name=f'custom_qa_function_{os.getenv('LANGCHAIN_PROJECT')}')
def answer_question(question:str)->str:
    prompt = f'다음 질문에 대해서 100자 이내로 요약해서 답변해주세요 : {question}'
    chain = llm | StrOutputParser()
    return chain.invoke(prompt)

result = answer_question('프로그래밍 전문가가 되는 방법 및 가이드')
print(f'answer_question : {result}')

# client 직접사용
client = Client()
# 프로젝트 목록조회
print('프로젝트 목록조회')
project_lists = list(client.list_projects())
for project in project_lists:
    print(project.name)


# 데이터 셋 생성
dataset_name=f"{os.getenv('LANGCHAIN_PROJECT')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description=os.getenv('LANGCHAIN_PROJECT') + '_QA 평가용 데이터셋'
)
# 평가용 예제
examples = [
    {
        "inputs": {"question": "Python이란 무엇인가요?"},
        "outputs": {"answer": "Python은 프로그래밍 언어입니다."}
    },
    {
        "inputs": {"question": "1+1은?"},
        "outputs": {"answer": "2입니다."}
    },
    {
        "inputs": {"question": "AI란?"},
        "outputs": {"answer": "인공지능입니다."}
    }
]
for ex in examples:
    client.create_example(
        inputs=ex['inputs'],
        outputs=ex['outputs'],
        dataset_id=dataset.id
    )
print(f'    {len(examples)}개 예제 추가 완료')

from langsmith.evaluation import evaluate
client = Client()
# 평가모델 정의
llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
#평가 함수 실행
def predict(inputs:str)->Dict[str,str]:
    q = inputs['question']
    result = llm.invoke(f'{q} 간단히 답해줘')
    return {'answer':result.content}
def simple_correctness(run, example):
    """run.outputs 로 모델 답변을 가져오는 방식"""

    gold = example.outputs["answer"]
    pred = run.outputs["answer"]

    score = 1.0 if gold in pred else 0.0

    return {
        "key": "correctness",
        "score": score,
        "comment": f"gold={gold} | pred={pred}"
    }
# 평가실행
results = evaluate(
    predict,
    data=dataset_name,            
    evaluators=[simple_correctness]
)

print('\n평가 결과 요약')
print(results)



# 정리 (테스트 후 삭제)
# client.delete_dataset(dataset_id=dataset.id)
# print(' 데이터셋 삭제완료')