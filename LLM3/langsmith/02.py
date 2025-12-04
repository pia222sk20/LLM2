# 이론부분의 sample 코드에 대한 완전히 구현한 코드
import os
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from typing import Any,Dict,List
from dotenv import load_dotenv

# langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# langsmith
from langsmith import Client
from langsmith.run_helpers import traceable

# 환경설정
load_dotenv()
def check_environment():
    '''환경변수 확인'''
    missing_keys = []
    if not os.getenv('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.getenv('LANGCHAIN_API_KEY'):
        missing_keys.append('LANGCHAIN_API_KEY')
    if missing_keys:
        print('필요한 API키가 없습니다.')
        for key in missing_keys:
            print(f' --------- {key}')
        raise ValueError('필수 키 누락')
    
    # langsmith 추적 활성화
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'llm_rag_example'
    print('환경설정 완료!')

# langsmith  자동추적
def auto_tracing():
    '''langsmith 기본 사용법'''
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ('system','당신은 친절한 ai 에이전트입니다. 사용자의 요구사항에 맞게 한글로 설명해주세요'),
        ('human', '간단히 설명해주세요: {topic}')
    ])
    chain = prompt | llm | StrOutputParser()
    topics = ['Python','AI']
    for topic in topics:
        response = chain.invoke({'topic':topic})
        print(f'   {topic} : {response[:50]}...')
    print('자동추적 완료')

def traceable_decorator():
    '''커스텀함수에 @traceable 데코레이터를 사용해서 추적'''
    llm = ChatOpenAI(model = 'gpt-4o-mini',temperature=0)

    @traceable(name='custom_qa_function')
    def answer_question(question:str) -> str:
        '''질문에 답변하는 함수(langsmith에서 추적됨)'''
        prompt = f'질문에 간단히 답해주세요 : {question}'
        response = llm.invoke(prompt)
        return response.content
    @traceable(name="multi_step_analysis")
    def analyze_topic(topic:str)->Dict[str,str]:
        '''여러 단계로 주제를 분석(중첩 추적)'''
        # 단계 1: 정의
        definition = answer_question(f'{topic}이란 무엇인가요?')
        # 단계 2 : 장점
        advantage = answer_question(f'{topic}의 장점은?')

        return {
            'topic':topic,
            'definition' : definition[:100],
            'advantage' : advantage[:100],
        }
    print('\n@traceable 테스트')
    result = analyze_topic('LangChain')
    print(f"    주제 : {result['topic']}")
    print(f"    정의 : {result['definition']}")
    print(f"    강점 : {result['advantage']}")
    print('\n @traceable 데코레이터 완료!')

# 메탇이터 와 태그 추가    
def metadata_tag():
    '''추적에 메타데이터와 태그를 추가해서 필터링/분석에 활용'''
    from langchain_core.runnables import RunnableConfig
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
    prompt = ChatPromptTemplate.from_template('{question}')
    chain = prompt | llm | StrOutputParser()
    # 메타데이터와 태그 설정
    config = RunnableConfig(
        metadata = {
            'user_id' : 'user_123',
            'session_id' : 'sees_456',
            'environment' : 'development',
            'version':"1.0.0"
        },
        tags = ['example','qa','test']
    )
    print('\n메타데이터 / 태그  테스트')
    response = chain.invoke(
        {'question':'RAG란 무엇인가요?'},
        config = config
    )
    print('\n메타데이터와 태그 추가 완료')


# langSmith Client 직접 사용
def langsmith_client():
    '''LangSmith client를 직접사용해서 데이터를 조회'''
    client = Client()
    print('\n프로젝트 목록 조회')
    try:
        projects = client.list_projects(limit=5)
        if projects:
            for project in projects:
                print(f'    - {project.name}')
    except Exception as e:
        print(f'프로젝트 조회중 오류 발생 : {e}')
    print('\n최근 실행기록')
    try:
        project_name = os.getenv('LANGCHAIN_PROJECT','default')
        runs = list(client.list_runs(
            project_name=project_name,
            limit=5
        ))
        if runs:
            for run in runs:
                status = 'success' if run.status == 'success' else 'faile'
                duration = run.end_time - run.start_time   if run.start_time and run.end_time else 'N/A'
                print(f'    {status} {run.name}  |  {duration}')
    except Exception as e:
        print(f'최근 실행기록 조회중 오류 발생 : {e}')
    print('\n langSmith Client 사용 완료')

def dataset_evaluation():
    '''langSmith에서 평가용 데이터셋을 생성하고 모델을 평가'''
    client = Client()
    # 데이터셋이름 생성(고유하게)
    dataset_name = f"qa_eval_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"    
    
    print(f'\n데이터셋 생성: {dataset_name}')

    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description='QA 시스템 평가용 데이터셋'
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
        # 데이터셋 생성 내용 확인
        print('데이터셋 생성 내용 확인')
        saved_examples = client.list_examples(dataset_id=dataset.id)
        for i, ex in enumerate(saved_examples,1):
            question = ex.inputs.get('question', 'N/A')
            print(f'  {i}  {question}')
        
        # 테스트 로직
        from langsmith.evaluation import evaluate
        client = Client()
        # 평가모델 정의
        llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
        #평가 함수 실행
        def predict(inputs:str)->Dict[str,str]:
            q = inputs['question']
            result = llm.invoke(f'{q} 간단히 답해줘')
            return {'answer':result.content}
        # 평가실행
        results = evaluate(
            dataset_name=dataset_name,
            target=predict,
            evaluators=['qa']  # langSmith 내장 평가기
        )

        print('\n평가 결과 요약')
        print(results['summary'])



        # 정리 (테스트 후 삭제)
        client.delete_dataset(dataset_id=dataset.id)
        print(' 데이터셋 삭제완료')
    except Exception as e:
        print(f' 평가용 데이터셋 오류발생 : {e}')



if __name__ =='__main__':
    # check_environment()  #  환경체크
    # auto_tracing() # 자동 추적
    # traceable_decorator() # 커스텀 함수 추적
    # metadata_tag() # 메타데이터 와 태그 추가
    # langsmith_client()  # 데이터조회  client 사용
    dataset_evaluation()  # 평가용 데이터셋 생성 및 확인 그리고 삭제
    
    
    # @traceable(name="create_traking")
    # def add(a, b):
    #     return a + b

    # add(1, 2)