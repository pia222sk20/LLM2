# RAG 체인
'''
사용자 질문
임베딩 변환      : 벡터로 변환
VectorDB 검색   : 유사한 문서 검색
문서 포멧팅     : 검색된 문서를 텍스트로 정리
프롬프트 구성 : 컨텍스트 + 질문 결합
LLM 호출    :  답변생성
출력파싱    : 문자열로 변환
최종 답변
'''

# 프롬프트 템플릿 : 재사용 가능한 프롬프트 구조를 정의
from langchain_core.prompts import ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ('system','당신은 {role} 입니다.'),
    ('human',"{question}")
])

# 변수 채우기
prompt = template.invoke({
    'role' : 'AI 어시스턴트',
    'question' : 'RAG란 무엇인가요?'
})

# 프롬프트 유형
# 단일 문자열
from langchain_core.prompts import PromptTemplate
template = PromptTemplate.from_template('''
다음질문에 답변하세요
질문 : {question}                                        
답변 : ''')
# 채팅 형식
from langchain_core.prompts import ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ("system", "시스템 지시사항"),
    ("human", "사용자 질문: {question}"),
    ("assistant", "이전 답변 (선택)"),
    ("human", "후속 질문")
])