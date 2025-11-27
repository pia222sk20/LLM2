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

# 프롬프트 설계 원칙
'''
1. 역할 정의 (Role Definition)
     "당신은 전문적인 기술 문서 Q&A 시스템입니다."    
2. 컨텍스트 제공 (Context)                          
    "다음은 참조할 문서입니다: {context}"        
3. 명확한 지시 (Instructions)                       
    - 컨텍스트 내 정보만 사용                      
    - 모르면 모른다고 답변                          
    - 한국어로 답변                                  
4. 질문 (Question)                                  
    "질문: {question}"                           
5. 출력 형식 (Output Format)                        
    "답변은 구조화된 형태로 작성하세요."      
'''
# 효과적인 RAG 프롬프트 작성 예시
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', ''' 당신은 제공된 문맥(Context)을 바탕으로 질문에 답변하는 AI 어시스턴트입니다.
## 규칙
1. 제공된 문맥 내의 정보만을 사용하여 답변하세요.
2. 문맥에 없는 정보는 추측하지 말고 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요.
3. 답변은 한국어로 명확하고 간결하게 작성하세요.
4. 가능하면 구조화된 형태(목록, 번호 등)로 답변하세요.
5. 확실하지 않은 내용은 그 점을 명시하세요
'''),
("human",'''## 참조문맥
 {context}

 ## 질문
 {question}

 ## 답변''')
])

# LCEL(LangChain Expression Language)
# | 파이프연산자를 이용해서 직관적으로 연결

# 전통적인 lagecy 방식
result = parser.parse(llm.invoke(prompt.format(question='질문')))

#  LCEL 방식
chain = prompt | llm | parser
result = chain.invoke({'question' : '질문'})

# 핵심 Runnable 컴포넌트
from langchain_core.runnable import RunnablePassThrough
#질문을 그대로 전달하면서 context는 별도 처리
chain = {
    "context" : retriever | fortmat_docs,
    'question' : RunnablePassThrough
} | prompt | llm