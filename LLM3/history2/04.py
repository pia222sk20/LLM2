from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_redis import RedisVectorStore, RedisChatMessageHistory

from dotenv import load_dotenv
load_dotenv()

class LangChainRAG:
    def __init__(self,redis_url="redis://localhost:6379"):
        self.redis_url = redis_url
        
        # OpenAI 임베딩
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        
        # OpenAI LLM
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        
        # Redis Vector Store
        self.vector_store = None
        print("LangChain RAG 초기화 완료")

    def add_documents(self, docs):
        """
        문서 추가
        docs: [Document(page_content='...', metadata={'source':'...'}), ...]
        """
        self.vector_store = RedisVectorStore.from_documents(
            documents=docs,
            embedding=self.embeddings,
            redis_url=self.redis_url,
            index_name="rag_index"
        )
        print(f"{len(docs)}개 문서 추가 완료")
    def get_retriever(self, k=2):
        """문서 검색기 반환"""
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def get_message_history(self, session_id):
        """세션별 메시지 히스토리"""
        return RedisChatMessageHistory(
            session_id=session_id,
            redis_url=self.redis_url,
            # ttl=3600  # 1시간
        )
    def create_chain(self):
        """LCEL 체인 생성"""
        # 프롬프트
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 문서를 기반으로 답변하는 AI입니다. 문서 내용을 바탕으로 정확히 답변하세요."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "문서:\n{context}\n\n질문: {question}")
        ])
        
        # 체인: prompt | llm | output_parser
        chain = prompt | self.llm | StrOutputParser()
        
        # 메시지 히스토리와 결합
        chain_with_history = RunnableWithMessageHistory(
            chain,
            self.get_message_history,
            input_messages_key="question",
            history_messages_key="history"
        )
        
        return chain_with_history

    def ask(self, session_id, question):
        """질문하기"""
        # 1. 문서 검색
        retriever = self.get_retriever(k=2)
        docs = retriever.invoke(question)
        context = "\n\n".join([f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs])
        
        # 2. 체인 실행
        chain = self.create_chain()
        answer = chain.invoke(
            {"question": question, "context": context},
            config={"configurable": {"session_id": session_id}}
        )
        
        return answer, docs        

def main():
    # 초기화
    rag = LangChainRAG(redis_url="redis://localhost:6379")
    
    # 문서 추가 (Document 형식)
    docs = [
        Document(
            page_content="재택근무는 주 3일까지 가능합니다. VPN 사용이 필수입니다.",
            metadata={'source': 'company_policy.pdf'}
        ),
        Document(
            page_content="복지 제도: 점심 식대 12,000원, 자기계발비 연 100만원 지원",
            metadata={'source': 'benefits.pdf'}
        ),
        Document(
            page_content="3분기 실적: 매출 450억원, 영업이익률 18.5%",
            metadata={'source': 'q3_report.pdf'}
        ),
    ]
    
    rag.add_documents(docs)
    
    print("\n" + "="*60)
    
    # User1 대화
    print("\n[User1 - Alice 세션]")
    ans1, docs1 = rag.ask('alice', '재택근무 정책은?')
    print(f"Q: 재택근무 정책은?")
    print(f"A: {ans1}")
    print(f"출처: {[d.metadata['source'] for d in docs1]}\n")
    
    ans2, docs2 = rag.ask('alice', 'VPN은 왜 필요해?')
    print(f"Q: VPN은 왜 필요해?")
    print(f"A: {ans2}\n")
    
    # User2 대화 (독립 세션)
    print("\n[User2 - Bob 세션]")
    ans3, docs3 = rag.ask('bob', '복지 제도는?')
    print(f"Q: 복지 제도는?")
    print(f"A: {ans3}")
    print(f"출처: {[d.metadata['source'] for d in docs3]}\n")
    
    ans4, docs4 = rag.ask('bob', '식대는 얼마야?')
    print(f"Q: 식대는 얼마야?")
    print(f"A: {ans4}\n")
    
    # 대화 내역 확인
    print("\n" + "="*60)
    print("대화 내역 확인")
    print("="*60)
    
    alice_history = rag.get_message_history('alice')
    print("\nAlice 대화:")
    for msg in alice_history.messages:
        print(f"  {msg.type}: {msg.content[:50]}...")
    
    bob_history = rag.get_message_history('bob')
    print("\nBob 대화:")
    for msg in bob_history.messages:
        print(f"  {msg.type}: {msg.content[:50]}...")


if __name__ == '__main__':
    main()    