"""
RAG Summary Evaluator - 온라인 문서 버전
PDF 파일 없이 웹 문서나 텍스트로 테스트 가능
LangChain 0.3.x, LangSmith 최신 버전 기준

핵심 기능:
1. Summary Evaluator를 통한 전체 실험 수준 평가
2. 질문-컨텍스트, 답변-컨텍스트 관련성 평가 (커스텀 구현)
3. 다중 LLM 모델 성능 비교 (GPT-4o-mini, GPT-4o, Ollama)
4. LangSmith를 통한 실험 추적 및 시각화
"""

# ============================================================================
# 1. 환경 설정 및 라이브러리 임포트
# ============================================================================

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain 최신 버전 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# LangSmith 평가 관련
from langsmith import Client
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate

# 환경 변수 로드
load_dotenv()

# LangSmith 클라이언트 초기화
ls_client = Client()


# ============================================================================
# 2. 관련성 평가를 위한 Pydantic 모델 정의
# ============================================================================

class RetrievalQuestionGrade(BaseModel):
    """질문-컨텍스트 관련성 평가 결과"""
    score: str = Field(
        description="질문과 컨텍스트의 관련성. 'yes' 또는 'no'"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="평가 이유"
    )


class RetrievalAnswerGrade(BaseModel):
    """답변-컨텍스트 관련성 평가 결과"""
    score: str = Field(
        description="답변이 컨텍스트에 기반했는지 여부. 'yes' 또는 'no'"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="평가 이유"
    )


# ============================================================================
# 3. 커스텀 관련성 평가자 클래스
# ============================================================================

class RelevanceGrader:
    """
    LangChain 표준 기능만 사용한 관련성 평가자
    """
    
    def __init__(self, llm: Any, target: str):
        """
        Args:
            llm: 평가에 사용할 LLM
            target: 평가 대상 ('retrieval-question' 또는 'retrieval-answer')
        """
        self.llm = llm
        self.target = target
        self.chain = self._create_grader_chain()
    
    def _create_grader_chain(self):
        """평가 체인 생성"""
        if self.target == "retrieval-question":
            return self._create_question_grader()
        elif self.target == "retrieval-answer":
            return self._create_answer_grader()
        else:
            raise ValueError(f"지원하지 않는 target: {self.target}")
    
    def _create_question_grader(self):
        """질문-컨텍스트 관련성 평가 체인"""
        
        system_prompt = """당신은 질문과 검색된 문서의 관련성을 평가하는 평가자입니다.

다음 기준으로 평가하세요:
- 검색된 문서(context)가 질문(question)에 답하는데 필요한 정보를 포함하고 있는가?
- 문서가 질문의 주제나 키워드와 관련이 있는가?

관련성이 있다면 'yes', 없다면 'no'를 반환하세요."""

        human_prompt = """질문: {input}

컨텍스트:
{context}

위 컨텍스트가 질문에 답하는데 관련이 있습니까?"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        llm_with_structure = self.llm.with_structured_output(RetrievalQuestionGrade)
        chain = prompt | llm_with_structure
        
        return chain
    
    def _create_answer_grader(self):
        """답변-컨텍스트 관련성 평가 체인"""
        
        system_prompt = """당신은 생성된 답변이 제공된 컨텍스트에 기반했는지 평가하는 평가자입니다.

다음 기준으로 평가하세요:
- 답변의 내용이 컨텍스트에서 직접 도출될 수 있는가?
- 답변이 컨텍스트에 없는 정보를 포함하고 있지 않은가? (환각 검사)
- 답변이 컨텍스트를 정확하게 반영하고 있는가?

답변이 컨텍스트에 기반했다면 'yes', 아니면 'no'를 반환하세요."""

        human_prompt = """컨텍스트:
{context}

생성된 답변: {input}

위 답변이 컨텍스트에 충실하게 기반했습니까?"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        llm_with_structure = self.llm.with_structured_output(RetrievalAnswerGrade)
        chain = prompt | llm_with_structure
        
        return chain
    
    def invoke(self, inputs: Dict[str, str]) -> Any:
        """평가 실행"""
        try:
            result = self.chain.invoke(inputs)
            return result
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            if self.target == "retrieval-question":
                return RetrievalQuestionGrade(score="no", reasoning="평가 실패")
            else:
                return RetrievalAnswerGrade(score="no", reasoning="평가 실패")


# ============================================================================
# 4. RAG 시스템 클래스 정의
# ============================================================================

class ModernRAGSystem:
    """
    최신 LangChain을 사용한 RAG 시스템
    웹 문서 또는 직접 텍스트 입력 지원
    """
    
    def __init__(
        self, 
        llm: Any,
        source_type: str = "text",  # "text", "web", "documents"
        source_data: Any = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4
    ):
        """
        Args:
            llm: 사용할 LLM 모델
            source_type: 데이터 소스 타입
                - "text": 직접 텍스트 입력
                - "web": 웹 URL
                - "documents": Document 객체 리스트
            source_data: 소스 데이터
                - text: 문자열
                - web: URL 문자열 또는 URL 리스트
                - documents: Document 객체 리스트
        """
        self.llm = llm
        self.source_type = source_type
        self.source_data = source_data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        
    def load_documents(self) -> List[Document]:
        """소스 타입에 따라 문서 로드"""
        
        if self.source_type == "text":
            # 텍스트를 Document 객체로 변환
            if isinstance(self.source_data, str):
                return [Document(page_content=self.source_data)]
            elif isinstance(self.source_data, list):
                return [Document(page_content=text) for text in self.source_data]
        
        elif self.source_type == "web":
            # 웹 URL에서 로드
            urls = [self.source_data] if isinstance(self.source_data, str) else self.source_data
            loader = WebBaseLoader(urls)
            return loader.load()
        
        elif self.source_type == "documents":
            # 이미 Document 객체인 경우
            return self.source_data
        
        else:
            raise ValueError(f"지원하지 않는 source_type: {self.source_type}")
    
    def load_and_split_documents(self) -> List[Document]:
        """문서를 로드하고 청크로 분할"""
        documents = self.load_documents()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        return splits
    
    def create_vectorstore(self):
        """벡터 스토어 생성"""
        splits = self.load_and_split_documents()
        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        return self.vectorstore
    
    def create_retriever(self):
        """검색기 생성"""
        if self.vectorstore is None:
            self.create_vectorstore()
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        return retriever
    
    def create_chain(self, retriever):
        """LCEL을 사용한 RAG 체인 생성"""
        
        template = """당신은 질문에 답변하는 AI 어시스턴트입니다.
다음 컨텍스트를 기반으로 질문에 답변해주세요.
컨텍스트에 관련 정보가 없다면 모른다고 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    @staticmethod
    def _format_docs(docs):
        """문서 리스트를 문자열로 포맷팅"""
        return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# 5. 샘플 데이터 생성
# ============================================================================

def create_sample_ai_documents():
    """
    테스트용 AI 관련 샘플 문서 생성
    실제 PDF 대신 사용할 수 있는 샘플 텍스트
    """
    
    sample_texts = [
        """
        인공지능(AI)의 발전과 현황
        
        2024년은 생성형 AI의 혁신적인 발전이 이루어진 해입니다. 
        OpenAI의 GPT-4는 자연어 처리 분야에서 획기적인 성능을 보여주었으며,
        Claude, Gemini 등 다양한 대형 언어 모델들이 등장했습니다.
        
        특히 멀티모달 AI 기술이 주목받고 있습니다. 
        텍스트뿐만 아니라 이미지, 음성, 비디오를 이해하고 생성할 수 있는 
        AI 모델들이 상용화되면서 실생활에서의 활용도가 크게 증가했습니다.
        """,
        
        """
        한국의 AI 산업 동향
        
        삼성전자는 자체 개발한 생성형 AI '삼성 가우스(Samsung Gauss)'를 공개했습니다.
        삼성 가우스는 언어, 코드, 이미지 생성 기능을 제공하며,
        온디바이스 AI 구현을 목표로 개발되었습니다.
        
        네이버는 초거대 AI '하이퍼클로바X'를 출시하여 
        한국어 특화 서비스를 제공하고 있습니다.
        카카오도 'KoGPT'를 기반으로 다양한 AI 서비스를 선보이고 있습니다.
        """,
        
        """
        AI 윤리와 규제
        
        유럽연합(EU)은 세계 최초로 포괄적인 AI 규제법인 'AI Act'를 통과시켰습니다.
        이 법안은 AI 시스템을 위험도에 따라 분류하고, 
        고위험 AI에 대해서는 엄격한 규제를 적용합니다.
        
        미국, 한국 등 주요 국가들도 AI 윤리 가이드라인과 
        규제 프레임워크를 마련하고 있습니다.
        AI의 투명성, 공정성, 책임성 확보가 중요한 과제로 대두되고 있습니다.
        """,
        
        """
        AI 기술의 실용화 사례
        
        의료 분야에서는 AI를 활용한 질병 진단, 신약 개발이 활발히 진행되고 있습니다.
        금융 분야에서는 AI 기반 리스크 관리, 자산 관리 서비스가 확대되고 있습니다.
        
        제조업에서는 AI를 활용한 품질 검사, 예지 정비가 보편화되고 있으며,
        물류 분야에서는 자율주행 로봇과 배송 최적화 시스템이 도입되고 있습니다.
        
        교육 분야에서도 AI 튜터, 맞춤형 학습 플랫폼이 
        학생들의 학습 효율을 높이는데 기여하고 있습니다.
        """
    ]
    
    return sample_texts


# ============================================================================
# 6. RAG 질문 응답 함수 생성
# ============================================================================

def create_rag_qa_function(llm: Any, source_data: Any, source_type: str = "text"):
    """
    RAG 시스템을 사용한 질문-응답 함수 생성
    
    Args:
        llm: 사용할 LLM
        source_data: 소스 데이터 (텍스트, URL 등)
        source_type: 소스 타입 ("text", "web", "documents")
    
    Returns:
        질문-응답 함수
    """
    rag_system = ModernRAGSystem(
        llm=llm,
        source_type=source_type,
        source_data=source_data
    )
    retriever = rag_system.create_retriever()
    rag_chain = rag_system.create_chain(retriever)
    
    def qa_function(inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs["question"]
        
        # 컨텍스트 검색
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 답변 생성
        answer = rag_chain.invoke(question)
        
        return {
            "question": question,
            "context": context,
            "answer": answer
        }
    
    return qa_function


# ============================================================================
# 7. 관련성 평가자 설정
# ============================================================================

def setup_relevance_graders(model_name: str = "gpt-4o-mini"):
    """
    질문-컨텍스트 및 답변-컨텍스트 관련성 평가자 생성
    """
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    rq_grader = RelevanceGrader(llm=llm, target="retrieval-question")
    ra_grader = RelevanceGrader(llm=llm, target="retrieval-answer")
    
    return rq_grader, ra_grader


# ============================================================================
# 8. Summary Evaluator 정의
# ============================================================================

def create_relevance_summary_evaluator(rq_grader, ra_grader):
    """
    관련성 점수를 종합하는 Summary Evaluator 생성
    
    핵심 개념:
    - 개별 실행이 아닌 전체 실험 수준에서 평가
    - 모든 runs와 examples를 받아서 종합 점수 계산
    - 질문-컨텍스트 관련성 + 답변-컨텍스트 관련성의 평균
    """
    
    def relevance_summary_evaluator(
        runs: List[Run], 
        examples: List[Example]
    ) -> Dict[str, Any]:
        """전체 실험에 대한 관련성 점수 계산"""
        
        rq_scores = 0
        ra_scores = 0
        total_runs = len(runs)
        
        print(f"\n[평가 시작] 총 {total_runs}개 실행에 대한 관련성 평가")
        print("-" * 60)
        
        for idx, (run, example) in enumerate(zip(runs, examples), 1):
            question = example.inputs["question"]
            context = run.outputs.get("context", "")
            prediction = run.outputs.get("answer", "")
            
            # 질문-컨텍스트 관련성 평가
            rq_result = rq_grader.invoke({
                "input": question,
                "context": context,
            })
            
            # 답변-컨텍스트 관련성 평가
            ra_result = ra_grader.invoke({
                "input": prediction,
                "context": context,
            })
            
            if rq_result.score == "yes":
                rq_scores += 1
            if ra_result.score == "yes":
                ra_scores += 1
            
            print(f"  [{idx}/{total_runs}] Q-관련성: {rq_result.score}, A-관련성: {ra_result.score}")
        
        # 최종 점수 계산
        rq_avg = rq_scores / total_runs
        ra_avg = ra_scores / total_runs
        final_score = (rq_avg + ra_avg) / 2
        
        print("-" * 60)
        print(f"[평가 완료]")
        print(f"  질문-컨텍스트 관련성: {rq_avg:.2%}")
        print(f"  답변-컨텍스트 관련성: {ra_avg:.2%}")
        print(f"  종합 관련성 점수: {final_score:.2%}\n")
        
        return {
            "key": "relevance_score",
            "score": final_score,
            "metadata": {
                "question_context_relevance": rq_avg,
                "answer_context_relevance": ra_avg,
                "total_evaluated": total_runs
            }
        }
    
    return relevance_summary_evaluator


# ============================================================================
# 9. LangSmith 추적 활성화
# ============================================================================

def enable_langsmith_tracing(project_name: str):
    """LangSmith 추적 활성화"""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    print(f"LangSmith 추적 활성화")
    print(f"프로젝트명: {project_name}\n")


# ============================================================================
# 10. 실험 실행 함수
# ============================================================================

def run_evaluation_experiment(
    source_data: Any,
    source_type: str,
    dataset_name: str,
    models_config: List[Dict[str, Any]],
    experiment_prefix: str = "RAG-EVAL"
):
    """
    다중 모델에 대한 RAG 평가 실험 실행
    
    models_config 예시:
    [
        {"name": "gpt-4o-mini", "type": "openai"},      # OpenAI의 mini 모델
        {"name": "gpt-4o", "type": "openai"},           # OpenAI의 풀 모델
        {"name": "llama3:8b", "type": "ollama"}         # Ollama 로컬 모델
    ]
    
    주석을 풀면:
    - 여러 모델이 동일한 데이터셋에 대해 순차적으로 평가됨
    - 각 모델의 성능이 별도로 측정되어 LangSmith에 기록됨
    - 결과를 비교하여 어떤 모델이 더 우수한지 확인 가능
    """
    
    # 관련성 평가자 설정
    rq_grader, ra_grader = setup_relevance_graders()
    summary_evaluator = create_relevance_summary_evaluator(rq_grader, ra_grader)
    
    results = []
    
    for model_config in models_config:
        model_name = model_config["name"]
        model_type = model_config["type"]
        
        print("\n" + "=" * 60)
        print(f"모델 평가 시작: {model_name} ({model_type})")
        print("=" * 60)
        
        # LLM 생성
        if model_type == "openai":
            llm = ChatOpenAI(model=model_name, temperature=0)
        elif model_type == "ollama":
            # Ollama 사용 시:
            # 1. Ollama를 로컬에 설치 (https://ollama.ai)
            # 2. 터미널에서 모델 다운로드: ollama pull llama3:8b
            # 3. Ollama 서버 실행: ollama serve
            llm = ChatOllama(model=model_name)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
        
        # RAG QA 함수 생성
        qa_function = create_rag_qa_function(llm, source_data, source_type)
        
        # 평가 실행
        experiment_result = evaluate(
            qa_function,
            data=dataset_name,
            summary_evaluators=[summary_evaluator],
            experiment_prefix=experiment_prefix,
            metadata={
                "model_name": model_name,
                "model_type": model_type,
                "variant": f"{model_name} 모델의 Summary Evaluator 기반 관련성 평가"
            }
        )
        
        results.append({
            "model": model_name,
            "result": experiment_result
        })
    
    return results


# ============================================================================
# 11. 데이터셋 생성
# ============================================================================

def create_evaluation_dataset(dataset_name: str, examples: List[Dict]):
    """LangSmith 평가 데이터셋 생성"""
    client = Client()
    
    if client.has_dataset(dataset_name=dataset_name):
        print(f"데이터셋 '{dataset_name}'이 이미 존재합니다.")
        return
    
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )
    
    print(f"데이터셋 '{dataset_name}' 생성 완료 (예제 {len(examples)}개)")


# ============================================================================
# 12. 메인 실행 코드
# ============================================================================

def main():
    """메인 실행 함수"""
    
    # LangSmith 추적 활성화
    enable_langsmith_tracing("RAG-Summary-Evaluation-2024")
    
    # 샘플 데이터 생성 (PDF 대신 사용)
    sample_texts = create_sample_ai_documents()
    
    # 데이터셋 설정
    DATASET_NAME = "RAG_EVAL_DATASET_SAMPLE"
    
    # 평가용 예제 데이터셋 생성 (최초 1회만 실행)
    examples = [
        {"inputs": {"question": "삼성전자가 개발한 생성형 AI의 이름은 무엇인가요?"}},
        {"inputs": {"question": "2024년 AI 분야의 주요 발전은 무엇인가요?"}},
        {"inputs": {"question": "유럽연합의 AI 규제법 이름은 무엇인가요?"}},
        {"inputs": {"question": "AI가 의료 분야에서 어떻게 활용되고 있나요?"}},
        {"inputs": {"question": "네이버의 초거대 AI 모델 이름은 무엇인가요?"}},
    ]
    
    # 데이터셋이 없으면 생성
    try:
        create_evaluation_dataset(DATASET_NAME, examples)
    except Exception as e:
        print(f"데이터셋 생성 중 오류 (이미 존재할 수 있음): {e}")
    
    # 평가할 모델 설정
    # 주석을 풀면 여러 모델을 동시에 비교 평가할 수 있습니다
    models_to_evaluate = [
        {
            "name": "gpt-4o-mini",      # 빠르고 저렴한 모델
            "type": "openai"
        },
        # 주석을 풀면 GPT-4o도 함께 평가됩니다 (더 강력하지만 비용이 높음)
        # {
        #     "name": "gpt-4o",
        #     "type": "openai"
        # },
        
        # Ollama 로컬 모델을 사용하려면:
        # 1. Ollama 설치: https://ollama.ai
        # 2. 모델 다운로드: ollama pull llama3:8b
        # 3. 아래 주석을 풀어서 활성화
        # {
        #     "name": "llama3:8b",        # 또는 "mistral:7b", "gemma:7b" 등
        #     "type": "ollama"
        # }
    ]
    
    # 평가 실행
    print("\n" + "=" * 60)
    print("RAG Summary Evaluator 실험 시작")
    print("=" * 60)
    print(f"\n평가할 모델 수: {len(models_to_evaluate)}")
    print(f"데이터셋: {DATASET_NAME}")
    print(f"평가 문항 수: {len(examples)}\n")
    
    results = run_evaluation_experiment(
        source_data=sample_texts,
        source_type="text",
        dataset_name=DATASET_NAME,
        models_config=models_to_evaluate,
        experiment_prefix="SUMMARY-EVAL"
    )
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("전체 실험 결과 요약")
    print("=" * 60)
    
    for result in results:
        print(f"\n모델: {result['model']}")
        print(f"  평가 완료")
        print(f"  LangSmith에서 상세 결과 확인 가능")


# ============================================================================
# 13. 테스트 함수들
# ============================================================================

def quick_test_graders():
    """관련성 평가자 빠른 테스트"""
    
    print("\n[관련성 평가자 테스트]")
    print("=" * 60)

    rq_grader, ra_grader = setup_relevance_graders()

    # 테스트 1: 질문-컨텍스트 관련성
    print("\n1. 질문-컨텍스트 관련성 평가 테스트")
    result1 = rq_grader.invoke({
        "input": "삼성전자가 자체 개발한 생성형 AI의 이름은?",
        "context": "삼성전자는 생성형 AI '삼성 가우스'를 공개했습니다."
    })
    print(f"   결과: {result1.score}")
    print(f"   이유: {result1.reasoning}")

    # 테스트 2: 답변-컨텍스트 관련성
    print("\n2. 답변-컨텍스트 관련성 평가 테스트")
    result2 = ra_grader.invoke({
        "input": "삼성전자가 자체 개발한 생성형 AI는 삼성 가우스입니다.",
        "context": "삼성전자는 생성형 AI '삼성 가우스'를 공개했습니다."
    })
    print(f"   결과: {result2.score}")
    print(f"   이유: {result2.reasoning}")

    print("\n" + "=" * 60)

def quick_test_rag_system():
    """RAG 시스템 빠른 테스트"""
    print("\n[RAG 시스템 테스트]")
    print("=" * 60)

    # 샘플 데이터 사용
    sample_texts = create_sample_ai_documents()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_function = create_rag_qa_function(llm, sample_texts, "text")

    test_question = "삼성전자가 개발한 AI의 이름은 무엇인가요?"
    result = qa_function({"question": test_question})

    print(f"\n질문: {result['question']}")
    print(f"\n컨텍스트:")
    context_preview = result["context"][:300] + "..." if len(result["context"]) > 300 else result["context"]
    print(context_preview)
    print(f"\n답변:")
    print(result["answer"])
    print("\n" + "=" * 60)

    return result

def explain_multi_model_evaluation():
    """다중 모델 평가 설명"""
    explanation = """
    ============================================================
    다중 모델 평가 설명
    ============================================================

    models_to_evaluate 리스트에 여러 모델을 추가하면:

    1. 동일한 데이터셋으로 여러 모델을 순차적으로 평가
    2. 각 모델의 성능이 독립적으로 측정됨
    3. LangSmith에서 모델 간 성능 비교 가능

    예시:
    models_to_evaluate = [
        {"name": "gpt-4o-mini", "type": "openai"},    # 빠르고 저렴
        {"name": "gpt-4o", "type": "openai"},         # 더 강력
        {"name": "llama3:8b", "type": "ollama"}       # 로컬 무료
    ]

    결과:
    - gpt-4o-mini: 관련성 점수 85%
    - gpt-4o: 관련성 점수 92%
    - llama3:8b: 관련성 점수 78%

    ============================================================
    Ollama 사용 방법
    ============================================================

    1. Ollama 설치
    - 웹사이트: https://ollama.ai
    - macOS/Linux: curl https://ollama.ai/install.sh | sh
    - Windows: 설치 프로그램 다운로드

    2. 모델 다운로드
    $ ollama pull llama3:8b        # Llama 3 (8B 파라미터)
    $ ollama pull mistral:7b       # Mistral (7B)
    $ ollama pull gemma:7b         # Gemma (7B)

    3. 모델 목록 확인
    $ ollama list

    4. 코드에서 사용
    주석을 풀고 모델 이름을 정확히 입력
    {"name": "llama3:8b", "type": "ollama"}

    주의사항:
    - Ollama는 로컬에서 실행되므로 인터넷 불필요
    - 충분한 RAM 필요 (8B 모델: 최소 8GB RAM)
    - GPU가 있으면 더 빠름 (선택사항)

    ============================================================
    """

    print(explanation)    

if __name__ == "__main__":
    # 메인 실험 실행   
    main()

    # 또는 개별 테스트 실행
    # quick_test_graders()
    # quick_test_rag_system()
    # explain_multi_model_evaluation()