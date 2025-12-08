import os
import warnings
import json
import sqlite3
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

warnings.filterwarnings('ignore')
# 필수 라이브러리 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks import BaseCallbackHandler
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sqlite 기반 추적 시스템
class LocalTraceDB:
    '''SQLite 기반 로컬 추적시스템
    LangSmith 대신 로컬에서 모든 LLM 호출을 추적하고 저장
    '''
    def __init__(self, db_path:str = 'local_traces.db'):
        self.db_path = db_path
        self._init_db()
    def _init_db(self):
        '''데이터베이스 초기화 및 데이블 생성'''
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 실행 추적 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs(
                       id TEXT PRIMARY KEY,
                       name TEXT,
                       run_type TEXT,
                       start_time TEXT,
                       end_time TEXT,
                       duration_seconds REAL,
                       input_data TEXT,
                       output_data TEXT,
                       metadata TEXT,
                       status TEXT,
                       error TEXT
                       )
        ''')
        # 메트릭 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metircs(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       run_id TEXT,
                       metric_name TEXT,
                       metirc_value REAL,
                       recorded_at TEXT,
                       FOREIGN KEY(run_id) REFERENCES runs(id)
                       )
        ''')
        # 토큰사용량 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS token_usage(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       run_id TEXT,
                       prompt_tokens INTEGER,
                       completion_tokens INTEGER,
                       total_tokens INTEGER,
                       estimated_cost REAL,
                       model TEXT,
                       recorded_at TEXT,
                       FOREIGN KEY(run_id) REFERENCES runs(id)
                       )
        ''')     
        conn.commit()
        conn.close()

    def start_run(self, name:str, run_type:str, input_data:Any, metadata:Dict=None) -> str:
        '''새 실행 시작'''
        run_id = str(uuid.uuid4)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO runs(id, name, run_type, start_time, input_data, metadata,status)
                       values(?,?,?,?,?,?,?)'''
                       ,(
                        run_id,name,run_type,datetime.now().isoformat(),
                        json.dumps(input_data, ensure_ascii=False) if input_data else None,
                        json.dumps(metadata, ensure_ascii=False) if input_data else None,   
                        'running'
                       ))
        conn.commit()
        conn.close()
        return run_id
    
    def end_run(self, run_id:str, output_data:Any, status:str='success', error:str=None):
        '''실행완료'''
        conn=sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        #시작시간 가져오기
        cursor.execute('SELECT start_time FROM runs WHERE ID = ?', (run_id))
        result = cursor.fetchone()
        if result:
            start_time = datetime.fromisoformat(result[0])
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            cursor.execute('''
                UPDATE runs
                           SET end_time = ?, duration_seconds=?, output_data=?,status=?,error=?
                           WHERE id = ?''',
                           (
                               end_time.isoformat(),
                               duration,
                               json.dumps(output_data,ensure_ascii=False) if output_data else None,
                               status,
                               error,
                               run_id
                           ))
            conn.commit()
            conn.close()

    def record_token_usage(self, run_id:str, prompt_tokens:int ,completion_tokens:int, model:str='gpt-4o-mini'):
        '''토큰사용량'''
        total_tokens = prompt_tokens + completion_tokens
        # 비용추정( gpt-4o-mini 기준)
        cost_per_1k_input = 0.00015
        cost_per_1k_output = 0.0006
        estimated_cost = (prompt_tokens / 1000*cost_per_1k_input + 
                          completion_tokens / 1000 * cost_per_1k_output)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO token_usage(run_id, prompt_tokens, completion_tokens, total_tokens, estimated_cost, model, recorded_at)
                       values(?,?,?,?,?,?,?)'''
                       ,(
                           run_id,prompt_tokens,completion_tokens,total_tokens,estimated_cost,model,datetime.now().isoformat()
                       ))
        conn.commit()
        conn.close()

    def get_summary(self) -> Dict:
        """전체 요약 통계"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 총 실행 수
        cursor.execute("SELECT COUNT(*) FROM runs")
        total_runs = cursor.fetchone()[0]
        
        # 평균 응답 시간
        cursor.execute("SELECT AVG(duration_seconds) FROM runs WHERE status = 'success'")
        avg_duration = cursor.fetchone()[0] or 0
        
        # 성공률
        cursor.execute("SELECT COUNT(*) FROM runs WHERE status = 'success'")
        success_count = cursor.fetchone()[0]
        success_rate = (success_count / total_runs * 100) if total_runs > 0 else 0
        
        # 총 토큰 사용량
        cursor.execute("SELECT SUM(total_tokens), SUM(estimated_cost) FROM token_usage")
        token_result = cursor.fetchone()
        total_tokens = token_result[0] or 0
        total_cost = token_result[1] or 0
        
        conn.close()
        
        return {
            "total_runs": total_runs,
            "avg_duration_seconds": round(avg_duration, 2),
            "success_rate": round(success_rate, 1),
            "total_tokens": total_tokens,
            "total_estimated_cost": round(total_cost, 4)
        }
    
    def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        """최근 실행 기록"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, run_type, start_time, duration_seconds, status
            FROM runs
            ORDER BY start_time DESC
            LIMIT ?
        """, (limit,))
        
        runs = []
        for row in cursor.fetchall():
            runs.append({
                "id": row[0][:8] + "...",  # 짧게 표시
                "name": row[1],
                "type": row[2],
                "time": row[3][:19] if row[3] else None,
                "duration": f"{row[4]:.2f}s" if row[4] else None,
                "status": row[5]
            })
        
        conn.close()
        return runs

#콜벡핸들러
class LocalMonitoringHandler(BaseCallbackHandler):
    '''로컬모니터링을 위한 콜백 핸들러
    모든 llm 호출을 로컬 SQLite3 DB에 기록'''
    def __init__(self,trace_db:LocalTraceDB, log_to_console:bool = True):
        self.trace_db = trace_db
        self.log_to_console = log_to_console
        self.current_run_id = None
        self.start_time = None
    def on_llm_start(self, serialized:Dict[str, Any], prompts:List[str], **kwargs):
        '''LLM 호출 시작'''
        self.start_time = datetime.now()
        model_name = serialized.get('name','UnKown')
        # db에 실행시간 기록
        self.current_run_id = self.trace_db.start_run(
            name = f'llm_call_{model_name}',
            run_type='llm',
            input_data={'prompts':prompts[:1]},
            metadata={'model':model_name}
        )
        if self.log_to_console:
            print(f"    llm 호출시작        : {self.start_time.strftime('%H:%M:%S')}")
            print(f"    모델               : {model_name}")
            print(f"    프롬프트길이        : {len(prompts[0])}")
    def on_llm_end(self, response,  **kwargs):
        '''llm 호출 완료'''
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # 토큰 사용량 추출
        token_usage = {}
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
        
        # 출력 텍스트 추출
        output_text = ""
        if response.generations:
            output_text = response.generations[0][0].text if response.generations[0] else ""
        
        # DB에 기록
        if self.current_run_id:
            self.trace_db.end_run(
                self.current_run_id,
                output_data={"response": output_text[:500]},  # 처음 500자만
                status="success"
            )
            
            # 토큰 사용량 기록
            if token_usage:
                self.trace_db.record_token_usage(
                    self.current_run_id,
                    token_usage.get('prompt_tokens', 0),
                    token_usage.get('completion_tokens', 0)
                )
            
            # 응답 시간 메트릭
            self.trace_db.record_metric(
                self.current_run_id, "latency_seconds", duration
            )
        
        if self.log_to_console:
            print(f"   LLM 호출 완료: {duration:.2f}초 소요")
            if token_usage:
                print(f"      입력 토큰: {token_usage.get('prompt_tokens', 'N/A')}")
                print(f"      출력 토큰: {token_usage.get('completion_tokens', 'N/A')}")
    
    def on_llm_error(self, error: Exception, **kwargs):
        """LLM 오류 발생"""
        if self.current_run_id:
            self.trace_db.end_run(
                self.current_run_id,
                output_data=None,
                status="error",
                error=str(error)
            )
        
        if self.log_to_console:
            print(f"    LLM 오류: {error}")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """체인 시작"""
        if self.log_to_console:
            chain_name = serialized.get('name', 'Unknown')
            print(f"\n    체인 시작: {chain_name}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """체인 완료"""
        if self.log_to_console:
            print(f"    체인 완료")
    
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs):
        """리트리버 시작"""
        if self.log_to_console:
            print(f"\n    검색 시작: '{query[:50]}...'")
    
    def on_retriever_end(self, documents: List[Document], **kwargs):
        """리트리버 완료"""
        if self.log_to_console:
            print(f"    검색 완료: {len(documents)}개 문서 반환")    

class LocalMonitoringHandler(BaseCallbackHandler):
    """
    로컬 모니터링을 위한 콜백 핸들러
    
    모든 LLM 호출을 로컬 SQLite DB에 기록합니다.
    """
    
    def __init__(self, trace_db: LocalTraceDB, log_to_console: bool = True):
        self.trace_db = trace_db
        self.log_to_console = log_to_console
        self.current_run_id = None
        self.start_time = None
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """LLM 호출 시작"""
        self.start_time = datetime.now()
        model_name = serialized.get("name", "Unknown")
        
        # DB에 실행 기록 시작
        self.current_run_id = self.trace_db.start_run(
            name=f"llm_call_{model_name}",
            run_type="llm",
            input_data={"prompts": prompts[:1]},  # 첫 프롬프트만 저장
            metadata={"model": model_name}
        )
        
        if self.log_to_console:
            print(f"\n   LLM 호출 시작: {self.start_time.strftime('%H:%M:%S')}")
            print(f"      모델: {model_name}")
            print(f"      프롬프트 길이: {len(prompts[0])}자")
    
    def on_llm_end(self, response, **kwargs):
        """LLM 호출 완료"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # 토큰 사용량 추출
        token_usage = {}
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
        
        # 출력 텍스트 추출
        output_text = ""
        if response.generations:
            output_text = response.generations[0][0].text if response.generations[0] else ""
        
        # DB에 기록
        if self.current_run_id:
            self.trace_db.end_run(
                self.current_run_id,
                output_data={"response": output_text[:500]},  # 처음 500자만
                status="success"
            )
            
            # 토큰 사용량 기록
            if token_usage:
                self.trace_db.record_token_usage(
                    self.current_run_id,
                    token_usage.get('prompt_tokens', 0),
                    token_usage.get('completion_tokens', 0)
                )
            
            # 응답 시간 메트릭
            self.trace_db.record_metric(
                self.current_run_id, "latency_seconds", duration
            )
        
        if self.log_to_console:
            print(f"   LLM 호출 완료: {duration:.2f}초 소요")
            if token_usage:
                print(f"      입력 토큰: {token_usage.get('prompt_tokens', 'N/A')}")
                print(f"      출력 토큰: {token_usage.get('completion_tokens', 'N/A')}")
    
    def on_llm_error(self, error: Exception, **kwargs):
        """LLM 오류 발생"""
        if self.current_run_id:
            self.trace_db.end_run(
                self.current_run_id,
                output_data=None,
                status="error",
                error=str(error)
            )
        
        if self.log_to_console:
            print(f"   LLM 오류: {error}")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """체인 시작"""
        if self.log_to_console:
            chain_name = serialized.get('name', 'Unknown')
            print(f"\n    체인 시작: {chain_name}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """체인 완료"""
        if self.log_to_console:
            print(f"    체인 완료")
    
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs):
        """리트리버 시작"""
        if self.log_to_console:
            print(f"\n    검색 시작: '{query[:50]}...'")
    
    def on_retriever_end(self, documents: List[Document], **kwargs):
        """리트리버 완료"""
        if self.log_to_console:
            print(f"    검색 완료: {len(documents)}개 문서 반환")

        



if __name__ == '__main__':
    # 로컬 모니터링 적용 RAG 체인
    trace_db = LocalTraceDB()
    print('sqllite 데이터베이스 초기화 완료')

    # 콜백핸들러 인스턴스(객체) 생성
    monitoring_handler = LocalMonitoringHandler(trace_db=trace_db)

