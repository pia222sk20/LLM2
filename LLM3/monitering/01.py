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


if __name__ == '__main__':
    LocalTraceDB()