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
            CREATE TABLE IF NOT EXITST runs(
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
            CREATE TABLE IT NOT EXISTS metircs(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       run_id TEXT,
                       metric_name TEXT,
                       metirc_value REAL,
                       recorded_at TEXT
                       FOREIGN KEY(run_id) REFERENCES runs(id)
                       )
        ''')
        # 토큰사용량 테이블
        cursor.execute('''
            CREATE TABLE IT NOT EXISTS token_usage(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       run_id TEXT,
                       prompt_tokens INTEGER,
                       completion_tokens INTEGER,
                       total_tokens INTEGER,
                       estimated_cost REAL,
                       model TEXT,
                       recorded_at TEXT
                       FOREIGN KEY(run_id) REFERENCES runs(id)
                       )
        ''')     
        conn.commit()
        conn.close()
if __name__ == '__main__':
    LocalTraceDB()