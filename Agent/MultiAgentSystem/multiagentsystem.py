SAMPLE_MOVIES = [
    {
        "id": "m1",
        "title": "The Shawshank Redemption",
        "year": 1994,
        "director": "Frank Darabont",
        "genre": ["Drama", "Crime"],
        "actors": ["Tim Robbins", "Morgan Freeman"],
        "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
        "rating": 9.3
    },
    {
        "id": "m2",
        "title": "The Godfather",
        "year": 1972,
        "director": "Francis Ford Coppola",
        "genre": ["Crime", "Drama"],
        "actors": ["Marlon Brando", "Al Pacino"],
        "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        "rating": 9.2
    },
    {
        "id": "m3",
        "title": "The Dark Knight",
        "year": 2008,
        "director": "Christopher Nolan",
        "genre": ["Action", "Crime", "Drama"],
        "actors": ["Christian Bale", "Heath Ledger"],
        "plot": "When the menace known as the Joker wreaks havoc on Gotham, Batman must accept one of the greatest tests.",
        "rating": 9.0
    },
    {
        "id": "m4",
        "title": "Pulp Fiction",
        "year": 1994,
        "director": "Quentin Tarantino",
        "genre": ["Crime", "Drama"],
        "actors": ["John Travolta", "Samuel L. Jackson"],
        "plot": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.",
        "rating": 8.9
    },
    {
        "id": "m5",
        "title": "Inception",
        "year": 2010,
        "director": "Christopher Nolan",
        "genre": ["Action", "Sci-Fi", "Thriller"],
        "actors": ["Leonardo DiCaprio", "Joseph Gordon-Levitt"],
        "plot": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
        "rating": 8.8
    }
]

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any,List,Optional
from datetime import datetime
import uuid
import json
import os

import chromadb
from chromadb.config import Settings
import openai
import numpy as np

# 기본 구조
from defaultAgent import AgentState,Message,SpecializedAgent,Corrdinator

# RAG 특화 에이전트
class VectorDBAgent(SpecializedAgent):
    '''벡터 DB 검색 에이전트 (ChromaDB)'''
    def __init__(self, name:str):
        super().__init__(name, 'vector_search')
        # ChromaDB 초기화
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,  # 익명 사용정보 수집 및 전송 비활성
            allow_reset=True    # 외부에서 db를 reset 할수 있는 api 허용
        ))
        try:
            self.client.delete_collection('movies')
        except:
            pass
        self.collection = self.client.create_collection(
            name = 'movies',
            metadata = {'description':'movie information database'}
        )
        self._initialize_db()
    def _get_embedding(self, text:str) -> List[float]:
        '''openai 임베딩 생성'''
        try:
            response = openai.embeddings.create(
                model='text-embedding-3-small',
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f'임베딩 생성 실패 : {e}')
            return [0.0]*1536
    def _initialize_db(self):
        '''영화 데이터를 vectordb 저장'''
        for movie in SAMPLE_MOVIES:
            doc_text = f"{movie['title']} ({movie['year']}). {movie['plot']}"

            self.collection.add(
                ids=[movie['id']],
                documents=[doc_text],
                embeddings=[self._get_embedding(doc_text)],
                metadatas=[{
                    'title':movie['title'],
                    'year':movie['year'],
                    'director':movie['director'],
                    'rating':movie['rating']
                }]
            )
    def _handle_message(self, message:Message)->Dict[str, Any]:
        content = message.content
        query = content.get('query','')
        top_k = content.get('top_k',3)
        query_embedding = self._get_embedding(query)
        results =  self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        retrieved_docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0

                retrieved_docs.append({
                    'id' : results['ids'][0][i],
                    'content':doc,
                    'metadata' : metadata,
                    'similarity' : 1-distance
                })
        return {
            'status':'retrieved',
            'query' : query,
            'results':retrieved_docs,
            'count' : len(retrieved_docs)
        }
# pip install networkx
import networkx as nx
class KnowledgeGraphAgent(SpecializedAgent):
    '''지식그래프 검색 에이전트(Networkx)'''
    def __init__(self, name:str):
        super().__init__(name, 'knowledge_graph')
        self.graph = nx.DiGraph()
        self._initialize_graph()
    def _initialize_graph(self):
        '''영화 관계 그래프 생성'''
        for movie in SAMPLE_MOVIES:
            # 영화 노드
            self.graph.add_node(movie['id'],type='movie',title=movie['title'],year=movie['year'])
            