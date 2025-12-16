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
            # 감독 노드 및 관계
            director_id = f"dir_{movie['director'].replace(' ','_')}"
            self.graph.add_node(director_id,type='direct',name=movie['director'])
            self.graph.add_edge(movie['id'],director_id,relation = 'directed_by')

            # 배우노드 및 관계
            for actor in movie['actors']:
                actor_id = f"act_{actor.replace(' ', '_')}"
                self.graph.add_node(actor_id,type='actor',name=actor)
                self.graph.add_edge(movie['id'],actor_id,relation = 'starts')
            # 장르노드 및 관계
            for genre in movie['genre']:
                genre_id = f"gen_{genre}"
                self.graph.add_node(genre_id,type='genre',name=genre)
                self.graph.add_edge(movie['id'],genre_id,relation = 'genre')
    
    def _handle_message(self, message: Message) -> Dict[str, Any]:
        content = message.content
        query_type = content.get('query_type', 'related')
        entity_id = content.get('entity_id', '')
        
        print(f"\n[{self.name}] 그래프 검색: {query_type} for {entity_id}")
        
        if query_type == 'related':
            # 관련 엔티티 찾기
            if entity_id in self.graph:
                neighbors = list(self.graph.neighbors(entity_id))
                related = []
                
                for neighbor in neighbors:
                    node_data = self.graph.nodes[neighbor]
                    edge_data = self.graph.edges[entity_id, neighbor]
                    related.append({
                        'id': neighbor,
                        'type': node_data.get('type', ''),
                        'name': node_data.get('name', node_data.get('title', '')),
                        'relation': edge_data.get('relation', '')
                    })
                
                print(f"  {len(related)}개 관련 엔티티 발견")
                
                return {
                    'status': 'found',
                    'entity_id': entity_id,
                    'related': related,
                    'count': len(related)
                }
        
        elif query_type == 'find_by_director':
            director_name = content.get('director', '')
            director_id = f"dir_{director_name.replace(' ', '_')}"
            
            if director_id in self.graph:
                # 감독의 영화 찾기
                movies = [n for n in self.graph.predecessors(director_id) 
                         if self.graph.nodes[n]['type'] == 'movie']
                
                movie_list = []
                for movie_id in movies:
                    movie_data = self.graph.nodes[movie_id]
                    movie_list.append({
                        'id': movie_id,
                        'title': movie_data['title'],
                        'year': movie_data['year']
                    })
                
                print(f"  ✓ {len(movie_list)}개 영화 발견")
                
                return {
                    'status': 'found',
                    'director': director_name,
                    'movies': movie_list,
                    'count': len(movie_list)
                }
        
        return {'status': 'not_found'}
    
