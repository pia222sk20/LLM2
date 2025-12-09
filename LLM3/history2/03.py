"""
Redis + OpenAI 임베딩 + GPT
"""

import redis
import json
import numpy as np
from openai import OpenAI
from datetime import datetime


class SimpleRAG:
    def __init__(self, openai_api_key):
        self.redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.openai = OpenAI(api_key=openai_api_key)
        self.docs = []  # 문서 저장
        self.embeddings = []  # 임베딩 저장
    
    # 문서 추가
    def add_doc(self, content, source):
        emb = self.openai.embeddings.create(
            model='text-embedding-3-small',
            input=content
        ).data[0].embedding
        
        self.docs.append({'content': content, 'source': source})
        self.embeddings.append(emb)
        print(f" 문서 추가: {source}")
    
    # 문서 검색
    def search(self, query, top_k=2):
        query_emb = self.openai.embeddings.create(
            model='text-embedding-3-small',
            input=query
        ).data[0].embedding
        
        # 코사인 유사도
        scores = np.dot(self.embeddings, query_emb)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        
        return [self.docs[i] for i in top_idx]
    
    # 메시지 저장
    def save_msg(self, user_id, role, content):
        key = f"chat:{user_id}"
        msg = json.dumps({
            'role': role,
            'content': content,
            'time': datetime.now().isoformat()
        })
        self.redis.lpush(key, msg)
        self.redis.ltrim(key, 0, 9)  # 최근 10개만
    
    # 메시지 불러오기
    def load_msgs(self, user_id):
        key = f"chat:{user_id}"
        msgs = self.redis.lrange(key, 0, -1)
        result = [json.loads(m) for m in msgs]
        result.reverse()  # 시간순 정렬
        return result
    
    # 답변 생성
    def ask(self, user_id, question):
        # 1. 문서 검색
        docs = self.search(question)
        context = '\n'.join([f"[{d['source']}] {d['content']}" for d in docs])
        
        # 2. 대화 내역
        history = self.load_msgs(user_id)
        messages = [{'role': 'system', 'content': '문서 기반으로 답변하세요'}]
        messages.extend([{'role': h['role'], 'content': h['content']} for h in history[-4:]])
        
        # 3. GPT 호출
        messages.append({
            'role': 'user',
            'content': f"문서:\n{context}\n\n질문: {question}"
        })
        
        response = self.openai.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages
        )
        answer = response.choices[0].message.content
        
        # 4. 대화 저장
        self.save_msg(user_id, 'user', question)
        self.save_msg(user_id, 'assistant', answer)
        
        return answer



def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    # 초기화
    rag = SimpleRAG(openai_api_key=os.getenv('OPENAI_API_KEY'))
    
    # 문서 추가
    rag.add_doc(
        "재택근무는 주 3일 가능. VPN 필수 사용",
        "정책.pdf"
    )
    rag.add_doc(
        "복지: 식대 12,000원, 자기계발비 100만원",
        "복지.pdf"
    )
    
    print("\n" + "="*50)
    
    # User1 대화
    print("\n[User1 - Alice]")
    ans1 = rag.ask('alice', '재택근무 정책은?')
    print(f"Q: 재택근무 정책은?")
    print(f"A: {ans1}\n")
    
    ans2 = rag.ask('alice', 'VPN은 왜 필요해?')
    print(f"Q: VPN은 왜 필요해?")
    print(f"A: {ans2}\n")
    
    # User2 대화 (독립적)
    print("\n[User2 - Bob]")
    ans3 = rag.ask('bob', '복지 제도는?')
    print(f"Q: 복지 제도는?")
    print(f"A: {ans3}\n")
    
    ans4 = rag.ask('bob', '식대는 얼마야?')
    print(f"Q: 식대는 얼마야?")
    print(f"A: {ans4}\n")
    
    # 대화 내역 확인
    print("\n" + "="*50)
    print("대화 내역")
    print("="*50)
    
    print("\nAlice:")
    for msg in rag.load_msgs('alice'):
        print(f"  {msg['role']}: {msg['content'][:30]}...")
    
    print("\nBob:")
    for msg in rag.load_msgs('bob'):
        print(f"  {msg['role']}: {msg['content'][:30]}...")


if __name__ == '__main__':
    main()