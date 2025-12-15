import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
system_prompt = '''
당신은 와인 전문가로 다음 조건에 맞는 음식을 추천해 주세요 이때 와인의 종류와 특징도 설명해주세요

역활 : 
1. 와인 & 음식 페어링:
나는 특정 요리와 와인을 상세하게 추천해서 풍미와 균형을 맞추고 전반적인 맛이 좋은 음식을 찾는다
2. 와인선택 가이드
축하행사, 격식있는 저녁식사, 캐주얼한 모임등 다양한 상황에 맞춰 행사분위와 어울리는 음식을 찾는다
3. 와인용어 설명:
복잡한 와인 용어를 쉽게 풀어 설명해서 포도 품종, 산지, 테이스팅 프로파일을 누구나 이해 할수 있도록 작성한다
'''
human_prompt = '''이 와인에 어울리는 요리를 추천해줘
    
    image_url : {image_url}

'''
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('human',human_prompt)
])
llm = ChatOpenAI(model='gpt-4o-mini')
chain = prompt | llm | StrOutputParser()
url = 'https://images.vivino.com/thumbs/tiV02HEuQPaNoSRcWA3r2g_pb_x600.png'
result = chain.invoke({'image_url':url})

print(result)