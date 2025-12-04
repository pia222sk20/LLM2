# DuckDuckGo 무료웹검색
#pip install duckduckgo-search

# 설치 확인
# from duckduckgo_search import DDGS
from ddgs import DDGS
ddgs = DDGS()
# results = ddgs.text(
#     query='AI 인공지능',
#     region='ko-kr',
#     safesearch='moderate',  # 안전검색
#     max_results=10
#     )
# for r in results:
#     print(f"{r['title']}")
#     print(f"{r['href']}")
#     print(f"{r['body']}\n")

# results = ddgs.news(
#     query='AI 인공지능',
#     region='ko-kr',
#     safesearch='moderate',  # 안전검색
#     max_results=10
#     )
# for r in results:
#     print(f"{r}")    

# results = ddgs.images(
#     query='AI',
#     # region='ko-kr',
#     region='wt-wt',
#     safesearch='moderate',  # 안전검색
#     max_results=10
#     )
# for r in results:
#     print(f"{r}") 



results = ddgs.videos(
    query='LangGraph tutorial',
    # region='ko-kr',
    region='wt-wt',
    safesearch='moderate',  # 안전검색
    max_results=10
    )
for r in results:
    print(f"{r}")   


# 지역설정
print(''' 지역설정
ko-kr : 한국
ja-jp : 일본
en-us : 미국
en-gb : 영국
zh-cn : 중국
de-de : 독일
fr-fr : 프량스
wt-wt : 전세계                                               
''')


# RAG 하이브리드 
'''
사용자 질문 
내부문서 검색
문서평가 --> LLM으로 관련성 평가
if 충분함
    goto 답변생성
else
    duckduckgo 웹 검색    
    goto 답변생성

답변생성 --> LLM
'''

# Tavily VS duckduckgo
# Tavily : 검색 전체 데이터를 다 가져온다
# duckduckgo : 요약된 내용(하지만 출처도 있어서 필요하면 전체 내용을 크롤링)  빠르다. 어떠한 로그도 안 남는다(프라이버시 강함)
