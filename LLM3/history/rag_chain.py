from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
def build_rag_chain(vectordb,question):
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
    prompt = ChatPromptTemplate.from_template('''
당신은 RAG기반 쳇봇입니다.
다음 문맥을 참고해서 사용자의 질문에 답하세요

문맥:
{context}

질문:
{question}

답변:''')
    retriever = vectordb.as_retriever(search_kwargs={'k':3})
    docs = retriever.get_relevant_document(question)
    context = '\n\n'.join( [doc.page_contnet for doc in docs])
    message = prompt.format_messages(
        context = context,
        question =question
    )
    response = llm(message)
    return response.content