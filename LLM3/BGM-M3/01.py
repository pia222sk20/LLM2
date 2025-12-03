from langchain_huggingface import HuggingFaceEmbeddings

# BGE-M3 모델 로드
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={
        'device': 'cpu',  # GPU 사용 (없으면 'cpu')
        'trust_remote_code': True
    },
    encode_kwargs={
        'normalize_embeddings': True,  # 정규화
        'batch_size': 32
    }
)

# 임베딩 생성
vector = embeddings.embed_query("한국어 텍스트입니다.")
print(f"벡터 차원: {len(vector)}")  # 1024


from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 문장 인코딩
output = model.encode(
    ["한국어 임베딩 테스트"],
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True
)

# 세 가지 출력
dense_vecs = output['dense_vecs']      # (1, 1024)
sparse_vecs = output['lexical_weights'] # {토큰: 가중치}
colbert_vecs = output['colbert_vecs']  # (1, seq_len, 1024)

print(f'output : {output}')