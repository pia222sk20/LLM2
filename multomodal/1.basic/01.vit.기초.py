import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# 1. 패치분할
def patch_embedding():
    '''이미지를 패치로 분할하는 과정(patch embedding)'''
    # 설정
    image_isze = 224  # (224 x 224)
    patch_size = 16   # (16 x 16)
    channels = 3
    embedding_dim = 768

    # 패치수 계산
    num_patches = (image_isze // patch_size) **2
    print(f'    이미지크기 : {image_isze} x {image_isze}')
    print(f'    패치크기 : {patch_size} x {patch_size}')
    print(f'    채널수 : {channels}')
    print(f'    패치 수 : {image_isze // patch_size} x {image_isze // patch_size}')

    # 더미 이미지 생성
    dummy_image = torch.randn(1, channels, image_isze,image_isze)
    print(f'    더미 이미지 생성')
    print(f'    입력 이미지 shape : {dummy_image.shape}')  # [1, 3, 224, 224]

    # 패치분할(Conv2d 사용)
    # Conv2d stride = patch_size 겹치지 않는 패치 추출
    patch_embed = nn.Conv2d(in_channels=channels, out_channels=embedding_dim,kernel_size=patch_size,stride=patch_size)

    # 패치 임베딩 적용
    patches =  patch_embed(dummy_image)
    print(f'\n패치임베딩 후')
    print(f'    Conv2d 출력 sahpe : {patches.shape}')  # [1, 768, 14 , 14]

    # Flatten : (B,D,H,W) -> (B, N, D)  (1,196,768)
    patches_flat = patches.flatten(2).transpose(1,2)
    print(f'    Flatten 후 sahpe : {patches_flat.shape}')  # [1, 196, 768]

    # 각 패치는 768차원 벡터
    print(f'   \n패치수 : {patches_flat.shape[1]}')
    print(f'   각 패치의 임베딩 차원 수 : {patches_flat.shape[2]}')
    return patches_flat


# 2. 위치임베딩의 역활
def positional_embedding():
    '''위치 임베딩'''
    num_patches = 196
    embedding_dim = 768

    # 위치 임베딩 생성
    # 이 텐서는 학습대상 Optimizer의해 업데이트
    position_embedding =  nn.Parameter( torch.randn(1, num_patches+1,embedding_dim))   # +1은 CLS 토큰
    print(f'    위치 임베딩 shape : {position_embedding.shape}')
    print(f'    총 위치수 : {num_patches+1}  (패치 196 + cls토큰 1)')
    # 배치차원 제거  :각 위치를 하나의 벡터로 다루기위해 배치크기가 1인 형태는 분석시 불 필요
    pos_emb = position_embedding.squeeze(0)
    # 코사인 유사도 계산을 위해서 정규화는 필수
    pos_emb_norm = pos_emb/pos_emb.norm(dim=1,keepdim=True)  # 단위벡터로 나눔
    # 코사인 유사도 행렬
    similarity =  torch.mm(pos_emb_norm,pos_emb_norm.t())
    print(f'    유사도 행렬 shape : {similarity.shape}')   # [197 197]
    # 학습전이라서 랜덤이지만 학습후에는 인접 패치끼지 유사해짐
    return position_embedding

# 3. CLS 토큰
def cls_token():
    embedding_dim = 768
    num_paches = 196
    batch_size = 2
    #cls 토큰 생성
    cls_token = nn.Parameter(torch.rand(1,1,embedding_dim))  # 학습을 통해 이미지를 요약하는 벡터로 진화
    print(f'    CLS 토큰 shape : {cls_token.shape}')  # [1,1,768]
    # 배치크기에 맞게 CLS 토큰 확장
    cls_tokens = cls_token.expand(batch_size,-1,-1)  # 배치마다 동일한 cls 토큰이 필요, expand는 view만 확장
    print(f'    배치확장 후 shape : {cls_tokens.shape}')  # [2,1,768]
    #패치 임베딩 생성
    patch_embeddings = torch.randn(batch_size, num_paches,embedding_dim)
    print(f'    패치 임베딩 shape : {patch_embeddings.shape}')
    # CLS 토큰을 시퀀스 맨 앞에 추가
    embeddings = torch.cat([cls_tokens,patch_embeddings], dim=1)  # [cls | patch1.... patch196]
    print(f'결합후 shape : {embeddings.shape}') #[2,197,768]    
    return embeddings

# 4 self-attention
def self_attention():
    '''self-attention 메커니즘
    Mukti-Head Self-attention의 구성요소
        Q(query) K(key) V(value) 
    '''
    embedding_dim = 768
    num_heads = 12
    head_dim = embedding_dim//num_heads  # 각 헤드가 담당하는 차원
    seq_len  = 197 # 196패치 + 1 CLS 
    batch_size = 1  # 샘플1개
    # 더미데이터 생성
    x = torch.randn(batch_size,seq_len,embedding_dim)
    # QKV 선형 레이어
    qkv_proj = nn.Linear(embedding_dim,embedding_dim*3) # 하나의 Linear 연산으로 QKV 동시에 생성
    # QKV 계산
    qkv = qkv_proj(x)
    print(f'    QKV shape:{qkv.shape}')
    # Q K V  분리
    qkv = qkv.reshape(batch_size, seq_len,3, num_heads,head_dim)  # [B, N,3,  heads, head_dim]
    qkv = qkv.permute(2,0,3,1,4)  # [3,B,heads,N,head_dim]
    q, k , v = qkv[0],qkv[1],qkv[2]
    print(f'Q shape : {q.shape}') # 동일한 모양
    print(f'K shape : {k.shape}') # 동일한 모양
    print(f'V shape : {v.shape}') # 동일한 모양
    # Attention Score 계산
    # Attention Score 계산
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    print(f"\n[Attention Score]")
    print(f"  Attention shape: {attn.shape}")  # [1, 12, 197, 197]
    
    # Softmax
    attn = attn.softmax(dim=-1)
    print(f"  Softmax 후 합계 (각 행): {attn[0, 0, 0].sum().item():.4f}")  # 1.0
    
    # Value와 곱하기
    out = attn @ v
    print(f"\n[Attention 적용]")
    print(f"  출력 shape: {out.shape}")  # [1, 12, 197, 64]
    
    # 헤드 결합
    out = out.transpose(1, 2).reshape(batch_size, seq_len, embedding_dim)
    print(f"  헤드 결합 후: {out.shape}")  # [1, 197, 768]
    
    return attn

def mlp():
    '''mlp 블럭'''
    embedding_dim = 768
    mlp_dim = embedding_dim*4 # 일반적으로 4배 확장
    print(f'입력/출력 차원 : {embedding_dim}')
    print(f'히든 차원 : {mlp_dim}')
    # MLP 블럭 정의
    mlp = nn.Sequential(
        nn.Linear(embedding_dim,mlp_dim),
        nn.GELU(), # 단순선형을 비 선형으로 변형
        nn.Linear(mlp_dim,embedding_dim)
        )
     # 파라미터 수 계산
    total_params = sum(p.numel() for p in mlp.parameters())
    print(f"  MLP 파라미터 수: {total_params:,}")
    
    # 더미 입력
    x = torch.randn(1, 197, embedding_dim)
    print(f"\n[입력/출력]")
    print(f"  입력 shape: {x.shape}")
    
    # MLP 적용
    out = mlp(x)
    print(f"  출력 shape: {out.shape}")
    
    print(f"\n[GELU 활성화 함수]")
    print(f"  GELU(x) = x * Phi(x)")
    print(f"  ReLU보다 부드럽고, 음수 입력에도 작은 값 출력")
    
    return out

# Transformer Encoderder block 구조
def transformer_block():
    '''Transformer Encoder 블럭'''
    class TransformerBlock(nn.Module):
        """간단한 Transformer Block 구현"""
        def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim),
            )
        
        def forward(self, x):
            # Pre-norm 구조
            # Attention with residual
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            # MLP with residual
            x = x + self.mlp(self.norm2(x))
            return x
    
    # 블록 생성
    block = TransformerBlock()
    
    # 파라미터 수
    total_params = sum(p.numel() for p in block.parameters())
    print(f"\n[Transformer Block 구성]")
    print(f"  1. Layer Normalization")
    print(f"  2. Multi-Head Self-Attention")
    print(f"  3. Residual Connection")
    print(f"  4. Layer Normalization")
    print(f"  5. MLP (Feed-Forward)")
    print(f"  6. Residual Connection")
    print(f"\n  블록당 파라미터 수: {total_params:,}")
    
    # 더미 입력
    x = torch.randn(1, 197, 768)
    print(f"\n[입력/출력]")
    print(f"  입력 shape: {x.shape}")
    
    out = block(x)
    print(f"  출력 shape: {out.shape}")
    
    # ViT-Base는 12개 블록
    print(f"\n[ViT-Base 전체 파라미터]")
    print(f"  12개 블록 파라미터: {total_params * 12:,}")
    
    return block

# self attention이 어디를 주목하는지 시각화

def main():
    # 1 패치임베딩
    patcheds = patch_embedding()
    # 2 위치임베딩
    pos_embed = positional_embedding()
    # 3. CLS 토큰
    embeddings = cls_token()
    # 4. self-Attention
    attention = self_attention()
    # 5. MLP
    mlp_output = mlp()
    # 6. transformer block
    block = transformer_block()
def _visualization_cls_attention(attn, image_size=224, patch_size=16):
    '''cls토큰이 각 패치를 얼마나 주목하는지 시각화
    attn : [1, heads,197,197]
    '''
    # cls -> patch attention만 추출
    cls_attn = attn[0, :, 0 , 1: ] # [heads, 196]
    # head평균
    cls_attn_mean = cls_attn.mean(dim=0)  # [196]
    # 14 x 14 reshape
    grid_size = image_size // patch_size
    attn_map = cls_attn_mean.reshape(grid_size,grid_size)
    # 정규화
    attn_map = attn_map / attn_map.max()
    return attn_map.detach().cpu().numpy()

# 더미이미지 + Attention 시각화
import requests
from PIL import Image
from io import BytesIO
def demo_cls_attention_visualization(is_dummy:bool = True, url:str = None,image_size:str = 224):
    # 1. 더미 이미지
    if is_dummy:
        image = np.random.rand(224,224,3)
    elif url.startswith("http"):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = image.resize((image_size, image_size))
        image = np.array(image) / 255.0  # 정규화
    else:  # 로컬 이미지 로드
        image = Image.open(url).convert('RGB')
        image = image.resize((image_size, image_size))
        image = np.array(image) / 255.0  # 정규화

    # self - attention 계산
    attn =  self_attention()
    # 3. attention map 생성
    attn_map = _visualization_cls_attention(attn)
    # 시각화
    plt.imshow(image)
    plt.imshow(attn_map, cmap='jet', alpha=0.5)
    plt.axis('off')
    # plt.savefig('vit_visialization.png')
    plt.show()




if __name__=='__main__':
#    main()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    url = 'C:/2.Lecture/LLM2/multomodal/1.basic/img/Cat03.jpg'
    demo_cls_attention_visualization(False, url=url)