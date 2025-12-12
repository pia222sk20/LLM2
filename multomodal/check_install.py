# 작동원리
    # - 이미지를 패치로 나누기  ex)224x224 --> 16x16  196개 패치
    # - 패치를 벡터로 변환  
    # - 위치정보를 추가 : 각 패치가 이미지의 어디에 위치했는지 정보를 추가
    # - Transformer 처리 -  학습
    # - 분류
# CNN은 특정 이미지의 부분을 보는관점,  vit 이미지 전체를 한번에 볼수 있음
# 가상환경은 python 3.10 가장 안정

# conda create -n vit_env python=3.10 -y
# conda activate vit_env
# conda install pytorch torchvision transformers datasets 
# conda install pillow requests numpy scikit-learn matplotlib tqdm 
# pip install torch
# pip install accelerate

# 라이브러리 설치 확인
import torch
print(f'torch version : {torch.__version__}')
print(f'cuda  : {torch.cuda.is_available()}')
import transformers
print(f'transformers : {transformers.__version__}')

package_to_check = [
    ('numpy','Numpy'),
    ('PIL','Pillow'),
    ('matplotlib','Matpltlib'),
    ('sklearn','Sckit-learn'),
    ('requests','Requests'),
]
all_check = True
for module_name , install_name in package_to_check:
    try:
        module = __import__(module_name)
        version = getattr(module,"__version__","unknown")
        print(f'    {install_name} : {version}')
    except ImportError:
        print(f'    {install_name} : Not Found')
        all_check=False
    
if all_check:
    print('all import success')