import torch
import os
import sys

# ============================================================
# 환경 변수로 CPU/GPU 선택 가능
# ============================================================
# 
# GPU 사용 (기본값):
#   python script.py
#
# CPU 강제 사용:
#   $env:FORCE_CPU='1'; python script.py  (PowerShell)
#   set FORCE_CPU=1 && python script.py   (CMD)
#   FORCE_CPU=1 python script.py          (Linux/Mac)
#
# ============================================================

def get_device(force_cpu: bool = False, verbose: bool = True) -> torch.device:
    """
    최적의 디바이스를 자동으로 선택합니다.
    
    Args:
        force_cpu: True이면 GPU가 있어도 CPU 사용
        verbose: True이면 디바이스 정보 출력
    
    Returns:
        torch.device: 선택된 디바이스
    """
    # 환경 변수 확인
    env_force_cpu = os.environ.get('FORCE_CPU', '0').lower() in ('1', 'true', 'yes')
    
    if force_cpu or env_force_cpu:
        device = torch.device('cpu')
        if verbose:
            print(f"\n[디바이스 설정]")
            print(f"  모드: CPU (강제 지정)")
            print(f"  디바이스: {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            print(f"\n[디바이스 설정]")
            print(f"  모드: GPU")
            print(f"  디바이스: {device}")
            print(f"  GPU 이름: {torch.cuda.get_device_name(0)}")
            print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"  CUDA 버전: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        if verbose:
            print(f"\n[디바이스 설정]")
            print(f"  모드: CPU (GPU 없음)")
            print(f"  디바이스: {device}")
    
    return device


def check_gpu_compatibility():
    """GPU 호환성을 확인하고 권장 사항을 출력합니다."""
    print("\n" + "=" * 60)
    print(" GPU 호환성 확인")
    print("=" * 60)
    
    print(f"\n[PyTorch 정보]")
    print(f"  버전: {torch.__version__}")
    print(f"  CUDA 지원: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  cuDNN 버전: {torch.backends.cudnn.version()}")
        
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n[GPU 정보]")
        print(f"  이름: {gpu_name}")
        
        # RTX 50 시리즈 확인
        if '5070' in gpu_name or '5080' in gpu_name or '5090' in gpu_name:
            cuda_version = torch.version.cuda
            if cuda_version and float(cuda_version.split('.')[0]) >= 12 and float(cuda_version.split('.')[1]) >= 8:
                print(f"  상태:  RTX 50 시리즈 호환 (CUDA {cuda_version})")
            else:
                print(f"  상태:  RTX 50 시리즈는 CUDA 12.8+ 필요")
                print(f"  해결: pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128")
        else:
            print(f"  상태:  호환")
        
        # 간단한 GPU 테스트
        try:
            x = torch.randn(100, 100).cuda()
            y = x @ x.T
            del x, y
            torch.cuda.empty_cache()
            print(f"\n[GPU 테스트]")
            print(f"  연산 테스트:  성공")
        except Exception as e:
            print(f"\n[GPU 테스트]")
            print(f"  연산 테스트:  실패 - {e}")
    else:
        print(f"\n[권장 사항]")
        print(f"  GPU가 감지되지 않았습니다.")
        print(f"  CPU 모드로 실행됩니다 (속도가 느릴 수 있음).")


def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정"""
    import numpy as np
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimal_batch_size(model_size: str = 'base', device: torch.device = None) -> int:
    """
    GPU 메모리에 따라 최적의 배치 사이즈를 반환합니다.
    
    Args:
        model_size: 'tiny', 'small', 'base', 'large'
        device: torch.device
    
    Returns:
        int: 권장 배치 사이즈
    """
    if device is None:
        device = get_device(verbose=False)
    
    if device.type == 'cpu':
        # CPU는 메모리가 넉넉하지만 속도를 위해 작은 배치 사용
        batch_sizes = {'tiny': 32, 'small': 16, 'base': 8, 'large': 4}
    else:
        # GPU 메모리 확인
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory_gb >= 20:  # 24GB+ (RTX 3090, 4090, 5090)
            batch_sizes = {'tiny': 128, 'small': 64, 'base': 32, 'large': 16}
        elif gpu_memory_gb >= 10:  # 12GB (RTX 3080, 4070, 5070)
            batch_sizes = {'tiny': 64, 'small': 32, 'base': 16, 'large': 8}
        elif gpu_memory_gb >= 6:  # 8GB (RTX 3060, 4060)
            batch_sizes = {'tiny': 32, 'small': 16, 'base': 8, 'large': 4}
        else:  # 6GB 이하
            batch_sizes = {'tiny': 16, 'small': 8, 'base': 4, 'large': 2}
    
    return batch_sizes.get(model_size, 16)


# ============================================================
# 전역 설정 (import 시 자동 설정)
# ============================================================

# 기본 디바이스 (verbose=False로 조용히 설정)
DEVICE = get_device(verbose=False)

# GPU 사용 여부
USE_GPU = DEVICE.type == 'cuda'

# 기본 시드
DEFAULT_SEED = 42


# ============================================================
# 커맨드 라인 인터페이스
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ViT 학습 환경 설정 확인')
    parser.add_argument('--cpu', action='store_true', help='CPU 모드로 실행')
    parser.add_argument('--check', action='store_true', help='GPU 호환성 확인')
    args = parser.parse_args()
    
    if args.check:
        check_gpu_compatibility()
    else:
        device = get_device(force_cpu=args.cpu, verbose=True)
        print(f"\n[설정 완료]")
        print(f"  DEVICE: {DEVICE}")
        print(f"  USE_GPU: {USE_GPU}")