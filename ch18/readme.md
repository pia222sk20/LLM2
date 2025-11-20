# Chapter 18. 트랜스포머 모형을 이용한 문서 요약

내용
- 문서 요약의 개념과 종류(추출적 vs 생성적) 이해
- Hugging Face Transformers를 이용한 문서 요약 실습
- T5, BART 등 주요 요약 모델의 특징 파악
- 한글 문서 요약 모델 활용

---

## 문서 요약의 이해

### 문서 요약이란?
긴 문서의 핵심 내용을 짧고 간결하게 추출하는 NLP 작업

### 요약 방식
- **추출적 요약(Extractive)**: 원문에서 중요한 문장을 그대로 추출
- **생성적 요약(Abstractive)**: 원문의 의미를 이해하고 새로운 문장으로 재생성
  - 트랜스포머 모델은 주로 생성적 요약 수행

---

## 파이프라인을 이용한 문서 요약

### 핵심 개념
Hugging Face의 `pipeline`은 복잡한 NLP 작업을 단 몇 줄로 실행할 수 있게 해주는 고수준 API

### 주요 특징
- 자동으로 모델과 토크나이저 로드
- 전처리/후처리 자동 수행
- 기본 모델: `sshleifer/distilbart-cnn-12-6`

### 실습 포인트
- 원문과 요약문 길이 비교
- 요약 품질 평가

---

## T5 모형과 자동 클래스를 이용한 문서 요약

### T5 (Text-to-Text Transfer Transformer)
- Google에서 개발한 범용 텍스트 생성 모델
- 모든 NLP 작업을 "텍스트→텍스트" 형식으로 통일
- Task Prefix 필요: `"summarize: "`, `"translate English to German: "` 등

### 생성 파라미터
- `num_beams`: Beam Search의 빔 개수 (높을수록 품질↑, 속도↓)
- `no_repeat_ngram_size`: n-gram 반복 방지
- `min_length`, `max_length`: 요약문 토큰 수 범위
- `early_stopping`: EOS 토큰 만나면 생성 종료

---

## T5 모형과 트레이너를 이용한 미세조정학습

### 미세조정(Fine-tuning)이란?
사전학습된 모델을 특정 도메인/작업에 맞게 추가 학습

### BillSum 데이터셋
- 미국 법안 문서와 요약문 쌍으로 구성
- 전처리: 원문 앞에 `"summarize: "` 추가

### Seq2SeqTrainer
- Sequence-to-Sequence 작업 전용 트레이너
- ROUGE 메트릭으로 요약 품질 자동 평가
- 체크포인트 자동 저장

---

## 한글 문서 요약

### 한글 요약 모델
1. **gogamza/kobart-summarization**
   - KoBART 기반 한국어 요약 모델
   - SKT에서 공개한 KoBART를 미세조정

2. **csebuetnlp/mT5_multilingual_XLSum**
   - 다국어 지원 T5 모델
   - 44개 언어 지원 (한국어 포함)

### 실습
- 한글 텍스트 전처리
- 모델별 성능 비교
- 도메인 특성에 맞는 모델 선택

---

## 요약

| 방법 | 장점 | 단점 | 사용 사례 |
|------|------|------|-----------|
| Pipeline | 간편, 빠른 프로토타이핑 | 커스터마이징 제한 | 데모, 간단한 요약 |
| Auto Classes | 유연성, 파라미터 조정 | 코드 복잡도 증가 | 프로덕션, 세밀한 조정 |
| Fine-tuning | 최고 성능, 도메인 특화 | 시간/자원 소모 | 특정 도메인 요약 |

---