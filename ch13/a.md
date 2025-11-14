2. 인코더-디코더 구조
   
(1) 개념 정의

Encoder는 입력 시퀀스를 내부 상태로 변환하는 모듈, Decoder는 그 상태(컨텍스트)를 참조하며 출력 시퀀스를 생성하는 모듈입니다.

(2) 왜 필요한가

기능 분리: 이해(Encoding)와 생성(Decoding)을 모듈화하여 확장성 확보.

다양한 입력/출력 도메인(텍스트→텍스트, 텍스트→태그 등)에 쉽게 적용.

나중에 Attention 같은 추가 요소 삽입이 용이 (기본 Encoder-Decoder가 뼈대).

(3) 동작 원리

Encoder 단계:

입력 토큰 $x₁, x₂, …, x_T$

순환구조(RNN/LSTM/GRU)가 $h_t = f(x_t, h_{t-1})$ 계산

최종 $h_T$ 혹은 전체 ${h₁…h_T}$ 를 요약(기본 Seq2Seq는 $h_T$ 단일)

Decoder 단계:

초기 Hidden State: $h_T$ (Encoder 마지막 상태)

반복: $y_t = Softmax(W·h_t^dec)$

다음 입력: (Teacher Forcing 시 GT 토큰, 아니면 이전 예측)

(4) 실제 사용 예시

작업	Encoder 입력	Decoder 출력

번역	영어 문장 토큰	한국어 문장 토큰

질의응답	질문 토큰	답변 토큰

요약	긴 문서 토큰	요약 문장 토큰

형태소 분석	음절 시퀀스	품사 태그 시퀀스
