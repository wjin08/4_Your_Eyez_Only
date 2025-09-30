

<div align="center">

# 👕 4 Your Eyez Only

**시각장애인을 위한 AI 비전 어시스턴트**


</div>

---

## 📌 프로젝트 개요

4 Your Eyez Only는 시각장애인을 위한 **위험 감지**와 **옷차림 추천**을 제공하는 AI 비전 어시스턴트입니다.

### 주요 기능
- 🚨 **위험 감지**: 화재, 건설현장, 총기, 폭발 등 실시간 위험 상황 감지 및 음성 알림
- 👔 **코디 추천**: 카메라로 옷의 유형과 색상을 인식하고 LLM 기반 스타일링 조언 제공
- 📱 **모바일 연동**: 안드로이드 앱을 통한 편리한 사용자 인터페이스
- 🖥️ **Qt 서버 관리**: Qt 기반 서버 매니저로 시스템 모니터링 및 제어


<img width="688" height="532" alt="스크린샷 2025-09-16 151956" src="https://github.com/user-attachments/assets/c257826e-05e7-49da-a439-ed8293c81c1e" />



<img width="500" height="600" alt="4YourEyes System" src="https://github.com/user-attachments/assets/ada0de24-d63b-4cd6-a93c-ecc28dbea649" />

---

## 🎯 핵심 아이디어

### 옷 인식: Detection + Classification 체인

시각장애인에게는 **'이게 어떤 옷인가'**를 정확하게 알려주는 단계가 필수입니다.

- **Detection**: YOLOX-Tiny로 옷 영역을 안정적으로 탐지 → 배경/노이즈 영향 최소화
- **Classification**: EfficientNetV2-S로 `반팔/긴팔/후드/청바지/반바지` 5종 세분류
- **Intel Geti**: 전처리 → 학습 → 평가 → 배포를 일관되게 운용

### 위험 감지 시스템

실시간 카메라 스트리밍을 통해 위험 상황을 감지하고 즉각적인 음성 피드백 제공

---

## 📊 데이터셋 전략

### 문제점과 해결
- **초기**: 구글 크롤링 이미지 → 스톡/규격 이미지로 **실전 적합도 낮음**
- **보강**: **실사 200장 직접 촬영**(목동 현대백화점 -> 유니클로/탑텐/원더플레이스)

<img width="1235" height="1021" alt="KakaoTalk_20250917_172106770" src="https://github.com/user-attachments/assets/b85f1fdb-6d0d-4536-b7c9-e78084873ed7" />

![KakaoTalk_20250917_172356009](https://github.com/user-attachments/assets/d4fa2dda-5684-48c3-92dd-38d3a837c67d)


  - 조명/각도/배경이 실제 환경과 유사한 네이티브 샘플 확보
 
    

<img width="2368" height="1185" alt="직접 찍은 데이터셋 모음" src="https://github.com/user-attachments/assets/180faebe-23cb-4eb8-bfbf-793566b29d2a" />


    
- **효과**: 재학습 후 **리얼타임 테스트에서 오탐 감소, 정확도 향상**

**핵심**: "데이터가 문제" — 실제 환경과 가까운 데이터가 성능의 핵심

---

## 🏗️ 시스템 아키텍처

![4YEO](https://github.com/user-attachments/assets/199cfacd-f49f-4a1b-af84-d0a1eb243d0a)


### 전체 흐름

```text
[카메라 입력]
    │
    ├─→ [위험 감지 모델] → 위험 감지 시 음성 알림
    │
    └─→ [YOLOX-Tiny Detection] → [Crop] → [EfficientNetV2-S Classification]
            │
            ├─→ {옷 유형, 색상 추출}
            │
            └─→ [LLM 서버] → 코디 추천 문장 → TTS/화면 출력
                    ↕
            [Qt 서버 매니저] ← 시스템 모니터링 및 제어
```

### 컴포넌트 구성
- **Vision Client** (Jetson Nano): 카메라 입력, 탐지/분류, 색상 추출
- **LLM Server**: 옷 정보 입력 → 코디 추천 응답 (선호/계절/날씨 반영)
- **Qt 서버 매니저**: 소켓 통신을 통한 AI 서버 관리 및 시스템 상태 모니터링
- **Mobile App**: 안드로이드 앱을 통한 UI 제공
- **TTS Module**: 음성 피드백 시스템



---

## 🤖 모델 구성

### 옷 인식 체인
- **Detection**: YOLOX-Tiny
- **Classification**: EfficientNetV2-S
- **Pipeline**: Intel Geti Detection→Classification 체인
- **Labels(5)**: `ShortSleeve`, `LongSleeve`, `Hoodie`, `Jeans`, `Shorts`

### 위험 감지
- 화재, 건설현장, 총기, 폭발 등 다양한 위험 상황 감지
- 실시간 스트리밍 처리 최적화

---

## ⚡ 디플로이 & 런타임 비교 (Jetson Nano)

<img width="2421" height="1211" alt="KakaoTalk_20250930_095048240" src="https://github.com/user-attachments/assets/cd3215c3-f444-4ce7-be83-e48a0d504bc6" />



**운용 전략**: LLM 서버 방식 메인 + 온디바이스 Lite 모델 보조 + Qt 서버 매니저를 통한 중앙 관리


---

## 📡 API 프로토콜

### Vision → LLM Server

```json
{
  "type": "Hoodie",
  "color": "Black"
}
```

### LLM Server → Vision

```json
{
  "recommendation": "블랙 후드에는 연청 데님과 흰 스니커즈가 잘 어울립니다."
}
```

### Qt Server Manager ↔ System

Qt 서버 매니저는 소켓 통신(`socketclient.cpp`)을 통해 AI 서버와 통신하며, 탭 기반 UI(`tab1.cpp`)로 시스템 상태를 시각화합니다.

---

## 📁 리포지토리 구조

```
4YourEyes/
├── README.md
├── VIT_DETR_MODEL/
│   ├── dataset/
│   ├── infer_add_color.py
│   ├── models/
│   ├── train.py
│   ├── train_multi.py
│   └── utils.py
│
├── for_Jetson/
│   └── openvino/
│       ├── LICENSE
│       ├── README.md
│       ├── deployment/
│       └── example_code/
│
└── qt/
    └── 4YEO_Server_Manager/
        ├── 4YourEyesOnly/
        ├── 4YourEyesOnly.pro        # Qt 프로젝트 파일
        ├── main.cpp                 # 메인 엔트리 포인트
        ├── socketclient.cpp/.h      # 소켓 통신 모듈
        ├── tab1.cpp/.h/.ui          # UI 탭 컴포넌트
        ├── widget.cpp/.h/.ui        # 메인 위젯
        ├── ai_sever.c               # AI 서버 연동
        └── syntax/                  # 구문 정의
```

---

## 🎬 시연 영상

https://github.com/user-attachments/assets/a497f0ac-2e05-4e3b-bd7c-a2312b71a258

### 사용 시나리오

1. **옷 스타일링**: 거울 앞에서 캡처 → AI 코디 추천 수신
2. **위험 감지**: 실시간 위험 상황 감지 → 즉각 음성 알림
3. **서버 관리**: Qt 매니저를 통한 시스템 모니터링 및 제어

<img width="400" height="300" alt="Danger Detection" src="https://github.com/user-attachments/assets/6fe1b7ba-19e4-4d75-a5f1-92d297c0fa27" />

---

## ✨ 주요 성과

- ✅ **실사 데이터 보강**으로 현장 정확도/안정성 향상
- ✅ **Detection + Classification 체인**으로 안정적인 옷 유형 인식
- ✅ **실시간 위험 감지** 및 음성 피드백 시스템 구현
- ✅ **모바일 앱 연동**으로 사용자 편의성 극대화
- ✅ **Qt 기반 서버 매니저**로 시스템 통합 관리 구현
- ✅ **모듈형 아키텍처**로 유지보수 및 확장 용이



</div>
