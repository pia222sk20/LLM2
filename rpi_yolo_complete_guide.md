# YOLO8 라즈베리파이 원격 객체 인식 완벽 가이드

## 📋 목차
1. [필수 준비물](#필수-준비물)
2. [설치 방법](#설치-방법)
3. [코드 저장](#코드-저장)
4. [연결 방법](#연결-방법)
5. [실행 방법](#실행-방법)
6. [트러블슈팅](#트러블슈팅)

---

## 필수 준비물

- **라즈베리파이 4 이상** (권장: 라즈베리파이 4B 4GB 이상)
- **라즈베리파이 카메라 모듈** (또는 USB 웹캠)
- **윈도우 PC**
- **같은 WiFi 네트워크**
- **전원 공급** (라즈베리파이 충분한 전력 필요)

---

## 설치 방법

### 1단계: 라즈베리파이 OS 설치 및 기본 설정

```bash
# 라즈베리파이 OS 설치 (라즈베리파이 임저 사용)
# https://www.raspberrypi.com/software/ 다운로드 후 설치

# 라즈베리파이에 SSH로 접속
ssh pi@raspberrypi.local
# 또는
ssh pi@<라즈베리파이_IP주소>
# 기본 비밀번호: raspberry
```

### 2단계: 라즈베리파이 패키지 업데이트

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 3단계: 필수 라이브러리 설치

```bash
# Python3 및 pip 설치
sudo apt-get install python3-pip python3-dev -y

# OpenCV 설치 (라즈베리파이 최적화 버전)
sudo apt-get install libatlas-base-dev libjasper-dev libtiff5 -y
sudo apt-get install libharfbuzz0b libwebp6 libtiff5 libjasper1 libopenjp2-7 -y
pip3 install opencv-python

# YOLO8 및 필수 패키지 설치
pip3 install ultralytics torch torchvision
pip3 install numpy

# 카메라 지원 패키지
sudo apt-get install python3-picamera2 -y
```

### 4단계: 윈도우 설치 (옵션)

```powershell
# 윈도우 PowerShell에서 실행
pip install ultralytics opencv-python numpy

# SSH 클라이언트 설치 (이미 설치된 경우 건너뛰기)
# Windows 10/11은 기본으로 설치됨
```

---

## 코드 저장

### 라즈베리파이 서버 코드

**파일명:** `camera_server.py`

```bash
# 라즈베리파이 홈 디렉토리에 저장
~/camera_server.py
또는
/home/pi/camera_server.py
```

**코드 내용:**

```python
# 라즈베리파이에서 실행할 서버 코드
# 저장: ~/camera_server.py

import socket
import struct
import cv2
import pickle
import threading
import time

class CameraServer:
    def __init__(self, port=5000):
        """
        라즈베리파이 카메라 스트리밍 서버
        
        Args:
            port: 서버 포트 (기본값 5000)
        """
        self.port = port
        self.server_socket = None
        self.running = True
        self.frame = None
    
    def capture_camera(self):
        """카메라에서 프레임 캡처"""
        cap = cv2.VideoCapture(0)
        
        # 카메라 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("카메라 초기화:", cap.isOpened())
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                self.frame = frame
                time.sleep(0.01)
            else:
                print("프레임 캡처 실패")
                break
        
        cap.release()
    
    def handle_client(self, client_socket, addr):
        """클라이언트 연결 처리"""
        print(f"클라이언트 연결: {addr}")
        
        try:
            while self.running:
                if self.frame is not None:
                    ret, buffer = cv2.imencode('.jpg', self.frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, 80])
                    
                    if ret:
                        data = pickle.dumps(buffer)
                        message_size = struct.pack("Q", len(data))
                        client_socket.sendall(message_size + data)
                        time.sleep(0.05)
                else:
                    time.sleep(0.1)
        except Exception as e:
            print(f"전송 오류: {e}")
        finally:
            client_socket.close()
            print(f"클라이언트 종료: {addr}")
    
    def run(self):
        """서버 실행"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(1)
            
            import subprocess
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
            ip_address = result.stdout.strip()
            print(f"서버 시작: {ip_address}:{self.port}")
            print("윈도우에서 연결 대기 중...")
            
            camera_thread = threading.Thread(target=self.capture_camera, daemon=True)
            camera_thread.start()
            
            while self.running:
                client_socket, addr = self.server_socket.accept()
                
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, addr),
                    daemon=True
                )
                client_thread.start()
        
        except Exception as e:
            print(f"서버 오류: {e}")
        finally:
            self.running = False
            if self.server_socket:
                self.server_socket.close()
            print("서버 종료")

if __name__ == "__main__":
    server = CameraServer(port=5000)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n서버 중지")
        server.running = False
```

### 윈도우 클라이언트 코드

**파일명:** `yolo_client.py`

```bash
# 윈도우 PC의 원하는 디렉토리에 저장
C:\Users\YourUsername\Desktop\yolo_client.py
또는
C:\yolo_project\yolo_client.py
```

**코드 내용:**

```python
# 윈도우에서 실행할 클라이언트 코드
# 저장: yolo_client.py

import socket
import struct
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import threading

class RemoteYOLOClient:
    def __init__(self, rpi_ip, rpi_port=5000):
        """
        윈도우에서 라즈베리파이의 카메라 스트림을 받아서 YOLO 처리
        
        Args:
            rpi_ip: 라즈베리파이 IP 주소
            rpi_port: 라즈베리파이 서버 포트
        """
        self.rpi_ip = rpi_ip
        self.rpi_port = rpi_port
        self.model = YOLO('yolov8n.pt')
        self.running = True
        self.frame = None
        self.sock = None
    
    def connect_to_rpi(self):
        """라즈베리파이에 소켓 연결"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.rpi_ip, self.rpi_port))
            print(f"✓ 라즈베리파이 연결 성공: {self.rpi_ip}:{self.rpi_port}")
            return True
        except Exception as e:
            print(f"✗ 연결 실패: {e}")
            return False
    
    def receive_stream(self):
        """라즈베리파이에서 프레임 수신"""
        data = b""
        payload_size = struct.calcsize("Q")
        
        while self.running:
            try:
                while len(data) < payload_size:
                    data += self.sock.recv(4096)
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]
                
                while len(data) < msg_size:
                    data += self.sock.recv(4096)
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    self.frame = frame
            except Exception as e:
                print(f"수신 오류: {e}")
                break
    
    def detect_and_display(self):
        """YOLO 객체 인식 및 표시"""
        detection_count = 0
        
        while self.running:
            if self.frame is not None:
                results = self.model(self.frame, conf=0.5)
                annotated_frame = results[0].plot()
                
                detections = []
                for box in results[0].boxes:
                    cls_name = results[0].names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    detections.append(f"{cls_name}({conf:.2f})")
                
                if detections:
                    detection_count += 1
                    info_text = f"감지: {' | '.join(detections[:3])}"
                    cv2.putText(annotated_frame, info_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('YOLO8 - 라즈베리파이 원격 객체 인식', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
                    self.running = False
    
    def run(self):
        """원격 스트림 수신 및 YOLO 처리"""
        if not self.connect_to_rpi():
            return
        
        receive_thread = threading.Thread(target=self.receive_stream, daemon=True)
        detect_thread = threading.Thread(target=self.detect_and_display, daemon=True)
        
        receive_thread.start()
        detect_thread.start()
        
        try:
            while self.running:
                pass
        except KeyboardInterrupt:
            print("\n프로그램 중지 중...")
            self.running = False
        finally:
            if self.sock:
                self.sock.close()
            cv2.destroyAllWindows()
            print("연결 종료")

if __name__ == "__main__":
    print("=" * 50)
    print("YOLO8 라즈베리파이 원격 객체 인식")
    print("=" * 50)
    
    rpi_ip = input("\n라즈베리파이 IP 주소 입력 (예: 192.168.1.100): ").strip()
    
    if not rpi_ip:
        print("IP 주소를 입력해주세요.")
        exit()
    
    client = RemoteYOLOClient(rpi_ip=rpi_ip)
    print("\n연결 시작...\n")
    client.run()
```

---

## 연결 방법

### 라즈베리파이 IP 주소 확인

**방법 1: 라즈베리파이 터미널에서**

```bash
hostname -I
# 출력 예: 192.168.1.100
```

**방법 2: 윈도우 PowerShell에서**

```powershell
ping raspberrypi.local
# 또는 라우터 관리 페이지 확인
```

**방법 3: IP 스캔 도구 사용**
- Advanced IP Scanner 같은 도구로 스캔

### WiFi 네트워크 확인

- 라즈베리파이와 윈도우 PC가 **같은 WiFi 네트워크**에 연결되어 있어야 함
- 라우터 설정에서 DHCP 예약 설정 권장 (고정 IP)

---

## 실행 방법

### 방법 1: 순차 실행 (권장)

**Step 1: 라즈베리파이 서버 시작**

```bash
# 라즈베리파이 터미널에서
ssh pi@raspberrypi.local
cd ~
python3 camera_server.py

# 출력 예:
# 카메라 초기화: True
# 서버 시작: 192.168.1.100 5000
# 윈도우에서 연결 대기 중...
```

**Step 2: 윈도우 클라이언트 실행**

```powershell
# 윈도우 PowerShell에서
cd C:\Users\YourUsername\Desktop
python yolo_client.py

# 입력 프롬프트:
# 라즈베리파이 IP 주소 입력: 192.168.1.100

# 출력 예:
# ==================================================
# YOLO8 라즈베리파이 원격 객체 인식
# ==================================================
#
# 라즈베리파이 IP 주소 입력: 192.168.1.100
#
# 연결 시작...
# ✓ 라즈베리파이 연결 성공: 192.168.1.100:5000
```

**Step 3: 객체 인식 확인**

- 새 창에서 실시간 영상 표시
- 감지된 객체 표시
- ESC 키로 종료

### 방법 2: 백그라운드 실행 (라즈베리파이)

```bash
# 라즈베리파이에서 백그라운드 실행
nohup python3 ~/camera_server.py > ~/camera_server.log 2>&1 &

# 프로세스 확인
ps aux | grep camera_server

# 로그 확인
tail -f ~/camera_server.log

# 프로세스 종료
pkill -f camera_server
```

### 방법 3: 자동 시작 설정 (라즈베리파이)

```bash
# systemd 서비스 파일 생성
sudo nano /etc/systemd/system/camera-server.service
```

**다음 내용 입력:**

```ini
[Unit]
Description=Camera Server for YOLO
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi
ExecStart=/usr/bin/python3 /home/pi/camera_server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**저장 후 활성화:**

```bash
sudo systemctl enable camera-server.service
sudo systemctl start camera-server.service
sudo systemctl status camera-server.service
```

---

## 트러블슈팅

### 1. 연결 실패

**증상:** `연결 실패: Connection refused`

**해결:**
```bash
# 라즈베리파이 IP 확인
hostname -I

# 방화벽 확인 (Windows)
# 설정 > 개인정보 보호 및 보안 > Windows Defender 방화벽
# > 방화벽을 통해 앱 허용 > Python 허용

# 라즈베리파이 포트 확인
netstat -tuln | grep 5000
```

### 2. 카메라 인식 안 됨

**증상:** `카메라 초기화: False`

**해결:**
```bash
# 라즈베리파이에서 카메라 확인
libcamera-hello

# 또는
v4l2-ctl --list-devices

# 카메라 활성화 (라즈베리파이 설정)
sudo raspi-config
# Interface Options > Camera > Enable
```

### 3. 느린 성능

**원인:** 모델이 너무 큼

**해결:**
```python
# yolo_client.py 수정
self.model = YOLO('yolov8n.pt')  # nano 사용 (가장 가벼움)
# 또는
self.model = YOLO('yolov8s.pt')  # small
```

### 4. 프레임 끊김

**해결:**
```bash
# 라즈베리파이 리소스 모니터링
top

# CPU 온도 확인
vcgencmd measure_temp

# 냉각 필요 시 히트싱크 부착
```

### 5. 메모리 부족

**해결:**
```bash
# 불필요한 프로세스 종료
ps aux
kill -9 [PID]

# 스왑 메모리 증가 (선택사항)
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048 로 설정
sudo dphys-swapfile setup
sudo systemctl restart dphys-swapfile
```

---

## 파일 정리

최종 파일 구조:

**라즈베리파이:**
```
/home/pi/
├── camera_server.py          # 서버 코드
└── camera_server.log         # 로그 파일 (자동 생성)
```

**윈도우 PC:**
```
C:\Users\YourUsername\Desktop\
├── yolo_client.py            # 클라이언트 코드
└── (실행)
```

---

## 성능 최적화 팁

- **라즈베리파이 4B 기준:** 약 5-10 FPS
- **라즈베리파이 5 기준:** 약 15-20 FPS

**최적화 방법:**
1. 프레임 해상도 낮추기: 480p → 360p
2. FPS 감소: 15 → 10
3. JPEG 품질 낮추기: 80 → 60
4. 모델 간소화: nano 사용
5. USB 카메라 대신 라즈베리파이 카메라 사용

---

## 문제 발생 시 확인사항

1. ✓ 라즈베리파이와 윈도우 PC가 같은 네트워크?
2. ✓ 라즈베리파이 서버 실행 중?
3. ✓ IP 주소 올바름?
4. ✓ 방화벽 포트 5000 열어둠?
5. ✓ 파이썬 패키지 모두 설치?
6. ✓ 카메라 정상 작동?