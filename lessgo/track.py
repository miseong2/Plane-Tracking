import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('test_video.mp4')

# 추적할 객체의 클래스 이름 (COCO 데이터셋 기준: 'airplane')
TARGET_CLASS = 'airplane'

# 추적기 상태 변수
tracker = None
is_tracking = False

WINDOW_NAME = "Detection and Tracking"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 읽기 실패')
        break

    
    frame_height, frame_width = frame.shape[:2]

    if is_tracking:
        # 추적기가 활성화된 상태라면, 추적을 계속 수행
        success, bbox = tracker.update(frame)
        if success:
            # 추적 성공 시, 새로운 바운딩 박스를 화면에 그림
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 추적 실패 시, 추적기 리셋
            is_tracking = False
            tracker = None
    else:
        # 추적기가 비활성화된 상태라면, YOLO로 객체 탐지
        results = model(frame)
        for r in results:
            for box in r.boxes:
                # box.cls는 클래스 인덱스, model.names로 실제 이름 확인
                class_name = model.names[int(box.cls)]
                
                if class_name == TARGET_CLASS and box.conf > 0.6: # 신뢰도가 60% 이상인 'airplane'만
                    # 추적할 객체를 찾았음!
                    x1, y1, x2, y2 = box.xyxy[0]
                    #bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1)) # (x, y, w, h) 형식으로 변환
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(frame_width, int(x2))
                    y2 = min(frame_height, int(y2))
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    if w>0 and h>0:
                        bbox = (x1, y1, w, h)
                        # OpenCV 추적기 생성 및 초기화 (CSRT는 정확도가 높지만 조금 느림)
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, bbox)
                        is_tracking = True
                        print(f"Start tracking {TARGET_CLASS}")
                        break # 첫 번째 비행기만 찾고 루프 종료
            if is_tracking:
                break

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()