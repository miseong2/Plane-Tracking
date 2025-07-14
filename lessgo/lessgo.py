#그냥 동영상 잘 나오는지 확인
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture('test_video.mp4')

if not cap.isOpened():
    print("파일을 열 수 없음!")
    exit()
    
while True:
    #ret: 성공적으로 읽었는지 여부, frame: 실제 사진 데이터
    ret, frame = cap.read() #다음 사진을 꺼내는 함수
    
    if not ret:
        print("프레임을 더이상 못읽겠어")
        break
    
    cv2.imshow('Live', frame)
    
    
    if cv2.waitKey(10)&0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

