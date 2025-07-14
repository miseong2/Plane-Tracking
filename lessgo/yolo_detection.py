#yolo로 동영상에 잇는 객체 탐지 후 시각화
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('test_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        print("파일 읽기 실패!")
        break
    
    #이미지(frame)에서 뭐가 잇는지 찾아주는 model함수
    #model()은 탐지 결과를 리스트로 반환
    results = model(frame)
    
    #탐지 결과 시각화
    #results에 어짜피 한프레임만 잇으니 인덱스 0
    annotated_frame = results[0].plot()
    
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


