import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("test_video.mp4")
model = YOLO('yolov8n.pt')

TARGET_CLASS = 'airplane'

tracker = None
is_tracking = False

WINDOW_NAME = "Detection and Tracking"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        exit()
    frame_height, frame_width = frame.shape[:2]
    
    if is_tracking:
        success, bbox = tracker.update(frame)
        if success:
            (x,y,w,h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "Tracking", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        else:
            tracker = None
            is_tracking = False
            
    else:
        results = model(frame)
        for r in results:
            for box in r.boxes:
                class_name = model.names[int(box.cls)]
                if class_name == TARGET_CLASS and box.conf > 0.6:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(frame_width, int(x2))
                    y2 = min(frame_height, int(y2))
                    w = x2-x1
                    h = y2-y1
                    if w>0 and h>0:
                        bbox = (x1,y1,w,h)
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, bbox)
                        is_tracking = True
                        print("tracking strat")
                        break
            if is_tracking:
                break
        
    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(10)&0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()