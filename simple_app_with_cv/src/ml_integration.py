from ultralytics import YOLO
import math
import cv2
import numpy as np

def import_model(path: str="yolo11n.pt"):
        model = YOLO(path)
        return model
    
def inference_model(model, classNames:dict, frame:cv2.typing.MatLike)->cv2.typing.MatLike:
    prediction = model.predict(frame)
    
    for r in prediction:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # class confidence
            confidence = math.ceil((box.conf[0]*100))/100
            
            # class name
            cls = int(box.cls[0])
            color = (255/len(classNames)*cls, 255/len(classNames)*(len(classNames)-cls), 255*len(classNames)/(cls+1))
            
            # object details
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, classNames[cls], [x1, y1-10], font, fontScale, color, thickness)
            cv2.putText(frame, str(confidence), [x2-80, y1+30], font, fontScale, color, thickness)           
    
    return frame

def draw_gird(frame:cv2.typing.MatLike)->cv2.typing.MatLike:
    h, w = frame.shape[:2]
    mod_1 = 1
    mod_2 = 5
    cv2.line(frame, (0, int(h*mod_1/mod_2)), (w, int(h*mod_1/mod_2)), (255, 255, 255), thickness=1)
    cv2.line(frame, (0, int(h*(mod_2 - mod_1)/mod_2)), (w, int(h*(mod_2 - mod_1)/mod_2)), (255, 255, 255), thickness=1)
    cv2.line(frame, (int(w*mod_1/mod_2), 0), (int(w*mod_1/mod_2), h), (255, 255, 255), thickness=1)
    cv2.line(frame, (int(w*(mod_2 - mod_1)/mod_2), 0), (int(w*(mod_2 - mod_1)/mod_2), h), (255, 255, 255), thickness=1)
    
    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  # Создаем ОДИН объект
    print("Объект создан")
    width_success = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
    height_success = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"Width set: {width_success}, Actual width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")    
    print(f"Height set {height_success}, Height get: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    width_success = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    
    height_success = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"Width set: {width_success}, Actual width: {actual_width}")
    print(f"Height get: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    