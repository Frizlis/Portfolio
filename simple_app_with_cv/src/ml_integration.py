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


if __name__ == "__main__":
    import_model("yolo11n.pt")