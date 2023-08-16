import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import servo
import time
import torch

cap=cv2.VideoCapture(0)

servo.center()

model = torch.hub.load("ultralytics/yolov5","custom","demomodel.pt")
count =0
while True: 
    
    ret,frame = cap.read() 
    count+=1
    
    if count%5==0:
    
        frame=cv2.resize(frame,(1080,720))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pred = model(frame)
        
        
        if len(pred.pandas().xyxy[0]['confidence']) > 0 :
            predInd = pred.pandas().xyxy[0]['confidence'].argmax()
            if pred.pandas().xyxy[0]['confidence'][predInd]>0.5:
                print(f"Class = {pred.pandas().xyxy[0]['name'][predInd]} Confidence = {pred.pandas().xyxy[0]['confidence'][predInd]}")
                servo.dump(pred.pandas().xyxy[0]['name'][predInd].upper())

    
    
    
cap.release()
