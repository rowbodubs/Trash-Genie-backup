import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from yolov5 import detect
import servo
import time
import timeit
import torch
from models.common import DetectMultiBackend

model = torch.hub.load("ultralytics/yolov5","custom","best.pt")
while True:  
	ret,frame = cap.read()  
	frame=cv2.resize(frame,(640,480))
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
	pred = model(frame)
    
    
    
	if not pred.pandas().xyxy[0].empty():
		greatestconf = 0
		for i in range (len(pred.pandas().xyxy[0])):
			if pred.pandas().xyxy[0]['confidence'][greatestconf] > pred.pandas().xy[0]['confidence']:
				greatestconf = i
                
		servo.dump(pred.pandas().xyxy[0]['name'][greatestconf]
		time.sleep(5)
        

    
cap.release()
