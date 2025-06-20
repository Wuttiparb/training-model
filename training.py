import ultralytics
import os
from roboflow import Roboflow
from ultralytics import YOLO
import torch
import time

def main():
    print(torch.cuda.is_available())
    ultralytics.checks()


    rf = Roboflow(api_key="Cx14JqJobLYVjeCovFtb")
    project = rf.workspace("phupha").project("new_cassette")
    version = project.version(1)
    dataset = version.download("yolov11")
                
    model = YOLO('yolo11n.pt')
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=300,
        imgsz=640,
        plots=True,
        device=0  # 0 for first GPU, 1 for second, 'cpu' for CPU
    )
def runtime():
    start = time.time()
    main()
    end = time.time()
    print(f'Runtime: {end-start:.3f} second')

if __name__ == "__main__":
    runtime()