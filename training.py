import ultralytics
import os
from roboflow import Roboflow
from ultralytics import YOLO
import torch

def main():
    print(torch.cuda.is_available())
    ultralytics.checks()

    rf = Roboflow(api_key="EHBMka1dvuHyFvLJmoo2")
    project = rf.workspace("cascade-label").project("cassette-trackingg")
    version = project.version(6)
    dataset = version.download("yolov11")

    model = YOLO('yolo11n.pt')
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=300,
        imgsz=640,
        plots=True,
        device=0  # 0 for first GPU, 1 for second, 'cpu' for CPU
    )

if __name__ == "__main__":
    main()