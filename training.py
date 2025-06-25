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
    version = project.version(3)
    dataset = version.download("yolov11")
                
    model = YOLO('yolo11n.pt')  # Load a pretrained YOLOv11 model
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=300,
        imgsz=640,
        plots=True,
        device=0 , # 0 for first GPU, 1 for second, 'cpu' for CPU
        resume=False,  # Set to True to resume training from a previous checkpoint
        save=True,
        project=r'C:\Users\Lenovo\Desktop\Project tracking\New folder\training-model\data\runs',
        patience=50,
        bach=16,  # Batch size
        name='11n300epV3'  # Name of the training run
    )
def runtime():
    start = time.time()
    main()
    end = time.time()
    print(f'Runtime: {end-start:.3f} second')

if __name__ == "__main__":
    runtime()