import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/11/yolo11-ACmix-CCFM.yaml")

    optimizer_params = {
        # 'lr0': 0.007,
        # 'momentum': 0.937,
        # 'weight_decay': 0.0005,
    }

    results = model.train(
        data="/content/Fyp_dataset/hrsid.yaml",
        # data='../dataset/HRSID/hrsid.yaml',
        # close_mosaic=0,
        epochs=150,
        imgsz=640,
        device=0,
        optimizer="SGD",
        batch=16,
        project="runs/ACyolo",
        amp=False,
    )
