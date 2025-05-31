from ultralytics import YOLO

model = YOLO("yolo11m-obb.pt")
model.train(data='mydataobb.yaml',
        imgsz=640,
        epochs=250,
        batch=32,
        device=['0', '1'],
        optimizer='Adamw',
        project='runs/train',
        name='exp',
        )
#metrix= model.val(data='/data/hlr/parkinglots/mydataobb.yaml', imgsz=640, conf=0.001, iou=0.65)