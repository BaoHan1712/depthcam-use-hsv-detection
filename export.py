from ultralytics import YOLO

model = YOLO("best.pt") 
model.export(
    format="engine",
    dynamic=True,
    int8=True,
    data="phone.yaml", # specify your dataset configuration here
)
0
