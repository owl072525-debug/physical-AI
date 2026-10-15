from ultralytics import YOLO
model = YOLO("yolo26n.pt")
path = model.export(format="onnx")
