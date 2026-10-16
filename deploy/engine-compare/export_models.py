"""yolo26n.pt -> ONNX / OpenVINO / MNN / NCNN / TFLite(LiteRT) 내보내기. 결과는 models/ 아래에 모읍니다.

- MNN 변환기는 한글 경로를 열지 못하므로 ASCII 경로의 임시 폴더에서 내보낸 뒤 복사합니다.
  (프로젝트 루트의 yolo26n.onnx 등은 건드리지 않습니다.)
- TFLite: ultralytics는 Windows에서 LiteRT 내보내기를 막고, Python 3.14용 TF 툴체인은 호환이 깨져 있어
  같은 ONNX를 onnx2tf로 직접 변환합니다. onnx2tf가 동작하는 Python(예: 3.12 + tensorflow 2.19)을 --tf-python으로 지정하세요.

    python export_models.py --tf-python C:/Temp/tfenv/Scripts/python.exe
    python export_models.py --formats ncnn        # 특정 형식만 (나머지 모델 파일은 그대로)
"""
import argparse, glob, shutil, subprocess, tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]  # physical-AI/
OUT = Path(__file__).resolve().parent / "models"
CALIB = ROOT.parent / "datasets" / "coco128" / "images" / "train2017"

ap = argparse.ArgumentParser()
ap.add_argument("--tf-python", default="", help="onnx2tf가 설치된 Python 경로 (비우면 TFLite 생략)")
ap.add_argument("--formats", nargs="*", default=["onnx", "openvino", "mnn", "ncnn"])
a = ap.parse_args()

OUT.mkdir(exist_ok=True)
tmp = Path(tempfile.mkdtemp(prefix="yolo_export_", dir="C:/Temp" if Path("C:/Temp").exists() else None))
pt = tmp / "yolo26n.pt"
shutil.copy2(ROOT / "yolo26n.pt", pt)


def collect(src):
    dst = OUT / src.name
    if dst.is_dir():
        shutil.rmtree(dst)
    shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)


# end2end=False: 모든 엔진이 같은 (1, 84, 8400) 출력을 내도록 맞추고 NMS는 공통 후처리에서 수행
for fmt in a.formats:
    collect(Path(YOLO(pt).export(format=fmt, imgsz=640, batch=1, end2end=False)))

if a.tf_python:
    # onnx2tf 변환 검증용 샘플(20x128x128x3, 0~1). 없으면 GitHub에서 받으려다 실패하므로 coco128로 직접 생성
    fs = sorted(glob.glob(str(CALIB / "*.jpg")))[:20]
    np.save(tmp / "calibration_image_sample_data_20x128x128x3_float32.npy", np.stack(
        [cv2.resize(cv2.imdecode(np.fromfile(f, np.uint8), 1), (128, 128))[..., ::-1] for f in fs]).astype(np.float32) / 255)
    code = ("import onnx2tf; onnx2tf.convert(input_onnx_file_path='yolo26n.onnx', output_folder_path='saved_model', "
            "not_use_onnxsim=True, verbosity='error', disable_group_convolution=True, enable_batchmatmul_unfold=True)")
    subprocess.run([a.tf_python, "-c", code], cwd=tmp, check=True)
    for f in (tmp / "saved_model").glob("*.tflite"):  # FP32·FP16(가중치) 두 개 생성
        collect(f)
shutil.rmtree(tmp, ignore_errors=True)

for p in sorted(OUT.iterdir()):
    size = sum(f.stat().st_size for f in p.rglob("*")) if p.is_dir() else p.stat().st_size
    print(f"{p.name:40s} {size / 2**20:7.2f} MB")
