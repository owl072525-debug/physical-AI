"""추론 엔진 비교: ONNX Runtime / OpenVINO / MNN / NCNN / TFLite(LiteRT) — 속도·정확도(mAP)·모델 용량.

전처리(letterbox)와 후처리(NMS)는 pi5-docker/infer_bench.py 것을 그대로 공유하므로 엔진 차이만 비교됩니다.
엔진마다 별도 프로세스로 실행해 로드 시간·메모리가 서로 섞이지 않게 합니다.

    python compare_engines.py                     # 전체 엔진 비교 → results/
    python compare_engines.py --only openvino     # 특정 엔진만
    python compare_engines.py --name run3         # results/run3.md, run3.json 으로 저장
"""
import argparse, glob, json, os, platform, subprocess, sys, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pi5-docker"))
from infer_bench import preprocess, postprocess  # noqa: E402

MODELS = HERE / "models"
# 이름: (모델 경로, 설명)
ENGINES = {
    "onnxruntime": ("yolo26n.onnx", "ONNX Runtime FP32"),
    "openvino": ("yolo26n_openvino_model", "OpenVINO FP32"),
    "mnn": ("yolo26n.mnn", "MNN FP32"),
    "ncnn": ("yolo26n_ncnn_model", "NCNN FP32"),
    "litert_fp32": ("yolo26n_float32.tflite", "LiteRT FP32"),
    "litert_fp16": ("yolo26n_float16.tflite", "LiteRT FP16 가중치"),
}
IOUV = np.linspace(0.5, 0.95, 10)


# ---------------------------------------------------------------- 엔진 로더: 반환값 = (NCHW float32 -> (1, 84, 8400)) 함수
def load_engine(name, path, threads):
    if name == "onnxruntime":
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        return lambda x: sess.run(None, {inp: x})[0], f"onnxruntime {ort.__version__}"

    if name == "openvino":
        import openvino as ov
        core = ov.Core()
        # FP32 고정: 기본값은 CPU에 따라 bf16/fp16으로 바뀔 수 있어 공정 비교를 위해 명시
        cm = core.compile_model(str(next(Path(path).glob("*.xml"))), "CPU", {
            "INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"})
        req = cm.create_infer_request()
        return lambda x: req.infer({0: x})[0].copy(), f"openvino {ov.__version__.split('-')[0]}"

    if name == "mnn":
        import MNN
        rt = MNN.nn.create_runtime_manager(({"backend": "CPU", "precision": "normal", "numThread": threads},))
        net = MNN.nn.load_module_from_file(str(path), [], [], runtime_manager=rt)

        def run(x):
            y = net.onForward([MNN.expr.const(x, x.shape, MNN.expr.NCHW)])[0]
            return MNN.expr.convert(y, MNN.expr.NCHW).read().copy()
        return run, f"MNN {MNN.version()}"

    if name == "ncnn":
        import ncnn
        net = ncnn.Net()
        net.opt.num_threads = threads
        # FP32 고정: 기본값은 fp16/bf16 저장·연산을 켤 수 있어 공정 비교를 위해 끔
        net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = False
        net.opt.use_bf16_storage = False
        net.load_param_mem((Path(path) / "model.ncnn.param").read_text())  # 한글 경로를 못 열어 메모리로 로드
        net.load_model_mem((Path(path) / "model.ncnn.bin").read_bytes())

        def run(x):
            with net.create_extractor() as ex:
                ex.input("in0", ncnn.Mat(x[0]).clone())
                return np.array(ex.extract("out0")[1])[None].copy()  # (84, 8400) -> (1, 84, 8400)
        return run, f"ncnn {ncnn.__version__}"

    if name.startswith("litert"):
        from ai_edge_litert.interpreter import Interpreter
        import ai_edge_litert
        it = Interpreter(model_content=Path(path).read_bytes(), num_threads=threads)  # 한글 경로를 못 열어 바이트로 로드
        it.allocate_tensors()
        i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

        def run(x):
            it.set_tensor(i, np.ascontiguousarray(x.transpose(0, 2, 3, 1)))  # TFLite 입력은 NHWC
            it.invoke()
            return it.get_tensor(o).copy()  # onnx2tf 변환본은 ONNX와 같은 픽셀 좌표 출력
        return run, f"ai-edge-litert {getattr(ai_edge_litert, '__version__', '?')}"

    raise ValueError(name)


# ---------------------------------------------------------------- 정확도 (ultralytics 검증기와 같은 매칭 방식)
def box_iou(a, b):
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match(pred, gt):
    """pred (N,6) conf 내림차순, gt (M,5) [cls x1 y1 x2 y2] -> (N,10) bool"""
    correct = np.zeros((len(pred), len(IOUV)), bool)
    if not len(pred) or not len(gt):
        return correct
    iou = box_iou(gt[:, 1:], pred[:, :4]) * (gt[:, :1] == pred[None, :, 5])
    matched = np.zeros((len(gt), len(IOUV)), bool)
    for j in np.flatnonzero((iou >= IOUV.min()).any(0)):
        avail = np.where(matched, 0, iou[:, j, None])
        k = avail.argmax(0)
        correct[j] = avail[k, range(len(IOUV))] >= IOUV
        matched[k, range(len(IOUV))] |= correct[j]
    return correct


def load_labels(img_path, shape):
    lbl = Path(str(img_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
    if not lbl.exists():
        return np.zeros((0, 5), np.float32)
    a = np.loadtxt(lbl, ndmin=2, dtype=np.float32)
    h, w = shape
    xy, wh = a[:, 1:3] * [w, h], a[:, 3:5] * [w, h] / 2
    return np.concatenate([a[:, :1], xy - wh, xy + wh], 1)


def map_score(stats):
    from ultralytics.utils.metrics import ap_per_class
    tp, conf, pcls, tcls = (np.concatenate(s, 0) for s in zip(*stats))
    ap = ap_per_class(tp, conf, pcls, tcls)[5]  # (nc, 10)
    return float(ap[:, 0].mean()), float(ap.mean())


# ---------------------------------------------------------------- 엔진 하나 측정 (자식 프로세스)
def run_one(a):
    import psutil
    proc = psutil.Process()
    files = sorted(glob.glob(os.path.join(a.images, "*.jpg")))[: a.n]
    imgs = [cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR) for f in files]  # 한글 경로 대응
    path = MODELS / ENGINES[a.engine][0]
    rss0 = proc.memory_info().rss

    t0 = time.perf_counter()
    infer, ver = load_engine(a.engine, path, a.threads)
    load_s = time.perf_counter() - t0

    for i in range(a.warmup):
        infer(preprocess(imgs[i % len(imgs)])[0])
    mem_mb = (proc.memory_info().rss - rss0) / 2**20  # 엔진 라이브러리 + 모델 + 실행 버퍼 (이미지 제외)

    inf_ms, e2e_ms, stats, dets = [], [], [], []
    for rnd in range(a.rounds):
        for im, f in zip(imgs, files):
            t0 = time.perf_counter(); x, r, pad = preprocess(im)
            t1 = time.perf_counter(); y = infer(x)
            t2 = time.perf_counter(); d = postprocess(y, r, pad, im.shape[:2])  # 배포 설정 conf 0.25
            t3 = time.perf_counter()
            inf_ms.append((t2 - t1) * 1000); e2e_ms.append((t3 - t0) * 1000)
            if rnd == 0:  # 정확도는 첫 회차 출력으로 (mAP 계산용 conf 0.001)
                dv = postprocess(y, r, pad, im.shape[:2], conf_thres=0.001)
                gt = load_labels(f, im.shape[:2])
                stats.append((match(dv, gt), dv[:, 4], dv[:, 5], gt[:, 0]))
                dets.append(d.tolist())

    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size
    map50, map5095 = map_score(stats)
    inf, e2e = np.array(inf_ms), np.array(e2e_ms)
    res = dict(engine=a.engine, desc=ENGINES[a.engine][1], version=ver, model=path.name, size_mb=size / 2**20,
               load_s=load_s, mem_mb=mem_mb,
               inf_mean=inf.mean(), inf_p50=np.median(inf), inf_p95=np.percentile(inf, 95), inf_std=inf.std(),
               e2e_p50=np.median(e2e), fps=1000 / np.median(inf), fps_e2e=1000 / np.median(e2e),
               map50=map50, map5095=map5095, n_det=sum(len(d) for d in dets), dets=dets, files=[Path(f).name for f in files])
    Path(a.json).write_text(json.dumps({k: (float(v) if isinstance(v, np.floating) else v) for k, v in res.items()}))


# ---------------------------------------------------------------- 결과 비교·보고
def agreement(ref, other, iou_thr=0.9):
    """ORT(conf 0.25) 검출 중 같은 클래스·IoU≥0.9로 재현된 비율"""
    hit = tot = 0
    for r, o in zip(ref, other):
        r, o = np.array(r).reshape(-1, 6), np.array(o).reshape(-1, 6)
        tot += len(r)
        if len(r) and len(o):
            iou = box_iou(r[:, :4], o[:, :4]) * (r[:, 5:6] == o[None, :, 5])
            hit += int((iou.max(1) >= iou_thr).sum())
    return hit / max(tot, 1)


def cpu_name():
    if platform.system() == "Windows":
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith(("model name", "Model")))
    except (OSError, StopIteration):
        return platform.machine()


def report(results, a):
    ref = next((r for r in results if r["engine"] == "onnxruntime"), results[0])
    for r in results:
        r["agree"] = agreement(ref["dets"], r["dets"])
    head = (f"{cpu_name()} | {platform.system()} {platform.release()} | "
            f"Python {platform.python_version()} | threads {a.threads} | warmup {a.warmup} | "
            f"{a.n}장 x {a.rounds}회 | batch 1, 640x640")
    lines = [f"> {head}", "",
             "| 엔진 | 버전 | 모델 파일 | 용량(MB) | 로드(s) | 추론 p50(ms) | 추론 p95(ms) | FPS(추론) | FPS(전체) | mAP50 | mAP50-95 | 검출 수 | ORT 일치율 | 메모리 증가(MB) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['desc']} | {r['version']} | `{r['model']}` | {r['size_mb']:.2f} | {r['load_s']:.2f} | "
                     f"{r['inf_p50']:.1f} | {r['inf_p95']:.1f} | {r['fps']:.1f} | {r['fps_e2e']:.1f} | "
                     f"{r['map50']:.4f} | {r['map5095']:.4f} | {r['n_det']} | {r['agree'] * 100:.1f}% | {r['mem_mb']:.0f} |")
    md = "\n".join(lines)
    print("\n" + md)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{a.name}.md").write_text(md + "\n", encoding="utf-8")
    (out / f"{a.name}.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("dets", "files")} for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(HERE.parents[2] / "datasets" / "coco128" / "images" / "train2017"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3, help="속도 측정 반복 횟수 (100장 x rounds)")
    ap.add_argument("--threads", type=int, default=4, help="Pi 5 = 4코어")
    ap.add_argument("--only", nargs="*", default=list(ENGINES))
    ap.add_argument("--name", default="summary", help="결과 파일 이름 (results/<name>.md, .json)")
    ap.add_argument("--engine", help=argparse.SUPPRESS)  # 내부용: 자식 프로세스
    ap.add_argument("--json", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.engine:
        return run_one(a)

    results, tmp = [], HERE / "results"
    tmp.mkdir(exist_ok=True)
    for name in a.only:
        if not (MODELS / ENGINES[name][0]).exists():
            print(f"[skip] {name}: {ENGINES[name][0]} 없음 (export_models.py 먼저 실행)")
            continue
        js = tmp / f"_{name}.json"
        print(f"[run] {name} ...", flush=True)
        cmd = [sys.executable, __file__, "--engine", name, "--json", str(js), "--images", a.images, "--n", str(a.n),
               "--warmup", str(a.warmup), "--rounds", str(a.rounds), "--threads", str(a.threads)]
        if subprocess.run(cmd).returncode == 0:
            results.append(json.loads(js.read_text()))
            js.unlink()
        else:
            print(f"[fail] {name}")
    if results:
        report(results, a)


if __name__ == "__main__":
    main()
