#!/usr/bin/env python3
"""
measure_system.py — Đo các con số hệ thống còn thiếu/nghi ngờ trong báo cáo.

Chạy TRÊN JETSON (thư mục gốc dự án):
    cd ~/Desktop/skin_disease_rag
    python measure_system.py

Đo:
  1. Kích thước file model thật (classifier, và các model nếu tìm thấy)
  2. FPS camera thật (đọc 100 frame, tính FPS)
  3. Thời gian classifier thật (1 ảnh, và multi-frame TTA 5 ảnh)
  4. Thời gian VAD + STT trên câu ngắn thực tế
"""
import os, time, glob


def sizeof(path):
    if os.path.exists(path):
        mb = os.path.getsize(path) / (1024*1024)
        return f"{mb:.1f} MB"
    return "KHÔNG TÌM THẤY"


def main():
    print("="*60)
    print("ĐO HỆ THỐNG THINKMED — số liệu cho báo cáo")
    print("="*60)

    # ── 1. Kích thước model ──
    print("\n[1] KÍCH THƯỚC FILE MODEL")
    print(f"  Classifier .pt: {sizeof('models/skin_cls_effb0.pt')}")
    # Tìm các model khác
    for p in glob.glob(os.path.expanduser("~/phowhisper-small-ct2/*")):
        if p.endswith(('.bin','.onnx')):
            print(f"  {os.path.basename(p)}: {sizeof(p)}")
    # ChromaDB
    if os.path.exists("chroma_db"):
        total = sum(os.path.getsize(os.path.join(d,f))
                    for d,_,fs in os.walk("chroma_db") for f in fs)
        print(f"  ChromaDB (cả thư mục): {total/(1024*1024):.1f} MB")

    # ── 2. FPS camera thật ──
    print("\n[2] FPS CAMERA THẬT (đọc 60 frame)")
    try:
        from csi_camera import CSICamera
        cap = CSICamera(sensor_id=0, width=1280, height=720)
        # bỏ 10 frame đầu (warmup)
        for _ in range(10):
            cap.read()
        t0 = time.time()
        n = 0
        for _ in range(60):
            ok, f = cap.read()
            if ok: n += 1
        dt = time.time() - t0
        print(f"  Đọc {n} frame trong {dt:.2f}s → {n/dt:.1f} FPS")
        cap.release()
    except Exception as e:
        print(f"  Không đo được camera: {e}")

    # ── 3. Thời gian classifier ──
    print("\n[3] THỜI GIAN CLASSIFIER")
    try:
        from classifier import SkinClassifier
        import glob as g
        clf = SkinClassifier()
        # tìm 1 ảnh test
        imgs = g.glob("test_image/*.jpg") + g.glob("captures/*.jpg")
        if imgs:
            img = imgs[0]
            # warmup
            clf.detect(img)
            # đo 1 ảnh, 5 lần
            ts = []
            for _ in range(5):
                t0 = time.time()
                clf.detect(img)
                ts.append(time.time()-t0)
            print(f"  1 ảnh (trung bình 5 lần): {sum(ts)/len(ts)*1000:.0f} ms")
            print(f"  → Nếu multi-frame TTA 5 frame: ~{sum(ts)/len(ts)*5*1000:.0f} ms")
        else:
            print("  Không tìm thấy ảnh test trong test_image/ hoặc captures/")
    except Exception as e:
        print(f"  Không đo được classifier: {e}")

    print("\n" + "="*60)
    print("Gửi các con số này để cập nhật báo cáo cho chính xác.")
    print("="*60)


if __name__ == "__main__":
    main()
