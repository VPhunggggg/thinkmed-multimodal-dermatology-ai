#!/usr/bin/env python3
"""
analyze_threshold.py — Chọn CONF_THRESHOLD bằng dữ liệu, không bằng cảm tính.

Ý TƯỞNG
-------
Kiosk KHÔNG bắt buộc trả lời mọi ca. Nó chỉ đưa ra chẩn đoán khi:
      argmax ∈ {5 bệnh}   VÀ   max_prob >= τ
Mọi trường hợp khác -> "tôi chưa đủ chắc chắn, bạn nên đi khám" = AN TOÀN.

Ba con số cho mỗi ngưỡng τ:
  ĐỘ PHỦ     % số ca kiosk dám đưa ra chẩn đoán
  ĐỘ CHÍNH XÁC  trong số đó, bao nhiêu % đúng   <- đây là lời hứa với bệnh nhân
  TỈ LỆ NGUY HIỂM  % ca `other` (bất thường ngoài 5 bệnh, gồm ung thư da)
                   bị gán một chẩn đoán TỰ TIN và SAI

Tăng τ -> chính xác hơn, nguy hiểm ít hơn, nhưng phủ ít hơn.
Đây là đánh đổi mà BẠN phải quyết định, không phải model.

    python analyze_threshold.py --data out
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Trả lời "lành tính, không cần lo" — sai ở đây nguy hiểm hơn sai ở chỗ khác,
# vì nó chủ động trấn an một bệnh nhân có thể đang mang tổn thương ác tính.
REASSURING = {"wart", "acne"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target-precision", type=float, default=0.85,
                    help="Độ chính xác tối thiểu bạn chấp nhận hứa với bệnh nhân")
    ap.add_argument("--final-check", action="store_true",
                    help="Xác minh ngưỡng trên MODEL FINAL (không phải OOF của CV)")
    args = ap.parse_args()

    root = Path(args.data)

    if args.final_check:
        # Logits của chính model sẽ deploy. Tập calibration 10%, đã có nhãn.
        fc = json.load(open(root / "final_calib.json"))
        classes = fc["classes"]
        T = fc["temperature_final"]
        logits = np.load(root / "final_cal_logits.npy")
        y = np.load(root / "final_cal_labels.npy")
        print(">>> CHẾ ĐỘ --final-check: xác minh ngưỡng trên MODEL FINAL")
        print(f"    (tập hiệu chỉnh 10%, {len(y)} ảnh — nhỏ hơn CV, con số sẽ nhiễu hơn)")
        df = None
    else:
        df = pd.read_csv(root / "manifest.csv")
        logits = np.load(root / "oof_logits.npy")
        cv = json.load(open(root / "cv_results.json"))
        classes = cv["classes"]
        T = cv["temperature_oof"]
        y = df["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()

    c2i = {c: i for i, c in enumerate(classes)}
    oi = c2i["other"]

    z = logits / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    pred, conf = p.argmax(1), p.max(1)

    is_other = y == oi
    print(f"\nT = {T:.3f} (từ CV)   |   {len(y)} ảnh   |   `other` = {is_other.sum()}")
    print("\n" + "=" * 76)
    print(f"{'τ':>5s} {'ĐỘ PHỦ':>9s} {'C.XÁC':>8s} {'NGUY HIỂM':>11s} {'trong đó':>10s}")
    print(f"{'':>5s} {'(trả lời)':>9s} {'(khi trả lời)':>8s} {'(other bị đoán)':>11s} {'"lành tính"':>10s}")
    print("=" * 76)

    rows = []
    for t in np.arange(0.30, 0.96, 0.05):
        say = (pred != oi) & (conf >= t)                 # kiosk dám chẩn đoán
        cov = say.mean()
        prec = (pred[say] == y[say]).mean() if say.sum() else float("nan")
        danger = (say & is_other).sum() / is_other.sum()  # other -> chẩn đoán tự tin
        reassure = sum((say & is_other & (pred == c2i[c])).sum()
                       for c in REASSURING if c in c2i) / is_other.sum()
        rows.append((t, cov, prec, danger, reassure))
        mark = ""
        if not np.isnan(prec) and prec >= args.target_precision:
            mark = "  <-- đạt mục tiêu"
        print(f"{t:5.2f} {cov:8.1%} {prec:8.1%} {danger:10.1%} {reassure:9.1%}{mark}")

    print("=" * 76)

    ok = [r for r in rows if not np.isnan(r[2]) and r[2] >= args.target_precision]
    if ok:
        t, cov, prec, danger, reassure = ok[0]
        print(f"\n>>> ĐỀ XUẤT:  CONF_THRESHOLD = {t:.2f}")
        print(f"    Kiosk chẩn đoán {cov:.0%} số ca, đúng {prec:.0%} khi nó nói.")
        print(f"    {1-cov:.0%} còn lại -> 'chưa đủ chắc chắn, hãy đi khám' (an toàn).")
        print(f"    {danger:.1%} ca `other` vẫn bị gán chẩn đoán sai tự tin.")
    else:
        print(f"\n>>> KHÔNG ngưỡng nào đạt độ chính xác {args.target_precision:.0%}.")
        best = max((r for r in rows if not np.isnan(r[2])), key=lambda r: r[2])
        print(f"    Cao nhất: {best[2]:.1%} @ τ={best[0]:.2f}, độ phủ chỉ {best[1]:.1%}.")
        print(f"    Model chưa đủ tốt để hứa {args.target_precision:.0%}. Hạ kỳ vọng,")
        print(f"    hoặc cải thiện model (thêm epoch / tách lớp tinea) rồi chạy lại.")

    # Độ chính xác theo từng lớp tại ngưỡng đề xuất
    t = ok[0][0] if ok else 0.60
    print(f"\n--- Tại τ = {t:.2f}, độ chính xác của TỪNG chẩn đoán ---")
    print(f"{'khi kiosk nói...':22s} {'số ca':>7s} {'đúng':>8s}")
    for c in classes:
        if c == "other":
            continue
        m = (pred == c2i[c]) & (conf >= t)
        if m.sum():
            acc = (y[m] == c2i[c]).mean()
            warn = "  ⚠ trấn an sai" if c in REASSURING and acc < 0.8 else ""
            print(f"{'“' + c + '”':22s} {m.sum():7d} {acc:7.1%}{warn}")
        else:
            print(f"{'“' + c + '”':22s} {0:7d} {'—':>8s}")

    print(f"\n--- Ca `other` bị gán chẩn đoán tự tin tại τ = {t:.2f} ---")
    for c in classes:
        if c == "other":
            continue
        n = (is_other & (pred == c2i[c]) & (conf >= t)).sum()
        if n:
            tag = "  ⚠ NGUY HIỂM — nói 'lành tính' với ca bất thường" if c in REASSURING else ""
            print(f"   -> “{c}”: {n} ca{tag}")

    # ══════════════════════════════════════════════════════════════════
    #  UNG THƯ DA — con số quan trọng nhất của toàn dự án
    # ══════════════════════════════════════════════════════════════════
    CANCER = (r"melanoma|nevus|nevi|mole|basal.?cell|bcc|carcinoma|actinic|"
              r"squamous|malignant|bowen|keratoacanth|lentigo")
    if df is None:
        print("\n(--final-check: bỏ qua phân tích ung thư — cần tên file, chỉ có ở chế độ OOF)")
        is_cancer = np.zeros(len(y), bool)
    else:
        is_cancer = df["file"].str.lower().str.contains(CANCER, regex=True).to_numpy() & is_other
    if is_cancer.sum():
        print("\n" + "=" * 76)
        print(f"UNG THƯ DA / NỐT RUỒI — {is_cancer.sum()} ảnh trong `other`")
        print("=" * 76)
        print(f"\n{'τ':>5s} {'-> other':>10s} {'-> BỆNH LÀNH':>14s} {'-> "mụn cóc"/"trứng cá"':>24s}")
        print(f"{'':>5s} {'(an toàn)':>10s} {'(tự tin, sai)':>14s} {'(TRẤN AN SAI)':>24s}")
        print("-" * 76)
        for t2 in np.arange(0.50, 0.96, 0.05):
            safe = ((pred == oi) | (conf < t2))[is_cancer].mean()
            wrong = ((pred != oi) & (conf >= t2))[is_cancer].mean()
            fatal = sum(((pred == c2i[c]) & (conf >= t2) & is_cancer).sum()
                        for c in REASSURING if c in c2i)
            fr = fatal / is_cancer.sum()
            bar = "  <<< " + "!" * min(int(fr * 100), 12) if fr > 0.02 else ""
            print(f"{t2:5.2f} {safe:9.1%} {wrong:13.1%} {fr:15.1%} ({fatal:3d} ca){bar}")
        print("-" * 76)
        print("\n  'An toàn' = định tuyến vào `other` HOẶC dưới ngưỡng -> đều dẫn tới")
        print("  'hãy đi khám bác sĩ'. Cột cuối là số ca ung thư được kiosk CHỦ ĐỘNG")
        print("  trấn an. Con số đó phải tiến về 0. Đây là chỉ số an toàn của bạn.")

    # ══════════════════════════════════════════════════════════════════
    #  NGƯỠNG RIÊNG TỪNG LỚP
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("NGƯỠNG RIÊNG TỪNG LỚP")
    print("=" * 76)
    print("\n  `wart`/`acne` = giá trị chẩn đoán THẤP, thiệt hại khi sai CAO.")
    print("  `tinea`/`psoriasis`/`atopic` = giá trị CAO, thiệt hại khi sai THẤP.")
    print("  => siết `wart`/`acne`, nới các lớp còn lại.\n")
    print(f"{'lớp':20s} {'τ':>5s} {'số ca':>7s} {'đúng':>7s} {'ung thư bị trấn an':>20s}")
    print("-" * 76)
    rec = {}
    for c in classes:
        if c == "other":
            continue
        goal = 0.93 if c in REASSURING else 0.85     # siết lớp "trấn an"
        pick = None
        for t2 in np.arange(0.40, 0.96, 0.05):
            m = (pred == c2i[c]) & (conf >= t2)
            if m.sum() < 30:
                break
            if (y[m] == c2i[c]).mean() >= goal:
                pick = t2
                break
        if pick is None:
            pick = 0.90
        m = (pred == c2i[c]) & (conf >= pick)
        acc = (y[m] == c2i[c]).mean() if m.sum() else float("nan")
        nc = (m & is_cancer).sum() if is_cancer.sum() else 0
        flag = "  ⚠" if (c in REASSURING and nc) else ""
        rec[c] = round(float(pick), 2)
        print(f"{c:20s} {pick:5.2f} {m.sum():7d} {acc:6.1%} {nc:16d} ca{flag}")
    print("-" * 76)
    print("\n  Dán vào config.py:\n")
    print(f"  CONF_THRESHOLD = {json.dumps(rec, ensure_ascii=False)}")
    print("\n  (classifier.py cần sửa để nhận dict — xem chat.)")


if __name__ == "__main__":
    main()
