#!/usr/bin/env python3
"""
run_kiosk.py — Luồng kiosk hoàn chỉnh: CAMERA -> phân loại -> hội thoại RAG (text).

Khác main.py (nhận --image tĩnh) và test_camera.py (chỉ classifier, không RAG):
đây là luồng thật của kiosk — người dùng đưa da vào khung, chụp, rồi HỎI ĐÁP
bằng text về kết quả.

    cd ~/Desktop/skin_disease_rag
    python run_kiosk.py --flip 2

PHÍM (trong cửa sổ camera):
    SPACE  chụp khung trong khung ngắm -> phân loại -> vào hội thoại
    c      đo độ nét (hiệu chỉnh BLUR_MIN)
    q      thoát

Sau khi chụp và có kết quả, cửa sổ camera đóng, chuyển sang hỏi-đáp text.
Gõ 'chụp lại' để mở camera chụp vùng da khác. Gõ 'thoát' để kết thúc.
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

import cv2

from config import CROP_BOX, CLASSIFIER_MODEL_PATH
from classifier_bridge import ClassifierBridge
from rag.retriever import Retriever
from rag.query_rewrite import build_search_query
from rag.response_generator import generate_response
from rag.vector_store import get_collection_info

# Dùng lại CSICamera cũ (subprocess gst-launch, có ISP preset da liễu).
# Nó cho ảnh NÉT HƠN pipeline cv2 trần: ee-strength=0.3 làm rõ viền tổn
# thương, tnr giữ texture da. API giống cv2.VideoCapture.
from csi_camera import CSICamera, DEFAULT_SKIN_ISP
from test_came_new_model import draw_frame_guide

# Tái dùng session + loop từ main.py (KHÔNG viết lại logic hội thoại)
from main import (
    ConversationSession, _detect_disease_in_text,
    EXIT_KEYWORDS, OPENING_NO_IMAGE,
)
from config import DISEASE_VI_NAMES, CONVERSATION_ONLY_DISEASES


RECAPTURE_KEYWORDS = {"chụp lại", "chup lai", "chụp", "ảnh khác", "quét lại"}


def check_rag_ready() -> bool:
    info = get_collection_info()
    if info["status"] != "ok" or info["count"] == 0:
        print("❌ ChromaDB chưa build. Chạy: python build_rag_full.py")
        return False
    print(f"[Kiosk] ChromaDB OK — {info['count']} chunks")
    return True


def open_camera(source, sensor_id, width, height, flip):
    if source == "csi":
        # CSICamera tự chạy gst-launch subprocess — KHÔNG cần cv2 GStreamer.
        # ISP preset da liễu (DEFAULT_SKIN_ISP) cho ảnh nét & màu chuẩn hơn.
        # Lưu ý: CSICamera hiện KHÔNG nhận flip; nếu ảnh lộn ngược, xoay bằng
        # cv2.flip trong vòng đọc (xem capture_and_classify).
        return CSICamera(sensor_id=sensor_id, width=width, height=height, fps=30)
    elif source.isdigit():
        return cv2.VideoCapture(int(source))
    else:
        return cv2.VideoCapture(source)


def capture_and_classify(bridge, cap, flip=2):
    """
    Mở vòng lặp camera tới khi user bấm SPACE (chụp) hoặc q (thoát).
    Trả về ChanDoanResult, hoặc None nếu user thoát.

    flip: IMX219 lắp lộn ngược -> xoay 180° bằng cv2 (vì CSICamera không
    nhận flip-method như pipeline cv2). flip=2 ~ xoay 180°.
    """
    Path("captures").mkdir(exist_ok=True)
    print("\n[Camera] SPACE=chụp & phân tích   c=đo nét   q=thoát\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            # CSICamera cần ~1.5s warmup — vài frame đầu có thể None
            time.sleep(0.05)
            continue
        # Xoay bù chiều lắp camera
        if flip == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif flip == 1:
            frame = cv2.flip(frame, 1)
        elif flip == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        disp, _ = draw_frame_guide(frame.copy())
        cv2.putText(disp, "SPACE=chup  q=thoat", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("ThinkMed Kiosk", disp)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            return None

        if k == ord("c"):
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"   Laplacian={cv2.Laplacian(g, cv2.CV_64F).var():7.1f}  "
                  f"brightness={g.mean():6.1f}")

        if k == ord(" "):
            path = f"captures/kiosk_{int(time.time())}.jpg"
            cv2.imwrite(path, frame)
            result = bridge.run(path)
            print(f"\n[CLS] reason={result.reason}", end="")
            if result.has_diagnosis:
                vi = DISEASE_VI_NAMES.get(result.disease_key, result.disease_key)
                print(f" -> {vi} ({result.confidence:.0%})")
            else:
                print(" -> không chẩn đoán từ ảnh")
            for kk, vv, pp in result.top3[:3]:
                print(f"       • {vv}: {pp:.0%}")
            return result


def conversation_turn(session, retriever, user_input, use_llm_rewrite):
    """Một lượt hỏi-đáp text (rút gọn từ main.start_conversation)."""
    mentioned = _detect_disease_in_text(user_input)
    if mentioned and mentioned != session.active_disease:
        session.context_disease = mentioned
        vi = DISEASE_VI_NAMES.get(mentioned, mentioned)
        note = " (tư vấn qua mô tả — cần bác sĩ khám để chẩn đoán)" \
               if mentioned in CONVERSATION_ONLY_DISEASES else ""
        print(f"   💡 Nhận ra: {vi}{note}")

    search_query = build_search_query(
        user_query=user_input, disease_key=session.active_disease,
        use_llm=use_llm_rewrite, history=session.recent_history(3),
    )
    print(f"   🔍 [{search_query}]")
    chunks = retriever.query(query_text=search_query, disease_key=session.active_disease)
    found = ", ".join(sorted(set(c["disease_vi"] for c in chunks)))
    print(f"   📄 {len(chunks)} chunks — {found}\n")

    session.add_turn("user", user_input)
    response = generate_response(
        user_query=user_input, retrieved_chunks=chunks,
        disease_key=session.active_disease, yolo_confidence=session.yolo_confidence,
        history=session.recent_history(4)[:-1], voice_mode=False,
    )
    session.add_turn("assistant", response)
    print()


def main():
    ap = argparse.ArgumentParser(description="ThinkMed Kiosk — camera + RAG (text)")
    ap.add_argument("--source", default="csi", help="'csi' | index webcam | video path")
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--flip", type=int, default=2, help="flip-method 0..3 (IMX219 lộn ngược -> 2)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--llm-rewrite", action="store_true")
    args = ap.parse_args()

    if not Path(CLASSIFIER_MODEL_PATH).exists():
        sys.exit(f"❌ Không tìm thấy model: {CLASSIFIER_MODEL_PATH}")
    if not check_rag_ready():
        sys.exit(1)

    print("[Kiosk] Tải classifier...")
    bridge = ClassifierBridge()
    retriever = Retriever()

    print(f"\n{'═'*60}")
    print(f"  🏥  THINKMED KIOSK  [⌨️  text mode]")
    print(f"{'═'*60}")

    while True:  # vòng ngoài: mỗi lần = một lần quét + hội thoại
        cap = open_camera(args.source, args.sensor_id, args.width, args.height, args.flip)
        if not cap.isOpened():
            sys.exit("❌ Không mở được camera.\n"
                     "   Nếu vừa Ctrl+C: sudo systemctl restart nvargus-daemon")

        result = capture_and_classify(bridge, cap, flip=args.flip)
        cap.release()
        cv2.destroyAllWindows()

        if result is None:      # user bấm q ở camera
            print("\n🤖  Kết thúc. Chúc bạn sức khoẻ!")
            break

        # Ảnh không đạt -> báo và mở lại camera
        if result.reason in ("blurry", "too_dark", "too_bright"):
            print(f"\n🤖  {result.opening}\n")
            continue

        # Có kết quả (ok / other / low_confidence) -> vào phiên hội thoại
        session = ConversationSession(
            image_path=None,
            detected_disease=result.disease_key if result.has_diagnosis else None,
            yolo_confidence=result.confidence,
            opening_text=result.opening,
        )
        print(f"\n🤖  {result.opening}\n")
        print("   (Gõ 'chụp lại' để quét vùng khác, 'thoát' để kết thúc)\n")
        session.add_turn("assistant", result.opening)

        # Vòng trong: hỏi-đáp text
        recapture = False
        while True:
            try:
                user_input = input("👤  ").strip()
            except (KeyboardInterrupt, EOFError):
                user_input = None

            if user_input is None or user_input.lower() in EXIT_KEYWORDS:
                print("\n🤖  Cảm ơn bạn. Chúc bạn sức khoẻ!")
                return
            if not user_input:
                continue
            if user_input.lower() in RECAPTURE_KEYWORDS:
                recapture = True
                break

            conversation_turn(session, retriever, user_input, args.llm_rewrite)

        if not recapture:
            break


if __name__ == "__main__":
    main()
