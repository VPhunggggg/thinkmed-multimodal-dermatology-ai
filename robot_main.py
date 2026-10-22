"""
robot_main.py — Entry point robot ThinkMed (CLASSIFIER version).

THAY ĐỔI so với bản YOLO:
  - YOLO detect  ->  SkinClassifier (EfficientNet, 6 lớp + ngưỡng an toàn)
  - Multi-frame: thay "3 frame liên tiếp thấy box" bằng "gom N frame,
    TRUNG BÌNH xác suất" (multi-frame TTA) — ổn định hơn 1 frame đơn.
  - 4 tình huống: ok / other / low_confidence / ảnh mờ (thay vì có/không bệnh)
  - Lời thoại AN TOÀN: mọi chẩn đoán đều kèm khuyến cáo tái khám.
  - Không còn bounding box -> UI hiện 1 kết quả (không "N vị trí").

Kiến trúc threading GIỮ NGUYÊN:
  MAIN THREAD   -> ui.run()
  WORKER THREAD -> _worker()

Camera: CSICamera (gst-launch subprocess + ISP preset da liễu).

Usage:
    python robot_main.py             # PIR sensor
    python robot_main.py --no-pir    # tap màn hình
    python robot_main.py --no-pir --no-voice   # text mode
"""

import argparse
import os
import sys
import threading
import time

# Tự động cấu hình mã hóa UTF-8 cho console (đặc biệt trên Windows cmd.exe)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ["PYTHONUTF8"] = "1"
from collections import defaultdict

import cv2
import numpy as np

from robot_ui import ThinkMedUI, State
from csi_camera import CSICamera
from config import (
    CLASSIFIER_MODEL_PATH, DISEASE_VI_NAMES, DISEASE_ALIASES,
    CROP_BOX, CONVERSATION_ONLY_DISEASES,
)
from rag.retriever import Retriever
from rag.query_rewrite import build_search_query
from rag.response_generator import generate_response
from rag.vector_store import get_collection_info


CAMERA_SENSOR_ID   = 0
CAMERA_WIDTH       = 1280
CAMERA_HEIGHT      = 720
PIR_GPIO           = 17

# Multi-frame TTA: gom N frame, trung bình xác suất. Thay cho "3 frame liên
# tiếp" của YOLO. Nhiều frame -> softmax mượt -> kết quả ổn định hơn.
TTA_FRAMES         = 5
VIEWFINDER_WARMUP_SEC = 3.0   # cho auto-exposure hội tụ + user đưa da vào khung

FIRST_LISTEN_WAIT_SEC = 12.0
CHECKIN_WAIT_SEC      = 10.0

EXIT_KEYWORDS = {
    "thoát", "thoat", "exit", "kết thúc", "kết thức", "cắt thúc",
    "tạm biệt", "dừng lại", "dừng ở đây", "đủ rồi", "xong rồi",
    "không hỏi nữa", "kết thúc hội thoại", "cảm ơn tạm biệt",
}

def _is_exit(text: str) -> bool:
    if not text:
        return False
    low = text.lower().strip()
    if low in {"q", "không", "không.", "hết", "thôi"}:
        return True
    return any(kw in low for kw in EXIT_KEYWORDS)

CHECKIN_QUESTION = "Bạn còn muốn hỏi thêm gì về tình trạng da của mình không?"

_CONTINUE_HINTS = ["có", "còn muốn", "cho tôi hỏi", "muốn hỏi thêm", "một câu nữa",
                   "còn câu hỏi", "chưa", "?"]
_ENDING_HINTS = ["không cần", "không có gì thêm", "không còn gì", "không hỏi gì thêm",
                 "không hỏi thêm", "thôi cảm ơn", "cảm ơn bạn", "cảm ơn nhiều",
                 "đủ rồi", "vậy thôi", "vậy là đủ", "hết rồi", "xong rồi",
                 "được rồi", "dừng lại", "kết thúc", "tạm biệt", "không ạ",
                 "không đâu", "khỏi cần", "không"]

def _wants_to_continue(text: str) -> bool:
    lower = text.lower().strip()
    if any(h in lower for h in _CONTINUE_HINTS):
        return True
    if any(h in lower for h in _ENDING_HINTS):
        return False
    return len(lower.split()) >= 5


# ── PIR ───────────────────────────────────────────────────────────────────────

def _setup_pir():
    try:
        import Jetson.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIR_GPIO, GPIO.IN)
        return GPIO
    except ImportError:
        print("⚠️  Jetson.GPIO không có — cài: pip install Jetson.GPIO")
        return None

def _pir_detected(gpio) -> bool:
    if gpio is None:
        return False
    try:
        return gpio.input(PIR_GPIO) == gpio.HIGH
    except Exception:
        return False


# ── CLASSIFIER wrapper (thay YOLO) ────────────────────────────────────────────

_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        from classifier import SkinClassifier
        _classifier = SkinClassifier()
    return _classifier


def _classify_frame(frame: np.ndarray):
    """
    Phân loại 1 frame -> (probs_dict, reason, best_key, best_conf).
    Dùng classifier.detect() nhưng trên frame trong bộ nhớ (ghi tạm ra file
    vì detect() nhận image_path — giữ interface classifier không đổi).
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, frame)
    tmp.close()
    try:
        res = _get_classifier().detect(tmp.name)
        return res
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def _check_skin_presence(frame: np.ndarray) -> float:
    """
    Kiểm tra tỷ lệ màu da người trong vùng crop trung tâm (khung ngắm).
    Dùng không gian màu kép HSV + YCrCb (chuẩn cho màu da người Việt/châu Á).
    Trả về tỷ lệ diện tích pixel da (0.0 -> 1.0).
    """
    from classifier import SkinClassifier
    crop, _ = SkinClassifier._crop(frame)
    if crop is None or crop.size == 0:
        return 0.0

    small = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)

    # Dải màu da người chuẩn
    mask_hsv = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 40)
    mask_ycrcb = (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 173) & (ycrcb[:, :, 2] >= 77) & (ycrcb[:, :, 2] <= 127)

    skin_mask = mask_hsv & mask_ycrcb
    ratio = float(np.count_nonzero(skin_mask)) / skin_mask.size
    return ratio


def _multiframe_classify(frames: list):
    """
    Multi-frame TTA: chạy classifier trên N frame, TRUNG BÌNH xác suất.
    Trả về (avg_probs, reason, best_key, best_conf, best_vi).
    """
    from config import CLASS_NAMES, OTHER_CLASS, CONF_THRESHOLD, CLASS_TO_DISEASE

    if not frames:
        return None, "blurry", None, 0.0, None

    # 1) Kiểm tra xem có da người thật trong khung hình không
    skin_ratios = [_check_skin_presence(f) for f in frames]
    avg_skin_ratio = sum(skin_ratios) / len(skin_ratios)
    print(f"[CLS] Tỷ lệ da trong khung: {avg_skin_ratio:.1%}", flush=True)
    if avg_skin_ratio < 0.12:
        print("[CLS] ⚠️  Không phát hiện vùng da nào trong khung hình (chỉ thấy tường/áo/vật thể).", flush=True)
        return None, "no_skin", None, 0.0, None

    # 2) Gom probs của các frame ĐỌC ĐƯỢC (bỏ frame ảnh mờ)
    prob_sum = defaultdict(float)
    n_valid = 0
    n_blur = 0
    for f in frames:
        res = _classify_frame(f)
        if res.reason in ("blurry", "too_dark", "too_bright"):
            n_blur += 1
            continue
        if not res.probs:
            continue
        for k, v in res.probs.items():
            prob_sum[k] += v
        n_valid += 1

    # Đa số frame mờ -> yêu cầu chụp lại
    if n_valid == 0:
        return None, "blurry", None, 0.0, None

    avg = {k: prob_sum[k] / n_valid for k in prob_sum}
    top_key = max(avg, key=avg.get)
    top_conf = avg[top_key]

    # OOD: bất thường ngoài 5 bệnh
    if top_key == OTHER_CLASS:
        return avg, "other", None, top_conf, None

    # Ngưỡng riêng từng lớp (áp sau trung bình)
    if isinstance(CONF_THRESHOLD, dict):
        thr = CONF_THRESHOLD.get(top_key, 0.75)
    else:
        thr = CONF_THRESHOLD
    if top_conf < thr:
        return avg, "low_confidence", None, top_conf, None

    disease_key = CLASS_TO_DISEASE.get(top_key, top_key)
    vi = DISEASE_VI_NAMES.get(disease_key, top_key)
    return avg, "ok", disease_key, top_conf, vi


# ── Lời thoại an toàn ─────────────────────────────────────────────────────────

SAFETY_TAIL = "Lưu ý, đây chỉ là công cụ sàng lọc, không thay thế bác sĩ."


def _announce(reason, vi, conf, avg, voice_mode, ui=None, disease_key=None):
    """Đọc kết quả theo tình huống kèm gợi ý phòng khám. Trả về (disease_key_for_rag, spoken_text)."""
    from config import CLASS_TO_DISEASE, get_clinic_room
    clinic_room = get_clinic_room(disease_key)

    if reason == "ok":
        if disease_key == "healthy":
            name_prefix = f"Chào bạn {ui.logged_in_user['name']}, " if (ui and ui.logged_in_user) else ""
            text = (
                f"{name_prefix}kết quả phân tích cho thấy vùng da của bạn hoàn toàn khỏe mạnh, "
                f"độ tin cậy {conf:.0%}, không phát hiện dấu hiệu bệnh lý da liễu bất thường. "
                f"Bạn hãy tiếp tục duy trì chăm sóc da thật tốt nhé! {SAFETY_TAIL}"
            )
        else:
            r_num = clinic_room.get("room_num", "")
            r_name = clinic_room.get("room_name", "")
            r_loc = clinic_room.get("location", "")
            fallback_hint = f"Hệ thống gợi ý bạn nên đến {r_num} ({r_name}) tại {r_loc} để được bác sĩ chuyên khoa thăm khám."
            room_voice = f" {clinic_room.get('voice_hint', fallback_hint)}"
            text = f"Hình ảnh gợi ý khả năng {vi}, độ tin cậy {conf:.0%}.{room_voice} {SAFETY_TAIL}"
    elif reason == "other":
        text = ("Tình trạng da của bạn có thể là da khỏe mạnh, hoặc là một vấn đề "
                "ngoài năm bệnh mà tôi được huấn luyện để nhận diện. Vì vậy tôi "
                "chưa thể đưa ra nhận định cụ thể. Nếu bạn thấy vùng da có triệu "
                "chứng bất thường, hãy mô tả để tôi hỗ trợ, hoặc đến gặp bác sĩ "
                "da liễu để được khám trực tiếp.")
    elif reason == "low_confidence":
        text = ("Tôi chưa đủ chắc chắn để đưa ra nhận định về vùng da này. "
                "Bạn có thể mô tả thêm triệu chứng như cảm giác ngứa, đau, thời "
                "gian xuất hiện, hoặc đến gặp bác sĩ da liễu để được khám trực tiếp.")
    elif reason == "no_skin":
        text = ("Tôi chưa nhận thấy vùng da nào trong khung hình. Bạn vui lòng đưa "
                "vùng da cần kiểm tra vào giữa ô vuông và thử lại nhé.")
    else:  # blurry
        text = ("Ảnh chưa đủ rõ. Bạn đưa vùng da lại gần khung hình và giữ ổn định "
                "giúp tôi, rồi thử lại nhé.")

    if ui:
        # Báo hiệu rõ ràng trên màn hình là Robot đang phát biểu kết quả kèm gợi ý phòng khám
        if reason == "ok":
            ui.set_state(State.RESULT_OK, vi_name=vi, confidences=[conf], speaking=True, disease_key=disease_key, clinic_room=clinic_room)
        else:
            ui.set_state(State.RESULT_NONE, speaking=True)

    if voice_mode:
        from audio.tts import speak
        speak(text, stop_event=ui.exit_event if ui else None)
    else:
        print(f"\n🤖  {text}\n")

    if ui and ui.exit_event.is_set():
        return text

    if ui:
        # Đọc xong -> sẵn sàng nhận câu hỏi
        if reason == "ok":
            ui.set_state(State.RESULT_OK, vi_name=vi, confidences=[conf], speaking=False, disease_key=disease_key, clinic_room=clinic_room)
        else:
            ui.set_state(State.RESULT_NONE, speaking=False)

    return text


# ── Lọc câu hỏi ngoài phạm vi y khoa (KHÔNG gọi thêm API) ────────────────────
# Dùng RAG score làm tín hiệu: câu hỏi ngoài y khoa (xe cộ, chính trị...) không
# chunk da liễu nào khớp -> score cao nhất thấp. RẺ (không thêm lời gọi LLM).
#
# CẢNH BÁO thiết kế: ngưỡng để LỎNG. Từ chối nhầm 1 câu hỏi sức khỏe THẬT tệ
# hơn trả lời 1 câu vu vơ. Chỉ chặn khi score RẤT thấp (rõ ràng ngoài phạm vi).

OFFTOPIC_SCORE_MAX = 0.72   # score cao nhất < ngưỡng này -> coi là ngoài y khoa
                            # (chunk da liễu khớp thường >0.80; điều chỉnh nếu cần)

def _is_medical_query(chunks) -> bool:
    """True nếu câu hỏi có vẻ liên quan y khoa (ít nhất 1 chunk khớp đủ tốt)."""
    if not chunks:
        return False
    top_score = max((c.get("score", 0.0) for c in chunks), default=0.0)
    return top_score >= OFFTOPIC_SCORE_MAX

OFFTOPIC_REPLY = ("Xin lỗi, tôi là trợ lý tư vấn về các bệnh da liễu nên chỉ có "
                  "thể hỗ trợ các câu hỏi liên quan đến da và bệnh da. Bạn vui "
                  "lòng đặt lại câu hỏi về tình trạng da của mình nhé.")


# ── Đặt lịch khám bệnh phòng chuyên khoa (Hàng đợi STT tăng tiến tự động) ─────

_CURRENT_STT = 14

def _peek_stt() -> int:
    """Xem trước số thứ tự tiếp theo sắp được cấp."""
    global _CURRENT_STT
    return _CURRENT_STT

def _claim_next_stt() -> int:
    """Cấp số thứ tự chính thức và tự động tăng cho lượt tiếp theo."""
    global _CURRENT_STT
    stt = _CURRENT_STT
    _CURRENT_STT += 1
    return stt


def _is_booking_yes(text: str) -> bool:
    if not text:
        return False
    lower = text.lower().strip()
    for kw in ["không", "khong", "thôi", "thoi", "khỏi", "khoi", "chưa", "chua", "đừng", "hủy", "huy", "no"]:
        if kw in lower:
            return False
    for kw in ["có", "co", "đặt", "dat", "lấy số", "lay so", "stt", "đồng ý", "dong y", "được", "duoc", "ừ", "u", "ok", "yes", "chắc chắn"]:
        if kw in lower:
            return True
    return False


def _is_booking_no(text: str) -> bool:
    if not text:
        return False
    lower = text.lower().strip()
    words = lower.split()
    if len(words) >= 4 and any(w in lower for w in ["lây", "chữa", "thuốc", "bệnh", "sao", "thế nào", "bao lâu", "gì", "tại sao"]):
        return False
    for kw in ["không", "khong", "thôi", "thoi", "khỏi", "khoi", "chưa", "chua", "để sau", "de sau", "hủy", "huy", "no", "khỏi cần"]:
        if kw in lower:
            return True
    return False


def _handle_booking_flow(ui: ThinkMedUI, disease_key: str, clinic_room: dict, vi_name: str, conf: float, voice_mode: bool):
    """
    Hỏi người dùng có muốn đặt lịch khám tại phòng gợi ý không.
    Hỗ trợ cả giọng nói (Có / Không) và chạm nút trên màn hình cảm ứng Kiosk.
    Số thứ tự tự động tăng tiến theo từng lượt đặt lịch thành công.
    """
    if ui.exit_event.is_set():
        return None

    patient_name = ui.logged_in_user.get("name", "Bệnh nhân") if ui.logged_in_user else "Bệnh nhân"
    stt_num = _peek_stt()

    # 1. Chuyển UI sang trạng thái hỏi đặt lịch
    ui.booking_event.clear()
    ui.booking_choice = None
    ui.set_state(
        State.RESULT_OK,
        vi_name=vi_name,
        confidences=[conf],
        disease_key=disease_key,
        clinic_room=clinic_room,
        booking_stage="ask",
        stt_num=stt_num,
        patient_name=patient_name,
        speaking=True,
    )

    ask_voice = (
        f"Bạn có muốn đặt lịch khám tại {clinic_room['room_num']} không? "
        "Bạn có thể nói Có hoặc Không, hoặc chạm nút trên màn hình."
    )
    print(f"\n🤖 Robot: {ask_voice}", flush=True)

    if voice_mode:
        from audio.tts import speak
        speak(ask_voice, stop_event=ui.exit_event)

    if ui.exit_event.is_set():
        return None

    # Sau khi đọc xong, bật trạng thái sẵn sàng lắng nghe câu trả lời
    ui.set_state(
        State.RESULT_OK,
        vi_name=vi_name,
        confidences=[conf],
        disease_key=disease_key,
        clinic_room=clinic_room,
        booking_stage="ask",
        stt_num=stt_num,
        patient_name=patient_name,
        speaking=False,
    )

    user_choice = None
    initial_user_query = None

    if voice_mode:
        from audio.recorder import record_vad
        from audio.stt_parakeet import transcribe

        stop_listen = threading.Event()

        def _monitor_touch():
            while not stop_listen.is_set():
                if ui.exit_event.is_set() or ui.booking_event.is_set():
                    stop_listen.set()
                    break
                time.sleep(0.05)

        t_mon = threading.Thread(target=_monitor_touch, daemon=True)
        t_mon.start()

        wav = record_vad(
            on_listening_start=lambda: None,
            on_energy_update=lambda energy, is_spk: ui.update_audio_level(energy, is_spk),
            stop_event=stop_listen,
            max_wait_sec=8.0,
        )
        stop_listen.set()

        if ui.exit_event.is_set():
            return None

        # Kiểm tra nếu người dùng đã chạm nút trên màn hình
        if ui.booking_event.is_set() and ui.booking_choice:
            user_choice = ui.booking_choice
            print(f"   [Booking] 🖱️ Nhận thao tác cảm ứng: {user_choice.upper()}", flush=True)
        elif wav:
            _t = time.time()
            text = transcribe(wav)
            print(f"   ⏱  STT: {time.time()-_t:.2f}s -> '{text}'", flush=True)
            if _is_exit(text):
                ui.exit_event.set()
                return None
            if _is_booking_yes(text):
                user_choice = "yes"
            elif _is_booking_no(text):
                user_choice = "no"
            else:
                print(f"   💡 Người dùng đặt câu hỏi y khoa trực tiếp: '{text}'", flush=True)
                user_choice = "no"
                initial_user_query = text
    else:
        # Chế độ terminal test (không mic)
        print(f"\n[Booking] Bạn có muốn đặt lịch khám tại {clinic_room['room_num']} không? (chạm nút màn hình hoặc chờ 6s)")
        t0 = time.time()
        while time.time() - t0 < 6.0 and not ui.exit_event.is_set():
            if ui.booking_event.is_set() and ui.booking_choice:
                user_choice = ui.booking_choice
                break
            time.sleep(0.1)

    if ui.exit_event.is_set():
        return None

    # 3. Phản hồi kết quả đặt lịch
    if user_choice == "yes":
        stt_num = _claim_next_stt()
        print(f"\n   ✅ [Booking] ĐẶT LỊCH THÀNH CÔNG: Cấp STT {stt_num} tại {clinic_room['room_num']} (Lượt tiếp theo: {_peek_stt()})", flush=True)
        ui.set_state(
            State.RESULT_OK,
            vi_name=vi_name,
            confidences=[conf],
            disease_key=disease_key,
            clinic_room=clinic_room,
            booking_stage="confirmed",
            stt_num=stt_num,
            patient_name=patient_name,
            speaking=True,
        )

        if ui.logged_in_user:
            confirm_voice = (
                f"Bạn đã được đặt lịch thành công! Chào bạn {patient_name}, bạn sẽ là số thứ tự {stt_num} "
                f"ở {clinic_room['room_num']} ({clinic_room['room_name']}) tại {clinic_room['location']}. "
                "Xin mời bạn chuẩn bị di chuyển đến phòng khám nhé."
            )
        else:
            confirm_voice = (
                f"Bạn đã được đặt lịch thành công! Bạn sẽ là số thứ tự {stt_num} "
                f"ở {clinic_room['room_num']} ({clinic_room['room_name']}) tại {clinic_room['location']}. "
                "Xin mời bạn chuẩn bị di chuyển đến phòng khám nhé."
            )

        print(f"🤖 Robot: {confirm_voice}\n", flush=True)
        if voice_mode:
            from audio.tts import speak
            speak(confirm_voice, stop_event=ui.exit_event)

        # Giữ màn hình phiếu khám 3 giây để bệnh nhân nhìn rõ số thứ tự
        time.sleep(3.0)

    else:
        print("\n   ℹ️ [Booking] Người dùng chọn không đặt lịch.", flush=True)
        ui.set_state(
            State.RESULT_OK,
            vi_name=vi_name,
            confidences=[conf],
            disease_key=disease_key,
            clinic_room=clinic_room,
            booking_stage="declined",
            stt_num=stt_num,
            patient_name=patient_name,
            speaking=True,
        )

        if not initial_user_query:
            decline_voice = "Dạ vâng, bạn có thể tự đến phòng khám khi thuận tiện nhé."
            print(f"🤖 Robot: {decline_voice}\n", flush=True)
            if voice_mode:
                from audio.tts import speak
                speak(decline_voice, stop_event=ui.exit_event)
            time.sleep(1.0)

    # 4. Chuyển tiếp vào hội thoại tư vấn
    if not initial_user_query and not ui.exit_event.is_set():
        next_prompt = "Bạn có câu hỏi nào cần tư vấn thêm về tình trạng da này không?"
        ui.set_state(
            State.RESULT_OK,
            vi_name=vi_name,
            confidences=[conf],
            disease_key=disease_key,
            clinic_room=clinic_room,
            booking_stage="confirmed" if user_choice == "yes" else "declined",
            stt_num=stt_num,
            patient_name=patient_name,
            speaking=True,
        )
        if voice_mode:
            from audio.tts import speak
            speak(next_prompt, stop_event=ui.exit_event)

    ui.set_state(
        State.RESULT_OK,
        vi_name=vi_name,
        confidences=[conf],
        disease_key=disease_key,
        clinic_room=clinic_room,
        booking_stage="confirmed" if user_choice == "yes" else "declined",
        stt_num=stt_num,
        patient_name=patient_name,
        speaking=False,
    )

    return initial_user_query


# ── Conversation (GIỮ NGUYÊN logic check-in) ─────────────────────────────────

def _conversation_loop(ui, retriever, disease_key, conf, voice_mode, initial_query=None):
    history     = []
    context_dis = disease_key

    def _listen_once(max_wait_sec):
        if ui.exit_event.is_set():
            return None
        if voice_mode:
            from audio.recorder import record_vad
            from audio.stt_parakeet import transcribe
            wav = record_vad(
                on_listening_start=lambda: ui.set_state(State.LISTENING),
                on_energy_update=lambda energy, is_spk: ui.update_audio_level(energy, is_spk),
                on_speech_end=lambda: ui.set_state(State.ANALYZING),
                stop_event=ui.exit_event,
                max_wait_sec=max_wait_sec,
            )
            if ui.exit_event.is_set():
                return None
            if not wav:
                return None
            _t = time.time()
            text = transcribe(wav)
            print(f"   ⏱  STT: {time.time()-_t:.2f}s", flush=True)
            return text
        else:
            try:
                return input("👤  ").strip()
            except (KeyboardInterrupt, EOFError):
                return None

    def _process_turn(user_input):
        nonlocal context_dis
        if ui.exit_event.is_set():
            return
        _t = time.time()
        lower = user_input.lower()
        for dk, aliases in DISEASE_ALIASES.items():
            if any(a in lower for a in aliases):
                context_dis = dk
                if dk in CONVERSATION_ONLY_DISEASES:
                    print(f"   💡 {DISEASE_VI_NAMES.get(dk, dk)} — tư vấn qua mô tả "
                          f"(cần bác sĩ khám để chẩn đoán)")
                break
        active = context_dis or disease_key

        _t0 = time.time()
        q = build_search_query(user_input, active, use_llm=False, history=history[-8:])
        print(f"   ⏱  query_rewrite (Rule-based): {time.time()-_t0:.3f}s", flush=True)

        _t0 = time.time()
        retriever_dk = None if active == "healthy" else active
        chunks = retriever.query(q, disease_key=retriever_dk)
        print(f"   ⏱  RAG retrieve (ChromaDB): {time.time()-_t0:.2f}s", flush=True)

        if active is None and not _is_medical_query(chunks):
            ui.set_state(State.SPEAKING, text=OFFTOPIC_REPLY)
            if voice_mode:
                from audio.tts import speak
                speak(OFFTOPIC_REPLY, stop_event=ui.exit_event)
            else:
                print(f"\n🤖  {OFFTOPIC_REPLY}\n")
            return

        history.append({"role": "user", "content": user_input})
        ui.set_state(State.SPEAKING)
        _t0 = time.time()
        response = generate_response(
            user_query=user_input, retrieved_chunks=chunks,
            disease_key=active, yolo_confidence=conf,
            history=history[:-1], voice_mode=voice_mode,
            on_text_update=lambda partial_text: ui.set_state(State.SPEAKING, text=partial_text),
            stop_event=ui.exit_event,
        )
        if ui.exit_event.is_set():
            return

        _dt_gen = time.time() - _t0
        n_words = len(response.split()) if response else 0
        n_chars = len(response) if response else 0
        print(f"   ⏱  generate_response (Groq #2 + TTS): {_dt_gen:.2f}s", flush=True)
        print(f"   📏  câu trả lời: {n_words} từ / {n_chars} ký tự", flush=True)
        print(f"   ⏱  TỔNG từ lúc có text người dùng: {time.time()-_t:.2f}s", flush=True)
        history.append({"role": "assistant", "content": response})
        # Hiển thị phụ đề / nội dung câu trả lời lên màn hình Kiosk
        ui.set_state(State.SPEAKING, text=response)
        if not voice_mode:
            print(f"\n🤖  {response}\n")
        time.sleep(0.3)

    if initial_query:
        _process_turn(initial_query)

    while True:
        if ui.exit_event.is_set():
            break
        user_input = _listen_once(FIRST_LISTEN_WAIT_SEC)
        if _is_exit(user_input) or ui.exit_event.is_set():
            break
        if user_input:
            _process_turn(user_input)
            continue

        # Im lặng -> check-in
        if ui.exit_event.is_set():
            break
        print(f"\n🤖  {CHECKIN_QUESTION}")
        ui.set_state(State.SPEAKING, text=CHECKIN_QUESTION)
        if voice_mode:
            from audio.tts import speak
            speak(CHECKIN_QUESTION, stop_event=ui.exit_event)
        if ui.exit_event.is_set():
            break
        reply = _listen_once(CHECKIN_WAIT_SEC)
        if not reply or _is_exit(reply) or ui.exit_event.is_set():
            break
        if _wants_to_continue(reply):
            _process_turn(reply)
        else:
            break


def _handle_qr_login(ui: ThinkMedUI, cap, voice_mode: bool):
    """Mở khung quét mã QR nhận diện người dùng và chào hỏi bằng giọng nói."""
    print("\n[Worker] 🔑 Bắt đầu quét mã QR nhận diện người dùng...", flush=True)
    if cap is None or not cap.isOpened():
        print("[Worker] ⚠️ Không có camera hoạt động để quét mã QR.", flush=True)
        if voice_mode:
            from audio.tts import speak
            speak("Chưa kết nối được camera để quét mã QR. Bạn có thể sử dụng trực tiếp các tính năng.", stop_event=ui.exit_event)
        ui.set_state(State.IDLE)
        return None

    from qr_login import QRScanner
    scanner = QRScanner()
    ui.set_state(State.QR_LOGIN, qr_user=None, qr_pts=None, success=False)
    ui.exit_event.clear()

    print("[Worker] 📷 Khung quét mã QR đã mở — Chờ người dùng hướng mã QR...", flush=True)
    recognized_user = None

    while ui._running and not ui.exit_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        user, raw_text, pts = scanner.scan_frame(frame)
        ui.update_viewfinder(frame)

        if user is not None:
            recognized_user = user
            print(f"\n   ✅ [QR] ĐÃ NHẬN DIỆN THÀNH CÔNG: {user['id']} -> {user['name']}", flush=True)
            ui.logged_in_user = user
            ui.set_state(State.QR_LOGIN, qr_user=user, qr_pts=pts, success=True)

            greeting_msg = f"Chào bạn {user['name']}! Chúc bạn một ngày tốt lành. Mời bạn chọn khám da hoặc tư vấn."
            print(f"   🤖 Robot: {greeting_msg}", flush=True)

            if voice_mode:
                from audio.tts import speak
                speak(greeting_msg, stop_event=ui.exit_event)
            else:
                time.sleep(2.0)

            time.sleep(1.2)
            break

        time.sleep(0.03)

    ui.set_state(State.IDLE)
    return recognized_user


# ── Worker thread ─────────────────────────────────────────────────────────────

def _worker(ui: ThinkMedUI, use_pir=True, voice_mode=True, camera_source="auto", require_qr_login=False):
    if camera_source == "auto":
        camera_source = "0" if os.name == "nt" else "csi"

    gpio = _setup_pir() if use_pir else None
    if gpio is None and use_pir:
        print("[Worker] ℹ️ Không có PIR phần cứng — tự động chuyển sang click chuột / phím Space.", flush=True)
        use_pir = False

    cap = None
    print(f"[Camera] Khởi động camera (source={camera_source})...", flush=True)
    try:
        if camera_source == "csi":
            cap = CSICamera(
                sensor_id=CAMERA_SENSOR_ID,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=30,
            )
        elif camera_source.isdigit():
            idx = int(camera_source)
            if os.name == "nt":
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        else:
            cap = cv2.VideoCapture(camera_source)
    except Exception as e:
        print(f"[Camera] ⚠️ Lỗi khởi tạo camera: {e}", flush=True)

    if cap is None or not cap.isOpened():
        print("[Camera] ⚠️ Camera chưa sẵn sàng. Chức năng 'Khám da' yêu cầu kết nối camera; chức năng 'Tư vấn' vẫn hoạt động 100% bình thường!", flush=True)
    else:
        print("[Camera] ✅ Camera đã kết nối thành công.", flush=True)

    retriever = Retriever()

    # Warmup classifier (lần đầu CUDA/YOLO biên dịch kernel)
    print("[Boot] Warmup classifier...", flush=True)
    _t = time.time()
    try:
        _get_classifier()   # __init__ đã tự warmup
        print(f"[Boot] ✅ Classifier sẵn sàng sau {time.time()-_t:.1f}s", flush=True)
    except Exception as e:
        print(f"[Boot] ⚠️ Warmup lỗi: {e}", flush=True)

    print("[Worker] Sẵn sàng — chờ người dùng chạm thẻ 'Khám da', 'Tư vấn' hoặc 'Đăng nhập QR'...", flush=True)
    ui.set_state(State.IDLE)

    # Nếu bật tùy chọn --qr-login ngay khi khởi động
    if require_qr_login and ui.logged_in_user is None:
        _handle_qr_login(ui, cap, voice_mode)

    try:
        while ui._running:
            action = "kham_da"
            if use_pir:
                while ui._running and not _pir_detected(gpio):
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            ui.update_viewfinder(frame)
                    time.sleep(0.1)
            else:
                print("\n[Worker] Chờ người dùng chạm thẻ 'Khám da', 'Tư vấn' hoặc 'Đăng nhập QR'...", flush=True)
                try:
                    action = ui.wait_for_tap()
                except RuntimeError as e:
                    print(f"[Worker] ❌ {e}", flush=True)
                    break

            if not ui._running:
                break

            ui.exit_event.clear()

            # ────────────────────────────────────────────────────────
            #  CHỨC NĂNG 0: ĐĂNG NHẬP QR (Quét mã nhận diện người dùng)
            # ────────────────────────────────────────────────────────
            if action == "qr_login":
                _handle_qr_login(ui, cap, voice_mode)
                continue

            # ────────────────────────────────────────────────────────
            #  CHỨC NĂNG 1: KHÁM DA (Camera -> YOLO11s AI -> Hội thoại)
            # ────────────────────────────────────────────────────────
            if action == "kham_da":
                print("\n[Worker] 🩺 Người dùng chọn: KHÁM DA", flush=True)
                if cap is None or not cap.isOpened():
                    print("[Worker] ⚠️ Không có camera hoạt động.", flush=True)
                    if voice_mode:
                        from audio.tts import speak
                        speak("Chưa kết nối được camera. Bạn vui lòng cắm camera hoặc chọn Tư vấn để được hỗ trợ nhé.", stop_event=ui.exit_event)
                    ui.set_state(State.IDLE)
                    continue

                # GREETING Khám da
                ui.set_state(State.GREETING)
                user_greet = f"Xin chào bạn {ui.logged_in_user['name']}! " if ui.logged_in_user else "Xin chào! "
                if voice_mode:
                    from audio.tts import speak
                    speak(user_greet + "Vui lòng đưa vùng da cần kiểm tra vào khung hình.", stop_event=ui.exit_event)
                else:
                    time.sleep(1.5)

                if ui.exit_event.is_set():
                    ui.set_state(State.IDLE)
                    continue

                # Vòng lặp soi da và chẩn đoán (tự động thử lại nếu ảnh chưa rõ hoặc chưa thấy da)
                from qr_login import QRScanner
                qr_scanner = QRScanner()

                while ui._running and not ui.exit_event.is_set():
                    ui.set_state(State.VIEWFINDER)
                    ui.capture_event.clear()
                    t_skin_locked = None

                    print("\n[Worker] 📷 Khung ngắm mở — Đưa vùng da vào ô vuông. Khi đạt chuẩn sẽ tự đếm 3s để chụp!", flush=True)

                    while ui._running and not ui.exit_event.is_set():
                        ret, frame = cap.read()
                        if not ret:
                            time.sleep(0.03)
                            continue

                        ui.update_viewfinder(frame)

                        # Kiểm tra mã QR trực tiếp trong khung ngắm nếu người dùng đưa mã vào
                        qr_u, raw_qr, pts = qr_scanner.scan_frame(frame)
                        if qr_u is not None:
                            ui.logged_in_user = qr_u
                            print(f"\n   🎯 [Camera] Quét thấy mã QR: {qr_u['name']} ({qr_u['id']})", flush=True)

                        # Kiểm tra vùng da hoàn hảo
                        skin_ratio = _check_skin_presence(frame)
                        is_skin_perfect = False

                        if skin_ratio >= 0.20:
                            from classifier import SkinClassifier, BLUR_MIN, BRIGHTNESS_RANGE
                            crop, _ = SkinClassifier._crop(frame)
                            if crop is not None and crop.size > 0:
                                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                                mean = float(gray.mean())
                                if var >= BLUR_MIN and BRIGHTNESS_RANGE[0] <= mean <= BRIGHTNESS_RANGE[1]:
                                    is_skin_perfect = True

                        if is_skin_perfect:
                            if t_skin_locked is None:
                                t_skin_locked = time.time()
                                print("   🎯 [Camera] ĐÃ KHÓA VÙNG DA ĐẠT CHUẨN! Đang giữ để chụp...", flush=True)

                            elapsed_locked = time.time() - t_skin_locked
                            remaining = max(1, int(3.0 - elapsed_locked + 0.99))
                            ui.set_state(State.VIEWFINDER, countdown=remaining, locked=True)

                            if elapsed_locked >= 3.0:
                                print("   📸 [Camera] Đủ 3s giữ ổn định -> TỰ ĐỘNG CHỤP!", flush=True)
                                break
                        else:
                            if t_skin_locked is not None:
                                print("   ⚠️ [Camera] Vùng da bị lệch hoặc rung -> Hủy đếm ngược, căn chỉnh lại.", flush=True)
                                t_skin_locked = None
                            ui.set_state(State.VIEWFINDER, countdown=None, locked=False)

                        if ui.capture_event.is_set():
                            print("   📸 [Camera] Người dùng bấm chụp ngay!", flush=True)
                            break

                        time.sleep(0.04)

                    if not ui._running or ui.exit_event.is_set():
                        break

                    # Gom TTA_FRAMES (5 frame) liên tiếp để phân loại
                    collected = []
                    while ui._running and len(collected) < TTA_FRAMES and not ui.exit_event.is_set():
                        ret, frame = cap.read()
                        if ret:
                            collected.append(frame.copy())
                        time.sleep(0.08)

                    if not ui._running or ui.exit_event.is_set():
                        break

                    # ANALYZING: multi-frame TTA với best-yolo11s.pt
                    ui.set_state(State.ANALYZING)

                    avg, reason, disease_key, conf, vi = _multiframe_classify(collected)
                    print(f"[CLS] reason={reason}"
                          + (f" -> {vi} ({conf:.0%})" if reason == "ok" else ""), flush=True)
                    if avg:
                        for k in sorted(avg, key=avg.get, reverse=True)[:4]:
                            print(f"       • {k}: {avg[k]:.0%}", flush=True)

                    # Hiển thị + đọc kết quả
                    _announce(reason, vi, conf, avg, voice_mode, ui=ui, disease_key=disease_key)

                    # Ảnh mờ hoặc không thấy da -> thông báo nhắc nhở và tự mở lại camera
                    if reason in ("blurry", "no_skin"):
                        print("   🔄 [Camera] Đang mở lại camera để bạn căn chỉnh và thử lại...", flush=True)
                        time.sleep(1.2)
                        continue

                    # Nếu chẩn đoán thành công (và không phải da khỏe mạnh) -> Hỏi đặt lịch khám tại phòng bệnh gợi ý (Có / Không, STT 14)
                    initial_query = None
                    if reason == "ok" and disease_key != "healthy":
                        from config import get_clinic_room
                        clinic_room = get_clinic_room(disease_key)
                        initial_query = _handle_booking_flow(ui, disease_key, clinic_room, vi, conf, voice_mode)

                    if not ui._running or ui.exit_event.is_set():
                        break

                    # Vào hội thoại tư vấn theo bệnh vừa phát hiện
                    _conversation_loop(ui, retriever, disease_key, conf, voice_mode, initial_query=initial_query)
                    break

            # ────────────────────────────────────────────────────────
            #  CHỨC NĂNG 2: TƯ VẤN (Bỏ qua Camera -> Mở Mic + RAG)
            # ────────────────────────────────────────────────────────
            elif action == "tu_van":
                print("\n[Worker] 💬 Người dùng chọn: TƯ VẤN TRỰC TIẾP", flush=True)
                ui.set_state(State.GREETING)
                user_greet = f"Xin chào bạn {ui.logged_in_user['name']}! " if ui.logged_in_user else "Xin chào! "
                greeting_tu_van = (
                    f"{user_greet}Tôi là trợ lý tư vấn da liễu ThinkMed. "
                    "Mời bạn đặt câu hỏi hoặc mô tả triệu chứng để tôi hỗ trợ nhé."
                )
                if voice_mode:
                    from audio.tts import speak
                    speak(greeting_tu_van, stop_event=ui.exit_event)
                else:
                    print(f"\n🤖  {greeting_tu_van}\n")
                    time.sleep(1.5)

                if not ui.exit_event.is_set():
                    # Mở thẳng màn hình lắng nghe micro & RAG (không cần camera)
                    _conversation_loop(ui, retriever, disease_key=None, conf=0.0, voice_mode=voice_mode)

            if voice_mode and not ui.exit_event.is_set():
                from audio.tts import speak
                speak("Cảm ơn bạn. Chúc bạn sức khoẻ!")
            ui.set_state(State.IDLE)
            ui.exit_event.clear()
            time.sleep(1.5)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[Worker] ❌ Lỗi: {e}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        if cap is not None:
            print("[Worker] Đang giải phóng camera...", flush=True)
            cap.release()
        ui.stop()
        if gpio:
            try: gpio.cleanup()
            except Exception: pass


def run_robot(use_pir=True, voice_mode=True, camera_source="auto", fullscreen=True, require_qr_login=False):
    ui = ThinkMedUI(fullscreen=fullscreen)
    worker = threading.Thread(target=_worker, args=(ui, use_pir, voice_mode, camera_source, require_qr_login), daemon=True)
    worker.start()
    try:
        ui.run()
    except KeyboardInterrupt:
        print("\n[Robot] Ctrl-C — thoát...", flush=True)
    finally:
        print("[Robot] Đang dừng giao diện và giải phóng tài nguyên...", flush=True)
        ui.stop()
        ui._stop_idle_audio()
        worker.join(timeout=3.5)
        if worker.is_alive():
            print("[Robot] ⚠️ Worker giải phóng tài nguyên chậm sau 3.5s.", flush=True)
        print("[Robot] Đã đóng hệ thống an toàn.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="ThinkMed Robot (Classifier)")
    parser.add_argument("--source",   default="auto", help="'auto' | 'csi' | index webcam (0, 1...)")
    parser.add_argument("--windowed", action="store_true", help="Chạy dạng cửa sổ thay vì toàn màn hình (tiện cho laptop)")
    parser.add_argument("--no-pir",   action="store_true")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--qr-login", action="store_true", help="Yêu cầu quét mã QR đăng nhập khi khởi động")
    args = parser.parse_args()

    info = get_collection_info()
    if info["status"] != "ok":
        print("❌ ChromaDB chưa build. Chạy: python build_rag_full.py")
        sys.exit(1)

    if not Path(CLASSIFIER_MODEL_PATH).exists():
        print(f"❌ Không tìm thấy classifier: {CLASSIFIER_MODEL_PATH}")
        sys.exit(1)

    if not args.no_voice:
        try:
            from audio.stt_parakeet import preload
            print("[Boot] Pre-loading STT model...")
            preload()
        except Exception as e:
            print(f"[Boot] ℹ️ Không khởi động được STT audio ({e}) -> chuyển sang chế độ gõ Text.", flush=True)
            args.no_voice = True

        try:
            from audio.tts import get_piper_voice
            print("[Boot] Pre-loading Piper TTS voice model...")
            get_piper_voice()
        except Exception as e:
            print(f"[Boot] ℹ️ Không khởi động trước được Piper TTS: {e}", flush=True)

    run_robot(
        use_pir=not args.no_pir,
        voice_mode=not args.no_voice,
        camera_source=args.source,
        fullscreen=not args.windowed,
        require_qr_login=args.qr_login,
    )


from pathlib import Path

if __name__ == "__main__":
    main()
