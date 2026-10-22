import cv2
import numpy as np
from robot_ui import ThinkMedUI, State, BG
from config import get_clinic_room

ui = ThinkMedUI(fullscreen=False)
test_frame = np.full((480, 640, 3), (200, 180, 160), dtype=np.uint8)
cv2.putText(test_frame, "Skin Area Test", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2)

disease_key = "acne"
vi_name = "Mụn trứng cá"
conf = 0.95
room = get_clinic_room(disease_key)

# Test 1: Ask state
canvas1 = np.full((ui.H, ui.W, 3), BG, dtype=np.uint8)
ui.set_state(
    State.RESULT_OK,
    vi_name=vi_name,
    confidences=[conf],
    disease_key=disease_key,
    clinic_room=room,
    booking_stage="ask",
    stt_num=14,
    patient_name="Nguyễn Việt Phương",
    speaking=False,
)
ui._draw_result_ok(canvas1, test_frame, ui.result_data)
cv2.imwrite("assets/test_stage_ask.png", canvas1)

# Test 2: Confirmed state (STT 14) with QR logged in user
canvas2 = np.full((ui.H, ui.W, 3), BG, dtype=np.uint8)
ui.set_state(
    State.RESULT_OK,
    vi_name=vi_name,
    confidences=[conf],
    disease_key=disease_key,
    clinic_room=room,
    booking_stage="confirmed",
    stt_num=14,
    patient_name="Nguyễn Việt Phương",
    speaking=False,
)
ui._draw_result_ok(canvas2, test_frame, ui.result_data)
cv2.imwrite("assets/test_stage_confirmed_qr.png", canvas2)

# Test 3: Confirmed state (STT 14) for guest user (Khách vãng lai)
canvas3 = np.full((ui.H, ui.W, 3), BG, dtype=np.uint8)
ui.set_state(
    State.RESULT_OK,
    vi_name=vi_name,
    confidences=[conf],
    disease_key=disease_key,
    clinic_room=room,
    booking_stage="confirmed",
    stt_num=14,
    patient_name="Bệnh nhân vãng lai",
    speaking=False,
)
ui._draw_result_ok(canvas3, test_frame, ui.result_data)
cv2.imwrite("assets/test_stage_confirmed_guest.png", canvas3)

# Test 4: Declined state
canvas4 = np.full((ui.H, ui.W, 3), BG, dtype=np.uint8)
ui.set_state(
    State.RESULT_OK,
    vi_name=vi_name,
    confidences=[conf],
    disease_key=disease_key,
    clinic_room=room,
    booking_stage="declined",
    stt_num=14,
    patient_name="Nguyễn Việt Phương",
    speaking=False,
)
ui._draw_result_ok(canvas4, test_frame, ui.result_data)
cv2.imwrite("assets/test_stage_declined.png", canvas4)

print("All 4 booking test screens saved successfully!")
