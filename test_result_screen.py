import cv2
import numpy as np
from robot_ui import ThinkMedUI, State, BG
from config import get_clinic_room

ui = ThinkMedUI(fullscreen=False)
test_frame = np.full((480, 640, 3), (200, 180, 160), dtype=np.uint8)
cv2.putText(test_frame, "Skin Area Test", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2)

test_cases = [
    ("acne", "Mụn trứng cá", 0.95),
    ("atopic_dermatitis", "Viêm da cơ địa", 0.88),
    ("psoriasis", "Vảy nến", 0.92),
    ("tinea", "Nấm da", 0.84),
    ("wart", "Mụn cóc", 0.90),
    ("healthy", "Da khỏe mạnh", 0.98),
]

for disease_key, vi_name, conf in test_cases:
    room = get_clinic_room(disease_key)
    canvas = np.full((ui.H, ui.W, 3), BG, dtype=np.uint8)
    data = {
        "vi_name": vi_name,
        "confidences": [conf],
        "disease_key": disease_key,
        "clinic_room": room,
        "speaking": False,
    }
    ui._draw_result_ok(canvas, test_frame, data)
    out_path = f"assets/test_result_{disease_key}.png"
    cv2.imwrite(out_path, canvas)
    print(f"Generated {out_path} -> {room['room_num']}: {room['room_name']}")

print("All test screens rendered successfully!")
