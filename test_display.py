"""
test_display.py — Test rendering riêng biệt, KHÔNG load RAG/YOLO.
Mục đích: xác định chính xác dòng nào treo (window thường hay fullscreen).

Usage:
    DISPLAY=:0 python test_display.py
"""

import sys
import time

import cv2
import numpy as np

print("=" * 50, flush=True)
print("TEST 1: Window thường (không fullscreen)", flush=True)
print("=" * 50, flush=True)

print("Gọi cv2.namedWindow...", flush=True)
cv2.namedWindow("Test1", cv2.WINDOW_NORMAL)
print("namedWindow xong.", flush=True)

print("Gọi cv2.resizeWindow...", flush=True)
cv2.resizeWindow("Test1", 640, 480)
print("resizeWindow xong.", flush=True)

img = np.zeros((480, 640, 3), dtype=np.uint8)
img[:] = (50, 50, 200)  # đỏ (BGR)
cv2.putText(img, "TEST 1 OK", (200, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

print("Vào vòng lặp imshow — nếu thấy ô màu ĐỎ thì TEST 1 PASS.", flush=True)
print("Nhấn ESC để qua TEST 2, hoặc đợi 6s tự chuyển.", flush=True)

start = time.time()
while time.time() - start < 6:
    cv2.imshow("Test1", img)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

print("TEST 1 hoàn tất — destroy window...", flush=True)
cv2.destroyAllWindows()
cv2.waitKey(1)
print("✅ TEST 1 PASS (basic rendering hoạt động)\n", flush=True)


print("=" * 50, flush=True)
print("TEST 2: Fullscreen property", flush=True)
print("=" * 50, flush=True)

print("Gọi cv2.namedWindow...", flush=True)
cv2.namedWindow("Test2", cv2.WINDOW_NORMAL)
print("namedWindow xong.", flush=True)

print("Gọi cv2.setWindowProperty(FULLSCREEN)...", flush=True)
print(">>> NẾU KHÔNG THẤY DÒNG TIẾP THEO → TREO Ở ĐÂY <<<", flush=True)
cv2.setWindowProperty("Test2", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
print("setWindowProperty XONG — không bị treo!", flush=True)

img2 = np.zeros((720, 1280, 3), dtype=np.uint8)
img2[:] = (50, 200, 50)  # xanh
cv2.putText(img2, "TEST 2 OK - FULLSCREEN", (350, 360),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)

print("Vào vòng lặp imshow fullscreen — nếu thấy full màn hình XANH thì PASS.", flush=True)
start = time.time()
while time.time() - start < 6:
    cv2.imshow("Test2", img2)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
print("✅ TEST 2 PASS\n", flush=True)
print("HOÀN TẤT TẤT CẢ TEST.", flush=True)
