#!/bin/bash
# setup_nemo_test.sh — Dựng môi trường TÁCH BẠCH để test Parakeet Vietnamese (NeMo).
#
# TÁCH BẠCH HOÀN TOÀN khỏi môi trường ThinkMed:
#   - Dùng virtualenv riêng ~/nemo_test_env
#   - KHÔNG cài đè lên môi trường đang chạy robot
#   - Nếu hỏng, chỉ cần xóa ~/nemo_test_env, hệ thống chính không ảnh hưởng
#
# Chạy:  bash setup_nemo_test.sh

set -e
echo "═══════════════════════════════════════════════════════"
echo "Dựng môi trường test NeMo — TÁCH BẠCH khỏi ThinkMed"
echo "═══════════════════════════════════════════════════════"

# Đẩy hiệu năng tối đa để đo đúng trần tốc độ
echo "[1/5] Đặt Jetson về chế độ hiệu năng tối đa (MAXN)..."
echo "  (chạy tay nếu cần sudo): sudo nvpmodel -m 0 && sudo jetson_clocks"

# venv riêng
echo "[2/5] Tạo virtualenv riêng ~/nemo_test_env..."
python3 -m venv ~/nemo_test_env
source ~/nemo_test_env/bin/activate

# PyTorch cho Jetson — QUAN TRỌNG: phải dùng bản build riêng cho Jetson,
# KHÔNG dùng pip torch generic (nhớ bài học: pip generic thiếu CUDA cho Tegra)
echo "[3/5] LƯU Ý PyTorch:"
echo "  NeMo cần PyTorch. Trên Jetson PHẢI dùng bản build sẵn của NVIDIA,"
echo "  KHÔNG dùng 'pip install torch' thường (thiếu CUDA cho sm_120)."
echo "  Xem: https://developer.download.nvidia.com/compute/redist/jp/"
echo "  hoặc dùng bản torch từ jetson-containers / forums.developer.nvidia.com"

# NeMo ASR (bản nhẹ, chỉ ASR — không cài full NeMo)
echo "[4/5] Cài nemo_toolkit[asr]..."
echo "  pip install nemo_toolkit[asr]"
echo "  (Nếu lỗi dependency, thử: pip install nemo_toolkit[asr]==<version tương thích>)"

echo "[5/5] Xong. Kích hoạt bằng: source ~/nemo_test_env/bin/activate"
echo ""
echo "⚠  CẢNH BÁO: PyTorch trên Jetson là bước khó nhất. Nếu import torch"
echo "   báo lỗi CUDA, phải cài đúng bản Jetson trước khi cài NeMo."
