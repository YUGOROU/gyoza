# GYOZA 盛り付けデモ Space（Docker SDK・GPU T4）
# MuJoCo は EGL でヘッドレス GPU 描画。RL(PPO) は CPU。エージェント LLM は HF inference へ外部呼び出し。
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    HF_HOME=/tmp/hf PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgles2 libosmesa6 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY space/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# コード + アセット（gyoza パッケージ・SO-101 MJCF・veg メッシュ）を同梱
COPY gyoza /app/gyoza
COPY third_party/mujoco_menagerie/robotstudio_so101 /app/third_party/mujoco_menagerie/robotstudio_so101
COPY space /app/space
# RL モデルはビルド時に同梱（別途 space/best_model.zip を配置）

ENV RL_MODEL=/app/space/best_model.zip
EXPOSE 7860
WORKDIR /app/space
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
