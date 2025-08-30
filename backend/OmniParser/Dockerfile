FROM --platform=linux/amd64 pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安装基本的系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 创建模型和权重文件夹
RUN mkdir -p weights/icon_detect_v1_5 \
    && mkdir -p weights/icon_caption_florence

EXPOSE 8000

CMD ["uvicorn", "omni:app", "--host", "0.0.0.0", "--port", "8000"] 