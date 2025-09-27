FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgraphviz-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 安装uv工具
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 复制项目文件
COPY . /app/

# 安装Python依赖
RUN ~/.cargo/bin/uv pip install --no-cache-dir -r requirements.txt

# 确保图形社区检测库正确安装
RUN ~/.cargo/bin/uv pip install --no-cache-dir leidenalg igraph

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 创建数据和日志目录
RUN mkdir -p /app/data /app/logs

# 暴露端口
EXPOSE 8501

# 设置入口点
ENTRYPOINT ["python", "init.py"]

# 默认命令
CMD ["python", "graphRAG_query.py"]