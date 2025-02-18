FROM ubuntu:latest
LABEL authors="REALMS"

# 使用官方 Python 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器
COPY . /app

# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 启动 Streamlit 应用
CMD ["streamlit", "run", "streamlit_app.py"]