FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --uid 10001 --no-create-home gateway
USER 10001

ENV TESTBENCH_HOST=0.0.0.0 \
    TESTBENCH_PORT=8000 \
    PYTHONUNBUFFERED=1
EXPOSE 8000

# python 直接做 1 号进程：uvicorn 收到 SIGTERM 后停止接新连接，等进行中的请求处理完再退出
CMD ["python", "-c", "from testbench.gateway.app import main; main()"]
