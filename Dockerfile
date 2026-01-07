FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

#to use the uvicorn command without say uv run uvicorn
ENV PATH="/app/.venv/bin:$PATH"

COPY ["pyproject.toml", "uv.lock", ".python-version","./"]
RUN uv sync --locked

COPY ["src/api/predict.py","./"]
COPY ["src/scripts/densenet_isic.pth","./src/scripts/"]
COPY ["src/modules/model.py","./src/modules/"]

EXPOSE 9696

ENTRYPOINT [ "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696" ]
