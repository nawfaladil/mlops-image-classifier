###################################################
# Stage 1: Builder — wheel up everything you need
###################################################
FROM python:3.8-slim AS builder

WORKDIR /app

# Copy just requirements and build wheels
COPY requirements.txt ./

# Create venv & install everything
RUN python -m venv .venv \
 && .venv/bin/pip install --upgrade pip \
 && .venv/bin/pip install --no-cache-dir \
      -r requirements.txt \
      torch==2.2.1+cpu \
      torchvision==0.17.1+cpu \
      -f https://download.pytorch.org/whl/torch_stable.html

COPY . .
###################################################
# Stage 2: Runtime — only Python + your wheels + code
###################################################
FROM python:3.8-slim

WORKDIR /app

COPY --from=builder /app/.venv .venv

# Put venv on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy your app code
COPY --from=builder /app .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]