# visualq-vrai triage service — the image deployed as `visualq-ml` on ECS Fargate.
# Build:  docker build --platform linux/amd64 -t visualq-vrai .
# Run:    docker run -p 8090:8090 visualq-vrai
FROM python:3.12-slim

WORKDIR /app

# Torch CPU wheels are enough for TabICL inference and keep the image small.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY schema ./schema

RUN pip install --no-cache-dir .

EXPOSE 8090

ENV VRAI_BUNDLES_DIR=/data/bundles

# --factory honors VRAI_URL_PREFIX (mount under a path prefix behind an ALB).
CMD ["uvicorn", "visualq_vrai.service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8090"]
