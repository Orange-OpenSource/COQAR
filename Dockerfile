FROM python:3.8-slim AS base
LABEL name="qrew"
LABEL description="Question rewriting"
LABEL maintainer="diana@orange.com"
LABEL url="https://gitlab.tech.orange/generation4dial/qrew.git"

ARG http_proxy
ARG https_proxy
ARG no_proxy

# hadolint ignore=DL3005,DL3008,DL3009
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends curl \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

FROM base as builder

COPY requirements.txt /app/
COPY server/requirements.txt /app/server/requirements.txt

# hadolint ignore=DL3005,DL3008,DL3009,SC1091
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends build-essential git \
    && python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --no-cache-dir pip==21.3.1 \
    && pip install --no-cache-dir -r /app/requirements.txt \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && pip install --no-cache-dir -r /app/server/requirements.txt \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY rewriting/ /app/rewriting/
COPY server/ /app/server/
COPY api.py /app/

FROM base AS runtime

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app


ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

RUN groupadd abc \
    && useradd --create-home -d /home/abc -s /bin/bash -g abc abc

RUN chgrp -R 0 /home/abc && \
    chmod -R g=u /home/abc

USER abc
WORKDIR /app/

ARG port=8090
ENV API_PORT ${port}

EXPOSE ${API_PORT}
# hadolint ignore=DL3025
#HEALTHCHECK CMD curl --fail  http://127.0.0.1:${API_PORT}/api/status || exit 1
# hadolint ignore=DL3025
CMD python api.py --port ${API_PORT}  --static-folder /app/server/static/
# hadolint ignore=DL3025
#CMD PYTHONPATH=/app python /app/main.py ${AGENT_CONFIG}
