# The ONNX Runtime local embedding links, from a pinned copy: our release
# first, ort-sys's CDN after, taken only as pinned (scripts/deps). Its own
# layer, so that it is fetched again only when the pins change.
FROM debian:trixie-slim AS onnxruntime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/deps/onnxruntime.sh scripts/deps/onnxruntime.tsv /deps/
RUN /deps/onnxruntime.sh fetch "$(uname -m)-unknown-linux-gnu" /opt/onnxruntime

FROM rust:1.95-slim-trixie AS build
# The AWS SDK's TLS and the ONNX Runtime that local embedding links are
# native code.
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake g++ make perl pkg-config \
    && rm -rf /var/lib/apt/lists/*
# ort-sys links the pinned library and never downloads one: a build that
# would fails instead.
COPY --from=onnxruntime /opt/onnxruntime /opt/onnxruntime
ENV ORT_LIB_LOCATION=/opt/onnxruntime ORT_SKIP_DOWNLOAD=1
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p vellev \
    && cp target/release/vellev /usr/local/bin/vellev

FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libstdc++6 tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home vellev \
    && install -d -o vellev -g vellev /var/lib/vellev/models /var/lib/vellev/cache
COPY --from=build /usr/local/bin/vellev /usr/local/bin/vellev
# The binary holds ONNX Runtime, so its licence and notices go with it.
COPY scripts/deps/onnxruntime/ /usr/share/doc/onnxruntime/
USER vellev
ENV VELLEV_CACHE_DIR=/var/lib/vellev/cache VELLEV_MODEL_DIR=/var/lib/vellev/models
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD ["vellev", "health"]
ENTRYPOINT ["tini", "--", "vellev"]
CMD ["serve"]
