FROM rust:1.95-slim-trixie AS build
# The AWS SDK's TLS and the ONNX Runtime that local embedding links are
# native code.
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake g++ make perl pkg-config \
    && rm -rf /var/lib/apt/lists/*
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
USER vellev
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD ["vellev", "health"]
ENTRYPOINT ["tini", "--", "vellev"]
CMD ["serve"]
