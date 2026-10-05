FROM rust:1.95-slim-trixie AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p vellev \
    && cp target/release/vellev /usr/local/bin/vellev

FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home vellev
COPY --from=build /usr/local/bin/vellev /usr/local/bin/vellev
USER vellev
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD ["vellev", "health"]
ENTRYPOINT ["tini", "--", "vellev"]
CMD ["serve"]
