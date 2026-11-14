# ONNX Runtime

vellev's image and binary statically link ONNX Runtime 1.28.0 for local
embedding. The library is the build ort-sys downloads, made by pyke
(https://ort.pyke.io) from Microsoft's ONNX Runtime
(https://github.com/microsoft/onnxruntime). It is pinned in
`scripts/deps/onnxruntime.tsv`.

`LICENSE` and `ThirdPartyNotices.txt` are ONNX Runtime's own, as of its
v1.28.0 tag. The image carries them in `/usr/share/doc/onnxruntime/`.
