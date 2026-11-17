# The same environment as an image: build.sh on Ubuntu 24.04, sources from fetch_sources.sh as a named context.
#   bash env/slime/fetch_sources.sh /path/to/slime-sources
#   docker build -f env/slime/Dockerfile --build-context sources=/path/to/slime-sources -t pear-slime:0.5.21 .
# Mirrors as in build.sh: --build-arg PYPI_INDEX_URL=... TORCH_FIND_LINKS=... RUSTUP_DIST_SERVER=...
# RUSTUP_UPDATE_ROOT=... CARGO_REGISTRY_MIRROR=... UV_PYTHON_INSTALL_MIRROR=...
FROM ubuntu:24.04

ARG PYPI_INDEX_URL= TORCH_FIND_LINKS= RUSTUP_DIST_SERVER= RUSTUP_UPDATE_ROOT= CARGO_REGISTRY_MIRROR= \
    UV_PYTHON_INSTALL_MIRROR= CUDA_ARCHS="9.0;10.0;12.0" NVTE_CUDA_ARCHS="90;100f;120f" JOBS=32
ENV DEBIAN_FRONTEND=noninteractive

# compilers for the CUDA extensions, protoc for SGLang's gRPC extension, RDMA libraries for NCCL over InfiniBand/RoCE
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates pkg-config protobuf-compiler libnuma1 libnuma-dev \
      libibverbs1 ibverbs-providers librdmacm1 numactl \
    && rm -rf /var/lib/apt/lists/*

COPY env/slime /build/env/slime
RUN --mount=type=bind,from=sources,target=/build/sources \
    PYPI_INDEX_URL=$PYPI_INDEX_URL TORCH_FIND_LINKS=$TORCH_FIND_LINKS RUSTUP_DIST_SERVER=$RUSTUP_DIST_SERVER \
    RUSTUP_UPDATE_ROOT=$RUSTUP_UPDATE_ROOT CARGO_REGISTRY_MIRROR=$CARGO_REGISTRY_MIRROR \
    UV_PYTHON_INSTALL_MIRROR=$UV_PYTHON_INSTALL_MIRROR CUDA_ARCHS=$CUDA_ARCHS NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS \
    JOBS=$JOBS bash /build/env/slime/build.sh /build/sources \
    && rm -rf /opt/slime/.cache /opt/slime/rust /opt/slime/src/sglang/rust/target

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
SHELL ["/bin/bash", "-c"]
RUN echo 'source /opt/slime/activate.sh' >> /etc/bash.bashrc
CMD ["bash"]
