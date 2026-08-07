#!/bin/sh
# WSLg's default GL context is llvmpipe (software). Force the D3D12 GPU path.
cd "$(dirname "$0")/.." || exit 1
GALLIUM_DRIVER=d3d12 exec ./host_app "$@"
