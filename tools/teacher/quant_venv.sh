#!/usr/bin/env bash
# A disposable venv for the NVFP4 quantization: it inherits the verl environment (torch, transformers) and adds
# llm-compressor without touching that environment. Tested combination: llmcompressor 0.12.0, compressed-tensors 0.17.1
# (0.15 needs newer transformers and compressed-tensors than PEAR's verl environment has).
#   WORK_DIR=<work dir> bash tools/teacher/quant_venv.sh
set -euo pipefail
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
V=${QUANT_VENV:-$WORK_DIR/envs/quant}
"$WORK_DIR/envs/verl/bin/python" -m venv --system-site-packages "$V"
"$V/bin/pip" install -q --no-deps llmcompressor==0.12.0 compressed-tensors==0.17.1
"$V/bin/pip" show llmcompressor | grep -E "^(Version|Requires)"
# llm-compressor's light dependencies, installed one by one without touching the inherited packages
for i in 1 2 3 4 5 6; do
  out=$("$V/bin/python" -c "from llmcompressor import oneshot; from llmcompressor.modifiers.quantization import QuantizationModifier; print('IMPORT_OK')" 2>&1 | tail -1) || true
  echo "$out"
  case "$out" in
    IMPORT_OK) break;;
    *"No module named"*) m=$(echo "$out" | sed -E "s/.*No module named '([^'.]+).*/\1/"); echo "installing $m"; "$V/bin/pip" install -q --no-deps "$m";;
    *) break;;
  esac
done
