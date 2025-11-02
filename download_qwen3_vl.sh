#!/bin/bash
# Download Qwen3-VL-2B-Instruct-FP8 with wget (resumable)
# Usage: bash download_qwen3_vl.sh

set -e

MODEL_NAME="Qwen3-VL-2B-Instruct-FP8"
BASE_URL="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8/resolve/main"
DOWNLOAD_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct-FP8/snapshots/main"

# Create directory
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo "Downloading $MODEL_NAME to $DOWNLOAD_DIR"
echo "Downloads are resumable - you can Ctrl+C and re-run this script"
echo ""

# List of files to download
FILES=(
    ".gitattributes"
    "README.md"
    "chat_template.json"
    "config.json"
    "generation_config.json"
    "model-00001-of-00001.safetensors"
    "model.safetensors.index.json"
    "preprocessor_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "video_preprocessor_config.json"
    "vocab.json"
)

# Download each file with resume capability
for file in "${FILES[@]}"; do
    echo "-----------------------------------"
    echo "Downloading: $file"

    # Check if file already exists and is complete
    if [ -f "$file" ]; then
        echo "  ✓ Already exists: $file"
    else
        # wget with resume (-c), timeout, retries
        wget -c --timeout=30 --tries=5 --retry-connrefused \
             --progress=bar:force \
             "${BASE_URL}/${file}" \
             -O "$file" || {
            echo "  ⚠ Failed to download $file (will retry on next run)"
        }
    fi
done

echo ""
echo "====================================="
echo "Download summary:"
echo "====================================="

# Check what's downloaded
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ✗ $file (missing)"
    fi
done

echo ""
echo "Download location: $DOWNLOAD_DIR"
echo ""
echo "To resume failed downloads, just run this script again!"
