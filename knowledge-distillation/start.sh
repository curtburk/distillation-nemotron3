#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# ZGX Fury KD — Corpus Construction Pipeline
#
# Usage:
#   ./start.sh teacher     Launch DeepSeek V4 Flash teacher (single vLLM instance)
#   ./start.sh sandbox     Build the code execution sandbox image
#   ./start.sh pipeline    Run steps 01-06 sequentially
#   ./start.sh status      Show teacher health + memory budget + progress
#   ./start.sh stop        Stop the teacher container
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLUE}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }

# ── Preflight ─────────────────────────────────────────────────────────────────

preflight() {
    command -v python3 >/dev/null || { err "python3 not found"; exit 1; }
    command -v docker >/dev/null || { err "docker not found"; exit 1; }
    python3 -c "import httpx, datasets" 2>/dev/null || {
        warn "Installing Python deps..."
        pip install -r requirements.txt --break-system-packages -q
    }
}

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_teacher() {
    preflight

    if docker ps --format '{{.Names}}' | grep -q '^vllm-teacher$'; then
        warn "Teacher already running. Use './start.sh stop' first."
        exit 1
    fi

    log "Launching DeepSeek V4 Flash NVFP4 teacher on port 8091 (GPU 1 / GB300)..."

    docker run -d \
        --name vllm-teacher \
        --runtime nvidia \
        --gpus '"device=1"' \
        -p 8091:8000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/vllm:/root/.cache/vllm \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -e HF_HUB_OFFLINE=1 \
        --ipc=host \
        vllm/vllm-openai:latest \
        --model nvidia/DeepSeek-V4-Flash-NVFP4 \
        --trust-remote-code \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --reasoning-parser deepseek_v3 \
        --port 8000

    log "Waiting for teacher to become ready (up to 20 min)..."
    for i in $(seq 1 240); do
        if curl -sf "http://localhost:8091/v1/models" >/dev/null 2>&1; then
            ok "Teacher ready on port 8091"
            return 0
        fi
        sleep 5
    done

    err "Teacher did not become ready. Last 50 log lines:"
    docker logs vllm-teacher --tail 50
    exit 1
}

cmd_stop() {
    if docker ps -a --format '{{.Names}}' | grep -q '^vllm-teacher$'; then
        log "Stopping teacher..."
        docker stop vllm-teacher >/dev/null 2>&1 || true
        docker rm vllm-teacher >/dev/null 2>&1 || true
        ok "Teacher stopped"
    else
        log "Teacher not running"
    fi
}

cmd_sandbox() {
    log "Building sandbox image..."
    docker build -f Dockerfile.sandbox -t fury-kd-sandbox:latest .
    ok "Sandbox image built: fury-kd-sandbox:latest"
}

cmd_pipeline() {
    preflight

    # Verify teacher is running
    curl -sf "http://localhost:8091/v1/models" >/dev/null 2>&1 || {
        err "Teacher not responding on port 8091. Run './start.sh teacher' first."
        exit 1
    }

    # Verify sandbox exists
    docker image inspect fury-kd-sandbox:latest >/dev/null 2>&1 || {
        warn "Sandbox image missing. Building..."
        cmd_sandbox
    }

    log "Starting corpus construction pipeline (est. 5-7 days)..."

    for step in 01_download_datasets 02_generate_seed_problems 03_generate_traces \
                04_execute_and_filter 05_score_reasoning 06_assemble_dataset; do
        echo ""
        log "→ scripts/${step}.py"
        python3 "scripts/${step}.py"
    done

    ok "Pipeline complete. Dataset ready in data/final/"
}

cmd_status() {
    echo ""
    log "Teacher health:"
    if curl -sf "http://localhost:8091/v1/models" >/dev/null 2>&1; then
        MODEL=$(curl -sf "http://localhost:8091/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
        ok "  Port 8091: ${MODEL}"
    else
        err "  Port 8091: not responding"
    fi

    echo ""
    log "HBM budget:"
    python3 -c "
import sys; sys.path.insert(0, '.')
import config
b = config.hbm_budget()
status = '✓' if b['fits_hbm'] else '✗'
print(f\"  {status} Teacher: {b['teacher_gb']} GB | KV cache: {b['kv_cache_gb']} GB | Overhead: {b['overhead_gb']} GB\")
print(f\"  {status} Total: {b['total_used_gb']} GB / {b['hbm_gb']} GB HBM  (headroom: {b['hbm_headroom_gb']} GB)\")
"

    echo ""
    log "Pipeline progress:"
    for step in raw seeds traces filtered scored final; do
        count=$(find "data/$step" -type f -name "*.jsonl" 2>/dev/null | wc -l)
        lines=$(cat data/$step/*.jsonl 2>/dev/null | wc -l)
        printf "  data/%-10s  %2d files  %8d records\n" "$step" "$count" "$lines"
    done
    echo ""
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

case "${1:-help}" in
    teacher)   cmd_teacher ;;
    stop)      cmd_stop ;;
    sandbox)   cmd_sandbox ;;
    pipeline)  cmd_pipeline ;;
    status)    cmd_status ;;
    *)
        cat <<'EOF'
Usage: ./start.sh <command>

Commands:
  teacher    Launch DeepSeek V4 Flash teacher (vLLM, single instance)
  sandbox    Build the code execution sandbox image
  pipeline   Run corpus construction pipeline (steps 01-06)
  status     Show teacher health, HBM budget, pipeline progress
  stop       Stop the teacher container

Typical workflow:
  ./start.sh sandbox            # one-time
  ./start.sh teacher            # ~20 min first launch (model download)
  ./start.sh status             # verify budget + health
  ./start.sh pipeline           # ~5-7 days unattended
  ./start.sh stop               # when done
EOF
        ;;
esac