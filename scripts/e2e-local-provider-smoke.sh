#!/usr/bin/env bash
set -euo pipefail

# Containerized E2E smoke test for the local Anthropic provider workspace.
#
# This borrows the same basic flow as amplifier-core/scripts/e2e-smoke-test.sh
# but avoids first-run auto-install pulling provider-anthropic from GitHub main
# by pre-seeding ~/.amplifier/settings.yaml inside the container.
#
# Workspace layout defaults:
#   ../amplifier-core
#   ../amplifier-app-cli
#   ../amplifier-foundation
#   ./   (this provider repo)
#
# Prerequisites:
#   - Docker installed and running
#   - maturin installed
#   - ANTHROPIC_API_KEY exported or stored in ~/.amplifier/keys.env
#
# Environment variables:
#   SMOKE_PROMPT         Prompt to run (default is a fast deterministic smoke prompt)
#   SMOKE_MODEL          Model to use for the live run (default: claude-sonnet-4-6)
#   SMOKE_TIMEOUT        Timeout in seconds (default: 180)
#   SMOKE_BUNDLE         Bundle to run (default: foundation)
#   AMPLIFIER_CORE_REPO  Override path to amplifier-core checkout
#   AMPLIFIER_APP_CLI_REPO
#   AMPLIFIER_FOUNDATION_REPO
#   AMPLIFIER_PROVIDER_REPO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDER_REPO_DIR="${AMPLIFIER_PROVIDER_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKSPACE_DIR="$(cd "$PROVIDER_REPO_DIR/.." && pwd)"
CORE_REPO_DIR="${AMPLIFIER_CORE_REPO:-$WORKSPACE_DIR/amplifier-core}"
APP_CLI_REPO_DIR="${AMPLIFIER_APP_CLI_REPO:-$WORKSPACE_DIR/amplifier-app-cli}"
FOUNDATION_REPO_DIR="${AMPLIFIER_FOUNDATION_REPO:-$WORKSPACE_DIR/amplifier-foundation}"

WHEEL_DIR="$CORE_REPO_DIR/dist"
CONTAINER_NAME="amplifier-provider-e2e-$$"
SKIP_BUILD=false
SMOKE_PROMPT="${SMOKE_PROMPT:-Reply with the exact text local-provider-smoke-ok.}"
SMOKE_MODEL="${SMOKE_MODEL:-claude-sonnet-4-6}"
SMOKE_BUNDLE="${SMOKE_BUNDLE:-foundation}"
TIMEOUT_SECONDS="${SMOKE_TIMEOUT:-180}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[provider-smoke]${NC} $*"; }
info() { echo -e "${CYAN}[provider-smoke]${NC} $*"; }
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

usage() {
    cat <<EOF
Usage: $0 [--skip-build]

Runs a real amplifier session in a clean Docker container while forcing
provider-anthropic to resolve from the local workspace checkout.

Options:
  --skip-build   Reuse the existing amplifier-core wheel in dist/
  --help         Show this help text

Environment:
  SMOKE_PROMPT          Prompt to execute
                        Example richer prompt:
                        Ask recipe author to run one of its example recipes
  SMOKE_MODEL           Model to run (default: claude-sonnet-4-6)
  SMOKE_TIMEOUT         Timeout in seconds
  SMOKE_BUNDLE          Bundle name (default: foundation)
  AMPLIFIER_*_REPO      Override repo locations if not using the standard workspace
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=true; shift ;;
        --help) usage; exit 0 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done

cleanup() {
    log "Cleaning up container $CONTAINER_NAME..."
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

require_dir() {
    local path="$1"
    local label="$2"
    [[ -d "$path" ]] || fail "$label not found: $path"
}

require_dir "$CORE_REPO_DIR" "amplifier-core repo"
require_dir "$APP_CLI_REPO_DIR" "amplifier-app-cli repo"
require_dir "$FOUNDATION_REPO_DIR" "amplifier-foundation repo"
require_dir "$PROVIDER_REPO_DIR" "provider repo"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    KEYS_ENV="$HOME/.amplifier/keys.env"
    if [[ -f "$KEYS_ENV" ]]; then
        log "Loading API keys from $KEYS_ENV..."
        # shellcheck disable=SC1090
        set -a
        source "$KEYS_ENV"
        set +a
    fi
fi

[[ -z "${ANTHROPIC_API_KEY:-}" ]] && fail "ANTHROPIC_API_KEY not set. Export it or store it in ~/.amplifier/keys.env"
command -v docker >/dev/null 2>&1 || fail "Docker not installed or not in PATH"
command -v maturin >/dev/null 2>&1 || fail "maturin not installed or not in PATH"

if [[ "$SKIP_BUILD" == "false" ]]; then
    log "Building local amplifier-core wheel..."
    mkdir -p "$WHEEL_DIR"
    rm -f "$WHEEL_DIR"/amplifier_core-*.whl
    (
        cd "$CORE_REPO_DIR"
        maturin build --release --out "$WHEEL_DIR"
    ) || fail "Wheel build failed"
else
    log "Skipping wheel build (--skip-build)"
fi

WHEEL="$(ls "$WHEEL_DIR"/amplifier_core-*.whl 2>/dev/null | head -1)"
[[ -n "$WHEEL" ]] || fail "No amplifier-core wheel found in $WHEEL_DIR"
WHEEL_BASENAME="$(basename "$WHEEL")"

log "Creating clean container..."
docker run -d --name "$CONTAINER_NAME" \
    -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    -e AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-}" \
    python:3.12-slim \
    sleep 3600 >/dev/null \
    || fail "Container creation failed"

info "Container: $CONTAINER_NAME"

log "Bootstrapping container (git + uv)..."
docker exec "$CONTAINER_NAME" bash -lc "
    apt-get update -qq &&
    apt-get install -y -qq git >/dev/null 2>&1 &&
    pip install -q uv
" || fail "Container bootstrap failed"

log "Installing amplifier from GitHub main..."
docker exec "$CONTAINER_NAME" bash -lc "
    export PATH=/root/.local/bin:\$PATH
    uv tool install git+https://github.com/microsoft/amplifier@main >/tmp/amplifier-install.log 2>&1
    tail -5 /tmp/amplifier-install.log
" || fail "Amplifier install failed"

log "Injecting local amplifier-core wheel..."
docker cp "$WHEEL" "$CONTAINER_NAME:/tmp/$WHEEL_BASENAME" || fail "Failed to copy core wheel"
CORE_OVERRIDE_OUT="$(docker exec "$CONTAINER_NAME" bash -lc "
    uv pip install \
        --python /root/.local/share/uv/tools/amplifier/bin/python3 \
        --force-reinstall --no-deps \
        '/tmp/$WHEEL_BASENAME' 2>&1
")" || fail "Failed to install local core wheel"
log "Core override: $(echo "$CORE_OVERRIDE_OUT" | tail -1)"

docker exec "$CONTAINER_NAME" mkdir -p /tmp/local-sources || fail "Failed to create local source directory"

install_local_repo() {
    local host_path="$1"
    local install_deps="${2:-false}"
    local repo_name
    local container_path
    local install_out
    local extra_flags="--no-deps"

    repo_name="$(basename "$host_path")"
    container_path="/tmp/local-sources/$repo_name"

    if [[ "$install_deps" == "true" ]]; then
        extra_flags=""
    fi

    info "Copying $repo_name -> $container_path"
    docker cp "$host_path" "$CONTAINER_NAME:$container_path" || fail "Failed to copy $repo_name"

    info "Installing $repo_name into the amplifier tool environment"
    install_out="$(docker exec "$CONTAINER_NAME" bash -lc "
        uv pip install \
            --python /root/.local/share/uv/tools/amplifier/bin/python3 \
            --force-reinstall $extra_flags \
            -e '$container_path' 2>&1
    ")" || fail "Failed to install $repo_name"
    log "$repo_name: $(echo "$install_out" | tail -1)"
}

install_local_repo "$APP_CLI_REPO_DIR"
install_local_repo "$FOUNDATION_REPO_DIR"
install_local_repo "$PROVIDER_REPO_DIR" true

log "Writing container settings to force the local Anthropic provider"
docker exec "$CONTAINER_NAME" bash -lc "mkdir -p /root/.amplifier"
docker exec "$CONTAINER_NAME" bash -lc "cat > /root/.amplifier/settings.yaml <<'EOF'
sources:
  modules:
    provider-anthropic: /tmp/local-sources/amplifier-module-provider-anthropic
config:
  providers:
    - module: provider-anthropic
      source: /tmp/local-sources/amplifier-module-provider-anthropic
      config:
        default_model: $SMOKE_MODEL
        raw: true
        fallback_on_overload: true
        fallback_retry_count: 1
        fallback_cooldown_seconds: 300
        persist_fallback_state: false
EOF" || fail "Failed to write settings.yaml"

log "Verifying installed versions and provider path..."
INSTALLED_VERSION="$(docker exec "$CONTAINER_NAME" bash -lc "
    export PATH=/root/.local/bin:\$PATH
    amplifier --version
")" || fail "Failed to read amplifier version"
info "Installed: $INSTALLED_VERSION"

PROVIDER_PATH="$(docker exec "$CONTAINER_NAME" bash -lc "
    /root/.local/share/uv/tools/amplifier/bin/python3 - <<'PY'
import amplifier_module_provider_anthropic
print(amplifier_module_provider_anthropic.__file__)
PY
")" || fail "Failed to resolve provider import path"
info "Provider import: $PROVIDER_PATH"

if [[ "$PROVIDER_PATH" != *"/tmp/local-sources/amplifier-module-provider-anthropic/"* ]]; then
    fail "Provider import path is not the local source checkout"
fi

SETTINGS_OUT="$(docker exec "$CONTAINER_NAME" bash -lc "
    cat /root/.amplifier/settings.yaml
")" || fail "Failed to read container settings"
info "Container settings:"
echo "$SETTINGS_OUT"

echo ""
log "============================================================"
log " LOCAL PROVIDER SMOKE TEST START"
log " Bundle: $SMOKE_BUNDLE"
log " Model: $SMOKE_MODEL"
log " Prompt: '$SMOKE_PROMPT'"
log " Timeout: ${TIMEOUT_SECONDS}s"
log "============================================================"
echo ""

SMOKE_EXIT_CODE=0
SMOKE_OUTPUT="$(docker exec "$CONTAINER_NAME" bash -lc "
    export PATH=/root/.local/bin:\$PATH
    export AMPLIFIER_MODULE_PROVIDER_ANTHROPIC=/tmp/local-sources/amplifier-module-provider-anthropic
    timeout $TIMEOUT_SECONDS amplifier run \
        --bundle '$SMOKE_BUNDLE' \
        --provider anthropic \
        --model '$SMOKE_MODEL' \
        '$SMOKE_PROMPT' 2>&1
" 2>&1)" || SMOKE_EXIT_CODE=$?

echo "============================================================"
echo " SMOKE TEST OUTPUT (last 40 lines):"
echo "============================================================"
echo "$SMOKE_OUTPUT" | tail -40
echo "============================================================"
echo ""

ERROR_PATTERNS="Traceback|TypeError|AttributeError|no attribute|object has no attribute|ImportError|ModuleNotFoundError|RuntimeError|KeyError|ValueError"
if echo "$SMOKE_OUTPUT" | grep -qE "$ERROR_PATTERNS"; then
    echo "============================================================"
    echo " ERRORS DETECTED:"
    echo "============================================================"
    echo "$SMOKE_OUTPUT" | grep -E "$ERROR_PATTERNS" | head -20
    echo "============================================================"
    fail "Smoke test failed with Python exceptions"
fi

TOOL_FAILURE_COUNT="$(echo "$SMOKE_OUTPUT" | grep -cE "Tool .+ failed:" || true)"
if [[ "$TOOL_FAILURE_COUNT" -gt 0 ]]; then
    echo "============================================================"
    echo " TOOL FAILURES DETECTED ($TOOL_FAILURE_COUNT):"
    echo "============================================================"
    echo "$SMOKE_OUTPUT" | grep -E "Tool .+ failed:" | head -10
    echo "============================================================"
    fail "Smoke test failed with tool failures"
fi

if [[ "$SMOKE_EXIT_CODE" -eq 124 ]]; then
    fail "Smoke test timed out after ${TIMEOUT_SECONDS}s"
fi

if [[ "$SMOKE_EXIT_CODE" -ne 0 ]]; then
    fail "Smoke test exited non-zero ($SMOKE_EXIT_CODE)"
fi

pass "========================================================"
pass " LOCAL PROVIDER SMOKE TEST PASSED"
pass " $INSTALLED_VERSION"
pass " Provider import: $PROVIDER_PATH"
pass "========================================================"
