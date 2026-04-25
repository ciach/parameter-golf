#!/usr/bin/env bash
set -Eeuo pipefail

# Local wrapper: clone the repo on a fresh RunPod, upload this working-tree harness,
# then run the normal pod setup/preflight script.
# Usage:
#   SSH_HOST=root@1.2.3.4 SSH_PORT=12345 ./setup_runpod_autoresearch.sh

SSH_HOST="${SSH_HOST:-}"
SSH_PORT="${SSH_PORT:-}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
REPO_DIR="${REPO_DIR:-/workspace/parameter-golf}"
REPO_URL="${REPO_URL:-https://github.com/ciach/parameter-golf.git}"
RUN_REMOTE_SETUP="${RUN_REMOTE_SETUP:-1}"

[ -n "${SSH_HOST}" ] || { echo "SSH_HOST is required, e.g. root@103.207.149.70" >&2; exit 2; }
[ -f "${SSH_KEY}" ] || { echo "SSH_KEY not found: ${SSH_KEY}" >&2; exit 2; }

ssh_args=(-o StrictHostKeyChecking=accept-new -i "${SSH_KEY}")
scp_args=(-o StrictHostKeyChecking=accept-new -i "${SSH_KEY}")
if [ -n "${SSH_PORT}" ]; then
  ssh_args=(-o StrictHostKeyChecking=accept-new -p "${SSH_PORT}" -i "${SSH_KEY}")
  scp_args=(-o StrictHostKeyChecking=accept-new -P "${SSH_PORT}" -i "${SSH_KEY}")
fi

log() { printf '[sync] %s\n' "$*"; }

log "preparing remote repo ${SSH_HOST}:${REPO_DIR}"
ssh "${ssh_args[@]}" "${SSH_HOST}" "set -euo pipefail
mkdir -p \"\$(dirname \"${REPO_DIR}\")\"
if [ ! -d \"${REPO_DIR}/.git\" ]; then
  if [ -e \"${REPO_DIR}\" ] && [ -n \"\$(find \"${REPO_DIR}\" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1 || true)\" ]; then
    mv \"${REPO_DIR}\" \"${REPO_DIR}.skeleton.\$(date +%s)\"
  fi
  git clone \"${REPO_URL}\" \"${REPO_DIR}\"
fi
mkdir -p \"${REPO_DIR}/tools\"
"

root_files=(
  train_gpt_frontier.py
  train_gpt_record1060.py
  run_experiments_r13_fast_tree.sh
  setup_runpod_r13.sh
  experiment_queue.json
  research_state.json
  program.md
  EXPERIMENT_TODO.md
  experiment_plan.md
)

tool_files=(
  tools/minify_train_script.py
  tools/autoresearch_loop.py
  tools/parse_experiment_log.py
)

log "uploading patched root files"
scp "${scp_args[@]}" "${root_files[@]}" "${SSH_HOST}:${REPO_DIR}/"
log "uploading tools"
scp "${scp_args[@]}" "${tool_files[@]}" "${SSH_HOST}:${REPO_DIR}/tools/"

log "remote compile/queue validation"
ssh "${ssh_args[@]}" "${SSH_HOST}" "set -euo pipefail
cd \"${REPO_DIR}\"
chmod +x setup_runpod_r13.sh run_experiments_r13_fast_tree.sh
python3 -m py_compile train_gpt_frontier.py tools/minify_train_script.py tools/autoresearch_loop.py tools/parse_experiment_log.py
python3 -m py_compile train_gpt_record1060.py
python3 -m json.tool experiment_queue.json >/tmp/experiment_queue.ok
python3 -m json.tool research_state.json >/tmp/research_state.ok
python3 tools/autoresearch_loop.py validate
"

if [ "${RUN_REMOTE_SETUP}" = "1" ]; then
  log "running remote setup/preflight"
  ssh "${ssh_args[@]}" "${SSH_HOST}" "set -euo pipefail
cd \"${REPO_DIR}\"
REPO_DIR=\"${REPO_DIR}\" ./setup_runpod_r13.sh
"
else
  log "skipping remote setup because RUN_REMOTE_SETUP=${RUN_REMOTE_SETUP}"
fi

log "done"
