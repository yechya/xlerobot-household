#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Install the software-only robot-agent user service.

Usage:
  install-software.sh [options]

Options:
  --prefix PATH       installation prefix (default: ~/.local/share/robot-agent)
  --config-dir PATH   user config directory (default: ~/.config/robot-agent)
  --state-dir PATH    state directory (default: ~/.local/state/robot-agent)
  --runtime-dir PATH  runtime directory (default: $XDG_RUNTIME_DIR/robot-agent)
  --python PATH       Python executable used to create the venv (default: python3)
  --with-camera       install and configure the three confirmed RGB camera service
  --enable             enable and start the systemd user service
  --no-service         do not install the systemd user unit
  --skip-install       only create deployment files (useful for CI validation)
  -h, --help           show this help
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${HOME}/.local/share/robot-agent"
config_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/robot-agent"
state_dir="${XDG_STATE_HOME:-${HOME}/.local/state}/robot-agent"
runtime_dir="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}/robot-agent-${UID}}/robot-agent"
python_bin="python3"
enable_service=0
install_service=1
skip_install=0
with_camera=0

while (($# > 0)); do
    case "$1" in
        --prefix)
            (($# >= 2)) || { echo "--prefix requires a path" >&2; exit 2; }
            prefix="$2"
            shift 2
            ;;
        --config-dir)
            (($# >= 2)) || { echo "--config-dir requires a path" >&2; exit 2; }
            config_dir="$2"
            shift 2
            ;;
        --state-dir)
            (($# >= 2)) || { echo "--state-dir requires a path" >&2; exit 2; }
            state_dir="$2"
            shift 2
            ;;
        --runtime-dir)
            (($# >= 2)) || { echo "--runtime-dir requires a path" >&2; exit 2; }
            runtime_dir="$2"
            shift 2
            ;;
        --python)
            (($# >= 2)) || { echo "--python requires a path" >&2; exit 2; }
            python_bin="$2"
            shift 2
            ;;
        --with-camera)
            with_camera=1
            shift
            ;;
        --enable)
            enable_service=1
            shift
            ;;
        --no-service)
            install_service=0
            shift
            ;;
        --skip-install)
            skip_install=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ((install_service == 0 && enable_service == 1)); then
    echo "--enable cannot be combined with --no-service" >&2
    exit 2
fi

if ((with_camera == 1 && install_service == 0)); then
    echo "--with-camera cannot be combined with --no-service" >&2
    exit 2
fi

if ((with_camera == 1)) && ! command -v ffmpeg >/dev/null 2>&1; then
    echo "--with-camera requires ffmpeg in PATH" >&2
    exit 1
fi

for path in "$prefix" "$config_dir" "$state_dir" "$runtime_dir"; do
    case "$path" in
        *[[:space:]]*)
            echo "paths containing whitespace are not supported: $path" >&2
            exit 2
            ;;
    esac
done

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin" >&2
    exit 1
fi

mkdir -p -m 700 "$config_dir" "$state_dir" "$runtime_dir"
chmod 700 "$config_dir" "$state_dir" "$runtime_dir"

venv="$prefix/.venv"
if ((skip_install == 0)); then
    mkdir -p "$prefix"
    if [[ ! -x "$venv/bin/python" ]]; then
        "$python_bin" -m venv "$venv"
    fi
    "$venv/bin/python" -m pip install "$repo_root"
else
    mkdir -p "$prefix"
fi

env_file="$config_dir/robot-agent.env"
if [[ ! -e "$env_file" ]]; then
    umask 077
    cat >"$env_file" <<EOF
ROBOT_AGENT_STATE_DIR=$state_dir
ROBOT_AGENT_RUNTIME_DIR=$runtime_dir
ROBOT_AGENT_SOCKET=$runtime_dir/robotd.sock
ROBOT_AGENT_LOG_LEVEL=INFO
ROBOT_AGENT_REQUIRE_OBSERVATION=false
ROBOT_AGENT_REQUIRE_POLICY_BENCHMARK=false
ROBOT_AGENT_REQUIRE_POLICY_EVIDENCE=false
ROBOT_AGENT_REQUIRE_SAFETY_INTERLOCK=false
ROBOT_AGENT_MOTION_BACKEND=disabled
ROBOT_AGENT_EMERGENCY_STOP_COMMAND=
ROBOT_AGENT_POLICY_REGISTRY=
ROBOT_AGENT_DATASET_RECORDING_CONFIG=
ROBOT_AGENT_TASK_CATALOG=
ROBOT_AGENT_MISSION_CATALOG=
ROBOT_AGENT_SEMANTIC_MAP_CONFIG=
ROBOT_AGENT_EPISODE_CONTEXT=
ROBOT_AGENT_BENCHMARK_REPORT=
ROBOT_AGENT_HEAD_CAMERA_SNAPSHOT_URL=
ROBOT_AGENT_LEFT_CAMERA_SNAPSHOT_URL=
ROBOT_AGENT_RIGHT_CAMERA_SNAPSHOT_URL=
VLM_API_TOKEN=
EOF
else
    echo "Keeping existing environment file: $env_file"
fi
chmod 600 "$env_file"

set_env_default() {
    local key="$1"
    local value="$2"
    if grep -q "^${key}=" "$env_file"; then
        if ! grep -q "^${key}=.+" "$env_file"; then
            sed -i "s|^${key}=.*$|${key}=${value}|" "$env_file"
        fi
    else
        printf '%s=%s\n' "$key" "$value" >>"$env_file"
    fi
}

if ((with_camera == 1)); then
    set_env_default ROBOT_AGENT_CAMERA_HOST 127.0.0.1
    set_env_default ROBOT_AGENT_CAMERA_PORT 8766
    set_env_default ROBOT_AGENT_CAMERA_FFMPEG ffmpeg
    set_env_default ROBOT_AGENT_CAMERA_WIDTH 640
    set_env_default ROBOT_AGENT_CAMERA_HEIGHT 480
    set_env_default ROBOT_AGENT_CAMERA_FPS 15
    set_env_default ROBOT_AGENT_CAMERA_INPUT_FORMAT mjpeg
    set_env_default ROBOT_AGENT_HEAD_CAMERA_DEVICE /dev/video2
    set_env_default ROBOT_AGENT_LEFT_CAMERA_DEVICE /dev/video4
    set_env_default ROBOT_AGENT_RIGHT_CAMERA_DEVICE /dev/video0
    set_env_default ROBOT_AGENT_HEAD_CAMERA_SNAPSHOT_URL \
        http://127.0.0.1:8766/cameras/head_cam/snapshot
    set_env_default ROBOT_AGENT_LEFT_CAMERA_SNAPSHOT_URL \
        http://127.0.0.1:8766/cameras/left_cam/snapshot
    set_env_default ROBOT_AGENT_RIGHT_CAMERA_SNAPSHOT_URL \
        http://127.0.0.1:8766/cameras/right_cam/snapshot
    set_env_default ROBOT_AGENT_HEAD_CAMERA_MJPEG_URL \
        http://127.0.0.1:8766/cameras/head_cam/stream
    set_env_default ROBOT_AGENT_LEFT_CAMERA_MJPEG_URL \
        http://127.0.0.1:8766/cameras/left_cam/stream
    set_env_default ROBOT_AGENT_RIGHT_CAMERA_MJPEG_URL \
        http://127.0.0.1:8766/cameras/right_cam/stream
fi

if ((install_service == 1)); then
    unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
    unit_path="$unit_dir/robotd-software.service"
    mkdir -p -m 700 "$unit_dir"
    sed \
        -e "s|@PREFIX@|$prefix|g" \
        -e "s|@CONFIG_DIR@|$config_dir|g" \
        -e "s|@STATE_DIR@|$state_dir|g" \
        -e "s|@RUNTIME_DIR@|$runtime_dir|g" \
        "$repo_root/deploy/robotd-software.service.in" >"$unit_path"
    chmod 600 "$unit_path"

    if ((with_camera == 1)); then
        camera_unit_path="$unit_dir/robot-camera.service"
        sed \
            -e "s|@PREFIX@|$prefix|g" \
            -e "s|@CONFIG_DIR@|$config_dir|g" \
            "$repo_root/deploy/robot-camera.service.in" >"$camera_unit_path"
        chmod 600 "$camera_unit_path"
    fi

    if command -v systemctl >/dev/null 2>&1; then
        if ! systemctl --user daemon-reload; then
            if ((enable_service == 1)); then
                echo "systemd user manager is unavailable; cannot enable the service" >&2
                exit 1
            fi
            echo "Unit installed; systemd user manager is unavailable" >&2
        elif ((enable_service == 1)); then
            if ((with_camera == 1)); then
                systemctl --user enable --now robot-camera.service
                systemctl --user restart robot-camera.service
                systemctl --user restart robotd-software.service
            else
                systemctl --user enable --now robotd-software.service
            fi
        fi
    elif ((enable_service == 1)); then
        echo "systemctl is unavailable; cannot enable the user service" >&2
        exit 1
    fi
fi

cat <<EOF
Software-only robot-agent deployment is ready.
  prefix:  $prefix
  config:  $env_file
  state:   $state_dir
  socket:  $runtime_dir/robotd.sock

Load the client environment in a shell:
  set -a; . "$env_file"; set +a
Then check it with:
  "$venv/bin/robotctl" health
  "$venv/bin/robotctl" capabilities
  "$venv/bin/robotctl" dry-run pick_up_bottle
EOF

if ((with_camera == 1)); then
    cat <<EOF
Camera service:
  unit:    robot-camera.service
  health:  curl http://127.0.0.1:8766/health
  preview: http://127.0.0.1:8766/cameras/head_cam/stream
EOF
fi
