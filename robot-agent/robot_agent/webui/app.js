const state = {
  tokenSecret: null,
  activeTaskId: null,
  visualPlan: null,
  events: [],
  capabilities: null,
  cameraPreviewVisible: false,
  cameraConfig: null,
  cameraMapping: null,
  runtimeState: null,
  recording: null,
  recordingDefaultsApplied: false,
  recordingRequestValidated: false,
};

const $ = (selector) => document.querySelector(selector);
const CAMERA_NAMES = ["head_cam", "left_cam", "right_cam"];
const CAMERA_LABELS = {
  head_cam: "头部",
  left_cam: "左臂",
  right_cam: "右臂",
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function callRobotd(command, params = {}) {
  const response = await requestJson("/api/command", {
    method: "POST",
    body: JSON.stringify({ command, params }),
  });
  rememberEvent(command, response);
  if (!response.success) {
    showToast(response.message || response.error || "操作被拒绝");
  }
  return response;
}

function rememberEvent(command, response) {
  state.events.unshift({ command, response, at: new Date() });
  state.events = state.events.slice(0, 20);
  renderEvents();
}

function displayResponse(response) {
  const clone = JSON.parse(JSON.stringify(response));
  redact(clone);
  return clone;
}

function redact(value) {
  if (Array.isArray(value)) {
    value.forEach(redact);
    return;
  }
  if (!value || typeof value !== "object") return;
  Object.keys(value).forEach((key) => {
    if (key === "token_secret") value[key] = "[held in browser memory]";
    else redact(value[key]);
  });
}

function renderEvents() {
  const container = $("#event-list");
  container.replaceChildren();
  if (!state.events.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "暂无操作记录";
    container.append(empty);
    return;
  }
  state.events.forEach((item) => {
    const entry = document.createElement("article");
    entry.className = `event-item ${item.response.success ? "success" : "failure"}`;
    const meta = document.createElement("div");
    meta.className = "event-meta";
    const command = document.createElement("span");
    command.textContent = item.command;
    const timestamp = document.createElement("time");
    timestamp.textContent = item.at.toLocaleTimeString();
    meta.append(command, timestamp);
    const message = document.createElement("p");
    message.className = "event-message";
    message.textContent = item.response.message || item.response.error || (item.response.success ? "完成" : "失败");
    entry.append(meta, message);
    container.append(entry);
  });
}

function showResponse(response) {
  $("#response-detail").textContent = JSON.stringify(displayResponse(response), null, 2);
  const event = state.events[0];
  if (!event) return;
  showToast(response.success ? "操作完成" : response.message || response.error || "操作被拒绝");
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("visible"), 2800);
}

function setConnection(online, message) {
  $("#connection-dot").className = `status-dot ${online ? "online" : "offline"}`;
  $("#connection-label").textContent = message;
}

function renderStatus(status) {
  state.runtimeState = status.state || null;
  $("#runtime-mode").textContent = status.state || "UNKNOWN";
  $("#status-state").textContent = status.state || "--";
  $("#status-task").textContent = status.active_task || "无";
  $("#status-executor").textContent = status.executor_active ? "运行中" : "空闲";
  const safety = status.safety_interlock || {};
  $("#status-safety").textContent = safety.required ? (safety.available ? "已接入" : "阻断") : "未启用";
  $("#status-plans").textContent = String(status.visual_plan_count ?? 0);
  $("#status-session").textContent = status.session_id ? status.session_id.slice(-12) : "--";
  $("#last-updated").textContent = `更新于 ${new Date().toLocaleTimeString()}`;
  updateRecordingControls();
}

async function refreshStatus() {
  try {
    const status = await requestJson("/api/status");
    renderStatus(status);
    setConnection(true, "robotd 已连接");
  } catch (error) {
    setConnection(false, "robotd 不可用");
    showToast(error.message);
  }
}

function renderTasks() {
  const container = $("#task-list");
  const tasks = state.capabilities?.task_catalog?.tasks || {};
  const entries = Object.entries(tasks).sort(([left], [right]) => left.localeCompare(right));
  $("#task-count").textContent = `${entries.length} 项`;
  container.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "未发现任务目录";
    container.append(empty);
    return;
  }
  entries.forEach(([taskId, task]) => {
    const row = document.createElement("article");
    row.className = "task-row";
    const copy = document.createElement("div");
    const name = document.createElement("div");
    name.className = "task-name";
    name.textContent = `${taskId} / ${task.name || ""}`;
    const description = document.createElement("div");
    description.className = "task-description";
    description.textContent = task.description || "";
    copy.append(name, description);
    const actions = document.createElement("div");
    actions.className = "task-actions";
    actions.append(
      makeButton("Dry-run", "button-secondary", () => taskDryRun(taskId)),
      makeButton("授权", "button-quiet", () => armTask(taskId)),
      makeButton("执行", "button-primary", () => executeTask(taskId)),
    );
    row.append(copy, actions);
    container.append(row);
  });
}

function makeButton(label, className, handler) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `button ${className}`;
  button.textContent = label;
  button.addEventListener("click", handler);
  return button;
}

function renderVisualPlan(plan) {
  state.visualPlan = plan;
  const panel = $("#visual-plan-panel");
  panel.hidden = false;
  $("#visual-plan-title").textContent = plan.goal || "视觉计划";
  $("#visual-plan-hash").textContent = plan.plan_sha256 ? plan.plan_sha256.slice(0, 16) : "无摘要";
  const steps = $("#visual-plan-steps");
  steps.replaceChildren();
  (plan.steps || []).forEach((step) => {
    const item = document.createElement("li");
    item.className = "plan-step";
    const copy = document.createElement("div");
    const task = document.createElement("strong");
    task.textContent = step.task_id;
    const reason = document.createElement("small");
    reason.textContent = step.reason;
    copy.append(task, reason);
    const confidence = document.createElement("span");
    confidence.className = "confidence";
    confidence.textContent = `${Math.round(Number(step.confidence) * 100)}%`;
    item.append(copy, confidence);
    steps.append(item);
  });
  const hasSteps = (plan.steps || []).length > 0;
  $("#visual-plan-arm").disabled = !hasSteps;
  $("#visual-plan-execute").disabled = !state.tokenSecret || !hasSteps;
}

async function loadCapabilities() {
  try {
    state.capabilities = await requestJson("/api/capabilities");
    renderTasks();
  } catch (error) {
    showToast(error.message);
  }
}

function renderCameraConfig(config) {
  state.cameraConfig = config;
  const cameras = config?.cameras || {};
  const configured = Object.values(cameras).filter((camera) => camera.configured).length;
  const status = $("#camera-status");
  status.textContent = config?.enabled ? "三路已连接" : `${configured}/3 路已配置`;
  Object.entries(cameras).forEach(([name, camera]) => {
    const health = $(`#camera-health-${name}`);
    if (health) health.textContent = camera.configured ? "已配置" : "未配置";
    const image = $(`#camera-image-${name}`);
    if (!image || !state.cameraPreviewVisible) return;
    image.src = camera.stream_url ? `${camera.stream_url}?ts=${Date.now()}` : "";
  });
}

async function loadCameraConfig() {
  try {
    renderCameraConfig(await requestJson("/api/cameras"));
  } catch (error) {
    $("#camera-status").textContent = "服务不可用";
    showToast(error.message);
  }
}

function renderCameraMapping(mapping) {
  state.cameraMapping = mapping || {};
  const cameras = state.cameraMapping.cameras || {};
  const options = $("#camera-device-options");
  options.replaceChildren();
  const available = state.cameraMapping.available_devices || [];
  available.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.device;
    options.append(option);
  });
  CAMERA_NAMES.forEach((name) => {
    const camera = cameras[name] || {};
    const input = $("#camera-map-" + name);
    if (input && document.activeElement !== input) input.value = camera.device || "";
    const status = $("#camera-map-status-" + name);
    if (!status) return;
    if (!camera.device) status.textContent = "未配置";
    else if (!camera.exists) status.textContent = camera.device + " 不存在";
    else if (!camera.consistent) status.textContent = "env / policy 不一致";
    else status.textContent = camera.device + " 已同步";
  });
  const consistentCount = CAMERA_NAMES.filter((name) => cameras[name]?.consistent).length;
  const existsCount = CAMERA_NAMES.filter((name) => cameras[name]?.exists).length;
  let label = "未配置";
  if (state.cameraMapping.configured && consistentCount === CAMERA_NAMES.length && existsCount === CAMERA_NAMES.length) {
    label = "三路已同步";
  } else if (state.cameraMapping.configured) {
    label = consistentCount + "/3 同步 · " + existsCount + "/3 可用";
  }
  $("#camera-mapping-availability").textContent = label;
  $("#camera-mapping-env").textContent = state.cameraMapping.env_file || "--";
  $("#camera-mapping-policy").textContent = state.cameraMapping.policy_registry_path || "--";
  updateCameraMappingControls();
}

async function loadCameraMapping(showErrors = false) {
  try {
    renderCameraMapping(await requestJson("/api/camera-mapping"));
  } catch (error) {
    $("#camera-mapping-availability").textContent = "读取失败";
    if (showErrors) showToast(error.message);
  }
}

function cameraMappingRequest() {
  const form = $("#camera-mapping-form");
  if (!form.reportValidity()) return null;
  const mapping = {};
  CAMERA_NAMES.forEach((name) => {
    mapping[name] = $("#camera-map-" + name).value.trim();
  });
  if (new Set(Object.values(mapping)).size !== CAMERA_NAMES.length) {
    showToast("三个摄像头端口不能重复");
    return null;
  }
  return mapping;
}

function updateCameraMappingControls() {
  const active = Boolean(state.recording?.active);
  $("#camera-mapping-form").querySelectorAll("input, button").forEach((control) => {
    control.disabled = active;
  });
  if (active) $("#camera-mapping-availability").textContent = "录制中锁定";
}

async function applyCameraMapping(event) {
  event.preventDefault();
  const mapping = cameraMappingRequest();
  if (!mapping) return;
  if (state.recording?.active) {
    showToast("数据集录制期间不能修改相机端口");
    return;
  }
  const summary = CAMERA_NAMES
    .map((name) => CAMERA_LABELS[name] + ": " + mapping[name])
    .join("\n");
  const restartServices = $("#camera-mapping-restart").checked;
  if (!window.confirm("确认保存相机端口映射？\n" + summary)) return;
  if (state.cameraPreviewVisible) closeCameraPreview();
  try {
    const response = await requestJson("/api/camera-mapping", {
      method: "POST",
      body: JSON.stringify({
        mapping,
        restart_services: restartServices,
        operator_confirmation: "CONFIRM CAMERA MAPPING",
      }),
    });
    rememberEvent("camera_mapping_apply", response);
    showResponse(response);
    renderCameraMapping(response.status || response);
    await loadCameraConfig();
    if (restartServices) showToast("映射已保存，服务已重启");
  } catch (error) {
    showToast(error.message);
  }
}

function closeCameraPreview() {
  state.cameraPreviewVisible = false;
  $("#camera-grid").hidden = true;
  $("#camera-toggle").textContent = "打开预览";
  Object.keys(state.cameraConfig?.cameras || {}).forEach((name) => {
    const image = $(`#camera-image-${name}`);
    if (image) image.removeAttribute("src");
  });
  $("#camera-status").textContent = "已关闭";
}

async function toggleCameraPreview() {
  if (state.recording?.active) {
    showToast("数据集录制期间相机由录制器占用");
    return;
  }
  if (state.cameraPreviewVisible) {
    closeCameraPreview();
    return;
  }
  state.cameraPreviewVisible = true;
  $("#camera-grid").hidden = false;
  $("#camera-toggle").textContent = "关闭预览";
  await loadCameraConfig();
  if (!state.cameraConfig?.enabled) {
    showToast("三路相机服务未就绪");
  }
}

function applyRecordingDefaults(recording) {
  const defaults = recording?.defaults || {};
  const limits = recording?.limits || {};
  if (!state.recordingDefaultsApplied && defaults.num_episodes) {
    $("#recording-episodes").value = defaults.num_episodes;
    $("#recording-episode-time").value = defaults.episode_time_s;
    $("#recording-reset-time").value = defaults.reset_time_s;
    state.recordingDefaultsApplied = true;
  }
  if (limits.num_episodes) $("#recording-episodes").max = limits.num_episodes;
  if (limits.episode_time_s) $("#recording-episode-time").max = limits.episode_time_s;
  if (limits.reset_time_s !== undefined) {
    $("#recording-reset-time").max = limits.reset_time_s;
  }
}

function updateRecordingControls() {
  const recording = state.recording || {};
  const active = Boolean(recording.active);
  const available = Boolean(recording.available);
  $("#recording-form").querySelectorAll("input").forEach((input) => {
    input.disabled = active || !available;
  });
  $("#recording-validate").disabled = active || !available;
  $("#recording-start").disabled =
    active
    || !available
    || !state.recordingRequestValidated
    || state.runtimeState !== "DRY_RUN";
  $("#recording-stop").disabled = !active;
  $("#camera-toggle").disabled = active;
  updateCameraMappingControls();
}

function renderRecordingStatus(recording) {
  const wasActive = Boolean(state.recording?.active);
  state.recording = recording || {};
  const active = Boolean(state.recording.active);
  applyRecordingDefaults(state.recording);

  $("#recording-state").textContent = String(state.recording.state || "UNKNOWN").toUpperCase();
  let availability = "未配置";
  if (state.recording.configured && !state.recording.enabled) availability = "配置已禁用";
  else if (state.recording.enabled && !state.recording.available) availability = "策略注册表不可用";
  else if (state.recording.available && state.runtimeState !== "DRY_RUN" && !active) {
    availability = "等待 DRY_RUN";
  } else if (state.recording.available) {
    if (active) availability = "录制中";
    else if (state.recordingRequestValidated) availability = "验证通过";
    else availability = "待验证";
  }
  $("#recording-availability").textContent = availability;

  const recorded = Number(state.recording.recorded_episodes || 0);
  const target = Number(state.recording.target_episodes || 0);
  const total = Number(state.recording.total_episodes || 0);
  const percent = Number(state.recording.progress_percent || 0);
  $("#recording-progress-text").textContent = `${recorded} / ${target} 新增回合 · 总计 ${total}`;
  $("#recording-progress-percent").textContent = `${percent.toFixed(1)}%`;
  $("#recording-progress").value = Math.max(0, Math.min(100, percent));
  $("#recording-repo").textContent = state.recording.repo_id || "--";
  $("#recording-frames").textContent = String(state.recording.total_frames || 0);
  $("#recording-session").textContent = state.recording.session_id
    ? state.recording.session_id.slice(-16)
    : "--";
  const datasetPath = state.recording.dataset_path || "--";
  $("#recording-path").textContent = datasetPath;
  $("#recording-path").title = datasetPath === "--" ? "" : datasetPath;
  const camera = state.recording.camera_service || {};
  let cameraLabel = "待机";
  if (camera.error) cameraLabel = "恢复失败";
  else if (camera.paused) cameraLabel = "录制占用";
  else if (camera.restored) cameraLabel = "已恢复";
  $("#recording-camera").textContent = cameraLabel;
  $("#recording-log").textContent = state.recording.log_tail || "暂无录制日志";

  if (active && state.cameraPreviewVisible) closeCameraPreview();
  if (active) $("#camera-status").textContent = "录制占用中";
  if (wasActive && !active) loadCameraConfig();
  updateRecordingControls();
}

async function loadRecordingStatus(showErrors = false) {
  try {
    const response = await requestJson("/api/recording");
    renderRecordingStatus(response.data || {});
  } catch (error) {
    $("#recording-availability").textContent = "状态不可用";
    if (showErrors) showToast(error.message);
  }
}

function recordingRequest() {
  const form = $("#recording-form");
  if (!form.reportValidity()) return null;
  return {
    dataset_name: $("#recording-dataset-name").value.trim(),
    task: $("#recording-task").value.trim(),
    num_episodes: Number($("#recording-episodes").value),
    episode_time_s: Number($("#recording-episode-time").value),
    reset_time_s: Number($("#recording-reset-time").value),
    resume: $("#recording-resume").checked,
  };
}

async function validateRecording() {
  const params = recordingRequest();
  if (!params) return;
  state.recordingRequestValidated = false;
  updateRecordingControls();
  try {
    const response = await callRobotd("dataset_recording_validate", params);
    state.recordingRequestValidated = Boolean(response.success);
    showResponse(response);
    renderRecordingStatus(state.recording);
  } catch (error) {
    showToast(error.message);
  }
}

async function startRecording(event) {
  event.preventDefault();
  const params = recordingRequest();
  if (!params) return;
  if (!state.recordingRequestValidated) {
    showToast("请先验证当前录制配置");
    return;
  }
  if (!window.confirm(`确认开始录制数据集 ${params.dataset_name}？`)) return;
  const previewWasVisible = state.cameraPreviewVisible;
  if (previewWasVisible) closeCameraPreview();
  state.recordingRequestValidated = false;
  updateRecordingControls();
  try {
    const response = await callRobotd("dataset_recording_start", {
      ...params,
      operator_confirmation: "CONFIRM RECORD DATASET",
    });
    showResponse(response);
    await loadRecordingStatus();
    await refreshStatus();
    if (!response.success && previewWasVisible) await toggleCameraPreview();
  } catch (error) {
    showToast(error.message);
    if (previewWasVisible) await toggleCameraPreview();
  }
}

async function stopRecording() {
  if (!window.confirm("确认停止当前数据集录制？")) return;
  try {
    const response = await callRobotd("dataset_recording_stop");
    showResponse(response);
    await loadRecordingStatus();
    await refreshStatus();
  } catch (error) {
    showToast(error.message);
  }
}

async function submitGoal(command) {
  const goal = $("#goal-input").value.trim();
  if (!goal) {
    showToast("请输入目标指令");
    return;
  }
  try {
    const response = await callRobotd(command, command === "observe" ? { instruction: goal } : { goal });
    if (response.success && response.data?.visual_plan) renderVisualPlan(response.data.visual_plan);
    showResponse(response);
  } catch (error) {
    showToast(error.message);
  }
}

async function taskDryRun(taskId) {
  try { showResponse(await callRobotd("dry_run", { goal: taskId })); }
  catch (error) { showToast(error.message); }
}

async function armTask(taskId) {
  if (!window.confirm(`确认授权任务 ${taskId}？`)) return;
  try {
    const response = await callRobotd("arm", { task_id: taskId, operator_confirmation: `CONFIRM ${taskId}` });
    if (response.success) {
      state.tokenSecret = response.data?.token_secret || null;
      state.activeTaskId = taskId;
      showToast("任务已授权，等待执行");
    }
    showResponse(response);
  } catch (error) { showToast(error.message); }
}

async function executeTask(taskId) {
  if (!state.tokenSecret || state.activeTaskId !== taskId) {
    showToast("请先授权同一任务");
    return;
  }
  if (!window.confirm(`确认执行任务 ${taskId}？`)) return;
  try {
    const response = await callRobotd("execute", { task_id: taskId, token: state.tokenSecret });
    state.tokenSecret = null;
    state.activeTaskId = null;
    showResponse(response);
    refreshStatus();
  } catch (error) { showToast(error.message); }
}

async function armVisualPlan() {
  if (!state.visualPlan?.plan_id || !window.confirm("确认授权当前视觉计划？")) return;
  const planId = state.visualPlan.plan_id;
  try {
    const response = await callRobotd("visual_plan_arm", {
      plan_id: planId,
      operator_confirmation: `CONFIRM plan:${planId}`,
    });
    if (response.success) {
      state.tokenSecret = response.data?.token_secret || null;
      state.activeTaskId = `plan:${planId}`;
      $("#visual-plan-execute").disabled = false;
      showToast("视觉计划已授权");
    }
    showResponse(response);
  } catch (error) { showToast(error.message); }
}

async function executeVisualPlan() {
  if (!state.visualPlan?.plan_id || !state.tokenSecret) {
    showToast("请先授权视觉计划");
    return;
  }
  if (!window.confirm("确认执行当前视觉计划？")) return;
  try {
    const response = await callRobotd("visual_plan_execute", {
      plan_id: state.visualPlan.plan_id,
      token: state.tokenSecret,
    });
    state.tokenSecret = null;
    state.activeTaskId = null;
    showResponse(response);
    refreshStatus();
  } catch (error) { showToast(error.message); }
}

async function safetyCommand(command, params = {}) {
  if (command === "estop" && !window.confirm("确认触发急停？")) return;
  if (command === "reset" && !window.confirm("确认复位 robotd？")) return;
  try { showResponse(await callRobotd(command, params)); refreshStatus(); }
  catch (error) { showToast(error.message); }
}

$("#goal-form").addEventListener("submit", (event) => { event.preventDefault(); submitGoal("plan"); });
$("#goal-dry-run").addEventListener("click", () => submitGoal("dry_run"));
$("#goal-observe").addEventListener("click", () => submitGoal("observe"));
$("#camera-toggle").addEventListener("click", toggleCameraPreview);
$("#camera-mapping-form").addEventListener("submit", applyCameraMapping);
$("#camera-mapping-refresh").addEventListener("click", () => loadCameraMapping(true));
$("#recording-form").addEventListener("submit", startRecording);
$("#recording-form").addEventListener("input", () => {
  state.recordingRequestValidated = false;
  renderRecordingStatus(state.recording);
});
$("#recording-validate").addEventListener("click", validateRecording);
$("#recording-stop").addEventListener("click", stopRecording);
$("#visual-plan-arm").addEventListener("click", armVisualPlan);
$("#visual-plan-execute").addEventListener("click", executeVisualPlan);
$("#refresh-button").addEventListener("click", () => { refreshStatus(); loadCapabilities(); loadCameraMapping(); });
$("#pause-button").addEventListener("click", () => safetyCommand("pause"));
$("#stop-button").addEventListener("click", () => safetyCommand("stop"));
$("#estop-button").addEventListener("click", () => safetyCommand("estop"));
$("#reset-button").addEventListener("click", () => safetyCommand("reset", { operator_confirmation: "RESET ESTOP" }));
$("#clear-events").addEventListener("click", () => { state.events = []; renderEvents(); });

loadCapabilities();
loadCameraConfig();
loadCameraMapping(true);
loadRecordingStatus(true);
refreshStatus();
window.setInterval(refreshStatus, 3000);
window.setInterval(() => loadRecordingStatus(false), 2000);
