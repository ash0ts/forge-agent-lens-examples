const dom = {
  connection: document.querySelector("#connection-status"),
  connectionLabel: document.querySelector(".connection-label"),
  lensLink: document.querySelector("#lens-link"),
  newChat: document.querySelector("#new-chat-button"),
  customerName: document.querySelector("#conversation-title"),
  customerAvatar: document.querySelector("#customer-avatar"),
  customerId: document.querySelector("#customer-id"),
  agentRoute: document.querySelector("#agent-route"),
  modelSelect: document.querySelector("#model-select"),
  welcome: document.querySelector("#welcome"),
  welcomeCopy: document.querySelector(".welcome-copy"),
  suggestions: document.querySelector("#suggestions"),
  messages: document.querySelector("#messages"),
  messageScroll: document.querySelector("#message-scroll"),
  typing: document.querySelector("#typing-indicator"),
  composer: document.querySelector("#composer"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  conversationCode: document.querySelector("#conversation-code"),
  traceRail: document.querySelector("#trace-rail"),
  traceToggle: document.querySelector("#trace-toggle"),
  traceState: document.querySelector("#trace-state"),
  traceSummary: document.querySelector("#mobile-trace-summary"),
  receiptEmpty: document.querySelector("#receipt-empty"),
  receiptList: document.querySelector("#receipt-list"),
  toolCount: document.querySelector("#tool-count"),
  toolTime: document.querySelector("#tool-time"),
  turnCount: document.querySelector("#turn-count"),
  telemetryFooter: document.querySelector("#telemetry-footer"),
  toasts: document.querySelector("#toast-region"),
};

const state = {
  config: null,
  conversationId: null,
  greeting: "",
  selectedRouteId: null,
  route: null,
  profile: null,
  jev: null,
  busy: false,
  userTurns: 0,
  toolEvents: [],
  totalToolMs: 0,
  receiptSequence: 0,
  jevRoutingShown: false,
};

const fallbackSuggestions = [
  "My refund for order NS-48291 still hasn’t arrived. Can you check what happened?",
  "The espresso cups in order NS-19064 arrived cracked. What can we do?",
  "Can I change the delivery address for order NS-73108 before it ships?",
];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function appendInlineMarkdown(parent, text) {
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let cursor = 0;
  for (const match of String(text).matchAll(pattern)) {
    if (match.index > cursor) {
      parent.append(document.createTextNode(text.slice(cursor, match.index)));
    }
    const token = match[0];
    const node = token.startsWith("**")
      ? el("strong", "", token.slice(2, -2))
      : el("code", "", token.slice(1, -1));
    parent.append(node);
    cursor = match.index + token.length;
  }
  if (cursor < text.length) {
    parent.append(document.createTextNode(text.slice(cursor)));
  }
}

function messageContent(role, content) {
  const container = el("div", "message-content");
  if (role === "user") {
    container.textContent = String(content);
    return container;
  }

  let list = null;
  for (const rawLine of String(content).split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      list = null;
      continue;
    }
    const listMatch = line.match(/^[-*]\s+(.+)$/);
    if (listMatch) {
      if (!list) {
        list = el("ul", "message-list");
        container.append(list);
      }
      const item = el("li");
      appendInlineMarkdown(item, listMatch[1]);
      list.append(item);
      continue;
    }
    list = null;
    const paragraph = el("p");
    appendInlineMarkdown(paragraph, line);
    container.append(paragraph);
  }
  return container;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const detail = typeof body === "object" ? body.detail || body.error : body;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return body;
}

function setConnection(status, label) {
  dom.connection.dataset.state = status;
  dom.connectionLabel.textContent = label;
  dom.connection.title = label;
}

function initials(name) {
  return String(name || "Customer")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "CU";
}

function routeIdentity(routeLike) {
  if (!routeLike) return null;
  if (typeof routeLike === "string") {
    return state.config?.routes?.find((route) => route.id === routeLike) || { id: routeLike, label: routeLike };
  }
  return routeLike;
}

function profileIdentity(profileLike) {
  if (!profileLike) return null;
  if (typeof profileLike === "string") return { id: profileLike, label: profileLike };
  const nested = profileLike.agent_profile || profileLike.agent || null;
  if (nested) return profileIdentity(nested);
  return {
    id: profileLike.id || profileLike.agent_id || "auto",
    label: profileLike.label || profileLike.name || profileLike.agent_label || profileLike.id || "Auto-route",
    role: profileLike.role || profileLike.description || "",
  };
}

function activeRoute() {
  return routeIdentity(state.route)
    || state.config?.routes?.find((route) => route.id === state.selectedRouteId)
    || null;
}

function updateAgentRoute(profileLike = state.profile, routingLike = state.jev?.routing) {
  const profile = profileIdentity(profileLike);
  const label = profile?.label || "Auto-route";
  dom.agentRoute.replaceChildren();
  const mark = el("span", "route-mark");
  mark.setAttribute("aria-hidden", "true");
  dom.agentRoute.append(mark, document.createTextNode(label));
  const routing = routingLike && routingLike.mode !== "off" ? routingLike : null;
  if (routing) {
    const mode = String(routing.mode || "shadow").toUpperCase();
    const source = routing.source === "mock"
      ? "MOCK JEV"
      : routing.source === "jev"
        ? "JEV"
        : "RULES";
    const badge = el("span", "route-source", `${source} · ${mode}`);
    dom.agentRoute.append(badge);
  }
  dom.agentRoute.title = [
    profile?.role || label,
    routing?.confidence !== undefined && routing?.confidence !== null
      ? `${routing.backend === "mock" ? "Mock Jev" : "Jev"} confidence ${Number(routing.confidence).toFixed(2)}`
      : null,
  ].filter(Boolean).join(" · ");
}

function applyCustomer(customer = {}) {
  const fallback = state.config?.demo_account || {};
  const merged = { ...fallback, ...customer };
  const name = merged.name || "Avery Arbor";
  dom.customerName.textContent = name;
  dom.customerAvatar.textContent = initials(name);
  dom.customerId.textContent = merged.email || merged.user_id || "Demo account";
  dom.customerId.title = [merged.email, merged.user_id].filter(Boolean).join(" · ");
}

function routeLabel(route) {
  const provider = String(route.provider || "provider");
  const model = String(route.model || route.label || route.id || "model");
  const shortModel = model.includes("/") ? model.split("/").at(-1) : model;
  return route.label || `${provider} · ${shortModel}`;
}

function applyConfig(config) {
  state.config = config || {};
  applyCustomer(config.demo_account);

  const routes = Array.isArray(config.routes) ? config.routes : [];
  dom.modelSelect.replaceChildren();
  if (!routes.length) {
    const option = el("option", "", "No model routes available");
    option.disabled = true;
    dom.modelSelect.append(option);
    dom.modelSelect.disabled = true;
  } else {
    routes.forEach((route) => {
      const option = el("option", "", routeLabel(route));
      option.value = route.id;
      option.disabled = route.available === false;
      if (route.available === false) option.textContent += " · unavailable";
      dom.modelSelect.append(option);
    });
    const configuredDefault = config.defaults?.route_id || config.defaults?.model_route_id;
    const firstAvailable = routes.find((route) => route.available !== false)?.id;
    state.selectedRouteId = routes.some((route) => route.id === configuredDefault && route.available !== false)
      ? configuredDefault
      : firstAvailable || routes[0].id;
    dom.modelSelect.value = state.selectedRouteId;
    dom.modelSelect.disabled = !firstAvailable;
  }

  const lens = config.lens || {};
  if (lens.enabled && lens.url) {
    dom.lensLink.href = lens.url;
    dom.lensLink.hidden = false;
  } else {
    dom.lensLink.hidden = true;
  }
  updateTelemetry({ enabled: Boolean(lens.enabled), project: lens.project, url: lens.url, flushed: false });
  renderSuggestions(config.suggestions);
}

function renderSuggestions(suggestions) {
  const values = Array.isArray(suggestions) && suggestions.length ? suggestions : fallbackSuggestions;
  dom.suggestions.replaceChildren();
  values.slice(0, 3).forEach((suggestion) => {
    const button = el("button", "suggestion", suggestion);
    button.type = "button";
    button.addEventListener("click", () => {
      dom.input.value = suggestion;
      resizeComposer();
      dom.input.focus();
      updateSendState();
    });
    dom.suggestions.append(button);
  });
}

async function boot() {
  setConnection("checking", "Connecting");
  try {
    const [health, config] = await Promise.all([
      api("/api/health"),
      api("/api/config"),
    ]);
    applyConfig(config);
    const healthy = health?.ok !== false && health?.status !== "error";
    setConnection(healthy ? "online" : "offline", healthy ? "Connected" : "Service unavailable");
    await startConversation({ removeExisting: false });
  } catch (error) {
    setConnection("offline", "Backend unavailable");
    dom.send.disabled = true;
    dom.input.disabled = true;
    dom.welcomeCopy.textContent = "Dispatch Desk could not reach its backend. Start the demo service, then reload this page.";
    toast(error.message || "Could not connect to Dispatch Desk", "error", 7000);
  }
}

async function startConversation({ removeExisting = true } = {}) {
  if (state.busy) return;
  state.busy = true;
  setTraceState("recording", "Starting");
  dom.send.disabled = true;
  dom.input.disabled = true;
  const previousId = state.conversationId;
  try {
    if (removeExisting && previousId) {
      await api(`/api/conversations/${encodeURIComponent(previousId)}`, { method: "DELETE" }).catch(() => null);
    }
    const body = state.selectedRouteId ? { route_id: state.selectedRouteId } : {};
    const conversation = await api("/api/conversations", {
      method: "POST",
      body: JSON.stringify(body),
    });
    resetLocalConversation();
    state.conversationId = conversation.id || conversation.conversation_id;
    state.greeting = conversation.greeting || "Tell us what happened and we’ll check the account.";
    state.route = conversation.route || routeIdentity(state.selectedRouteId);
    state.profile = conversation.profile || null;
    state.jev = conversation.jev || null;
    applyCustomer(conversation.customer);
    updateAgentRoute();
    dom.welcomeCopy.textContent = state.greeting;
    dom.conversationCode.textContent = state.conversationId
      ? `Conversation ${shortId(state.conversationId)}`
      : "Conversation ready";
    dom.input.disabled = false;
    setTraceState("idle", "Idle");
    updateSendState();
    dom.input.focus();
  } catch (error) {
    setTraceState("error", "Unavailable");
    dom.input.disabled = true;
    toast(error.message || "Could not start a conversation", "error", 6000);
  } finally {
    state.busy = false;
    updateSendState();
  }
}

function resetLocalConversation() {
  state.conversationId = null;
  state.greeting = "";
  state.route = null;
  state.profile = null;
  state.jev = null;
  state.userTurns = 0;
  state.toolEvents = [];
  state.totalToolMs = 0;
  state.receiptSequence = 0;
  state.jevRoutingShown = false;
  dom.messages.replaceChildren();
  dom.receiptList.replaceChildren();
  dom.receiptEmpty.hidden = false;
  dom.welcome.hidden = false;
  dom.input.value = "";
  resizeComposer();
  updateReceiptSummary();
  updateAgentRoute();
}

function setBusy(busy) {
  state.busy = busy;
  dom.typing.hidden = !busy;
  dom.input.disabled = busy;
  dom.modelSelect.disabled = busy || !state.config?.routes?.some((route) => route.available !== false);
  setTraceState(busy ? "recording" : "idle", busy ? "Recording" : "Recorded");
  updateSendState();
  if (busy) scrollMessages();
}

async function sendMessage(content) {
  const clean = String(content || "").trim();
  if (!clean || state.busy || !state.conversationId) return;

  dom.welcome.hidden = true;
  appendMessage({ role: "user", content: clean });
  dom.input.value = "";
  resizeComposer();
  state.userTurns += 1;
  updateReceiptSummary();
  setBusy(true);
  let turnFailed = false;

  try {
    const result = await api(`/api/conversations/${encodeURIComponent(state.conversationId)}/messages`, {
      method: "POST",
      body: JSON.stringify({ content: clean }),
    });
    state.profile = result.profile || state.profile;
    state.route = result.route || state.route;
    state.jev = result.jev || state.jev;
    updateAgentRoute(state.profile, state.jev?.routing);
    const assistant = typeof result.assistant === "string"
      ? { role: "assistant", content: result.assistant }
      : result.assistant || { role: "assistant", content: result.content || "" };
    appendMessage({
      role: assistant.role || "assistant",
      content: assistant.content || "The support agent completed the turn without a text response.",
      evidenceCount: Array.isArray(result.tool_events) ? result.tool_events.length : 0,
    });
    addReceiptRun(result);
    updateTelemetry(result.telemetry || {});
    setConnection("online", "Connected");
  } catch (error) {
    turnFailed = true;
    appendMessage({
      role: "assistant",
      content: `This turn could not be completed: ${error.message || "the support service was unavailable"}.`,
      error: true,
    });
    setConnection("offline", "Turn failed");
    toast(error.message || "The support turn failed", "error", 6000);
  } finally {
    setBusy(false);
    if (turnFailed) setTraceState("error", "Turn failed");
    dom.input.focus();
  }
}

function appendMessage({ role, content, evidenceCount = 0, error = false }) {
  const normalizedRole = role === "user" ? "user" : "assistant";
  const item = el("article", "message");
  item.dataset.role = normalizedRole;
  if (error) item.dataset.error = "true";

  if (normalizedRole === "assistant") {
    const avatar = el("span", "message-avatar", "N");
    avatar.setAttribute("aria-hidden", "true");
    item.append(avatar);
  }

  const bubble = el("div", "message-bubble");
  bubble.append(messageContent(normalizedRole, content));
  const meta = el("div", "message-meta");
  meta.append(el("span", "", normalizedRole === "user" ? "Customer" : "Northstar Care"));
  meta.append(el("span", "", formatClock(new Date())));
  if (normalizedRole === "assistant" && evidenceCount > 0) {
    const button = el("button", "evidence-link", `${evidenceCount} tool ${evidenceCount === 1 ? "call" : "calls"}`);
    button.type = "button";
    button.addEventListener("click", openEvidence);
    meta.append(button);
  }
  bubble.append(meta);
  item.append(bubble);
  dom.messages.append(item);
  scrollMessages();
}

function addReceiptRun(result) {
  const events = Array.isArray(result.tool_events) ? result.tool_events : [];
  const usage = result.usage || {};
  const route = routeIdentity(result.route) || activeRoute() || {};
  const profile = profileIdentity(result.profile) || profileIdentity(state.profile);
  const turn = state.userTurns;
  dom.receiptEmpty.hidden = true;

  const divider = el("li", "turn-divider", `Turn ${turn} · ${profile?.label || "Support agent"}`);
  dom.receiptList.append(divider);

  const jev = result.jev || {};
  const routing = jev.routing;
  if (routing && routing.mode !== "off" && !state.jevRoutingShown) {
    const selected = humanize(routing.selected_profile_id || "No route selected");
    const applied = humanize(routing.applied_profile_id || "Deterministic route");
    const confidence = Number(routing.confidence);
    const usedJev = routing.source === "jev" || routing.source === "mock";
    const isMock = routing.backend === "mock" || routing.source === "mock";
    const shadow = routing.mode === "shadow";
    const statusLabel = shadow
      ? `Shadow suggestion: ${selected}`
      : usedJev
        ? `Applied: ${selected}`
        : `Fallback applied: ${applied}`;
    dom.receiptList.append(receiptCard({
      kind: "decision",
      label: isMock ? "Mock Jev routing decision" : "Jev routing decision",
      subtitle: routing.returned_model || routing.requested_model || (isMock ? "Local decision mock" : "TypeSafe System One"),
      durationMs: routing.duration_ms,
      status: routing.fallback_reason ? "warning" : "ok",
      statusLabel: Number.isFinite(confidence)
        ? `${statusLabel} · ${confidence.toFixed(2)} confidence`
        : statusLabel,
      details: {
        mode: routing.mode,
        backend: routing.backend,
        source: routing.source,
        selected_agent: routing.selected_profile_id,
        applied_agent: routing.applied_profile_id,
        confidence: routing.confidence,
        probabilities: routing.probabilities,
        fallback_reason: routing.fallback_reason,
      },
    }));
    state.jevRoutingShown = true;
  }

  const assessment = jev.latest_assessment;
  if (assessment) {
    const isMock = assessment.backend === "mock" || assessment.source === "mock";
    const probabilities = assessment.probabilities || {};
    const strongest = Object.entries(probabilities)
      .filter(([, value]) => Number.isFinite(Number(value)))
      .sort((left, right) => Number(right[1]) - Number(left[1]))[0];
    const statusLabel = strongest
      ? `Highest signal · ${humanize(strongest[0])}: ${(Number(strongest[1]) * 100).toFixed(0)}%`
      : assessment.fallback_reason || "Assessment unavailable";
    dom.receiptList.append(receiptCard({
      kind: "decision",
      label: isMock ? "Mock Jev behavior check" : "Jev behavior check",
      subtitle: assessment.returned_model || assessment.requested_model || (isMock ? "Local decision mock" : "TypeSafe System One"),
      durationMs: assessment.duration_ms,
      status: ["jev", "mock"].includes(assessment.source) ? "ok" : "warning",
      statusLabel,
      details: {
        redundant_confirmation: probabilities.redundant_confirmation,
        partial_completion: probabilities.partial_completion,
        unsupported_definitive_answer: probabilities.unsupported_definitive_answer,
        backend: assessment.backend,
        source: assessment.source,
        fallback_reason: assessment.fallback_reason,
      },
    }));
  }

  const modelDetails = [
    usage.total_tokens ? `${Number(usage.total_tokens).toLocaleString()} tokens` : null,
    Number.isFinite(Number(usage.observed_cost_usd)) ? formatMoney(usage.observed_cost_usd) : null,
  ].filter(Boolean).join(" · ") || "Completed model response";
  dom.receiptList.append(receiptCard({
    kind: "model",
    label: route.model || route.label || "Model response",
    subtitle: route.provider || "inference",
    status: "ok",
    statusLabel: modelDetails,
    details: usage,
  }));

  events.forEach((event) => {
    const duration = Number(event.duration_ms || 0);
    state.toolEvents.push(event);
    state.totalToolMs += Number.isFinite(duration) ? duration : 0;
    const status = event.status === "error" || event.error_type ? "error" : "ok";
    dom.receiptList.append(receiptCard({
      kind: "tool",
      label: event.label || humanize(event.name || "Tool call"),
      subtitle: event.name || "mcp.tool",
      durationMs: duration,
      status,
      statusLabel: status === "error" ? humanize(event.error_type || "Tool error") : "Returned evidence",
      details: {
        arguments: event.arguments,
        result: event.result_preview ?? event.result,
      },
    }));
  });

  updateReceiptSummary();
  dom.traceSummary.textContent = `${state.toolEvents.length} tools · ${formatDuration(state.totalToolMs)}`;
  requestAnimationFrame(() => {
    const body = document.querySelector("#trace-body");
    body.scrollTo({ top: body.scrollHeight, behavior: "smooth" });
  });
}

function receiptCard({ kind, label, subtitle, durationMs, status, statusLabel, details }) {
  state.receiptSequence += 1;
  const item = el("li", "receipt-run");
  item.style.animationDelay = `${Math.min(state.receiptSequence * 35, 220)}ms`;
  const card = el("article", "receipt-card");
  card.dataset.kind = kind;
  card.dataset.status = status;

  const heading = el("div", "receipt-heading");
  const title = el("div", "receipt-title");
  title.append(el("span", "", label));
  title.append(el("small", "", subtitle));
  heading.append(title);
  if (durationMs !== undefined && Number.isFinite(Number(durationMs))) {
    heading.append(el("span", "receipt-duration", formatDuration(durationMs)));
  }
  card.append(heading);

  const statusLine = el("div", "receipt-status-line");
  statusLine.append(el("span", "receipt-status-dot"));
  statusLine.append(el("span", "", statusLabel));
  card.append(statusLine);

  const detailEntries = Object.entries(details || {})
    .filter(([, value]) => value !== undefined && value !== null && value !== "");
  if (detailEntries.length) {
    const disclosure = el("details", "receipt-details");
    disclosure.append(el("summary", "", "Inspect receipt"));
    const grid = el("div", "receipt-detail-grid");
    detailEntries.forEach(([key, value]) => {
      const row = el("div");
      row.append(el("span", "", humanize(key)));
      const code = el("code", "", safePreview(value));
      row.append(code);
      grid.append(row);
    });
    disclosure.append(grid);
    card.append(disclosure);
  }
  item.append(card);
  return item;
}

function updateReceiptSummary() {
  dom.toolCount.textContent = String(state.toolEvents.length);
  dom.toolTime.textContent = formatDuration(state.totalToolMs);
  dom.turnCount.textContent = String(state.userTurns);
}

function updateTelemetry(telemetry = {}) {
  const configured = state.config?.lens || {};
  const enabled = telemetry.enabled ?? configured.enabled ?? false;
  const project = telemetry.project || configured.project || "No project configured";
  const link = telemetry.url || configured.url;
  const strong = dom.telemetryFooter.querySelector("strong");
  const small = dom.telemetryFooter.querySelector("small");
  strong.textContent = enabled
    ? telemetry.flushed ? "Sent to Agent Lens" : "Agent Lens connected"
    : "Telemetry is off";
  small.textContent = project;
  if (link) {
    dom.lensLink.href = link;
    dom.lensLink.hidden = false;
  }
}

function setTraceState(stateName, label) {
  dom.traceState.dataset.state = stateName;
  dom.traceState.textContent = label;
  if (stateName === "recording") {
    dom.traceSummary.textContent = "Recording this turn";
  } else if (stateName === "error") {
    dom.traceSummary.textContent = "Turn needs attention";
  } else if (state.toolEvents.length) {
    dom.traceSummary.textContent = `${state.toolEvents.length} tools · ${formatDuration(state.totalToolMs)}`;
  } else {
    dom.traceSummary.textContent = "Waiting for a message";
  }
}

function openEvidence() {
  dom.traceRail.dataset.mobileOpen = "true";
  dom.traceToggle.setAttribute("aria-expanded", "true");
  const last = dom.receiptList.lastElementChild;
  last?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function toggleEvidence() {
  const open = dom.traceRail.dataset.mobileOpen !== "true";
  dom.traceRail.dataset.mobileOpen = String(open);
  dom.traceToggle.setAttribute("aria-expanded", String(open));
}

function resizeComposer() {
  dom.input.style.height = "auto";
  dom.input.style.height = `${Math.min(dom.input.scrollHeight, 144)}px`;
}

function updateSendState() {
  dom.send.disabled = state.busy || !state.conversationId || !dom.input.value.trim();
}

function scrollMessages() {
  requestAnimationFrame(() => {
    dom.messageScroll.scrollTo({ top: dom.messageScroll.scrollHeight, behavior: "smooth" });
  });
}

function toast(message, tone = "default", duration = 4000) {
  const item = el("div", "toast", message);
  item.dataset.tone = tone;
  dom.toasts.append(item);
  window.setTimeout(() => item.remove(), duration);
}

function shortId(value) {
  const text = String(value || "");
  return text.length > 12 ? `${text.slice(0, 8)}…${text.slice(-4)}` : text;
}

function humanize(value) {
  return String(value || "")
    .replace(/[_./-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function safePreview(value) {
  if (typeof value === "string") return value.slice(0, 1200);
  try {
    return JSON.stringify(value, null, 2).slice(0, 1200);
  } catch {
    return String(value).slice(0, 1200);
  }
}

function formatDuration(value) {
  const ms = Number(value || 0);
  if (!Number.isFinite(ms) || ms <= 0) return "0 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 10000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.round(ms / 1000)} s`;
}

function formatMoney(value) {
  const amount = Number(value);
  if (!Number.isFinite(amount)) return "";
  if (amount < 0.01) return `$${amount.toFixed(4)}`;
  return `$${amount.toFixed(2)}`;
}

function formatClock(date) {
  return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date);
}

dom.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(dom.input.value);
});

dom.input.addEventListener("input", () => {
  resizeComposer();
  updateSendState();
});

dom.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    dom.composer.requestSubmit();
  }
});

dom.newChat.addEventListener("click", () => startConversation({ removeExisting: true }));

dom.modelSelect.addEventListener("change", async () => {
  state.selectedRouteId = dom.modelSelect.value;
  const selected = state.config?.routes?.find((route) => route.id === state.selectedRouteId);
  await startConversation({ removeExisting: true });
  toast(`New chat started with ${routeLabel(selected || {})}`);
});

dom.traceToggle.addEventListener("click", toggleEvidence);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && dom.traceRail.dataset.mobileOpen === "true") {
    dom.traceRail.dataset.mobileOpen = "false";
    dom.traceToggle.setAttribute("aria-expanded", "false");
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    dom.input.focus();
  }
});

window.addEventListener("online", () => setConnection("online", "Connected"));
window.addEventListener("offline", () => setConnection("offline", "Browser offline"));

boot();
