"use strict";

/* DocuCite AI front end.
 * All text from the server (document names, passages, model answers) is inserted with
 * textContent / createTextNode, never innerHTML: it is untrusted and must not run as markup. */

const $ = (selector) => document.querySelector(selector);

let currentUser = null;

/* ---------- helpers ---------- */

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

function toast(message, isError = false) {
  const node = el("div", { class: "toast" + (isError ? " error" : ""), text: message });
  $("#toasts").append(node);
  setTimeout(() => node.remove(), isError ? 9000 : 5000);
}

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  if (response.status === 401 && path !== "/auth/login") {
    showLogin();
    throw new Error("Please sign in again.");
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
    } catch { /* body was not JSON */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function jsonRequest(method, body) {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

function formatDate(iso) {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/* ---------- authentication ---------- */

function showLogin() {
  currentUser = null;
  $("#app-view").hidden = true;
  $("#login-view").hidden = false;
  $("#login-password").value = "";
}

function showApp(user) {
  currentUser = user;
  $("#login-view").hidden = true;
  $("#app-view").hidden = false;
  $("#user-name").textContent = user.username;
  $("#user-role").textContent = user.role === "admin" ? "Administrator" : "User";
  $("#user-avatar").textContent = user.username.slice(0, 1);
  const isAdmin = user.role === "admin";
  document.querySelectorAll(".admin-only").forEach((node) => { node.hidden = !isAdmin; });
  switchView("ask");
}

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const errorBox = $("#login-error");
  errorBox.hidden = true;
  try {
    const user = await api("/auth/login", jsonRequest("POST", {
      username: $("#login-username").value.trim(),
      password: $("#login-password").value,
    }));
    showApp(user);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  }
});

$("#logout").addEventListener("click", async () => {
  await fetch("/auth/logout", { method: "POST", credentials: "same-origin" });
  $("#messages").querySelectorAll(".msg").forEach((node) => node.remove());
  $("#empty-state").hidden = false;
  showLogin();
});

/* ---------- navigation ---------- */

const loaders = { documents: loadDocuments, log: loadLog, users: loadUsers };

function switchView(name) {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((view) => { view.hidden = view.id !== `view-${name}`; });
  if (loaders[name]) loaders[name]().catch((error) => toast(error.message, true));
  if (name === "ask") {
    $("#question").focus();
    refreshAskHints();
  }
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});

/* ---------- chat ---------- */

const messages = $("#messages");
const questionBox = $("#question");

function scrollToBottom() { messages.scrollTop = messages.scrollHeight; }

function addUserMessage(text) {
  $("#empty-state").hidden = true;
  messages.append(el("div", { class: "msg user" }, el("div", { class: "bubble", text })));
  scrollToBottom();
}

function addTyping() {
  const node = el("div", { class: "msg assistant" },
    el("div", { class: "bubble" }, el("span", { class: "typing" }, el("i"), el("i"), el("i"))));
  messages.append(node);
  scrollToBottom();
  return node;
}

/* Inline formatting: **bold** and [n] citation chips, built from DOM nodes (never innerHTML). */
function renderInline(text, sourcesNode) {
  const fragment = document.createDocumentFragment();
  text.split(/(\*\*[^*\n]+\*\*|\[\d+\])/).forEach((part) => {
    const bold = part.match(/^\*\*([^*\n]+)\*\*$/);
    const cite = part.match(/^\[(\d+)\]$/);
    if (bold) {
      fragment.append(el("strong", { text: bold[1] }));
    } else if (cite && !sourcesNode) {
      fragment.append(el("span", { class: "chip static", text: cite[1] }));
    } else if (cite) {
      const number = cite[1];
      fragment.append(el("button", {
        class: "chip", type: "button", title: `Show source ${number}`, text: number,
        onclick: () => {
          const target = sourcesNode.querySelector(`[data-n="${number}"]`);
          if (!target) return;
          target.open = true;
          target.scrollIntoView({ behavior: "smooth", block: "nearest" });
          target.classList.remove("flash");
          void target.offsetWidth;
          target.classList.add("flash");
        },
      }));
    } else if (part) {
      fragment.append(document.createTextNode(part));
    }
  });
  return fragment;
}

/* Block formatting: paragraphs and "* item" / "- item" bullet lists. */
function renderAnswerText(text, sourcesNode) {
  const container = document.createDocumentFragment();
  let list = null;
  text.split("\n").forEach((line) => {
    const item = line.match(/^\s*[*-]\s+(.*)$/);
    if (item) {
      if (!list) { list = el("ul"); container.append(list); }
      list.append(el("li", {}, renderInline(item[1], sourcesNode)));
      return;
    }
    list = null;
    if (line.trim()) container.append(el("p", {}, renderInline(line.trim(), sourcesNode)));
  });
  return container;
}
function renderSources(citations) {
  const wrapper = el("div", { class: "sources" }, el("div", { class: "sources-title", text: "Sources" }));
  citations.forEach((citation) => {
    wrapper.append(el("details", { class: "source", "data-n": String(citation.number) },
      el("summary", {},
        el("span", { class: "source-num", text: String(citation.number) }),
        el("span", { class: "source-name", text: citation.filename }),
        el("span", { class: "source-page", text: `page ${citation.page_number}` })),
      el("blockquote", { text: citation.passage })));
  });
  return wrapper;
}

function renderAssistant(result) {
  const sources = renderSources(result.citations);
  const bubble = el("div", { class: "bubble" + (result.found ? " rich" : "") });
  bubble.append(result.found ? renderAnswerText(result.answer, sources) : document.createTextNode(result.answer));
  const node = el("div", { class: "msg assistant" + (result.found ? "" : " notfound") }, bubble);
  if (result.found && result.citations.length) node.append(sources);
  return node;
}

async function ask(question) {
  addUserMessage(question);
  const typing = addTyping();
  $("#send").disabled = true;
  try {
    const result = await api("/ask", jsonRequest("POST", { question }));
    typing.replaceWith(renderAssistant(result));
  } catch (error) {
    typing.replaceWith(el("div", { class: "msg assistant failed" },
      el("div", { class: "bubble", text: `Sorry, something went wrong: ${error.message}` })));
  } finally {
    $("#send").disabled = false;
    scrollToBottom();
    questionBox.focus();
  }
}

$("#ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionBox.value.trim();
  if (question.length < 3) return;
  questionBox.value = "";
  autoGrow();
  ask(question);
});

questionBox.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#ask-form").requestSubmit();
  }
});

function autoGrow() {
  questionBox.style.height = "auto";
  questionBox.style.height = `${Math.min(questionBox.scrollHeight, 160)}px`;
}
questionBox.addEventListener("input", autoGrow);

/* Suggestions and "no documents" notice, based on what has actually been uploaded. */
const GENERIC_SUGGESTIONS = [
  "What are the main rules or requirements?",
  "What deadlines or dates are mentioned?",
  "Who is responsible for what?",
];
const MAX_SUGGESTIONS = 4;

async function refreshAskHints() {
  let documents = [];
  try {
    documents = await api("/documents");
  } catch { return; }
  const isAdmin = currentUser && currentUser.role === "admin";
  $("#no-docs").hidden = documents.length > 0;
  $("#no-docs-admin").hidden = !isAdmin;
  $("#no-docs-user").hidden = isAdmin;
  const suggestions = $("#suggestions");
  suggestions.replaceChildren();
  if (documents.length === 0) return;
  const questions = [`Summarise “${documents[0].filename}” in a few bullet points`, ...GENERIC_SUGGESTIONS];
  questions.slice(0, MAX_SUGGESTIONS).forEach((text) => {
    suggestions.append(el("button", {
      class: "suggestion", type: "button", text, onclick: () => ask(text),
    }));
  });
}

document.querySelectorAll("[data-goto]").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.goto));
});

/* ---------- documents ---------- */

/* Indexing state: which documents are being embedded right now, and a poller that
 * refreshes the table every few seconds so the user sees real progress. */
const indexingIds = new Set();
let uploadInProgress = false;
let pollTimer = null;
let loadingDocuments = false;
const POLL_INTERVAL_MS = 3000;

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(() => loadDocuments().catch(() => {}), POLL_INTERVAL_MS);
}

function stopPolling() {
  if (indexingIds.size > 0 || uploadInProgress) return;
  clearInterval(pollTimer);
  pollTimer = null;
}

function indexingCell(doc) {
  const percent = doc.chunk_count ? Math.round((100 * doc.embedded_chunk_count) / doc.chunk_count) : 0;
  const bar = el("i");
  bar.style.width = `${percent}%`;
  return el("div", { class: "indexing" },
    el("div", { class: "indexing-line" },
      el("span", { class: "spinner", "aria-hidden": "true" }),
      el("span", { class: "indexing-text", text: `Indexing ${doc.embedded_chunk_count} / ${doc.chunk_count} (${percent}%)` })),
    el("div", { class: "progress", role: "progressbar", "aria-valuenow": String(percent), "aria-valuemin": "0", "aria-valuemax": "100" }, bar),
    el("div", { class: "muted small", text: "The free AI tier limits speed, so this can take a minute or two. You can leave this page open." }));
}

async function retryIndexing(doc) {
  indexingIds.add(doc.id);
  startPolling();
  loadDocuments().catch(() => {});
  try {
    await api(`/documents/${doc.id}/embeddings`, { method: "POST" });
    toast(`“${doc.filename}” is fully indexed.`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    indexingIds.delete(doc.id);
    stopPolling();
    loadDocuments().catch(() => {});
  }
}

async function loadDocuments() {
  if (loadingDocuments) return;
  loadingDocuments = true;
  try {
    renderDocuments(await api("/documents"));
  } finally {
    loadingDocuments = false;
  }
}

function renderDocuments(documents) {
  const body = $("#documents-body");
  body.replaceChildren();
  $("#documents-empty").hidden = documents.length > 0;
  const isAdmin = currentUser && currentUser.role === "admin";
  documents.forEach((doc) => {
    const complete = doc.embedded_chunk_count >= doc.chunk_count;
    const busy = !complete && (indexingIds.has(doc.id) || uploadInProgress);
    const status = el("td", {});
    if (complete) {
      status.append(el("span", { class: "badge ok", text: "Ready" }));
    } else if (busy) {
      status.append(indexingCell(doc));
    } else {
      status.append(el("span", {
        class: "badge warn", text: `Keyword only (${doc.embedded_chunk_count}/${doc.chunk_count})`,
      }));
      if (isAdmin) {
        status.append(" ", el("button", {
          class: "link small", type: "button",
          text: doc.embedded_chunk_count > 0 ? "Continue" : "Retry",
          onclick: () => retryIndexing(doc),
        }));
      }
    }
    body.append(el("tr", {},
      el("td", { class: "truncate", text: doc.filename }),
      el("td", { class: "num", text: String(doc.page_count) }),
      el("td", { class: "num", text: String(doc.chunk_count) }),
      status,
      el("td", { class: "muted", text: formatDate(doc.uploaded_at) }),
      el("td", {}, isAdmin && !busy ? el("button", {
        class: "link danger small", type: "button", text: "Delete",
        onclick: async () => {
          if (!confirm(`Delete “${doc.filename}”? Its passages and search data will be removed.`)) return;
          try {
            await api(`/documents/${doc.id}`, { method: "DELETE" });
            toast("Document deleted.");
          } catch (error) { toast(error.message, true); }
          loadDocuments().catch(() => {});
        },
      }) : null)));
  });
}

async function uploadFile(file) {
  const status = $("#upload-status");
  if (!file) return;
  if (!/\.pdf$/i.test(file.name)) {
    status.textContent = "Only PDF files are supported.";
    return;
  }
  uploadInProgress = true;
  startPolling();
  status.replaceChildren(el("span", { class: "spinner", "aria-hidden": "true" }),
    ` Uploading and indexing “${file.name}”… large documents can take a minute or two.`);
  $("#upload-card").classList.add("busy");
  const form = new FormData();
  form.append("file", file);
  try {
    const doc = await api("/documents", { method: "POST", body: form });
    const note = doc.embedded_chunk_count < doc.chunk_count
      ? " Search indexing is incomplete; use Continue to finish it." : "";
    toast(`Added “${doc.filename}” (${doc.page_count} pages, ${doc.chunk_count} passages).${note}`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    uploadInProgress = false;
    status.textContent = "";
    $("#upload-card").classList.remove("busy");
    stopPolling();
    loadDocuments().catch(() => {});
  }
}
const dropzone = $("#upload-card");
$("#browse").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", (event) => {
  uploadFile(event.target.files[0]);
  event.target.value = "";
});
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.add("over");
}));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.remove("over");
}));
dropzone.addEventListener("drop", (event) => uploadFile(event.dataTransfer.files[0]));

/* ---------- question log ---------- */

const LOG_CLAMP_CHARS = 280;
let logRows = [];
let logFilter = "all";
let logQuery = "";

async function loadLog() {
  logRows = await api("/questions");
  renderLog();
}

function renderLog() {
  const counts = { all: logRows.length, answered: logRows.filter((row) => row.found).length };
  counts.notfound = counts.all - counts.answered;
  Object.entries(counts).forEach(([key, value]) => { $(`#count-${key}`).textContent = String(value); });
  const query = logQuery.trim().toLowerCase();
  const rows = logRows.filter((row) =>
    (logFilter === "all" || (logFilter === "answered") === row.found)
    && (!query || `${row.question} ${row.answer} ${row.username}`.toLowerCase().includes(query)));
  $("#log-list").replaceChildren(...rows.map(renderLogItem));
  const empty = $("#log-empty");
  empty.hidden = rows.length > 0;
  empty.textContent = logRows.length === 0 ? "No questions yet." : "No entries match your filter.";
  $("#log-clear").disabled = logRows.length === 0;
}

function renderLogItem(row) {
  const body = el("div", { class: "log-answer rich" }, renderAnswerText(row.answer, null));
  const isLong = row.answer.length > LOG_CLAMP_CHARS;
  if (isLong) body.classList.add("clamped");
  const toggle = isLong ? el("button", {
    class: "link small", type: "button", text: "Show more",
    onclick: () => {
      toggle.textContent = body.classList.toggle("clamped") ? "Show more" : "Show less";
    },
  }) : null;
  return el("article", { class: "log-item" },
    el("header", { class: "log-head" },
      el("h3", { class: "log-question", text: row.question }),
      el("button", {
        class: "link danger small", type: "button", text: "Delete",
        onclick: async () => {
          if (!confirm("Delete this log entry? This cannot be undone.")) return;
          try {
            await api(`/questions/${row.id}`, { method: "DELETE" });
            logRows = logRows.filter((entry) => entry.id !== row.id);
            renderLog();
            toast("Log entry deleted.");
          } catch (error) { toast(error.message, true); }
        },
      })),
    el("div", { class: "log-meta" },
      el("span", { class: "badge " + (row.found ? "ok" : "warn"), text: row.found ? "Answered" : "Not found" }),
      el("span", { class: "muted small", text: `${row.username} · ${formatDate(row.created_at)}` })),
    body, toggle);
}

document.querySelectorAll(".seg").forEach((button) => button.addEventListener("click", () => {
  logFilter = button.dataset.filter;
  document.querySelectorAll(".seg").forEach((other) => other.classList.toggle("active", other === button));
  renderLog();
}));

$("#log-search").addEventListener("input", (event) => {
  logQuery = event.target.value;
  renderLog();
});

$("#log-clear").addEventListener("click", async () => {
  if (!confirm(`Delete all ${logRows.length} log entries? This removes the review trail and cannot be undone.`)) return;
  try {
    const result = await api("/questions", { method: "DELETE" });
    toast(`${result.deleted} log entries deleted.`);
    await loadLog();
  } catch (error) { toast(error.message, true); }
});
/* ---------- users ---------- */

async function loadUsers() {
  const users = await api("/users");
  const body = $("#users-body");
  body.replaceChildren();
  users.forEach((user) => {
    body.append(el("tr", {}, el("td", { text: user.username }), el("td", {},
      el("span", { class: "badge " + (user.role === "admin" ? "warn" : "ok"), text: user.role }))));
  });
}

$("#user-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/users", jsonRequest("POST", {
      username: $("#new-username").value.trim(),
      password: $("#new-password").value,
      role: $("#new-role").value,
    }));
    event.target.reset();
    toast("User added.");
    await loadUsers();
  } catch (error) { toast(error.message, true); }
});

/* ---------- start ---------- */

(async function start() {
  try {
    showApp(await api("/auth/me"));
  } catch {
    showLogin();
  }
})();
