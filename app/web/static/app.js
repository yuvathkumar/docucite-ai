"use strict";

/* DocuCite AI front end.
 * All text from the server (document names, passages, model answers) is inserted with
 * textContent / createTextNode, never innerHTML: it is untrusted and must not run as markup. */

const $ = (selector) => document.querySelector(selector);
const SUGGESTIONS = [
  "What is the policy on vacation days?",
  "Summarise the key points of the uploaded documents",
  "Which document mentions deadlines?",
];

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
  setTimeout(() => node.remove(), 5000);
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
  if (name === "ask") $("#question").focus();
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

SUGGESTIONS.forEach((text) => {
  $("#suggestions").append(el("button", {
    class: "suggestion", type: "button", text, onclick: () => ask(text),
  }));
});

/* ---------- documents ---------- */

async function loadDocuments() {
  const documents = await api("/documents");
  const body = $("#documents-body");
  body.replaceChildren();
  $("#documents-empty").hidden = documents.length > 0;
  const isAdmin = currentUser && currentUser.role === "admin";
  documents.forEach((doc) => {
    const complete = doc.embedded_chunk_count >= doc.chunk_count;
    const status = el("td", {},
      el("span", {
        class: "badge " + (complete ? "ok" : "warn"),
        text: complete ? "Ready" : `Keyword only (${doc.embedded_chunk_count}/${doc.chunk_count})`,
      }));
    if (!complete && isAdmin) {
      status.append(" ", el("button", {
        class: "link small", type: "button", text: "Retry",
        onclick: async () => {
          try {
            await api(`/documents/${doc.id}/embeddings`, { method: "POST" });
            toast("Search index updated.");
          } catch (error) { toast(error.message, true); }
          loadDocuments();
        },
      }));
    }
    body.append(el("tr", {},
      el("td", { class: "truncate", text: doc.filename }),
      el("td", { class: "num", text: String(doc.page_count) }),
      el("td", { class: "num", text: String(doc.chunk_count) }),
      status,
      el("td", { class: "muted", text: formatDate(doc.uploaded_at) }),
      el("td", {}, isAdmin ? el("button", {
        class: "link danger small", type: "button", text: "Delete",
        onclick: async () => {
          if (!confirm(`Delete “${doc.filename}”? Its passages and search data will be removed.`)) return;
          try {
            await api(`/documents/${doc.id}`, { method: "DELETE" });
            toast("Document deleted.");
          } catch (error) { toast(error.message, true); }
          loadDocuments();
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
  status.textContent = `Uploading ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  try {
    const doc = await api("/documents", { method: "POST", body: form });
    status.textContent = "";
    toast(`Added “${doc.filename}” (${doc.page_count} pages, ${doc.chunk_count} passages).`);
    await loadDocuments();
  } catch (error) {
    status.textContent = "";
    toast(error.message, true);
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

async function loadLog() {
  const rows = await api("/questions");
  const body = $("#log-body");
  body.replaceChildren();
  $("#log-empty").hidden = rows.length > 0;
  rows.forEach((row) => {
    body.append(el("tr", {},
      el("td", { class: "truncate" },
        el("div", { text: row.question }),
        el("div", { class: "log-answer", text: row.answer })),
      el("td", { text: row.username }),
      el("td", {}, el("span", {
        class: "badge " + (row.found ? "ok" : "warn"), text: row.found ? "Answered" : "Not found",
      })),
      el("td", { class: "muted", text: formatDate(row.created_at) })));
  });
}

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
