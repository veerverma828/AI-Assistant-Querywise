const API = "http://127.0.0.1:8000";

export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed (${res.status})`);
  return res.json();
}

export async function listDocuments() {
  const res = await fetch(`${API}/documents`);
  if (!res.ok) throw new Error(`Could not load documents (${res.status})`);
  return (await res.json()).documents;
}

export async function deleteDocument(name) {
  const res = await fetch(`${API}/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete failed (${res.status})`);
  return res.json();
}

export async function listReports() {
  const res = await fetch(`${API}/reports`);
  if (!res.ok) throw new Error(`Could not load reports (${res.status})`);
  return (await res.json()).reports;
}

export async function deleteReport(name) {
  const res = await fetch(`${API}/reports/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete failed (${res.status})`);
  return res.json();
}

export const reportUrl = (name) => `${API}/reports/${encodeURIComponent(name)}`;

// Reads a server-sent-events response and runs onEvent(event) for every event.
async function readStream(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop(); // last piece may be incomplete
    for (const part of parts) {
      if (part.startsWith("data: ")) onEvent(JSON.parse(part.slice(6)));
    }
  }
}

async function postStream(path, body, onEvent, signal) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  await readStream(res, onEvent);
}

// Streams the agent's steps. `signal` lets the caller cancel (Stop button).
// `history` = earlier messages of this chat [{role, content}], so follow-ups make sense.
export const askAgentStream = (question, onEvent, signal, history = []) =>
  postStream("/agent/stream", { question, history }, onEvent, signal);

// Saves a report straight away (the "Save as report" button), no agent involved.
export async function saveReportDirect(title, content) {
  const res = await fetch(`${API}/reports`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, content }),
  });
  if (!res.ok) throw new Error(`Save failed (${res.status})`);
  return res.json();
}

// Sends the user's decision on a paused action and streams the rest of the run.
// `edits` may carry a changed title/content from the approval card.
export const approveAction = (id, approved, edits, onEvent, signal) =>
  postStream("/agent/approve", { id, approved, ...edits }, onEvent, signal);
