// The recording part of the Python test_runpod_workflows.py: each job leaves its
// workflow, uploads, files and result in a folder of runpod_outputs/workflows,
// and index.html shows them all, in Portuguese like the Python gallery.
import assert from "node:assert/strict";
import { mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { basename, extname, join } from "node:path";

import type { ComfyUIWorkflow } from "../../src/nodes/base.js";
import { OUTPUT_DIR, type RunPodJob } from "../runpod.js";
import { Outputs, pngSize } from "./media.js";
import type { Check, Uploads } from "./registry.js";

export const GALLERY = join(OUTPUT_DIR, "workflows");

/** Runs jobs: the RunPod endpoint, or a stand-in in the tests. */
export interface Endpoint {
  run(workflow: unknown, timeoutSeconds: number, images?: Uploads): Promise<RunPodJob>;
}

type Result = Record<string, unknown>;

/** Whether Python takes the value as true: empty lists and objects are false too. */
function truthy(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (value !== null && typeof value === "object") {
    return Object.keys(value).length > 0;
  }
  return Boolean(value);
}

/** Python's html.escape. */
function escape(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#x27;");
}

/** JSON with two-space indents, as the Python suite writes it. */
export function writeJson(path: string, value: unknown): void {
  writeFileSync(path, JSON.stringify(value, null, 2), "utf8");
}

function keepFiles(job: RunPodJob, folder: string): [string, Buffer][] {
  const files: [string, Buffer][] = [];
  for (const item of job.output?.images ?? []) {
    // Only the base name, so a file name from the server stays inside the folder.
    const name = basename(item.filename);
    if (item.type !== "base64" || name === "." || name === "..") {
      continue;
    }
    const data = Buffer.from(item.data, "base64");
    writeFileSync(join(folder, name), data);
    files.push([name, data]);
  }
  return files;
}

export interface RecordOptions {
  /** Seconds the job may take, from when it is sent. */
  readonly timeout: number;
  readonly uploads?: Uploads;
  readonly check?: Check | undefined;
  /** What ComfyUI's error must say, for a job that must fail. */
  readonly failsWith?: readonly string[];
  /** The gallery's folder; runpod_outputs/workflows unless a test sets another. */
  readonly gallery?: string;
}

/** Runs a job, checks it and records it in the gallery, failed or not. */
export async function runAndRecord(
  endpoint: Endpoint,
  folderName: string,
  title: string,
  description: string,
  payload: ComfyUIWorkflow,
  options: RecordOptions,
): Promise<void> {
  const { timeout, uploads = {}, check, failsWith = [], gallery = GALLERY } = options;
  const folder = join(gallery, folderName);
  rmSync(folder, { recursive: true, force: true });
  mkdirSync(folder, { recursive: true });
  writeJson(join(folder, "workflow.json"), payload);
  for (const [name, data] of Object.entries(uploads)) {
    writeFileSync(join(folder, `entrada-${name}`), data);
  }
  const result: Result = {
    title,
    description,
    status: "running",
    nodes: Object.keys(payload).length,
    class_types: [...new Set(Object.values(payload).map((node) => node.class_type))].sort(),
    expected_failure: [...failsWith],
  };
  writeJson(join(folder, "result.json"), result);
  writeGallery(gallery);
  const started = performance.now();
  let job: Partial<RunPodJob> = {};
  try {
    job = await endpoint.run(payload, timeout, uploads);
    const files = keepFiles(job as RunPodJob, folder);
    if (failsWith.length > 0) {
      assert(job.status === "FAILED", `expected a failure, the job is ${job.status}`);
      const report = JSON.stringify(job);
      for (const text of failsWith) {
        assert(report.includes(text), `'${text}' is not in the job error: ${job.error}`);
      }
    } else {
      assert(job.status === "COMPLETED", `the job is ${job.status}: ${job.error} ${JSON.stringify(job.output)}`);
      check?.(new Outputs(files));
    }
    result["status"] = "passed";
  } catch (error) {
    result["status"] = "failed";
    result["message"] = (error instanceof Error ? error.message : String(error)).slice(0, 6000);
    throw error;
  } finally {
    const output = job.output ?? {};
    Object.assign(result, {
      seconds: Math.round((performance.now() - started) / 100) / 10,
      job_id: job.id ?? null,
      job_status: job.status ?? null,
      job_error: job.error ?? null,
      worker_messages: truthy(output.errors) ? output.errors : truthy(output.details) ? output.details : null,
      execution_ms: job.executionTime ?? null,
      queue_ms: job.delayTime ?? null,
    });
    writeJson(join(folder, "result.json"), result);
    writeGallery(gallery);
  }
}

const STATUS_LABELS: Readonly<Record<string, string>> = { passed: "passou", failed: "falhou", running: "rodando" };

/** How the gallery shows a file of a job's folder. */
export function mediaHtml(folder: string, name: string): string {
  const source = escape(`${basename(folder)}/${name}`);
  const caption = escape(name);
  const suffix = extname(name);
  let body: string;
  if (suffix === ".png" || suffix === ".webp") {
    const small = suffix === ".png" && pngSize(readFileSync(join(folder, name)).subarray(0, 24))[0] < 256;
    const css = small ? ' class="pixel"' : "";
    body = `<a href="${source}"><img src="${source}" loading="lazy"${css} alt="${caption}"></a>`;
  } else if (suffix === ".mp4" || suffix === ".webm" || suffix === ".mkv") {
    body = `<video src="${source}" controls loop playsinline preload="metadata"></video>`;
  } else if (suffix === ".wav") {
    body = `<audio src="${source}" controls preload="none"></audio>`;
  } else {
    body = `<a class="file" href="${source}">${caption}</a>`;
  }
  return `<figure>${body}<figcaption>${caption}</figcaption></figure>`;
}

function cardHtml(folder: string, result: Result): string {
  const status = String(result["status"] ?? "running");
  const media = readdirSync(folder)
    .sort()
    .filter((name) => statSync(join(folder, name)).isFile() && extname(name) !== ".json")
    .map((name) => mediaHtml(folder, name));
  const details: string[] = [];
  if (result["seconds"] != null) {
    details.push(`${String(result["seconds"])} s`);
  }
  details.push(`${String(result["nodes"] ?? "?")} nós`);
  if (truthy(result["expected_failure"])) {
    details.push("falha esperada");
  }
  const message = truthy(result["message"]) ? result["message"] : result["job_error"];
  const error = truthy(message) ? `<pre>${escape(String(message))}</pre>` : "";
  const worker = result["worker_messages"];
  let workerHtml = "";
  if (truthy(worker)) {
    const text = escape(JSON.stringify(worker, null, 2));
    workerHtml = `<details><summary>mensagens do worker</summary><pre>${text}</pre></details>`;
  }
  const types = ((result["class_types"] as string[] | undefined) ?? []).map(escape).join(", ");
  const name = escape(basename(folder));
  const links = `<a href="${name}/workflow.json">workflow.json</a> · <a href="${name}/result.json">result.json</a>`;
  const files = media.join("") || '<p class="meta">sem arquivos</p>';
  return (
    `<article class="card ${status}"><header><h2>${name}</h2>` +
    `<span class="badge">${STATUS_LABELS[status] ?? status}</span></header>` +
    `<p>${escape(String(result["description"] ?? ""))}</p>` +
    `<p class="meta">${details.join(" · ")} · ${links}</p>${error}${workerHtml}` +
    `<div class="media">${files}</div>` +
    `<details><summary>tipos de nó</summary><p class="meta">${types}</p></details></article>`
  );
}

const GALLERY_STYLE = `
:root { --bg: #f6f5f2; --card: #ffffff; --ink: #1d1d1f; --muted: #6b6b70; --line: #e3e1dc;
  --ok: #1f8a4c; --bad: #c0362c; --run: #b7791f; --accent: #3d5afe; }
@media (prefers-color-scheme: dark) { :root { --bg: #141416; --card: #1d1d21; --ink: #ececf1;
  --muted: #a0a0aa; --line: #2c2c33; --ok: #4cc38a; --bad: #ff6b5f; --run: #f0b54a;
  --accent: #8c9eff; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
.summary { color: var(--muted); margin: 0 0 24px; }
.summary b { color: var(--ink); }
.cards { display: grid; gap: 16px; }
.card { background: var(--card); border: 1px solid var(--line); border-left: 4px solid var(--run);
  border-radius: 10px; padding: 14px 16px; }
.card.passed { border-left-color: var(--ok); }
.card.failed { border-left-color: var(--bad); }
.card header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.card h2 { font-size: 16px; margin: 0; font-family: ui-monospace, monospace; word-break: break-all; }
.badge { font-size: 12px; font-weight: 600; padding: 2px 10px; border-radius: 99px; color: #fff;
  background: var(--run); white-space: nowrap; }
.passed .badge { background: var(--ok); }
.failed .badge { background: var(--bad); }
.card p { margin: 6px 0; }
.meta { color: var(--muted); font-size: 13px; }
.meta a { color: var(--accent); }
pre { background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 8px;
  overflow-x: auto; white-space: pre-wrap; font-size: 12px; }
.media { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px;
  margin-top: 10px; }
figure { margin: 0; }
figure audio { width: 100%; }
figure img, figure video { width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--line);
  background: repeating-conic-gradient(#8882 0 25%, transparent 0 50%) 0 0 / 12px 12px; }
img.pixel { image-rendering: pixelated; }
figcaption { font-size: 11px; color: var(--muted); word-break: break-all; }
a.file { display: block; padding: 24px 8px; text-align: center; border: 1px dashed var(--line);
  border-radius: 6px; color: var(--accent); }
details summary { cursor: pointer; color: var(--muted); font-size: 13px; margin-top: 6px; }
`;

function now(): string {
  const date = new Date();
  const two = (value: number) => String(value).padStart(2, "0");
  return (
    `${two(date.getDate())}/${two(date.getMonth() + 1)}/${date.getFullYear()} ` +
    `${two(date.getHours())}:${two(date.getMinutes())}:${two(date.getSeconds())}`
  );
}

let pages = 0;

/** Writes index.html with a card for each folder of the gallery. */
export function writeGallery(gallery = GALLERY): void {
  mkdirSync(gallery, { recursive: true });
  const cards: string[] = [];
  const counts: Record<string, number> = { passed: 0, failed: 0, running: 0 };
  for (const name of readdirSync(gallery).sort()) {
    const folder = join(gallery, name);
    if (!statSync(folder).isDirectory()) {
      continue;
    }
    let result: Result;
    try {
      result = JSON.parse(readFileSync(join(folder, "result.json"), "utf8")) as Result;
    } catch {
      continue;
    }
    const status = String(result["status"] ?? "running");
    counts[status] = (counts[status] ?? 0) + 1;
    cards.push(cardHtml(folder, result));
  }
  const summary =
    `<b>${cards.length}</b> workflows · <b>${counts["passed"]}</b> passaram · ` +
    `<b>${counts["failed"]}</b> falharam · <b>${counts["running"]}</b> rodando · ` +
    `atualizado em ${now()}`;
  const page =
    '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">' +
    '<meta name="viewport" content="width=device-width, initial-scale=1">' +
    `<title>Galeria de workflows</title><style>${GALLERY_STYLE}</style></head><body><main>` +
    "<h1>Galeria de workflows no RunPod</h1>" +
    `<p class="summary">${summary}</p><div class="cards">${cards.join("")}</div>` +
    "</main></body></html>";
  // Written aside and renamed, so a browser never reads half a page.
  pages += 1;
  const temporary = join(gallery, `.index-${process.pid}-${pages}.html`);
  writeFileSync(temporary, page, "utf8");
  renameSync(temporary, join(gallery, "index.html"));
}
