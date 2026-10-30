// Port of the RunPod part of the Python conftest.py: runs workflows on the
// RunPod serverless endpoint in .env, which serves ComfyUI. Every job takes
// paid GPU time, so the tests that use it only run with RUNPOD=1.
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const ENV_FILE = join(ROOT, ".env");
// Files returned by the RunPod jobs, one folder per test; ignored by git.
export const OUTPUT_DIR = join(ROOT, "runpod_outputs");
const RUNPOD_SETTINGS = ["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"] as const;

/** Whether the tests that send real jobs to RunPod run. */
export const RUNPOD = process.env["RUNPOD"] === "1";

/** A file the worker returns, such as an image or a video. */
export interface RunPodFile {
  readonly filename: string;
  readonly type: string;
  readonly data: string;
}

export interface RunPodJob {
  readonly id: string;
  readonly status: string;
  readonly output?: {
    readonly images?: readonly RunPodFile[];
    // What worker-comfyui says when ComfyUI rejects or fails a workflow.
    readonly errors?: unknown;
    readonly details?: unknown;
  } | null;
  readonly error?: string;
  readonly executionTime?: number;
  readonly delayTime?: number;
}

/** Runs workflows on a RunPod serverless endpoint that serves ComfyUI. */
export class RunPodEndpoint {
  readonly url: string;
  readonly #apiKey: string;

  constructor(endpoint: string, apiKey: string) {
    const url = endpoint.startsWith("http") ? endpoint : `https://api.runpod.ai/v2/${endpoint}`;
    this.url = url.replace(/\/+$/, "");
    this.#apiKey = apiKey;
  }

  async run(workflow: unknown, timeoutSeconds = 600, images?: Readonly<Record<string, Uint8Array>>): Promise<RunPodJob> {
    const input: Record<string, unknown> = { workflow };
    if (images !== undefined && Object.keys(images).length > 0) {
      // worker-comfyui uploads these to ComfyUI's input folder before the run.
      input["images"] = Object.entries(images).map(([name, data]) => ({
        name,
        image: Buffer.from(data).toString("base64"),
      }));
    }
    let job = await this.#request("POST", "/run", { input });
    const deadline = Date.now() + timeoutSeconds * 1000;
    while (job.status === "IN_QUEUE" || job.status === "IN_PROGRESS") {
      if (Date.now() > deadline) {
        throw new Error(`RunPod job ${job.id} did not finish in ${timeoutSeconds}s`);
      }
      await sleep(2000);
      job = await this.#request("GET", `/status/${encodeURIComponent(job.id)}`);
    }
    return job;
  }

  async #request(method: string, path: string, body?: unknown): Promise<RunPodJob> {
    const response = await fetch(`${this.url}${path}`, {
      method,
      headers: { Authorization: `Bearer ${this.#apiKey}`, "Content-Type": "application/json" },
      body: body === undefined ? null : JSON.stringify(body),
      signal: AbortSignal.timeout(60_000),
    });
    if (!response.ok) {
      throw new Error(`RunPod answered ${method} ${path} with ${response.status}: ${await response.text()}`);
    }
    return (await response.json()) as RunPodJob;
  }
}

export function readEnvFile(path: string): Record<string, string> {
  // Accepts both "NAME=value" and "NAME: value" lines.
  if (!existsSync(path)) {
    return {};
  }
  const settings: [string, string][] = [];
  for (const line of readFileSync(path, "utf8").split(/\r?\n/)) {
    const match = /^\s*(?:export\s+)?([A-Za-z_]\w*)\s*[:=]\s*(.*)$/.exec(line);
    if (match !== null && !line.trimStart().startsWith("#")) {
      settings.push([match[1]!, match[2]!.trim().replace(/^['"]+|['"]+$/g, "")]);
    }
  }
  return Object.fromEntries(settings);
}

/** Runs a test's jobs and keeps every file the worker returns in its folder. */
export class RunPodJobs {
  readonly endpoint: RunPodEndpoint;
  readonly folder: string;

  constructor(endpoint: RunPodEndpoint, folder: string) {
    this.endpoint = endpoint;
    this.folder = folder;
  }

  async run(workflow: unknown, timeoutSeconds = 600): Promise<RunPodJob> {
    const job = await this.endpoint.run(workflow, timeoutSeconds);
    for (const item of job.output?.images ?? []) {
      // Only the base name, so a file name from the server stays inside.
      const name = basename(item.filename);
      if (item.type === "base64" && name !== "." && name !== "..") {
        mkdirSync(this.folder, { recursive: true });
        writeFileSync(join(this.folder, name), Buffer.from(item.data, "base64"));
      }
    }
    return job;
  }
}

let endpoint: RunPodEndpoint | undefined;

/** The endpoint in .env or the environment, which all the RunPod tests share. */
export function runpodEndpoint(): RunPodEndpoint {
  if (endpoint === undefined) {
    const settings = { ...readEnvFile(ENV_FILE), ...process.env };
    const missing = RUNPOD_SETTINGS.filter((name) => !settings[name]);
    if (missing.length > 0) {
      throw new Error(`RUNPOD=1 needs ${missing.join(", ")} in .env or the environment`);
    }
    endpoint = new RunPodEndpoint(settings.RUNPOD_ENDPOINT_ID!, settings.RUNPOD_API_KEY!);
  }
  return endpoint;
}

/** The jobs of a test, whose files go to runpod_outputs/<test name>. */
export function runpodJobs(testName: string): RunPodJobs {
  const shared = runpodEndpoint();
  const folder = join(OUTPUT_DIR, testName.replace(/\W+/g, "_"));
  rmSync(folder, { recursive: true, force: true });
  return new RunPodJobs(shared, folder);
}
