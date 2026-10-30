// The registry of the Python test_runpod_workflows.py: each case builds a
// workflow and returns the check of what ComfyUI sends back.
import type { ComfyUIWorkflow } from "../../src/nodes/base.js";
import { Workflow } from "../../src/workflow.js";
import type { Outputs } from "./media.js";

export const TIMEOUT = 1200;
export const GPU_TIMEOUT = 1800;

/** Checks the files a job returned; it throws when they are wrong. */
export type Check = (out: Outputs) => void;
/** Files the worker puts in ComfyUI's input folder before the job, by name. */
export type Uploads = Readonly<Record<string, Buffer>>;

export class Case {
  readonly #build: (w: Workflow) => Check | void;
  readonly #raw: (() => ComfyUIWorkflow) | null;

  constructor(
    readonly name: string,
    readonly description: string,
    build: ((w: Workflow) => Check | void) | { raw: () => ComfyUIWorkflow },
    readonly uploads: Uploads = {},
    readonly failsWith: readonly string[] = [],
    readonly librarySays: string | null = null,
    readonly gpu = false,
  ) {
    this.#build = typeof build === "function" ? build : () => undefined;
    this.#raw = typeof build === "function" ? null : build.raw;
  }

  get timeout(): number {
    return this.gpu ? GPU_TIMEOUT : TIMEOUT;
  }

  /** The workflow sent to ComfyUI and the check of its outputs; none for a refused one. */
  payload(): [ComfyUIWorkflow, Check | undefined] {
    if (this.#raw !== null) {
      return [this.#raw(), undefined];
    }
    const workflow = new Workflow();
    const check = this.#build(workflow);
    return [sent(workflow), check ?? undefined];
  }
}

export const CASES: Case[] = [];

export interface CaseOptions {
  readonly uploads?: Uploads;
  /** What ComfyUI's error says, for a workflow it must fail. */
  readonly failsWith?: string | readonly string[];
  readonly gpu?: boolean;
}

/** Adds a case that builds its workflow with the library. */
export function addCase(
  name: string,
  description: string,
  options: CaseOptions,
  build: (w: Workflow) => Check | void,
): void {
  const { uploads = {}, failsWith = [], gpu = false } = options;
  const reasons = typeof failsWith === "string" ? [failsWith] : [...failsWith];
  CASES.push(new Case(name, description, build, uploads, reasons, null, gpu));
}

export interface RejectedOptions {
  /** What the library's error says when it loads the workflow. */
  readonly librarySays: string;
  /** What ComfyUI's error says. */
  readonly comfySays: string;
}

/** Adds a workflow written by hand, which the library refuses to load and ComfyUI refuses to run. */
export function addRejected(
  name: string,
  description: string,
  options: RejectedOptions,
  build: () => ComfyUIWorkflow,
): void {
  CASES.push(new Case(name, description, { raw: build }, {}, [options.comfySays], options.librarySays));
}

/** The workflow as it reaches ComfyUI, after a trip through JSON. */
export function sent(workflow: Workflow): ComfyUIWorkflow {
  return JSON.parse(JSON.stringify(workflow.getComfyUIWorkflowFormat())) as ComfyUIWorkflow;
}
