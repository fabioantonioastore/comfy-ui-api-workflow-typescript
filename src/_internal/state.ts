// What nodes and pipelines keep out of sight: the workflow reads and changes
// it, while users only reach it through the public members.
import type { BaseNode } from "../nodes/base.js";
import type { Link } from "../nodes/validation.js";
import type { Workflow } from "../workflow.js";

export interface NodeState {
  nodeId: string;
  /** The inputs that are set. Replaced on each change, never changed in place. */
  values: ReadonlyMap<string, unknown>;
  /** Told of each change by the workflow that holds the node, which can reject it. */
  onChange: ((node: BaseNode) => void) | null;
  /** While true, changes are type checked and linked, but the node is validated later. */
  batch: boolean;
}

export interface Rejection {
  /** The position of the node in the pipeline, from 1. */
  readonly position: number;
  readonly type: { readonly name: string };
  readonly error: unknown;
}

export interface PipelineState {
  /** The workflow adding the pipeline, only while it runs its build. */
  workflow: Workflow | null;
  readonly nodes: Map<string, BaseNode>;
  outputs: ReadonlyMap<string, Link>;
  /** The first node the workflow rejected while the build ran. */
  rejection: Rejection | null;
}

const nodes = new WeakMap<object, NodeState>();
const pipelines = new WeakMap<object, PipelineState>();

export function setNodeState(node: object, state: NodeState): void {
  nodes.set(node, state);
}

export function findNodeState(node: object): NodeState | undefined {
  return nodes.get(node);
}

export function nodeState(node: object): NodeState {
  const state = nodes.get(node);
  if (state === undefined) {
    throw new TypeError("This object is not a node made by its constructor.");
  }
  return state;
}

export function setPipelineState(pipeline: object, state: PipelineState): void {
  pipelines.set(pipeline, state);
}

export function pipelineState(pipeline: object): PipelineState {
  const state = pipelines.get(pipeline);
  if (state === undefined) {
    throw new TypeError("This object is not a pipeline made by its constructor.");
  }
  return state;
}
