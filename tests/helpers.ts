import { readFileSync } from "node:fs";
import { expect } from "vitest";

import type { Workflow } from "../src/workflow.js";

/** JSON with sorted keys, so equal workflows give equal text. */
export function canonical(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) =>
    item !== null && typeof item === "object" && !Array.isArray(item)
      ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
      : item,
  );
}

/** Runs the action, which must throw an error of the type, and returns that error. */
export function expectError<E extends Error>(
  action: () => unknown,
  type: abstract new (...args: any[]) => E,
  message?: string | RegExp,
): E {
  let caught: unknown;
  try {
    action();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(type);
  const error = caught as E;
  if (typeof message === "string") {
    expect(error.message).toBe(message);
  } else if (message !== undefined) {
    expect(error.message).toMatch(message);
  }
  return error;
}

/** What a rejected change must leave as it was: the nodes, what is sent, and the links between nodes. */
export function workflowState(workflow: Workflow) {
  return {
    nodes: [...workflow.nodes],
    format: workflow.getComfyUIWorkflowFormat(),
    links: Object.fromEntries(
      [...workflow.nodes.keys()].map((nodeId) => [
        nodeId,
        [workflow.dependencies(nodeId), workflow.dependents(nodeId)],
      ]),
    ),
  };
}

export type WorkflowState = ReturnType<typeof workflowState>;

/** Expects the workflow to be as it was: the same node objects, inputs and links. */
export function expectUnchanged(workflow: Workflow, before: WorkflowState): void {
  const after = workflowState(workflow);
  expect(after.nodes.map(([nodeId]) => nodeId)).toEqual(before.nodes.map(([nodeId]) => nodeId));
  after.nodes.forEach(([, node], index) => {
    expect(node).toBe(before.nodes[index]![1]);
  });
  expect(after.format).toEqual(before.format);
  expect(after.links).toEqual(before.links);
}

/** A seeded random number generator (mulberry32), so random tests repeat. */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function pick<T>(random: () => number, items: readonly T[]): T {
  const item = items[Math.floor(random() * items.length)];
  if (item === undefined) {
    throw new Error("Cannot pick from no items.");
  }
  return item;
}

/** A node as the Python library built it, and what it sent or why it refused. */
export interface NodeCase {
  readonly case: string;
  readonly class_type: string;
  readonly node_id: string;
  readonly inputs: Readonly<Record<string, unknown>>;
  readonly sent?: Readonly<Record<string, unknown>>;
  readonly error?: string;
  readonly message?: string;
}

export interface Golden {
  readonly source: string;
  readonly nodes: readonly NodeCase[];
  /** The workflow each example prints, by the name of the example. */
  readonly examples: Readonly<Record<string, unknown>>;
}

/** What the Python library sends ComfyUI, as scripts/export_golden.py records it. */
export function readGolden(): Golden {
  return JSON.parse(readFileSync(new URL("./fixtures/golden.json", import.meta.url), "utf8")) as Golden;
}
