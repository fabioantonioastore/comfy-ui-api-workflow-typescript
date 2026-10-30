import { repr } from "../_internal/format.js";
import { NodeException } from "../exceptions.js";

export function quoteNames(names: readonly string[]): string {
  const quoted = names.map((name) => repr(name));
  if (quoted.length === 1) {
    return quoted[0]!;
  }
  return `${quoted.slice(0, -1).join(", ")} and ${quoted.at(-1)}`;
}

export function describeDependents(dependents: readonly string[]): string {
  if (dependents.length === 1) {
    return `node ${quoteNames(dependents)} depends on it`;
  }
  return `nodes ${quoteNames(dependents)} depend on it`;
}

export class InvalidNodeParameterValue extends NodeException {}

export class NodeNotHaveOutput extends NodeException {}

export class NodeNotFoundError extends NodeException {}

export class NodeHasDependentsError extends NodeException {
  readonly nodeId: string;
  readonly dependents: readonly string[];

  constructor(nodeId: string, dependents: readonly string[]) {
    super(`Cannot remove node ${repr(nodeId)}: ${describeDependents(dependents)}.`);
    this.nodeId = nodeId;
    this.dependents = dependents;
  }
}

export class InvalidNodeChangeError extends NodeException {
  readonly nodeId: string;
  /** The input the change was to, or "nodeId". */
  readonly inputName: string;
  readonly reason: string;

  constructor(nodeId: string, inputName: string, reason: string, options?: ErrorOptions) {
    super(`Cannot change ${repr(inputName)} of node ${repr(nodeId)}: ${reason}.`, options);
    this.nodeId = nodeId;
    this.inputName = inputName;
    this.reason = reason;
  }
}

export class InvalidLinkError extends NodeException {
  readonly nodeId: string;
  /** The input that holds the link. */
  readonly inputName: string;
  readonly reason: string;

  constructor(nodeId: string, inputName: string, reason: string) {
    super(`Cannot add node ${repr(nodeId)}: input ${repr(inputName)} uses ${reason}.`);
    this.nodeId = nodeId;
    this.inputName = inputName;
    this.reason = reason;
  }
}
