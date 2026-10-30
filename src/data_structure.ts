import { repr } from "./_internal/format.js";
import { CycleError } from "./errors/data_structure.js";

const INSPECT: unique symbol = Symbol.for("nodejs.util.inspect.custom");

export class IncrementalDirectedAcyclicGraphNode<T = unknown> {
  readonly data: T;

  constructor(data: T) {
    this.data = data;
  }

  toString(): string {
    return repr(this.data);
  }

  [INSPECT](): string {
    return `${this.constructor.name}(${repr(this.data)})`;
  }
}

type GraphNode = IncrementalDirectedAcyclicGraphNode;

function missing(node: GraphNode): Error {
  return new Error(`${node} is not in the graph`);
}

// Keeps a topological order while edges are added one by one, moving only the
// nodes between the two ends of an edge that goes against the order.
export class IncrementalDirectedAcyclicGraph {
  readonly #successors = new Map<GraphNode, Set<GraphNode>>();
  readonly #predecessors = new Map<GraphNode, Set<GraphNode>>();
  readonly #order = new Map<GraphNode, number>();
  #next = 0;

  addNode(node: GraphNode): void {
    if (this.#order.has(node)) {
      return;
    }
    this.#successors.set(node, new Set());
    this.#predecessors.set(node, new Set());
    this.#order.set(node, this.#next);
    this.#next += 1;
  }

  removeNode(node: GraphNode): void {
    const successors = this.#successors.get(node);
    const predecessors = this.#predecessors.get(node);
    if (successors === undefined || predecessors === undefined) {
      throw missing(node);
    }
    for (const successor of successors) {
      this.#predecessors.get(successor)?.delete(node);
    }
    for (const predecessor of predecessors) {
      this.#successors.get(predecessor)?.delete(node);
    }
    this.#successors.delete(node);
    this.#predecessors.delete(node);
    this.#order.delete(node);
  }

  insertBetween(a: GraphNode, b: GraphNode, node: GraphNode, removeOriginal = true): void {
    const newNodes = [a, b, node].filter((other) => !this.has(other));
    const newEdge = !(this.#successors.get(a)?.has(node) ?? false);
    try {
      this.addNode(node);
      this.addEdge(a, node);
      this.addEdge(node, b);
    } catch (error) {
      if (error instanceof CycleError) {
        if (newEdge && this.has(a) && this.has(node)) {
          this.removeEdge(a, node);
        }
        for (const newNode of newNodes) {
          if (this.has(newNode)) {
            this.removeNode(newNode);
          }
        }
      }
      throw error;
    }
    if (removeOriginal) {
      this.removeEdge(a, b);
    }
  }

  addEdge(x: GraphNode, y: GraphNode): void {
    if (x === y) {
      throw new CycleError([x, x]);
    }
    this.addNode(x);
    this.addNode(y);
    if (this.#successors.get(x)!.has(y)) {
      return;
    }
    const orderX = this.#order.get(x)!;
    const orderY = this.#order.get(y)!;
    if (orderX < orderY) {
      this.#link(x, y);
      return;
    }
    const forward = this.#forwardSearch(y, x, orderX);
    const backward = this.#backwardSearch(x, orderY);
    this.#reorder(backward, forward);
    this.#link(x, y);
  }

  removeEdge(x: GraphNode, y: GraphNode): void {
    const successors = this.#successors.get(x);
    const predecessors = this.#predecessors.get(y);
    if (successors === undefined) {
      throw missing(x);
    }
    if (predecessors === undefined) {
      throw missing(y);
    }
    successors.delete(y);
    predecessors.delete(x);
  }

  topologicalOrder(): GraphNode[] {
    return [...this.#order.keys()].sort((a, b) => this.#order.get(a)! - this.#order.get(b)!);
  }

  dependencies(node: GraphNode): Set<GraphNode> {
    const successors = this.#successors.get(node);
    if (successors === undefined) {
      throw missing(node);
    }
    return new Set(successors);
  }

  dependents(node: GraphNode): Set<GraphNode> {
    const predecessors = this.#predecessors.get(node);
    if (predecessors === undefined) {
      throw missing(node);
    }
    return new Set(predecessors);
  }

  has(node: GraphNode): boolean {
    return this.#order.has(node);
  }

  get size(): number {
    return this.#order.size;
  }

  #link(x: GraphNode, y: GraphNode): void {
    this.#successors.get(x)!.add(y);
    this.#predecessors.get(y)!.add(x);
  }

  #forwardSearch(y: GraphNode, x: GraphNode, upperBound: number): GraphNode[] {
    const parent = new Map<GraphNode, GraphNode | null>([[y, null]]);
    const stack = [y];
    while (stack.length > 0) {
      const node = stack.pop()!;
      for (const successor of this.#successors.get(node)!) {
        if (successor === x) {
          const path = [node];
          let previous = parent.get(node) ?? null;
          while (previous !== null) {
            path.push(previous);
            previous = parent.get(previous) ?? null;
          }
          path.reverse();
          throw new CycleError([x, ...path, x]);
        }
        if (!parent.has(successor) && this.#order.get(successor)! < upperBound) {
          parent.set(successor, node);
          stack.push(successor);
        }
      }
    }
    return [...parent.keys()];
  }

  #backwardSearch(x: GraphNode, lowerBound: number): GraphNode[] {
    const visited = new Set([x]);
    const stack = [x];
    while (stack.length > 0) {
      const node = stack.pop()!;
      for (const predecessor of this.#predecessors.get(node)!) {
        if (!visited.has(predecessor) && this.#order.get(predecessor)! > lowerBound) {
          visited.add(predecessor);
          stack.push(predecessor);
        }
      }
    }
    return [...visited];
  }

  #reorder(backward: GraphNode[], forward: GraphNode[]): void {
    const key = (node: GraphNode): number => this.#order.get(node)!;
    backward.sort((a, b) => key(a) - key(b));
    forward.sort((a, b) => key(a) - key(b));
    const nodes = [...backward, ...forward];
    const slots = nodes.map(key).sort((a, b) => a - b);
    nodes.forEach((node, index) => {
      this.#order.set(node, slots[index]!);
    });
  }
}
