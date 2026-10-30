import type { IncrementalDirectedAcyclicGraphNode } from "../data_structure.js";
import { DataStructureException } from "../exceptions.js";

type GraphNode = IncrementalDirectedAcyclicGraphNode;

function describe(cycle: readonly GraphNode[]): string {
  const source = cycle[0];
  const target = cycle[1];
  if (source === target) {
    return `Cannot add edge ${source} -> ${target}: a node cannot have an edge to itself.`;
  }
  return (
    `Cannot add edge ${source} -> ${target}: a path from ${target} back to ${source} already exists ` +
    `(${cycle.slice(1).join(" -> ")}), ` +
    `so the new edge would create the cycle ${cycle.join(" -> ")}.`
  );
}

export class CycleError extends DataStructureException {
  /** The rejected edge followed by the path that already leads back to its source. */
  readonly cycle: readonly GraphNode[];

  constructor(cycle: readonly GraphNode[]) {
    super(describe(cycle));
    this.cycle = cycle;
  }

  /** The edge that would close the cycle: [source, target]. */
  get edge(): readonly [GraphNode, GraphNode] {
    return [this.cycle[0]!, this.cycle[1]!];
  }

  /** The path that already leads from the target of the edge back to its source. */
  get path(): readonly GraphNode[] {
    return this.cycle.slice(1);
  }
}
