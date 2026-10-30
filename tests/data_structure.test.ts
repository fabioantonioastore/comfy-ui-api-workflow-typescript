import { describe, expect, it } from "vitest";

import {
  IncrementalDirectedAcyclicGraph,
  IncrementalDirectedAcyclicGraphNode,
} from "../src/data_structure.js";
import { CycleError } from "../src/errors/data_structure.js";
import { ComfyUIAPIWorkflowException, DataStructureException } from "../src/exceptions.js";
import { expectError, pick, seededRandom } from "./helpers.js";

type GraphNode = IncrementalDirectedAcyclicGraphNode;

function makeNodes<const T extends readonly unknown[]>(...names: T): { [K in keyof T]: GraphNode } {
  return names.map((name) => new IncrementalDirectedAcyclicGraphNode(name)) as {
    [K in keyof T]: GraphNode;
  };
}

function edge(node: GraphNode, dependency: GraphNode): string {
  return `${String(node.data)}->${String(dependency.data)}`;
}

function edges(graph: IncrementalDirectedAcyclicGraph): Set<string> {
  const found = new Set<string>();
  for (const node of graph.topologicalOrder()) {
    for (const dependency of graph.dependencies(node)) {
      found.add(edge(node, dependency));
    }
  }
  return found;
}

function structure(graph: IncrementalDirectedAcyclicGraph): [GraphNode[], string[]] {
  const nodes = [...graph.topologicalOrder()].sort((a, b) => String(a.data).localeCompare(String(b.data)));
  return [nodes, [...edges(graph)].sort()];
}

function assertValidOrder(graph: IncrementalDirectedAcyclicGraph): void {
  const order = graph.topologicalOrder();
  const position = new Map(order.map((node, index) => [node, index]));
  for (const node of order) {
    for (const dependency of graph.dependencies(node)) {
      expect(position.get(node)!).toBeLessThan(position.get(dependency)!);
      expect(graph.dependents(dependency).has(node)).toBe(true);
    }
  }
}

function withEdges(
  graph: IncrementalDirectedAcyclicGraph,
  newEdges: readonly [GraphNode, GraphNode][],
): Map<GraphNode, Set<GraphNode>> {
  const adjacency = new Map<GraphNode, Set<GraphNode>>();
  for (const node of graph.topologicalOrder()) {
    adjacency.set(node, graph.dependencies(node));
  }
  for (const [source, target] of newEdges) {
    if (!adjacency.has(source)) {
      adjacency.set(source, new Set());
    }
    adjacency.get(source)!.add(target);
    if (!adjacency.has(target)) {
      adjacency.set(target, new Set());
    }
  }
  return adjacency;
}

function hasCycle(adjacency: Map<GraphNode, Set<GraphNode>>): boolean {
  const indegree = new Map([...adjacency.keys()].map((node) => [node, 0]));
  for (const targets of adjacency.values()) {
    for (const target of targets) {
      indegree.set(target, indegree.get(target)! + 1);
    }
  }
  const ready = [...indegree].filter(([, degree]) => degree === 0).map(([node]) => node);
  let visited = 0;
  while (ready.length > 0) {
    const node = ready.pop()!;
    visited += 1;
    for (const target of adjacency.get(node)!) {
      indegree.set(target, indegree.get(target)! - 1);
      if (indegree.get(target) === 0) {
        ready.push(target);
      }
    }
  }
  return visited !== adjacency.size;
}

function chain(...nodes: GraphNode[]): IncrementalDirectedAcyclicGraph {
  const graph = new IncrementalDirectedAcyclicGraph();
  for (let index = 0; index + 1 < nodes.length; index += 1) {
    graph.addEdge(nodes[index]!, nodes[index + 1]!);
  }
  return graph;
}

describe("IncrementalDirectedAcyclicGraph", () => {
  it("adds a node only once", () => {
    const [a, b] = makeNodes("a", "b");
    const graph = chain(a, b);
    graph.addNode(a);
    expect(graph.size).toBe(2);
    expect(graph.topologicalOrder()).toEqual([a, b]);
    expect(edges(graph)).toEqual(new Set([edge(a, b)]));
    expect(graph.dependents(b)).toEqual(new Set([a]));
  });

  it("adds the missing nodes of an edge", () => {
    const [a, b] = makeNodes("a", "b");
    const graph = chain(a, b);
    expect(graph.has(a) && graph.has(b)).toBe(true);
    expect(graph.dependencies(a)).toEqual(new Set([b]));
    expect(graph.dependents(b)).toEqual(new Set([a]));
  });

  it("reorders the nodes for an edge against the current order", () => {
    const graph = new IncrementalDirectedAcyclicGraph();
    const [a, b] = makeNodes("a", "b");
    graph.addNode(a);
    graph.addNode(b);
    graph.addEdge(b, a);
    expect(graph.topologicalOrder()).toEqual([b, a]);
  });

  it("moves the whole affected region", () => {
    const graph = new IncrementalDirectedAcyclicGraph();
    const [a, b, c, d] = makeNodes("a", "b", "c", "d");
    for (const node of [a, b, c, d]) {
      graph.addNode(node);
    }
    graph.addEdge(a, b);
    graph.addEdge(c, d);
    graph.addEdge(d, a);
    expect(graph.topologicalOrder()).toEqual([c, d, a, b]);
  });

  it("ignores a duplicate edge", () => {
    const [a, b] = makeNodes("a", "b");
    const graph = chain(a, b);
    const orderBefore = graph.topologicalOrder();
    graph.addEdge(a, b);
    expect(graph.topologicalOrder()).toEqual(orderBefore);
    expect(edges(graph)).toEqual(new Set([edge(a, b)]));
  });

  it("rejects an edge that closes a cycle, with the cycle", () => {
    const [a, b, c] = makeNodes("a", "b", "c");
    const graph = chain(a, b, c);
    const error = expectError(() => graph.addEdge(c, a), CycleError);
    expect(error.cycle).toEqual([c, a, b, c]);
  });

  it("leaves the graph unchanged when it rejects an edge", () => {
    const [a, b, c] = makeNodes("a", "b", "c");
    const graph = chain(a, b, c);
    const orderBefore = graph.topologicalOrder();
    const edgesBefore = edges(graph);
    expectError(() => graph.addEdge(c, a), CycleError);
    expect(graph.topologicalOrder()).toEqual(orderBefore);
    expect(edges(graph)).toEqual(edgesBefore);
  });

  it("rejects a self loop", () => {
    const graph = new IncrementalDirectedAcyclicGraph();
    const [node] = makeNodes("n");
    graph.addNode(node);
    const error = expectError(() => graph.addEdge(node, node), CycleError);
    expect(error.cycle).toEqual([node, node]);
    expect(graph.dependencies(node)).toEqual(new Set());
  });

  it("does not add a new node through a self loop", () => {
    const graph = new IncrementalDirectedAcyclicGraph();
    const [node] = makeNodes("n");
    expectError(() => graph.addEdge(node, node), CycleError);
    expect(graph.has(node)).toBe(false);
    expect(graph.size).toBe(0);
  });

  it("throws cycle errors of the library hierarchy", () => {
    expect(CycleError.prototype).toBeInstanceOf(DataStructureException);
    expect(CycleError.prototype).toBeInstanceOf(ComfyUIAPIWorkflowException);
    expect(CycleError.prototype).toBeInstanceOf(Error);
  });

  it("explains the rejected edge", () => {
    const [a, b, c] = makeNodes("a", "b", "c");
    const graph = chain(a, b, c);
    expectError(
      () => graph.addEdge(c, a),
      CycleError,
      "Cannot add edge 'c' -> 'a': a path from 'a' back to 'c' already exists ('a' -> 'b' -> 'c'), " +
        "so the new edge would create the cycle 'c' -> 'a' -> 'b' -> 'c'.",
    );
  });

  it("explains a self loop", () => {
    const graph = new IncrementalDirectedAcyclicGraph();
    const [node] = makeNodes("n");
    expectError(
      () => graph.addEdge(node, node),
      CycleError,
      "Cannot add edge 'n' -> 'n': a node cannot have an edge to itself.",
    );
  });

  it("exposes the rejected edge and the existing path", () => {
    const [a, b, c] = makeNodes("a", "b", "c");
    const graph = chain(a, b, c);
    const error = expectError(() => graph.addEdge(c, a), CycleError);
    expect(error.edge).toEqual([c, a]);
    expect(error.path).toEqual([a, b, c]);
    expect(error.name).toBe("CycleError");
  });

  it("drops the edges of a removed node", () => {
    const [a, b, c] = makeNodes("a", "b", "c");
    const graph = chain(a, b, c);
    graph.removeNode(b);
    expect(graph.has(b)).toBe(false);
    expect(graph.size).toBe(2);
    expect(graph.dependencies(a)).toEqual(new Set());
    expect(graph.dependents(c)).toEqual(new Set());
  });

  it("throws when removing a node it does not have", () => {
    const graph = new IncrementalDirectedAcyclicGraph();
    const [node] = makeNodes("n");
    expectError(() => graph.removeNode(node), Error, "'n' is not in the graph");
  });

  it("keeps both nodes of a removed edge", () => {
    const [a, b] = makeNodes("a", "b");
    const graph = chain(a, b);
    graph.removeEdge(a, b);
    expect(edges(graph)).toEqual(new Set());
    expect(graph.dependents(b)).toEqual(new Set());
    expect(graph.has(a) && graph.has(b)).toBe(true);
  });

  it("allows the reverse of a removed edge", () => {
    const [a, b] = makeNodes("a", "b");
    const graph = chain(a, b);
    graph.removeEdge(a, b);
    graph.addEdge(b, a);
    expect(graph.topologicalOrder()).toEqual([b, a]);
  });

  it("splices a node into an edge", () => {
    const [a, b, node] = makeNodes("a", "b", "n");
    const graph = chain(a, b);
    graph.insertBetween(a, b, node);
    expect(edges(graph)).toEqual(new Set([edge(a, node), edge(node, b)]));
    expect(graph.topologicalOrder()).toEqual([a, node, b]);
  });

  it("can keep the original edge when splicing", () => {
    const [a, b, node] = makeNodes("a", "b", "n");
    const graph = chain(a, b);
    graph.insertBetween(a, b, node, false);
    expect(edges(graph)).toEqual(new Set([edge(a, b), edge(a, node), edge(node, b)]));
  });

  it("leaves the graph unchanged when a splice would close a cycle", () => {
    const [a, b, c, node] = makeNodes("a", "b", "c", "n");
    const graph = chain(b, c, a);
    const before = structure(graph);
    expectError(() => graph.insertBetween(a, b, node), CycleError);
    expect(structure(graph)).toEqual(before);
    expect(graph.has(node)).toBe(false);
  });

  it("names the edge that closes the cycle of a splice", () => {
    const [a, b, c, node] = makeNodes("a", "b", "c", "n");
    const graph = chain(b, c, a);
    const error = expectError(() => graph.insertBetween(a, b, node), CycleError);
    expect(error.edge).toEqual([node, b]);
    expect(error.cycle).toEqual([node, b, c, a, node]);
  });

  it("removes a new end of a rejected splice", () => {
    const [a, b, m, node] = makeNodes("a", "b", "m", "n");
    const graph = chain(b, m, node);
    const before = structure(graph);
    expectError(() => graph.insertBetween(a, b, node), CycleError);
    expect(structure(graph)).toEqual(before);
    expect(graph.has(a)).toBe(false);
  });

  it("keeps an edge that was there before a rejected splice", () => {
    const [a, b, node] = makeNodes("a", "b", "n");
    const graph = new IncrementalDirectedAcyclicGraph();
    graph.addEdge(a, node);
    graph.addEdge(b, node);
    const before = structure(graph);
    expectError(() => graph.insertBetween(a, b, node), CycleError);
    expect(structure(graph)).toEqual(before);
  });

  it("returns copies of the dependencies and dependents", () => {
    const [a, b, c] = makeNodes("a", "b", "c");
    const graph = chain(a, b);
    graph.dependencies(a).add(c);
    graph.dependents(b).add(c);
    expect(graph.dependencies(a)).toEqual(new Set([b]));
    expect(graph.dependents(b)).toEqual(new Set([a]));
  });

  it("shows only the data of a node as text", () => {
    expect(String(new IncrementalDirectedAcyclicGraphNode("a"))).toBe("'a'");
    expect(String(new IncrementalDirectedAcyclicGraphNode(3))).toBe("3");
  });

  it.each([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])(
    "matches a brute force cycle check over random operations (seed %i)",
    (seed) => {
      const random = seededRandom(seed);
      const graph = new IncrementalDirectedAcyclicGraph();
      const pool = makeNodes(...Array.from({ length: 12 }, (_, index) => index));
      const byData = (x: GraphNode, y: GraphNode): number => Number(x.data) - Number(y.data);
      for (let step = 0; step < 300; step += 1) {
        const roll = random();
        if (roll < 0.55) {
          const source = pick(random, pool);
          const target = pick(random, pool);
          const before = structure(graph);
          if (hasCycle(withEdges(graph, [[source, target]]))) {
            expectError(() => graph.addEdge(source, target), CycleError);
            expect(structure(graph)).toEqual(before);
          } else {
            graph.addEdge(source, target);
            expect(graph.dependencies(source).has(target)).toBe(true);
          }
        } else if (roll < 0.75) {
          const a = pick(random, pool);
          const b = pick(random, pool);
          const node = pick(random, pool);
          const removeOriginal = random() < 0.5;
          const before = structure(graph);
          if (hasCycle(withEdges(graph, [[a, node], [node, b]]))) {
            expectError(() => graph.insertBetween(a, b, node, removeOriginal), CycleError);
            expect(structure(graph)).toEqual(before);
          } else {
            graph.insertBetween(a, b, node, removeOriginal);
            const expected = new Set([...before[1], edge(a, node), edge(node, b)]);
            if (removeOriginal) {
              expected.delete(edge(a, b));
            }
            expect(edges(graph)).toEqual(expected);
          }
        } else if (roll < 0.9) {
          const current = graph
            .topologicalOrder()
            .sort(byData)
            .flatMap((node) => [...graph.dependencies(node)].sort(byData).map((dependency) => [node, dependency] as const));
          if (current.length > 0) {
            const [node, dependency] = pick(random, current);
            graph.removeEdge(node, dependency);
          }
        } else if (graph.size > 0) {
          graph.removeNode(pick(random, graph.topologicalOrder().sort(byData)));
        }
        assertValidOrder(graph);
      }
    },
  );
});
