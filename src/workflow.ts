import { errorMessage, errorTypeName, repr, sentence, typeName } from "./_internal/format.js";
import { checkInputs, comfyName, defaultValue, isRequired } from "./_internal/inputs.js";
import { isRecord, own } from "./_internal/records.js";
import { NODE_TYPES } from "./_internal/registry.js";
import { nodeState, pipelineState } from "./_internal/state.js";
import {
  IncrementalDirectedAcyclicGraph,
  IncrementalDirectedAcyclicGraphNode,
} from "./data_structure.js";
import { CycleError } from "./errors/data_structure.js";
import {
  describeDependents,
  InvalidLinkError,
  InvalidNodeChangeError,
  InvalidNodeParameterValue,
  NodeHasDependentsError,
  NodeNotFoundError,
  quoteNames,
} from "./errors/node.js";
import { PipelineError } from "./errors/pipeline.js";
import { InvalidWorkflowError } from "./errors/workflow.js";
import type { BaseNode, ComfyUIWorkflow, NodeArgs, NodeOutput, NodeType } from "./nodes/base.js";
import { acceptsLink, isLink, validateType, type Link } from "./nodes/validation.js";
import { Pipeline, type PipelineArgs, type PipelineType } from "./pipeline.js";

type GraphNode = IncrementalDirectedAcyclicGraphNode;

const ANY_SOCKETS = new Set(["*", "COMFY_MATCHTYPE_V3"]);

function idOf(graphNode: GraphNode): string {
  return graphNode.data as string;
}

function typeOf(node: BaseNode): NodeType {
  return node.constructor as NodeType;
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function field<T>(
  nodeId: string,
  data: Record<string, unknown>,
  key: string,
  kind: string,
  accepts: (value: unknown) => value is T,
): T {
  if (!Object.hasOwn(data, key)) {
    throw new InvalidWorkflowError(nodeId, `${repr(key)} is missing`);
  }
  const value = data[key];
  if (!accepts(value)) {
    throw new InvalidWorkflowError(nodeId, `${repr(key)} must be ${kind}, got ${typeName(value)}`);
  }
  return value;
}

function loadValue(nodeId: string, name: string, value: unknown): unknown {
  if (isRecord(value) && Object.hasOwn(value, "__value__")) {
    return value["__value__"];
  }
  if (Array.isArray(value)) {
    if (!isLink(value)) {
      const reason = `input ${repr(name)} is an array but not a link [node_id, output_index]`;
      throw new InvalidWorkflowError(nodeId, reason);
    }
    return [value[0], value[1]];
  }
  return value;
}

function loadNode(nodeId: string, data: unknown): BaseNode {
  if (!isRecord(data)) {
    const reason = `expected an object with 'class_type' and 'inputs', got ${typeName(data)}`;
    throw new InvalidWorkflowError(nodeId, reason);
  }
  const classType = field(nodeId, data, "class_type", "a string", isString);
  const inputs = field(nodeId, data, "inputs", "an object", isRecord);
  const type = NODE_TYPES.get(classType);
  if (type === undefined) {
    throw new InvalidWorkflowError(nodeId, `unknown class_type ${repr(classType)}`);
  }
  // ComfyUI input name -> input name.
  const names = new Map(Object.keys(type.INPUTS).map((name) => [comfyName(type, name), name]));
  const given: Record<string, unknown> = {};
  for (const [name, value] of Object.entries(inputs)) {
    const input = names.get(name);
    if (input === undefined) {
      throw new InvalidWorkflowError(nodeId, `${classType} has no input ${repr(name)}`);
    }
    given[input] = loadValue(nodeId, name, value);
  }
  const missing = [...names]
    .filter(([, input]) => !Object.hasOwn(given, input) && isRequired(type.INPUTS[input]!))
    .map(([name]) => name);
  if (missing.length > 0) {
    const noun = missing.length === 1 ? "input" : "inputs";
    throw new InvalidWorkflowError(nodeId, `${classType} is missing the ${noun} ${quoteNames(missing)}`);
  }
  try {
    return new type(nodeId, given);
  } catch (error) {
    if (error instanceof InvalidNodeParameterValue) {
      throw new InvalidWorkflowError(nodeId, error.message, { cause: error });
    }
    throw error;
  }
}

function socketsMatch(outputType: string, inputType: string): boolean {
  const outputs = new Set(outputType.split(",").map((part) => part.trim()));
  const inputs = new Set(inputType.split(",").map((part) => part.trim()));
  for (const part of [...outputs, ...inputs]) {
    if (ANY_SOCKETS.has(part)) {
      return true;
    }
  }
  return [...outputs].some((part) => inputs.has(part));
}

function missingOutput(source: BaseNode, index: number): string | null {
  const sourceType = typeOf(source);
  if (!Object.values(sourceType.OUTPUTS).includes(index)) {
    return (
      `output ${index} of node ${repr(source.nodeId)}, ` +
      `but ${sourceType.CLASS_TYPE} has no output ${index}`
    );
  }
  return null;
}

function linkProblem(source: BaseNode, index: number, node: BaseNode, name: string): string | null {
  const missing = missingOutput(source, index);
  if (missing !== null) {
    return missing;
  }
  const sourceType = typeOf(source);
  const inputType = own(typeOf(node).INPUT_SOCKET_TYPES, name);
  const outputType = sourceType.RETURN_TYPES[index];
  if (inputType === undefined || outputType === undefined) {
    return null;
  }
  if (socketsMatch(outputType, inputType)) {
    return null;
  }
  return (
    `output ${index} of node ${repr(source.nodeId)}, which is ${outputType}, ` +
    `but the input takes ${inputType}`
  );
}

function* linksOf(node: BaseNode): Generator<[name: string, sourceId: string, index: number]> {
  for (const [name, value] of Object.entries(node.inputs())) {
    if (isLink(value)) {
      yield [name, value[0], value[1]];
    }
  }
}

function checkLink(
  nodes: ReadonlyMap<string, BaseNode>,
  node: BaseNode,
  name: string,
  sourceId: string,
  index: number,
): void {
  const nodeId = node.nodeId;
  const shown = comfyName(typeOf(node), name);
  if (sourceId === nodeId) {
    throw new InvalidWorkflowError(nodeId, `input ${repr(shown)} uses the node's own output`);
  }
  const source = nodes.get(sourceId);
  if (source === undefined) {
    const reason = `input ${repr(shown)} uses node ${repr(sourceId)}, which is not in the workflow`;
    throw new InvalidWorkflowError(nodeId, reason);
  }
  const problem = linkProblem(source, index, node, name);
  if (problem !== null) {
    throw new InvalidWorkflowError(nodeId, `input ${repr(shown)} uses ${problem}`);
  }
}

function newPipeline(type: unknown): [string, Pipeline] {
  if (type instanceof Pipeline) {
    const reason = "addPipeline takes the pipeline class, not an instance";
    throw new PipelineError(type.constructor.name, reason);
  }
  if (typeof type !== "function") {
    throw new PipelineError(typeName(type), "addPipeline takes a Pipeline subclass");
  }
  if (!(type === Pipeline || type.prototype instanceof Pipeline)) {
    throw new PipelineError(type.name, "addPipeline takes a Pipeline subclass");
  }
  const pipelineType = type as PipelineType;
  if (pipelineType.prototype.build === Pipeline.prototype.build) {
    throw new PipelineError(type.name, "it does not define build");
  }
  return [type.name, new pipelineType()];
}

function checkParameters(
  name: string,
  type: PipelineType,
  given: unknown,
  nodes: ReadonlyMap<string, BaseNode>,
): Record<string, unknown> {
  const parameters = given ?? {};
  if (!isRecord(parameters)) {
    throw new PipelineError(name, `its parameters must be an object, not ${typeName(parameters)}`);
  }
  const schema = type.PARAMETERS;
  if (schema === undefined) {
    return { ...parameters };
  }
  const named = Object.keys(schema);
  for (const key of Object.keys(parameters)) {
    if (!Object.hasOwn(schema, key)) {
      const takes = named.length > 0 ? `it takes ${quoteNames(named)}` : "it takes no parameters";
      throw new PipelineError(name, `it has no parameter ${repr(key)} (${takes})`);
    }
  }
  const missing = named.filter((key) => isRequired(schema[key]!) && own(parameters, key) === undefined);
  if (missing.length > 0) {
    const noun = missing.length === 1 ? "parameter" : "parameters";
    throw new PipelineError(name, `it is missing the ${noun} ${quoteNames(missing)}`);
  }
  const checked: [string, unknown][] = [];
  for (const [key, spec] of Object.entries(schema)) {
    const value = own(parameters, key);
    if (value === undefined || (value === null && spec.optional)) {
      if (spec.hasDefault) {
        checked.push([key, defaultValue(spec)]);
      }
      continue;
    }
    try {
      validateType(key, value, spec);
    } catch (error) {
      if (error instanceof InvalidNodeParameterValue) {
        throw new PipelineError(name, error.message, { cause: error });
      }
      throw error;
    }
    if (acceptsLink(spec) && isLink(value)) {
      const [sourceId, index] = value;
      const source = nodes.get(sourceId);
      if (source === undefined) {
        const reason = `parameter ${repr(key)} uses node ${repr(sourceId)}, which is not in the workflow`;
        throw new PipelineError(name, reason);
      }
      const problem = missingOutput(source, index);
      if (problem !== null) {
        throw new PipelineError(name, `parameter ${repr(key)} uses ${problem}`);
      }
    }
    checked.push([key, value]);
  }
  return Object.fromEntries(checked);
}

function pipelineError(name: string, pipeline: Pipeline, error: unknown): PipelineError {
  const state = pipelineState(pipeline);
  const rejection = state.rejection;
  if (rejection !== null) {
    const reason =
      `node ${rejection.position} of the pipeline (${rejection.type.name}) was rejected: ` +
      sentence(rejection.error);
    return new PipelineError(name, reason, { cause: rejection.error });
  }
  if (error instanceof InvalidNodeChangeError && state.nodes.has(error.nodeId)) {
    const position = [...state.nodes.keys()].indexOf(error.nodeId) + 1;
    const changed = state.nodes.get(error.nodeId)!.constructor.name;
    const reason =
      `a change to node ${position} of the pipeline (${changed}) was rejected: ` + sentence(error);
    return new PipelineError(name, reason, { cause: error });
  }
  let reason = `its build failed with ${errorTypeName(error)}`;
  if (errorMessage(error) !== "") {
    reason += `: ${sentence(error)}`;
  }
  return new PipelineError(name, reason, { cause: error });
}

export class Workflow {
  #nodes = new Map<string, BaseNode>();
  #nextIndex = 0;
  #graph = new IncrementalDirectedAcyclicGraph();
  #graphNodes = new Map<string, GraphNode>();

  /** The nodes by id, in the order they were added. */
  get nodes(): ReadonlyMap<string, BaseNode> {
    return this.#nodes;
  }

  getNodeOutput(nodeId: string): NodeOutput {
    const node = this.#nodes.get(nodeId);
    if (node === undefined) {
      throw new NodeNotFoundError(`node: ${nodeId} not found`);
    }
    return node.output;
  }

  /** The ids of the nodes whose outputs the node uses. */
  dependencies(nodeId: string): Set<string> {
    return new Set([...this.#graph.dependencies(this.#graphNode(nodeId))].map(idOf));
  }

  /** The ids of the nodes that use the outputs of the node. */
  dependents(nodeId: string): Set<string> {
    return new Set([...this.#graph.dependents(this.#graphNode(nodeId))].map(idOf));
  }

  addNode<T extends NodeType>(type: T, ...args: NodeArgs<T>): InstanceType<T> {
    const nodeId = this.#newNodeId(type);
    const node = new type(nodeId, ...args) as InstanceType<T>;
    const graphNode = new IncrementalDirectedAcyclicGraphNode(nodeId);
    this.#graph.addNode(graphNode);
    this.#graphNodes.set(nodeId, graphNode);
    try {
      this.#checkLinks(node);
      this.#link(graphNode, node);
    } catch (error) {
      this.#graph.removeNode(graphNode);
      this.#graphNodes.delete(nodeId);
      throw error;
    }
    this.#nodes.set(nodeId, node);
    this.#watch(graphNode, node);
    return node;
  }

  /** Adds every node of the pipeline, or none if any is rejected. */
  addPipeline<P extends PipelineType>(type: P, ...args: PipelineArgs<P>): InstanceType<P> {
    const [name, pipeline] = newPipeline(type);
    const [given] = args as unknown[];
    const parameters = checkParameters(name, type, given, this.#nodes);
    const nextIndex = this.#nextIndex;
    const state = pipelineState(pipeline);
    state.workflow = this;
    try {
      const outputs: unknown = pipeline.build(parameters);
      if (state.rejection !== null) {
        throw state.rejection.error;
      }
      state.outputs = this.#pipelineOutputs(name, outputs);
    } catch (error) {
      const failure = error instanceof PipelineError ? error : pipelineError(name, pipeline, error);
      for (const nodeId of [...state.nodes.keys()].reverse()) {
        this.removeNode(nodeId);
      }
      state.nodes.clear();
      this.#nextIndex = nextIndex;
      throw failure;
    } finally {
      state.workflow = null;
    }
    return pipeline as InstanceType<P>;
  }

  /** Changes several inputs of a node at once: the node is validated after all of them. */
  updateNode(nodeId: string, inputs: Readonly<Record<string, unknown>>): void {
    const node = this.#nodes.get(nodeId);
    if (node === undefined) {
      throw new NodeNotFoundError(`node: ${nodeId} not found`);
    }
    const type = typeOf(node);
    const state = nodeState(node);
    const before = state.values;
    state.batch = true;
    try {
      for (const [name, value] of Object.entries(inputs)) {
        if (!Object.hasOwn(type.INPUTS, name)) {
          throw new InvalidNodeChangeError(nodeId, name, `${type.name} has no input ${repr(name)}`);
        }
        (node as unknown as Record<string, unknown>)[name] = value;
      }
      checkInputs(type, state.values);
    } catch (error) {
      state.values = before;
      this.#link(this.#graphNode(nodeId), node);
      throw error;
    } finally {
      state.batch = false;
    }
  }

  removeNode(nodeId: string): void {
    const dependents = this.#dependentsInOrder(nodeId);
    if (dependents.length > 0) {
      throw new NodeHasDependentsError(nodeId, dependents);
    }
    this.#graph.removeNode(this.#graphNode(nodeId));
    this.#graphNodes.delete(nodeId);
    const node = this.#nodes.get(nodeId)!;
    this.#nodes.delete(nodeId);
    nodeState(node).onChange = null;
  }

  /** Replaces the nodes with the ones of a workflow in the API format, if all of them load. */
  loadWorkflow(workflow: unknown): void {
    if (!isRecord(workflow)) {
      throw new InvalidWorkflowError(null, `expected an object of nodes, got ${typeName(workflow)}`);
    }
    if (Array.isArray(workflow["nodes"]) && Array.isArray(workflow["links"])) {
      const reason =
        "it is in the ComfyUI UI format; export it with 'Export (API)' and load that file";
      throw new InvalidWorkflowError(null, reason);
    }
    const nodes = new Map<string, BaseNode>();
    for (const [nodeId, data] of Object.entries(workflow)) {
      nodes.set(nodeId, loadNode(nodeId, data));
    }
    const graph = new IncrementalDirectedAcyclicGraph();
    const graphNodes = new Map<string, GraphNode>();
    for (const nodeId of nodes.keys()) {
      const graphNode = new IncrementalDirectedAcyclicGraphNode(nodeId);
      graph.addNode(graphNode);
      graphNodes.set(nodeId, graphNode);
    }
    for (const [nodeId, node] of nodes) {
      for (const [name, sourceId, index] of linksOf(node)) {
        checkLink(nodes, node, name, sourceId, index);
        try {
          graph.addEdge(graphNodes.get(nodeId)!, graphNodes.get(sourceId)!);
        } catch (error) {
          if (!(error instanceof CycleError)) {
            throw error;
          }
          const reason =
            `input ${repr(comfyName(typeOf(node), name))} uses node ${repr(sourceId)}, ` +
            `which depends on ${repr(nodeId)}, ` +
            `so the workflow has the cycle ${error.cycle.join(" -> ")}`;
          throw new InvalidWorkflowError(nodeId, reason, { cause: error });
        }
      }
    }
    for (const node of this.#nodes.values()) {
      nodeState(node).onChange = null;
    }
    this.#nodes = nodes;
    this.#graph = graph;
    this.#graphNodes = graphNodes;
    this.#nextIndex = nodes.size;
    for (const [nodeId, node] of nodes) {
      this.#watch(graphNodes.get(nodeId)!, node);
    }
  }

  getComfyUIWorkflowFormat(): ComfyUIWorkflow {
    return Object.fromEntries(
      [...this.#nodes.values()].flatMap((node) => Object.entries(node.getComfyUIWorkflowFormat())),
    );
  }

  #newNodeId(type: NodeType): string {
    for (;;) {
      const nodeId = `${type.name}_${this.#nextIndex}`;
      this.#nextIndex += 1;
      if (!this.#nodes.has(nodeId)) {
        return nodeId;
      }
    }
  }

  #watch(graphNode: GraphNode, node: BaseNode): void {
    nodeState(node).onChange = (changed) => {
      this.#checkChange(graphNode, changed);
    };
  }

  #pipelineOutputs(name: string, outputs: unknown): Map<string, Link> {
    const links = new Map<string, Link>();
    if (outputs === undefined || outputs === null) {
      return links;
    }
    if (!isRecord(outputs)) {
      const reason = `its build must return an object of outputs or undefined, not ${typeName(outputs)}`;
      throw new PipelineError(name, reason);
    }
    for (const [output, value] of Object.entries(outputs)) {
      if (!isLink(value)) {
        const reason = `its output ${repr(output)} must be a link [nodeId, outputIndex], not ${typeName(value)}`;
        throw new PipelineError(name, reason);
      }
      const [sourceId, index] = value;
      const source = this.#nodes.get(sourceId);
      if (source === undefined) {
        const reason = `its output ${repr(output)} uses node ${repr(sourceId)}, which is not in the workflow`;
        throw new PipelineError(name, reason);
      }
      const missing = missingOutput(source, index);
      if (missing !== null) {
        throw new PipelineError(name, `its output ${repr(output)} uses ${missing}`);
      }
      links.set(output, Object.freeze([sourceId, index] as const));
    }
    return links;
  }

  #checkLinks(node: BaseNode): void {
    for (const [name, sourceId, index] of linksOf(node)) {
      const source = this.#nodes.get(sourceId);
      if (source === undefined) {
        continue;
      }
      const problem = linkProblem(source, index, node, name);
      if (problem !== null) {
        throw new InvalidLinkError(node.nodeId, name, problem);
      }
    }
  }

  #graphNode(nodeId: string): GraphNode {
    const graphNode = this.#graphNodes.get(nodeId);
    if (graphNode === undefined) {
      throw new NodeNotFoundError(`node: ${nodeId} not found`);
    }
    return graphNode;
  }

  #dependentsInOrder(nodeId: string): string[] {
    const dependents = this.dependents(nodeId);
    return [...this.#nodes.keys()].filter((otherId) => dependents.has(otherId));
  }

  #checkChange(graphNode: GraphNode, node: BaseNode): void {
    const nodeId = idOf(graphNode);
    if (node.nodeId !== nodeId) {
      const dependents = this.#dependentsInOrder(nodeId);
      const reason =
        dependents.length > 0
          ? describeDependents(dependents)
          : "a node keeps its id while it is in a workflow";
      throw new InvalidNodeChangeError(nodeId, "nodeId", reason);
    }
    const links = new Map<string, Link>();
    for (const [name, sourceId, index] of linksOf(node)) {
      links.set(name, [sourceId, index]);
    }
    for (const [name, [sourceId, index]] of links) {
      const source = this.#nodes.get(sourceId);
      if (!this.#graphNodes.has(sourceId) || source === undefined) {
        throw new InvalidNodeChangeError(nodeId, name, `node ${repr(sourceId)} is not in the workflow`);
      }
      const problem = linkProblem(source, index, node, name);
      if (problem !== null) {
        throw new InvalidNodeChangeError(nodeId, name, `it would use ${problem}`);
      }
    }
    try {
      this.#link(graphNode, node);
    } catch (error) {
      if (!(error instanceof CycleError)) {
        throw error;
      }
      const sourceId = idOf(error.edge[1]);
      const name = [...links].find(([, [linked]]) => linked === sourceId)![0];
      const reason =
        sourceId === nodeId
          ? "a node cannot use its own output"
          : `node ${repr(sourceId)} depends on ${repr(nodeId)}, ` +
            `so the change would create the cycle ${error.cycle.join(" -> ")}`;
      throw new InvalidNodeChangeError(nodeId, name, reason, { cause: error });
    }
  }

  #link(graphNode: GraphNode, node: BaseNode): void {
    const dependencies = new Set<GraphNode>();
    for (const [, sourceId] of linksOf(node)) {
      dependencies.add(this.#graphNode(sourceId));
    }
    for (const dependency of dependencies) {
      this.#graph.addEdge(graphNode, dependency);
    }
    for (const dependency of this.#graph.dependencies(graphNode)) {
      if (!dependencies.has(dependency)) {
        this.#graph.removeEdge(graphNode, dependency);
      }
    }
  }
}
