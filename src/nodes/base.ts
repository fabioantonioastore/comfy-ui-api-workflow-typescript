import { repr } from "../_internal/format.js";
import {
  checkInputs,
  comfyUIFormat,
  defaultValue,
  inputRecord,
  inputValue,
  isFactory,
  NOT_GIVEN,
  readInputs,
  type NodeSchema,
} from "../_internal/inputs.js";
import { outputProxy } from "../_internal/output.js";
import { own } from "../_internal/records.js";
import { findNodeState, nodeState, setNodeState, type NodeState } from "../_internal/state.js";
import { InvalidNodeParameterValue } from "../errors/node.js";
import { OutputNotFoundError } from "../errors/output.js";
import type { InputSpec, TypeSpec } from "./schema.js";
import { validateType, type Link } from "./validation.js";

export type { Link } from "./validation.js";

/** A node in the API format ComfyUI runs. */
export interface ComfyUINode {
  class_type: string;
  inputs: Record<string, unknown>;
}

/** A workflow in the API format: each node by its id. */
export type ComfyUIWorkflow = Record<string, ComfyUINode>;

/** What a node constructor takes after the node id: its inputs, optional when none is required. */
export type InputsArgs<I> = {} extends I ? [inputs?: I] : [inputs: I];

/** A node class: its constructor and what it tells about the ComfyUI node. */
export interface NodeType<N extends BaseNode = BaseNode> extends NodeSchema {
  new (nodeId: string, ...args: any[]): N;
  readonly prototype: N;
  readonly CATEGORY: string;
  readonly OUTPUTS: Readonly<Record<string, number>>;
  readonly RETURN_TYPES: readonly string[];
  readonly INPUT_SOCKET_TYPES: Readonly<Record<string, string>>;
}

/** The arguments a node class takes after the node id. */
export type NodeArgs<T> = T extends new (nodeId: string, ...args: infer A extends unknown[]) => BaseNode
  ? A
  : never;

/** The outputs of a node, each read as the link to it: output.LATENT or output["Audio VAE"]. */
export type NodeOutput<O extends string = string> = { readonly [K in O]: Link };

export const NodeOutput = class NodeOutput {
  readonly nodeId: string;
  readonly outputs: Readonly<Record<string, number>>;

  constructor(nodeId: string, outputs: Readonly<Record<string, number>>) {
    this.nodeId = nodeId;
    this.outputs = outputs;
    return outputProxy(
      this,
      (name) => {
        const index = own(outputs, name);
        return index === undefined ? undefined : Object.freeze([nodeId, index] as const);
      },
      (name) => new OutputNotFoundError(`node: ${nodeId} don't have the output: ${name}`),
    );
  }
} as unknown as new <O extends string = string>(
  nodeId: string,
  outputs: Readonly<Record<O, number>>,
) => NodeOutput<O>;

const NODE_ID: TypeSpec = { kinds: ["string"], options: null, optional: false };

// The input accessors go on the prototype of each class the first time it
// builds a node, so the hundreds of classes cost nothing until they are used.
const installed = new WeakSet<object>();

function install(type: NodeType): void {
  const prototype: object = type.prototype;
  if (installed.has(prototype)) {
    return;
  }
  for (const name of Object.keys(type.INPUTS)) {
    if (name in BaseNode.prototype) {
      throw new TypeError(`${type.name} cannot have an input named ${repr(name)}: nodes use that name.`);
    }
    if (!Object.hasOwn(prototype, name)) {
      Object.defineProperty(prototype, name, {
        configurable: true,
        get(this: object): unknown {
          return findNodeState(this)?.values.get(name);
        },
        set(this: BaseNode, value: unknown) {
          setInput(this, name, value);
        },
      });
    }
  }
  installed.add(prototype);
}

// The type check runs before the value is stored, so a wrong type never gets
// in, whether at construction or in a later change. A rejected change is
// undone, so the node stays as it was.
function commit(
  node: BaseNode,
  state: NodeState,
  type: NodeType,
  values: ReadonlyMap<string, unknown>,
): void {
  if (!state.batch) {
    checkInputs(type, values);
  }
  const previous = state.values;
  state.values = values;
  if (state.onChange === null) {
    return;
  }
  try {
    state.onChange(node);
  } catch (error) {
    state.values = previous;
    throw error;
  }
}

// Setting undefined removes an input, like Python's del; so does null when
// it stands for "not given". An input that is always sent cannot be removed.
function setInput(node: BaseNode, name: string, value: unknown): void {
  const state = nodeState(node);
  const type = node.constructor as NodeType;
  const spec: InputSpec = type.INPUTS[name]!;
  const values = new Map(state.values);
  if (value === null && isFactory(spec)) {
    values.set(name, defaultValue(spec));
  } else {
    const stored = inputValue(name, spec, value);
    if (stored !== NOT_GIVEN) {
      values.set(name, stored);
    } else if (!spec.optional) {
      throw new InvalidNodeParameterValue(`${name} cannot be removed because it is not optional`);
    } else if (!values.delete(name)) {
      return;
    }
  }
  commit(node, state, type, values);
}

/**
 * A ComfyUI node. Each class declares its inputs in INPUTS, and each input
 * is a property of the node, checked whenever it changes.
 */
export abstract class BaseNode<I extends object = object, O extends string = string> {
  /** The node type ComfyUI knows: the class name, unless the class sets another. */
  static get CLASS_TYPE(): string {
    return this.name;
  }

  static readonly CATEGORY: string = "";
  static readonly OUTPUTS: Readonly<Record<string, number>> = {};
  static readonly INPUT_NAMES: Readonly<Record<string, string>> = {};
  // ComfyUI socket types, as in object_info: RETURN_TYPES by output index and
  // INPUT_SOCKET_TYPES by input name. A node without them is not type checked.
  // A V1 combo, whose type in object_info is its list of options, is "V1 COMBO":
  // no output has that type, so as in ComfyUI only * and match type outputs fit.
  static readonly RETURN_TYPES: readonly string[] = [];
  static readonly INPUT_SOCKET_TYPES: Readonly<Record<string, string>> = {};
  /** The inputs, in the order they are sent. */
  static readonly INPUTS: Readonly<Record<string, InputSpec>> = {};

  /** Checks the set inputs together, once each one passed its own checks. */
  static validate(_inputs: Readonly<Record<string, unknown>>): void {}

  constructor(nodeId: string, ...args: InputsArgs<I>) {
    const type = new.target as unknown as NodeType;
    validateType("nodeId", nodeId, NODE_ID);
    install(type);
    const values = readInputs(type, args[0] ?? {});
    checkInputs(type, values);
    setNodeState(this, { nodeId, values, onChange: null, batch: false });
    Object.preventExtensions(this);
  }

  get nodeId(): string {
    return nodeState(this).nodeId;
  }

  set nodeId(nodeId: string) {
    validateType("nodeId", nodeId, NODE_ID);
    const state = nodeState(this);
    const previous = state.nodeId;
    state.nodeId = nodeId;
    if (state.onChange === null) {
      return;
    }
    try {
      state.onChange(this);
    } catch (error) {
      state.nodeId = previous;
      throw error;
    }
  }

  get output(): NodeOutput<O> {
    const type = this.constructor as NodeType;
    return new NodeOutput<O>(this.nodeId, type.OUTPUTS as Readonly<Record<O, number>>);
  }

  /** The inputs that are set, in the order the class declares them. */
  inputs(): Partial<I> {
    return inputRecord(this.constructor as NodeType, nodeState(this).values) as Partial<I>;
  }

  getComfyUIWorkflowFormat(): ComfyUIWorkflow {
    const state = nodeState(this);
    return comfyUIFormat(this.constructor as NodeType, state.nodeId, state.values);
  }
}
