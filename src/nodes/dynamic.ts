import { repr } from "../_internal/format.js";
import { comfyUIFormat } from "../_internal/inputs.js";
import { own } from "../_internal/records.js";
import { InvalidNodeParameterValue } from "../errors/node.js";
import { BaseNode, type ComfyUIWorkflow, type NodeType } from "./base.js";
import type { Option } from "./schema.js";
import { validateBool, validateChoice, validateNotEmpty, validateRange } from "./validation.js";

/** The set inputs of a node, by name. */
type Inputs = Readonly<Record<string, unknown>>;

export type Check = (name: string, value: unknown) => void;

export interface ValueOptions {
  readonly default?: unknown;
  readonly required?: boolean;
  readonly check?: Check | null;
}

// An input that only exists inside a branch of a dynamic combo or a row of a
// dynamic group. One with a default is sent with it when not given, as the
// ComfyUI frontend does; ComfyUI rejects a prompt missing a required input.
export class Value {
  readonly hasDefault: boolean;
  readonly default: unknown;
  readonly required: boolean;
  readonly check: Check | null;

  constructor(options: ValueOptions = {}) {
    this.hasDefault = options.default !== undefined;
    this.default = options.default;
    this.required = options.required ?? true;
    this.check = options.check ?? null;
  }

  validate(name: string, value: unknown): void {
    this.check?.(name, value);
  }
}

export interface RequiredOptions {
  readonly required?: boolean;
}

export function number(
  options: RequiredOptions & { readonly default?: number; readonly low?: number; readonly high?: number } = {},
): Value {
  const { low = null, high = null } = options;
  return new Value({
    default: options.default,
    required: options.required,
    check: (name, value) => validateRange(name, value, low, high),
  });
}

export function choice(
  options: readonly Option[],
  settings: RequiredOptions & { readonly default?: Option } = {},
): Value {
  const choices = [...options];
  return new Value({
    default: settings.default,
    required: settings.required,
    check: (name, value) => validateChoice(name, value, choices),
  });
}

export function serverChoice(options: RequiredOptions = {}): Value {
  return new Value({ required: options.required, check: validateNotEmpty });
}

export function flag(options: RequiredOptions & { readonly default?: boolean } = {}): Value {
  return new Value({ default: options.default, required: options.required, check: validateBool });
}

export function text(options: RequiredOptions & { readonly default?: string } = {}): Value {
  return new Value({ default: options.default, required: options.required });
}

export function link(options: RequiredOptions = {}): Value {
  return new Value({ required: options.required });
}

function oneOf(keys: readonly string[]): string {
  return keys.length === 1 ? repr(keys[0]) : `one of ${repr(keys)}`;
}

// Set values only: null counts as not given, as Python's None does.
function given(inputs: Inputs, attribute: string): unknown {
  return own(inputs, attribute) ?? null;
}

export type Branch = Readonly<Record<string, Value | Combo>>;

// A dynamic combo: each option opens its own inputs, sent as "<combo>.<input>"
// and set here as <combo>_<input>. An attribute can be in several branches, each
// with its own default and checks; the first option is the default one.
export class Combo {
  readonly branches: Readonly<Record<string, Branch>>;

  constructor(branches: Readonly<Record<string, Branch>>) {
    this.branches = branches;
  }

  get default(): string {
    const [first] = Object.keys(this.branches);
    if (first === undefined) {
      throw new RangeError("A combo without options has no default.");
    }
    return first;
  }

  attributes(branch: string | null = null): Set<string> {
    const keys = branch === null ? Object.keys(this.branches) : [branch];
    const found = new Set<string>();
    for (const key of keys) {
      for (const [name, spec] of Object.entries(own(this.branches, key) ?? {})) {
        found.add(name);
        if (spec instanceof Combo) {
          for (const attribute of spec.attributes()) {
            found.add(attribute);
          }
        }
      }
    }
    return found;
  }

  validate(inputs: Inputs, name: string, value: unknown): void {
    validateChoice(name, value, Object.keys(this.branches));
    const chosen = value as string;
    const inBranch = this.attributes(chosen);
    const others = [...this.attributes()].filter((attribute) => !inBranch.has(attribute)).sort();
    for (const attribute of others) {
      if (given(inputs, attribute) !== null) {
        const keys = Object.keys(this.branches).filter((key) => this.attributes(key).has(attribute));
        throw new InvalidNodeParameterValue(`${attribute} only applies when ${name} is ${oneOf(keys)}`);
      }
    }
    for (const [attribute, spec] of Object.entries(this.branches[chosen] ?? {})) {
      const set = given(inputs, attribute);
      if (spec instanceof Combo) {
        spec.validate(inputs, attribute, set ?? spec.default);
      } else if (set !== null) {
        spec.validate(attribute, set);
      } else if (spec.required && !spec.hasDefault) {
        throw new InvalidNodeParameterValue(`${attribute} is required when ${name} is ${repr(chosen)}`);
      }
    }
  }

  *defaults(inputs: Inputs, value: unknown): Generator<[string, unknown]> {
    const branch = typeof value === "string" ? own(this.branches, value) : undefined;
    for (const [attribute, spec] of Object.entries(branch ?? {})) {
      const set = given(inputs, attribute);
      if (spec instanceof Combo) {
        const chosen = set ?? spec.default;
        if (set === null) {
          yield [attribute, chosen];
        }
        yield* spec.defaults(inputs, chosen);
      } else if (set === null && spec.hasDefault) {
        yield [attribute, spec.default];
      }
    }
  }
}

// A dynamic group: rows of the same fields, sent as "<group>.<row>.<field>" and
// set here as <group>_<row>_<field>. A row is used when any of its fields is set;
// the rows a node needs at least have their required fields as inputs that are
// not optional, so they are always there.
export class Group {
  readonly rows: number;
  readonly fields: Readonly<Record<string, Value>>;

  constructor(rows: number, fields: Readonly<Record<string, Value>>) {
    this.rows = rows;
    this.fields = fields;
  }

  attribute(name: string, row: number, field: string): string {
    return `${name}_${row}_${field}`;
  }

  usedRows(inputs: Inputs, name: string): number[] {
    const used: number[] = [];
    for (let row = 0; row < this.rows; row += 1) {
      if (Object.keys(this.fields).some((field) => given(inputs, this.attribute(name, row, field)) !== null)) {
        used.push(row);
      }
    }
    return used;
  }

  validate(inputs: Inputs, name: string): void {
    for (const row of this.usedRows(inputs, name)) {
      for (const [field, spec] of Object.entries(this.fields)) {
        const attribute = this.attribute(name, row, field);
        const value = given(inputs, attribute);
        if (value !== null) {
          spec.validate(attribute, value);
        } else if (spec.required && !spec.hasDefault) {
          throw new InvalidNodeParameterValue(`${attribute} is required in row ${row} of ${name}`);
        }
      }
    }
  }

  *defaults(inputs: Inputs, name: string): Generator<[string, unknown]> {
    for (const row of this.usedRows(inputs, name)) {
      for (const [field, spec] of Object.entries(this.fields)) {
        const attribute = this.attribute(name, row, field);
        if (given(inputs, attribute) === null && spec.hasDefault) {
          yield [attribute, spec.default];
        }
      }
    }
  }
}

export function validateDynamic(
  dynamicInputs: Readonly<Record<string, Combo | Group>>,
  inputs: Inputs,
): void {
  for (const [name, spec] of Object.entries(dynamicInputs)) {
    if (spec instanceof Group) {
      spec.validate(inputs, name);
    } else {
      spec.validate(inputs, name, own(inputs, name));
    }
  }
}

// The base of the nodes with dynamic combos or groups: the ComfyUI format also
// carries the defaults of the chosen branches and rows, which are not stored,
// so changing a branch never leaves stale inputs behind.
export abstract class DynamicInputs<I extends object = object, O extends string = string> extends BaseNode<I, O> {
  static readonly DYNAMIC_INPUTS: Readonly<Record<string, Combo | Group>> = {};

  static override validate(inputs: Inputs): void {
    validateDynamic(this.DYNAMIC_INPUTS, inputs);
  }

  override getComfyUIWorkflowFormat(): ComfyUIWorkflow {
    const type = this.constructor as NodeType & typeof DynamicInputs;
    const inputs = this.inputs() as Inputs;
    const values = new Map(Object.entries(inputs));
    for (const [name, spec] of Object.entries(type.DYNAMIC_INPUTS)) {
      const defaults = spec instanceof Group ? spec.defaults(inputs, name) : spec.defaults(inputs, own(inputs, name));
      for (const [attribute, value] of defaults) {
        values.set(attribute, value);
      }
    }
    return comfyUIFormat(type, this.nodeId, values);
  }
}
