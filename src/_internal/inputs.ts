// How a node class reads, checks and sends its inputs, as its schema says.
import { InvalidNodeParameterValue, quoteNames } from "../errors/node.js";
import type { ComfyUIWorkflow } from "../nodes/base.js";
import type { InputSpec } from "../nodes/schema.js";
import {
  isLink,
  validateBool,
  validateChoice,
  validateNotEmpty,
  validateRange,
  validateType,
} from "../nodes/validation.js";
import { repr, typeName } from "./format.js";
import { deepCopy, isRecord, own } from "./records.js";

/** The static side of a node class, as far as its inputs go. */
export interface NodeSchema {
  readonly name: string;
  readonly CLASS_TYPE: string;
  readonly INPUTS: Readonly<Record<string, InputSpec>>;
  readonly INPUT_NAMES: Readonly<Record<string, string>>;
  validate(inputs: Readonly<Record<string, unknown>>): void;
}

export const NOT_GIVEN: unique symbol = Symbol("not given");

export function isRequired(spec: InputSpec): boolean {
  return !spec.optional && !spec.hasDefault;
}

// A Python constructor fills these in with `{} if value is None else value`.
export function isFactory(spec: InputSpec): boolean {
  return spec.hasDefault && typeof spec.default === "object" && spec.default !== null;
}

export function defaultValue(spec: InputSpec): unknown {
  return isFactory(spec) ? deepCopy(spec.default) : spec.default;
}

/**
 * What is stored for a value given to an input, after its type check: a link
 * as a frozen copy, and NOT_GIVEN where undefined or null mean leaving it out.
 */
export function inputValue(name: string, spec: InputSpec, value: unknown): unknown {
  if (value === undefined || (value === null && (spec.optional || isFactory(spec)))) {
    return NOT_GIVEN;
  }
  validateType(name, value, spec);
  return isLink(value) ? Object.freeze([value[0], value[1]] as const) : value;
}

/** Reads the inputs given to a constructor: each one is type checked, and defaults fill the rest. */
export function readInputs(type: NodeSchema, given: unknown): Map<string, unknown> {
  if (!isRecord(given)) {
    throw new InvalidNodeParameterValue(
      `${type.name} takes its inputs in an object, not ${typeName(given)}`,
    );
  }
  for (const name of Object.keys(given)) {
    if (!Object.hasOwn(type.INPUTS, name)) {
      throw new InvalidNodeParameterValue(`${type.name} has no input ${repr(name)}`);
    }
  }
  const specs = Object.entries(type.INPUTS);
  const missing = specs
    .filter(([name, spec]) => isRequired(spec) && own(given, name) === undefined)
    .map(([name]) => name);
  if (missing.length > 0) {
    const noun = missing.length === 1 ? "input" : "inputs";
    throw new InvalidNodeParameterValue(`${type.name} is missing the ${noun} ${quoteNames(missing)}`);
  }
  const values = new Map<string, unknown>();
  for (const [name, spec] of specs) {
    const value = inputValue(name, spec, own(given, name));
    if (value !== NOT_GIVEN) {
      values.set(name, value);
    } else if (spec.hasDefault) {
      values.set(name, defaultValue(spec));
    }
  }
  return values;
}

/** The inputs that are set, in the order the class declares them. */
export function inputRecord(
  type: NodeSchema,
  values: ReadonlyMap<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    Object.keys(type.INPUTS)
      .filter((name) => values.has(name))
      .map((name) => [name, values.get(name)]),
  );
}

function checkConditioning(values: ReadonlyMap<string, unknown>): void {
  const positive = values.get("positive");
  const negative = values.get("negative");
  if (
    isLink(positive) &&
    isLink(negative) &&
    positive[0] === negative[0] &&
    positive[1] === negative[1]
  ) {
    throw new InvalidNodeParameterValue("positive and negative must not use the same input");
  }
}

/** Runs every check of the class on inputs that already passed their type checks. */
export function checkInputs(type: NodeSchema, values: ReadonlyMap<string, unknown>): void {
  for (const [name, spec] of Object.entries(type.INPUTS)) {
    if (!values.has(name)) {
      continue;
    }
    const value = values.get(name);
    if (spec.bool) {
      validateBool(name, value);
    }
    if (spec.range !== null) {
      validateRange(name, value, spec.range[0], spec.range[1]);
    }
    if (spec.choice) {
      validateChoice(name, value, spec.options ?? []);
    }
    if (spec.notEmpty) {
      validateNotEmpty(name, value);
    }
  }
  type.validate(inputRecord(type, values));
  checkConditioning(values);
}

// ComfyUI reads every list as a link, so a list widget value goes wrapped.
export function apiValue(value: unknown): unknown {
  if (Array.isArray(value) && !isLink(value)) {
    return { __value__: value };
  }
  return value;
}

export function comfyName(type: NodeSchema, name: string): string {
  return own(type.INPUT_NAMES, name) ?? name;
}

/** The node in the API format, with its inputs under their ComfyUI names. */
export function comfyUIFormat(
  type: NodeSchema,
  nodeId: string,
  values: ReadonlyMap<string, unknown>,
): ComfyUIWorkflow {
  const inputs = Object.fromEntries(
    Object.keys(type.INPUTS)
      .filter((name) => values.has(name))
      .map((name) => [comfyName(type, name), apiValue(values.get(name))]),
  );
  return Object.fromEntries([[nodeId, { class_type: type.CLASS_TYPE, inputs }]]);
}
