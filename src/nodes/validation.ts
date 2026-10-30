import { repr, typeName } from "../_internal/format.js";
import { isRecord } from "../_internal/records.js";
import { InvalidNodeParameterValue } from "../errors/node.js";
import type { Kind, Option, TypeSpec } from "./schema.js";

/** The output of a node, as ComfyUI links it: [node id, output index]. */
export type Link = readonly [nodeId: string, outputIndex: number];

const KIND_NAMES: Readonly<Record<Kind, string | null>> = {
  int: "an integer",
  float: "a number",
  string: "a string",
  boolean: "a boolean",
  link: "a link",
  any: null,
};

export function isLink(value: unknown): value is Link {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    typeof value[0] === "string" &&
    Number.isInteger(value[1])
  );
}

function accepts(kind: Kind, value: unknown): boolean {
  switch (kind) {
    case "int":
      return Number.isSafeInteger(value);
    case "float":
      return typeof value === "number";
    case "string":
      return typeof value === "string";
    case "boolean":
      return typeof value === "boolean";
    case "link":
      return isLink(value);
    case "any":
      return true;
  }
}

function optionKind(option: Option): Kind {
  if (typeof option === "string") {
    return "string";
  }
  return Number.isInteger(option) ? "int" : "float";
}

// The options of a choice only give their types: validateChoice checks the value.
export function validateType(name: string, value: unknown, spec: TypeSpec): void {
  if (spec.optional && value == null) {
    return;
  }
  const kinds = [...(spec.options ?? []).map(optionKind), ...spec.kinds];
  if (kinds.some((kind) => accepts(kind, value))) {
    return;
  }
  const expected = new Set<string>();
  for (const kind of kinds) {
    const description = KIND_NAMES[kind];
    if (description !== null) {
      expected.add(description);
    }
  }
  if (spec.optional) {
    expected.add("undefined");
  }
  throw new InvalidNodeParameterValue(
    `${name} must be ${[...expected].join(" or ")}, not ${typeName(value)}`,
  );
}

export function acceptsLink(spec: TypeSpec): boolean {
  return spec.kinds.includes("link");
}

// Every validator lets a link through: its value only exists when ComfyUI runs.
export function validateRange(
  name: string,
  value: unknown,
  rangeMin: number | null = null,
  rangeMax: number | null = null,
): void {
  if (isLink(value)) {
    return;
  }
  if (typeof value !== "number") {
    throw new InvalidNodeParameterValue(`${name} must be a number, not ${typeName(value)}`);
  }
  // NaN fails every comparison, so no range would ever stop it.
  if (Number.isNaN(value)) {
    throw new InvalidNodeParameterValue(`${name} must not be NaN`);
  }
  // The API format is JSON, which has no infinity, even where ComfyUI sets no bound.
  if (!Number.isFinite(value)) {
    throw new InvalidNodeParameterValue(`${name} must be finite, not ${value}`);
  }
  if ((rangeMin !== null && value < rangeMin) || (rangeMax !== null && value > rangeMax)) {
    const lower = rangeMin ?? -Infinity;
    const upper = rangeMax ?? Infinity;
    throw new InvalidNodeParameterValue(`${name} must be in closed range: [${lower}, ${upper}]`);
  }
}

function compareText(a: string, b: string): number {
  if (a < b) {
    return -1;
  }
  return a > b ? 1 : 0;
}

export function validateChoice(name: string, value: unknown, choices: Iterable<Option>): void {
  if (isLink(value)) {
    return;
  }
  const options: unknown[] = [...choices];
  if (!options.includes(value)) {
    const sorted = options.sort((a, b) => compareText(String(a), String(b)));
    throw new InvalidNodeParameterValue(
      `Invalid ${name}, must be one of the following values: ${repr(sorted)}`,
    );
  }
}

// Empty as Python sees it: no text, no items, zero, false or nothing at all.
function isEmpty(value: unknown): boolean {
  if (value === null || value === undefined || value === false || value === 0 || value === "") {
    return true;
  }
  if (Array.isArray(value)) {
    return value.length === 0;
  }
  if (value instanceof Map || value instanceof Set) {
    return value.size === 0;
  }
  return isRecord(value) && Object.keys(value).length === 0;
}

export function validateNotEmpty(name: string, value: unknown): void {
  if (isLink(value)) {
    return;
  }
  if (isEmpty(value)) {
    throw new InvalidNodeParameterValue(`${name} must not be empty`);
  }
}

export function validateBool(name: string, value: unknown): void {
  if (isLink(value)) {
    return;
  }
  if (typeof value !== "boolean") {
    throw new InvalidNodeParameterValue(`${name} must be a boolean`);
  }
}
