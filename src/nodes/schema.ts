// The inputs of a node class: what each one takes and how it is checked.
// The same specs describe the parameters of a pipeline.
import { deepFreeze } from "../_internal/records.js";

/** A kind of value: "int" takes safe integers, "float" any number, "any" every value. */
export type Kind = "int" | "float" | "string" | "boolean" | "link" | "any";

/** An option of a choice. */
export type Option = string | number;

/** A bound of a closed range; null leaves that side open. */
export type Bound = number | null;

export type Range = readonly [min: Bound, max: Bound];

/** What a value is checked against before it is stored. */
export interface TypeSpec {
  /** The kinds taken besides the options, in the order messages name them. */
  readonly kinds: readonly Kind[];
  /** The options of a choice: the type of each one is taken. */
  readonly options: readonly Option[] | null;
  /** Can be left out: undefined and null then stand for "not given". */
  readonly optional: boolean;
}

export interface InputSpec extends TypeSpec {
  /** The value must also be one of the options. */
  readonly choice: boolean;
  readonly hasDefault: boolean;
  /**
   * Used when the input is not given. An object or array is copied for each
   * node, and null stands for it too.
   */
  readonly default: unknown;
  readonly range: Range | null;
  readonly notEmpty: boolean;
  /** The value must be a boolean. */
  readonly bool: boolean;
}

export interface CommonOptions {
  /** Also takes a link to the output of another node. */
  readonly link?: boolean;
  readonly optional?: boolean;
}

export interface NumberOptions extends CommonOptions {
  readonly default?: number;
  readonly range?: Range;
}

export interface StringOptions extends CommonOptions {
  readonly default?: string;
  readonly notEmpty?: boolean;
}

export interface BooleanOptions extends CommonOptions {
  readonly default?: boolean;
  readonly bool?: boolean;
}

export interface ChoiceOptions<V extends Option> extends CommonOptions {
  readonly default?: V;
}

export interface AnyOptions {
  readonly default?: unknown;
  readonly optional?: boolean;
}

export interface LinkOptions {
  readonly optional?: boolean;
}

interface Checks {
  readonly options?: readonly Option[];
  readonly choice?: boolean;
  readonly range?: Range;
  readonly notEmpty?: boolean;
  readonly bool?: boolean;
}

function input(
  kinds: readonly Kind[],
  given: CommonOptions & { readonly default?: unknown },
  checks: Checks = {},
): InputSpec {
  const hasDefault = given.default !== undefined;
  if (hasDefault && given.optional === true) {
    throw new TypeError("An input with a default is not optional.");
  }
  return Object.freeze({
    kinds: Object.freeze(given.link === true ? [...kinds, "link" as const] : [...kinds]),
    options: checks.options === undefined ? null : Object.freeze([...checks.options]),
    choice: checks.choice ?? false,
    optional: given.optional ?? false,
    hasDefault,
    default: deepFreeze(given.default),
    range: checks.range === undefined ? null : Object.freeze([checks.range[0], checks.range[1]] as const),
    notEmpty: checks.notEmpty ?? false,
    bool: checks.bool ?? false,
  });
}

/** An input that only takes a link. */
export function link(options: LinkOptions = {}): InputSpec {
  return input([], { ...options, link: true });
}

export function int(options: NumberOptions = {}): InputSpec {
  return input(["int"], options, { range: options.range });
}

export function float(options: NumberOptions = {}): InputSpec {
  return input(["float"], options, { range: options.range });
}

export function string(options: StringOptions = {}): InputSpec {
  return input(["string"], options, { notEmpty: options.notEmpty });
}

export function boolean(options: BooleanOptions = {}): InputSpec {
  return input(["boolean"], options, { bool: options.bool });
}

/** An input that takes every value, like a Python one annotated with Any. */
export function any(options: AnyOptions = {}): InputSpec {
  return input(["any"], options);
}

/** An input whose value must be one of the options. */
export function choice<const V extends readonly Option[]>(
  values: V,
  options: ChoiceOptions<V[number]> = {},
): InputSpec {
  return input([], options, { options: values, choice: true });
}

/**
 * An input typed by the options, whose value is not checked against them:
 * like a Python Literal without validate_choice.
 */
export function literal<const V extends readonly Option[]>(
  values: V,
  options: ChoiceOptions<V[number]> = {},
): InputSpec {
  return input([], options, { options: values });
}
