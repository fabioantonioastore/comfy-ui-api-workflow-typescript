import { describe, expect, it } from "vitest";

import { InvalidNodeParameterValue } from "../src/errors/node.js";
import { any, boolean, float, int, link, literal, string, type TypeSpec } from "../src/nodes/schema.js";
import {
  isLink,
  validateBool,
  validateChoice,
  validateNotEmpty,
  validateRange,
  validateType,
  type Link,
} from "../src/nodes/validation.js";
import { expectError } from "./helpers.js";

const LINK: Link = ["PrimitiveInt_0", 0];

describe("isLink", () => {
  it.each([[["Loader_0", 0]], [["129:94", 2]], [["Loader_0", 1]]])("takes %j as a link", (value) => {
    expect(isLink(value)).toBe(true);
  });

  it.each([[0], ["Loader_0"], [["Loader_0"]], [[0, 1]], [["Loader_0", "0"]], [["Loader_0", 1.5]], [[]], [[{ x: 0 }]]])(
    "does not take %j as a link",
    (value) => {
      expect(isLink(value)).toBe(false);
    },
  );
});

describe("validateRange", () => {
  it.each([
    [0, 0, 10],
    [10, 0, 10],
    [0.5, 0.0, 1.0],
    [-5, null, 10],
    [1e30, 0, null],
    [7, null, null],
  ])("accepts %d in [%s, %s]", (value, rangeMin, rangeMax) => {
    validateRange("steps", value, rangeMin, rangeMax);
  });

  const outside: [number, number | null, number | null, string][] = [
    [-1, 0, 10, "steps must be in closed range: [0, 10]"],
    [11, 0, 10, "steps must be in closed range: [0, 10]"],
    [11, null, 10, "steps must be in closed range: [-Infinity, 10]"],
    [-1, 0, null, "steps must be in closed range: [0, Infinity]"],
  ];

  it.each(outside)("rejects %d outside [%s, %s]", (value, rangeMin, rangeMax, message) => {
    expectError(() => validateRange("steps", value, rangeMin, rangeMax), InvalidNodeParameterValue, message);
  });

  it.each([true, false])("rejects the boolean %s", (value) => {
    expectError(() => validateRange("steps", value, 0, 10), InvalidNodeParameterValue, /^steps must be a number/);
  });

  it("rejects a boolean without bounds", () => {
    expectError(() => validateRange("index", true), InvalidNodeParameterValue, /^index must be a number/);
  });

  it.each([
    [0.0, 100.0],
    [null, null],
  ])("rejects NaN in [%s, %s]", (rangeMin, rangeMax) => {
    expectError(() => validateRange("cfg", Number.NaN, rangeMin, rangeMax), InvalidNodeParameterValue, "cfg must not be NaN");
  });

  it.each([
    [Infinity, 0.0, null],
    [Infinity, null, 1.0],
    [Infinity, null, null],
    [-Infinity, 0.0, null],
    [-Infinity, null, 1.0],
    [-Infinity, null, null],
  ])("rejects %d even without that bound", (value, rangeMin, rangeMax) => {
    expectError(
      () => validateRange("threshold", value, rangeMin, rangeMax),
      InvalidNodeParameterValue,
      `threshold must be finite, not ${value}`,
    );
  });

  it("lets links through", () => {
    validateRange("steps", LINK, 1, 10);
  });

  it.each([
    ["20", "string"],
    [null, "null"],
    [[1, 2], "array"],
    [{ steps: 20 }, "object"],
  ])("rejects %j, which is no number", (value, kind) => {
    expectError(() => validateRange("steps", value, 0, 10), InvalidNodeParameterValue, `steps must be a number, not ${kind}`);
  });
});

describe("validateChoice", () => {
  const crops = ["disabled", "center"] as const;
  const message = "Invalid crop, must be one of the following values: ['center', 'disabled']";

  it("accepts an option", () => {
    validateChoice("crop", "center", crops);
  });

  it("accepts options that are not strings", () => {
    validateChoice("bit_depth", 8, new Set(["auto", 8, 10]));
  });

  it("rejects an unknown option, listing the options", () => {
    expectError(() => validateChoice("crop", "middle", crops), InvalidNodeParameterValue, message);
  });

  it("sorts the options as text", () => {
    expectError(
      () => validateChoice("bit_depth", 12, ["auto", 8, 10]),
      InvalidNodeParameterValue,
      "Invalid bit_depth, must be one of the following values: [10, 8, 'auto']",
    );
  });

  it("lets links through", () => {
    validateChoice("crop", LINK, ["center"]);
  });

  it.each([[["center"]], [{ crop: "center" }]])("rejects %j, listing the options", (value) => {
    expectError(() => validateChoice("crop", value, crops), InvalidNodeParameterValue, message);
  });
});

describe("validateNotEmpty", () => {
  it("accepts a file name", () => {
    validateNotEmpty("lora_name", "detail.safetensors");
  });

  it.each([[""], [[]], [{}], [null], [undefined]])("rejects the empty value %j", (value) => {
    expectError(() => validateNotEmpty("lora_name", value), InvalidNodeParameterValue, "lora_name must not be empty");
  });

  it("lets links through", () => {
    validateNotEmpty("lora_name", LINK);
  });
});

describe("validateBool", () => {
  it.each([true, false])("accepts %s", (value) => {
    validateBool("enabled", value);
  });

  it.each([[1], [0], ["false"], [null]])("rejects %j", (value) => {
    expectError(() => validateBool("enabled", value), InvalidNodeParameterValue, "enabled must be a boolean");
  });

  it("lets links through", () => {
    validateBool("enabled", LINK);
  });
});

describe("validateType", () => {
  const cases: [string, TypeSpec, unknown][] = [
    ["int", int({ link: true }), 20],
    ["int-link", int({ link: true }), LINK],
    ["float-from-int", float({ link: true }), 8],
    ["float", float({ link: true }), 0.5],
    ["str", string({ link: true }), "model.safetensors"],
    ["bool", boolean({ link: true }), false],
    ["literal-int", literal(["auto", 8, 10], { link: true }), 8],
    ["literal-type", literal(["euler"], { link: true }), "heun"],
    ["link", link(), LINK],
    ["optional-link-null", link({ optional: true }), null],
    ["optional-link-undefined", link({ optional: true }), undefined],
    ["optional-int", int({ optional: true }), null],
    ["any", any(), new Map()],
  ];

  it.each(cases)("accepts a value of the type (%s)", (_, spec, value) => {
    validateType("steps", value, spec);
  });

  const rejected: [string, TypeSpec, unknown, string][] = [
    ["int-str", int({ link: true }), "20", "steps must be an integer or a link, not string"],
    ["int-float", int({ link: true }), 20.5, "steps must be an integer or a link, not float"],
    ["int-unsafe", int({ link: true }), 2 ** 64, "steps must be an integer or a link, not unsafe integer"],
    ["int-bool", int({ link: true }), true, "steps must be an integer or a link, not boolean"],
    ["float-null", float({ link: true }), null, "steps must be a number or a link, not null"],
    ["float-bool", float({ link: true }), false, "steps must be a number or a link, not boolean"],
    ["str-int", string({ link: true }), 5, "steps must be a string or a link, not integer"],
    ["bool-int", boolean({ link: true }), 1, "steps must be a boolean or a link, not integer"],
    ["literal-array", literal(["euler", "heun"], { link: true }), ["euler"], "steps must be a string or a link, not array"],
    [
      "literal-mixed",
      literal(["auto", 8, 10], { link: true }),
      8.5,
      "steps must be a string or an integer or a link, not float",
    ],
    ["link-str", link(), "model.safetensors", "steps must be a link, not string"],
    ["link-short", link(), ["Loader_0"], "steps must be a link, not array"],
    ["optional-link-int", link({ optional: true }), 0, "steps must be a link or undefined, not integer"],
    ["optional-int", int({ optional: true }), "20", "steps must be an integer or undefined, not string"],
  ];

  it.each(rejected)("rejects a value of another type (%s)", (_, spec, value, message) => {
    expectError(() => validateType("steps", value, spec), InvalidNodeParameterValue, message);
  });
});
