// Port of the Python test_node_classes.py: what every node class of the
// library does, read from the classes alone.
import { describe, expect, it } from "vitest";

import { isRequired } from "../src/_internal/inputs.js";
import { NODE_TYPES } from "../src/_internal/registry.js";
import { InvalidNodeParameterValue } from "../src/errors/node.js";
import type { BaseNode, Link, NodeType } from "../src/nodes/base.js";
import { Combo, type Group } from "../src/nodes/dynamic.js";
import { CurveEditor } from "../src/nodes/utilities.js";
import { expectError } from "./helpers.js";

const LINK: Link = ["Upstream_0", 0];
const TYPES = [...NODE_TYPES.values()].sort((a, b) => (a.CLASS_TYPE < b.CLASS_TYPE ? -1 : 1));
// Combos whose options depend on another input; dynamic_nodes.test.ts covers them.
const DEPENDENT_COMBOS = new Set(["SaveVideo.format_codec_encoding", "SaveVideo.codec_encoding"]);
// Choices that open the branches where every SaveVideo input exists.
const WHOLE_TREE: Readonly<Record<string, Record<string, unknown>>> = {
  SaveVideo: {
    format_codec: "h264",
    format_codec_encoding: "re-encode",
    codec: "h264",
    codec_encoding: "re-encode",
  },
};

function dynamicInputs(type: NodeType): Readonly<Record<string, Combo | Group>> {
  return (type as NodeType & { DYNAMIC_INPUTS?: Record<string, Combo | Group> }).DYNAMIC_INPUTS ?? {};
}

// Inputs of dynamic combo branches and group rows cannot all be set at once.
function branchInputs(type: NodeType): Set<string> {
  const found = new Set<string>();
  for (const [name, spec] of Object.entries(dynamicInputs(type))) {
    found.add(name);
    if (spec instanceof Combo) {
      spec.attributes().forEach((attribute) => found.add(attribute));
    } else {
      for (let row = 0; row < spec.rows; row += 1) {
        Object.keys(spec.fields).forEach((field) => found.add(spec.attribute(name, row, field)));
      }
    }
  }
  return found;
}

// Each input gets its own link, since positive and negative cannot share one.
function linkFor(name: string): Link {
  return [`Upstream_${name}`, 0];
}

function build(type: NodeType, values: Record<string, unknown> = {}): BaseNode {
  const required = Object.entries(type.INPUTS)
    .filter(([, spec]) => isRequired(spec))
    .map(([name]) => [name, linkFor(name)]);
  return new type(`${type.name}_0`, { ...Object.fromEntries(required), ...values });
}

describe("every node class", () => {
  it.each(TYPES.map((type) => [type.CLASS_TYPE, type] as const))("builds with its defaults (%s)", (_, type) => {
    const inputs = build(type).inputs() as Record<string, unknown>;
    const fixed = Object.fromEntries(
      Object.entries(type.INPUTS)
        .filter(([, spec]) => isRequired(spec) || spec.hasDefault)
        .map(([name, spec]) => [name, isRequired(spec) ? linkFor(name) : spec.default]),
    );

    expect(Object.fromEntries(Object.keys(fixed).map((name) => [name, inputs[name]]))).toEqual(fixed);
    expect(Object.keys(inputs).every((name) => Object.hasOwn(type.INPUTS, name))).toBe(true);
    expect(Object.values(inputs)).not.toContain(null);
  });

  it.each(TYPES.map((type) => [type.CLASS_TYPE, type] as const))(
    "sends every input that takes a link as one (%s)",
    (_, type) => {
      const branches = branchInputs(type);
      const linked = Object.fromEntries(
        Object.entries(type.INPUTS)
          .filter(([name, spec]) => (spec.kinds.includes("link") || spec.kinds.includes("any")) && !branches.has(name))
          .map(([name]) => [name, linkFor(name)]),
      );

      const node = build(type, { ...WHOLE_TREE[type.CLASS_TYPE], ...linked });
      const sent = node.getComfyUIWorkflowFormat()[node.nodeId]!;

      expect(sent.class_type).toBe(type.CLASS_TYPE);
      expect(
        Object.fromEntries(Object.keys(linked).map((name) => [name, sent.inputs[type.INPUT_NAMES[name] ?? name]])),
      ).toEqual(linked);
    },
  );
});

const COMBO_INPUTS = TYPES.flatMap((type) =>
  Object.entries(type.INPUTS)
    .filter(
      ([name, spec]) =>
        spec.options !== null && !DEPENDENT_COMBOS.has(`${type.CLASS_TYPE}.${name}`) && !branchInputs(type).has(name),
    )
    .map(([name, spec]) => [`${type.CLASS_TYPE}.${name}`, type, name, spec.options!] as const),
);

describe("a combo input", () => {
  it("is in many node classes", () => {
    expect(COMBO_INPUTS.length).toBeGreaterThan(100);
  });

  it.each(COMBO_INPUTS)("takes each option and rejects any other (%s)", (_, type, name, options) => {
    for (const option of options) {
      expect((build(type, { [name]: option }).inputs() as Record<string, unknown>)[name]).toBe(option);
    }
    expectError(
      () => build(type, { [name]: "__not_an_option__" }),
      InvalidNodeParameterValue,
      new RegExp(`${name}|${name.replaceAll("_", " ")}`),
    );
  });
});

describe("an input of a library node with an object default", () => {
  it("cannot be removed", () => {
    const node = build(CurveEditor) as CurveEditor;
    const before = node.inputs();

    expectError(
      () => {
        node.curve = undefined;
      },
      InvalidNodeParameterValue,
      "curve cannot be removed because it is not optional",
    );

    expect(node.inputs()).toEqual(before);
  });

  it("gets a copy of the default for each node", () => {
    const first = build(CurveEditor) as CurveEditor;
    const second = build(CurveEditor) as CurveEditor;

    expect(first.curve).toEqual(second.curve);
    expect(first.curve).not.toBe(second.curve);
    expect(first.curve).not.toBe(CurveEditor.INPUTS.curve.default);
  });
});

const CONDITIONED = TYPES.filter((type) => Object.hasOwn(type.INPUTS, "positive") && Object.hasOwn(type.INPUTS, "negative"));

describe("a node class with positive and negative", () => {
  it("is in many node classes", () => {
    expect(CONDITIONED.length).toBeGreaterThan(10);
  });

  it.each(CONDITIONED.map((type) => [type.CLASS_TYPE, type] as const))(
    "cannot use the same link for both (%s)",
    (_, type) => {
      expectError(
        () => build(type, { positive: LINK, negative: LINK }),
        InvalidNodeParameterValue,
        "positive and negative must not use the same input",
      );
    },
  );
});
