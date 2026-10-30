// The node classes against tests/fixtures/golden.json, which
// scripts/export_golden.py records from the Python library.
import { describe, expect, it } from "vitest";

import { NODE_TYPES } from "../src/_internal/registry.js";
import type { NodeType } from "../src/nodes/base.js";
import { readGolden } from "./helpers.js";

const golden = readGolden();

function nodeType(classType: string): NodeType {
  const type = NODE_TYPES.get(classType);
  if (type === undefined) {
    throw new Error(`There is no node class of type ${classType}.`);
  }
  return type;
}

// An input named like a member of every node, such as output, ends in "_" here.
function inputsOf(type: NodeType, inputs: Readonly<Record<string, unknown>>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(inputs).map(([name, value]) => [Object.hasOwn(type.INPUTS, name) ? name : `${name}_`, value]),
  );
}

describe("the node classes, against the Python library", () => {
  it("are the same node types", () => {
    expect(new Set(NODE_TYPES.keys())).toEqual(new Set(golden.nodes.map((found) => found.class_type)));
  });

  it.each(golden.nodes.map((found) => [`${found.class_type} (${found.case})`, found] as const))(
    "send what it sends: %s",
    (_, found) => {
      const type = nodeType(found.class_type);
      const build = () => new type(found.node_id, inputsOf(type, found.inputs));
      if (found.error !== undefined) {
        expect(build).toThrow(expect.objectContaining({ name: found.error }));
        return;
      }
      expect(build().getComfyUIWorkflowFormat()).toEqual({
        [found.node_id]: { class_type: found.class_type, inputs: found.sent },
      });
    },
  );
});
