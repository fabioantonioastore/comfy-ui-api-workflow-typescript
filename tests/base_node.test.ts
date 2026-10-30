import { describe, expect, it } from "vitest";

import { InvalidNodeParameterValue } from "../src/errors/node.js";
import { OutputNotFoundError } from "../src/errors/output.js";
import { BaseNode, type Link, type NodeOutput } from "../src/nodes/base.js";
import { any, int, link } from "../src/nodes/schema.js";
import { expectError } from "./helpers.js";

class TextNode extends BaseNode<{ text: unknown; clip: unknown }, "CONDITIONING"> {
  static override readonly CATEGORY = "test";
  static override readonly OUTPUTS = { CONDITIONING: 0 };
  static override readonly INPUTS = { text: any(), clip: any() };
}

class PartialNode extends BaseNode<{ set_input?: unknown; unset_input?: unknown }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUTS = {
    set_input: any({ default: 1 }),
    unset_input: any({ optional: true }),
  };
  declare unset_input: unknown;
}

class IntermediateNode<I extends { width: unknown } = { width: unknown }> extends BaseNode<I> {
  static override readonly INPUTS = { width: any() };
}

class LeafNode extends IntermediateNode<{ width: unknown; height: unknown }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUTS = { ...IntermediateNode.INPUTS, height: any() };
}

class RenamedNode extends BaseNode<{ model: Link }> {
  static override readonly CATEGORY = "test";
  static override readonly CLASS_TYPE = "Renamed Node";
  static override readonly INPUTS = { model: link() };
}

class DottedInputNode extends BaseNode<{ time_embed: unknown; output_: unknown }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUT_NAMES = { time_embed: "time_embed.", output_: "output" };
  static override readonly INPUTS = { time_embed: any(), output_: any() };
}

class ListWidgetNode extends BaseNode<{ editor_state: unknown; image: unknown }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUTS = { editor_state: any(), image: any() };
}

interface TypedNodeInputs {
  model: Link;
  steps?: number | Link;
  mask?: Link | null;
}

class TypedNode extends BaseNode<TypedNodeInputs, "LATENT"> {
  static override readonly CATEGORY = "test";
  static override readonly OUTPUTS = { LATENT: 0 };
  static override readonly INPUTS = {
    steps: int({ link: true, default: 20 }),
    model: link(),
    mask: link({ optional: true }),
  };
  declare model: Link;
  declare steps: number | Link;
  declare mask: Link | undefined;
}

class RangeNode extends BaseNode<{ low?: number; high?: number }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUTS = { low: int({ default: 0 }), high: int({ default: 1 }) };
  declare low: number;
  declare high: number;

  static override validate(inputs: Readonly<Record<string, unknown>>): void {
    if (Number(inputs["low"]) > Number(inputs["high"])) {
      throw new InvalidNodeParameterValue("low must not be above high");
    }
  }
}

class ShiftedRangeNode extends RangeNode {
  static override readonly INPUTS = { low: int({ default: 5 }), high: int({ default: 6 }) };
}

class PlainRangeNode extends RangeNode {}

class RequiredRangeNode extends BaseNode<{ low: number; high: number }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUTS = { low: int(), high: int() };
  declare low: number;

  static override validate(inputs: Readonly<Record<string, unknown>>): void {
    RangeNode.validate(inputs);
  }
}

class MutableDefaultNode extends BaseNode<{ state?: unknown; note?: Link | null }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUTS = {
    state: any({ default: {} }),
    note: link({ optional: true }),
  };
  declare state: unknown;
  declare note: Link | undefined;
}

class ConditioningNode extends BaseNode<{ positive: Link; negative: Link }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUTS = { positive: link(), negative: link(), strength: any({ default: 1.0 }) };
  declare negative: Link;
}

describe("BaseNode", () => {
  it("keys the ComfyUI format by node id, with the inputs", () => {
    const node = new TextNode("TextNode_0", { text: "a cat", clip: ["Loader_0", 1] });

    expect(node.getComfyUIWorkflowFormat()).toEqual({
      TextNode_0: { class_type: "TextNode", inputs: { text: "a cat", clip: ["Loader_0", 1] } },
    });
  });

  it("leaves unset inputs out", () => {
    const node = new PartialNode("PartialNode_0");

    expect(node.getComfyUIWorkflowFormat()["PartialNode_0"]!.inputs).toEqual({ set_input: 1 });
  });

  it("includes the inputs of intermediate classes", () => {
    const node = new LeafNode("LeafNode_0", { width: 512, height: 768 });

    expect(node.getComfyUIWorkflowFormat()["LeafNode_0"]!.inputs).toEqual({ width: 512, height: 768 });
  });

  it("links outputs by the outputs of the class", () => {
    const node = new TextNode("TextNode_0", { text: "a cat", clip: ["Loader_0", 1] });

    expect(node.output.CONDITIONING).toEqual(["TextNode_0", 0]);
  });

  it("has no output to link without outputs", () => {
    const node = new PartialNode("PartialNode_0");

    expectError(() => (node.output as NodeOutput)["LATENT"], OutputNotFoundError);
  });

  it("rejects undeclared properties", () => {
    const node = new TextNode("TextNode_0", { text: "a cat", clip: ["Loader_0", 1] });

    expect(() => {
      (node as unknown as Record<string, unknown>)["txet"] = "typo";
    }).toThrow(TypeError);
  });

  it("uses the class name as the class type by default", () => {
    expect(TextNode.CLASS_TYPE).toBe("TextNode");
    expect(IntermediateNode.CLASS_TYPE).toBe("IntermediateNode");
    expect(LeafNode.CLASS_TYPE).toBe("LeafNode");
  });

  it("sends an explicit class type", () => {
    const node = new RenamedNode("RenamedNode_0", { model: ["Loader_0", 0] });

    expect(node.getComfyUIWorkflowFormat()["RenamedNode_0"]!.class_type).toBe("Renamed Node");
  });

  it("maps inputs to their ComfyUI names", () => {
    const node = new DottedInputNode("DottedInputNode_0", { time_embed: 0.5, output_: "depth" });

    expect(node.getComfyUIWorkflowFormat()["DottedInputNode_0"]!.inputs).toEqual({
      "time_embed.": 0.5,
      output: "depth",
    });
  });

  it("wraps array values so ComfyUI does not read them as links", () => {
    const node = new ListWidgetNode("ListWidgetNode_0", {
      editor_state: [{ x: 0 }],
      image: ["LoadImage_0", 0],
    });

    expect(node.getComfyUIWorkflowFormat()["ListWidgetNode_0"]!.inputs).toEqual({
      editor_state: { __value__: [{ x: 0 }] },
      image: ["LoadImage_0", 0],
    });
  });

  it("wraps an empty array", () => {
    const node = new ListWidgetNode("ListWidgetNode_0", { editor_state: [], image: ["LoadImage_0", 0] });

    expect(node.getComfyUIWorkflowFormat()["ListWidgetNode_0"]!.inputs["editor_state"]).toEqual({
      __value__: [],
    });
  });

  it("removes an optional input set to undefined", () => {
    const node = new TypedNode("TypedNode_0", { model: ["Loader_0", 0], mask: ["LoadImage_1", 1] });

    node.mask = undefined;

    expect(node.inputs()).toEqual({ steps: 20, model: ["Loader_0", 0] });
  });

  it("removes an optional input set to null", () => {
    const node = new TypedNode("TypedNode_0", { model: ["Loader_0", 0], mask: ["LoadImage_1", 1] });

    (node as unknown as { mask: null }).mask = null;

    expect(node.inputs()).toEqual({ steps: 20, model: ["Loader_0", 0] });
  });

  it.each(["model", "steps"] as const)("cannot remove %s, which is not optional", (name) => {
    const node = new TypedNode("TypedNode_0", { model: ["Loader_0", 0] });
    const before = node.inputs();

    expectError(
      () => {
        (node as unknown as Record<string, unknown>)[name] = undefined;
      },
      InvalidNodeParameterValue,
      `${name} cannot be removed because it is not optional`,
    );
    expect(node.inputs()).toEqual(before);
  });

  it("takes values and links of the input types", () => {
    const node = new TypedNode("TypedNode_0", {
      model: ["Loader_0", 0],
      steps: ["PrimitiveInt_1", 0],
      mask: ["LoadImage_2", 1],
    });

    expect(node.inputs()).toEqual({
      steps: ["PrimitiveInt_1", 0],
      model: ["Loader_0", 0],
      mask: ["LoadImage_2", 1],
    });
  });

  it("rejects a value of another type", () => {
    expectError(
      () => new TypedNode("TypedNode_0", { model: ["Loader_0", 0], steps: "20" as unknown as number }),
      InvalidNodeParameterValue,
      "steps must be an integer or a link, not string",
    );
  });

  it("rejects a value for an input that only takes links", () => {
    expectError(
      () => new TypedNode("TypedNode_0", { model: "model.safetensors" as unknown as Link }),
      InvalidNodeParameterValue,
      "model must be a link, not string",
    );
  });

  it("rejects a missing input", () => {
    expectError(
      () => new TypedNode("TypedNode_0", {} as TypedNodeInputs),
      InvalidNodeParameterValue,
      "TypedNode is missing the input 'model'",
    );
  });

  it("rejects an unknown input", () => {
    expectError(
      () => new TypedNode("TypedNode_0", { model: ["Loader_0", 0], modle: 1 } as TypedNodeInputs),
      InvalidNodeParameterValue,
      "TypedNode has no input 'modle'",
    );
  });

  it("keeps the previous value when a value of another type is set", () => {
    const node = new TypedNode("TypedNode_0", { model: ["Loader_0", 0] });

    expectError(() => {
      node.steps = 2.5;
    }, InvalidNodeParameterValue);
    expect(node.steps).toBe(20);
  });

  it("leaves an unset input unset when a value of another type is set", () => {
    const node = new TypedNode("TypedNode_0", { model: ["Loader_0", 0] });

    expectError(
      () => {
        node.mask = "mask.png" as unknown as Link;
      },
      InvalidNodeParameterValue,
      "mask must be a link or undefined, not string",
    );
    expect("mask" in node.inputs()).toBe(false);
  });

  it("stores a link as a copy", () => {
    const model: [string, number] = ["Loader_0", 0];
    const node = new TypedNode("TypedNode_0", { model });

    model[1] = 5;

    expect(node.model).toEqual(["Loader_0", 0]);
    expect(Object.isFrozen(node.model)).toBe(true);
  });

  it("takes a string node id", () => {
    expectError(
      () => new TextNode(7 as unknown as string, { text: "a cat", clip: ["Loader_0", 1] }),
      InvalidNodeParameterValue,
      "nodeId must be a string, not integer",
    );
  });

  it("lets an input of any type take any value", () => {
    const node = new TextNode("TextNode_0", { text: 5, clip: null });

    expect(node.inputs()).toEqual({ text: 5, clip: null });
  });

  it("keeps the previous value when a change breaks the node validation", () => {
    const node = new RangeNode("RangeNode_0", { low: 0, high: 1 });

    expectError(
      () => {
        node.low = 2;
      },
      InvalidNodeParameterValue,
      "low must not be above high",
    );
    expect([node.low, node.high]).toEqual([0, 1]);
  });

  it("validates a change to a node whose inputs are all required", () => {
    const node = new RequiredRangeNode("RequiredRangeNode_0", { low: 0, high: 1 });

    expectError(
      () => {
        node.low = 2;
      },
      InvalidNodeParameterValue,
      "low must not be above high",
    );
    expect(node.inputs()).toEqual({ low: 0, high: 1 });
  });

  it("keeps a valid change", () => {
    const node = new RangeNode("RangeNode_0", { low: 0, high: 1 });

    node.high = 10;
    node.low = 5;

    expect([node.low, node.high]).toEqual([5, 10]);
  });

  it("validates changes to a subclass that declares nothing", () => {
    const node = new PlainRangeNode("PlainRangeNode_0");

    expectError(() => {
      node.low = 2;
    }, InvalidNodeParameterValue);
    expect(node.low).toBe(0);
  });

  it("uses the defaults of a subclass", () => {
    const node = new ShiftedRangeNode("ShiftedRangeNode_0");

    expect([node.low, node.high]).toEqual([5, 6]);
  });

  it("takes changes to an input the constructor leaves unset", () => {
    const node = new PartialNode("PartialNode_0");

    node.unset_input = 5;

    expect(node.inputs()).toEqual({ set_input: 1, unset_input: 5 });
  });

  it("rejects the same link for positive and negative", () => {
    const node = new ConditioningNode("ConditioningNode_0", {
      positive: ["Text_1", 0],
      negative: ["Text_2", 0],
    });

    expectError(
      () => {
        node.negative = ["Text_1", 0];
      },
      InvalidNodeParameterValue,
      "positive and negative must not use the same input",
    );
    expect(node.negative).toEqual(["Text_2", 0]);
  });

  it("cannot remove an input whose default is always sent", () => {
    const node = new MutableDefaultNode("MutableDefaultNode_0", { state: { zoom: 2 } });

    expectError(
      () => {
        node.state = undefined;
      },
      InvalidNodeParameterValue,
      "state cannot be removed because it is not optional",
    );
    expect(node.state).toEqual({ zoom: 2 });
  });

  it("can remove an input the constructor leaves out", () => {
    const node = new MutableDefaultNode("MutableDefaultNode_0", { note: ["Text_1", 0] });

    node.note = undefined;

    expect(node.inputs()).toEqual({ state: {} });
  });

  it("gives each node its own copy of an object default, which null stands for", () => {
    const first = new MutableDefaultNode("MutableDefaultNode_0");
    const second = new MutableDefaultNode("MutableDefaultNode_1", { state: null });

    (first.state as Record<string, unknown>)["zoom"] = 2;

    expect(second.state).toEqual({});
    expect(MutableDefaultNode.INPUTS.state.default).toEqual({});
    second.state = { zoom: 3 };
    second.state = null;
    expect(second.state).toEqual({});
  });

  it("rejects an input that clashes with a member of every node", () => {
    class ClashingNode extends BaseNode {
      static override readonly INPUTS = { output: any() };
    }

    expect(() => new ClashingNode("ClashingNode_0", { output: 1 })).toThrow(TypeError);
  });

  it("only works when its constructor made it", () => {
    const node = Object.create(TextNode.prototype) as TextNode;

    expectError(() => node.nodeId, TypeError, "This object is not a node made by its constructor.");
  });

  it("can be awaited, as reading then on its outputs gives undefined", async () => {
    const node = new TextNode("TextNode_0", { text: "a cat", clip: ["Loader_0", 1] });

    expect((node.output as unknown as { then: unknown }).then).toBeUndefined();
    await expect(Promise.resolve(node.output)).resolves.toBeDefined();
  });
});
