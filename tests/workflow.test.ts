// Port of the Python test_workflow.py. The pickle and copy tests have no
// counterpart: errors are plain JavaScript objects here.
import { describe, expect, it } from "vitest";

import { CycleError } from "../src/errors/data_structure.js";
import {
  InvalidLinkError,
  InvalidNodeChangeError,
  InvalidNodeParameterValue,
  NodeHasDependentsError,
  NodeNotFoundError,
} from "../src/errors/node.js";
import { NodeException } from "../src/exceptions.js";
import { BaseNode, type Link } from "../src/nodes/base.js";
import { CreateVideo, GetVideoComponents } from "../src/nodes/video.js";
import { EmptyImage } from "../src/nodes/image.js";
import { CLIPTextEncode } from "../src/nodes/model/conditioning.js";
import { EmptyLatentImage, VAEDecode } from "../src/nodes/model/latent.js";
import { LTXVEmptyLatentAudio } from "../src/nodes/model/latent/ltxv.js";
import { CheckpointLoaderSimple } from "../src/nodes/model/loaders.js";
import { KSampler, type KSamplerInputs } from "../src/nodes/model/sampling.js";
import { int, link, string } from "../src/nodes/schema.js";
import { GetItemFromList, PreviewAny } from "../src/nodes/utilities.js";
import { ComfySwitchNode } from "../src/nodes/utilities/logic.js";
import { PrimitiveFloat, PrimitiveInt, PrimitiveString } from "../src/nodes/utilities/primitive.js";
import { Workflow } from "../src/workflow.js";
import { expectError, expectUnchanged, workflowState as state } from "./helpers.js";

class TextNode extends BaseNode<{ text: string; clip: Link }, "CONDITIONING"> {
  static override readonly CATEGORY = "test";
  static override readonly OUTPUTS = { CONDITIONING: 0 };
  static override readonly INPUTS = { text: string(), clip: link() };
}

class LatentNode extends BaseNode<{ width?: number; height?: number }, "LATENT"> {
  static override readonly CATEGORY = "test";
  static override readonly OUTPUTS = { LATENT: 0 };
  static override readonly INPUTS = { width: int({ default: 512 }), height: int({ default: 512 }) };
}

class MaskNode extends BaseNode<{ mask?: Link | null; image?: Link | null }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUTS = { mask: link({ optional: true }), image: link({ optional: true }) };
  declare mask: Link | undefined;
  declare image: Link | undefined;
}

function addSources(workflow: Workflow) {
  const checkpoint = workflow.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
  return {
    checkpoint,
    positive: workflow.addNode(TextNode, { text: "a cat", clip: checkpoint.output.CLIP }),
    negative: workflow.addNode(TextNode, { text: "blurry", clip: checkpoint.output.CLIP }),
    latent: workflow.addNode(LatentNode),
  };
}

type Sources = ReturnType<typeof addSources>;

function sourceNodes(sources: Sources): BaseNode[] {
  return [sources.checkpoint, sources.positive, sources.negative, sources.latent];
}

function samplerInputs(sources: Sources, overrides: Partial<KSamplerInputs> = {}): KSamplerInputs {
  return {
    model: sources.checkpoint.output.MODEL,
    sampler_name: "euler",
    scheduler: "normal",
    positive: sources.positive.output.CONDITIONING,
    negative: sources.negative.output.CONDITIONING,
    latent_image: sources.latent.output.LATENT,
    ...overrides,
  };
}

describe("Workflow", () => {
  it("is empty when new", () => {
    const workflow = new Workflow();

    expect(workflow.nodes.size).toBe(0);
    expect(workflow.getComfyUIWorkflowFormat()).toEqual({});
  });

  it("returns the node addNode creates", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);

    const node = workflow.addNode(KSampler, samplerInputs(sources));

    expect(node).toBeInstanceOf(KSampler);
    expect(workflow.nodes.get(node.nodeId)).toBe(node);
  });

  it("names nodes after their class with a shared index", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const sampler = workflow.addNode(KSampler, samplerInputs(sources));

    expect([...sourceNodes(sources), sampler].map((node) => node.nodeId)).toEqual([
      "CheckpointLoaderSimple_0",
      "TextNode_1",
      "TextNode_2",
      "LatentNode_3",
      "KSampler_4",
    ]);
  });

  it("sends every node", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const sampler = workflow.addNode(KSampler, samplerInputs(sources));

    expect(workflow.getComfyUIWorkflowFormat()).toEqual(
      Object.assign({}, ...[...sourceNodes(sources), sampler].map((node) => node.getComfyUIWorkflowFormat())),
    );
  });

  it("sends the changes made after a node was added", () => {
    const workflow = new Workflow();
    const sampler = workflow.addNode(KSampler, samplerInputs(addSources(workflow)));

    sampler.steps = 30;

    expect(workflow.getComfyUIWorkflowFormat()[sampler.nodeId]!.inputs["steps"]).toBe(30);
  });

  it("chains nodes through named outputs", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const first = workflow.addNode(KSampler, samplerInputs(sources));
    const second = workflow.addNode(
      KSampler,
      samplerInputs(sources, { latent_image: first.output.LATENT, denoise: 0.5 }),
    );

    const sent = JSON.parse(JSON.stringify(workflow.getComfyUIWorkflowFormat()));

    expect(sent[first.nodeId].inputs.model).toEqual(["CheckpointLoaderSimple_0", 0]);
    expect(sent[second.nodeId].inputs.latent_image).toEqual(["KSampler_4", 0]);
  });

  it("returns the outputs of a node", () => {
    const workflow = new Workflow();
    const node = workflow.addNode(KSampler, samplerInputs(addSources(workflow)));

    expect(workflow.getNodeOutput(node.nodeId)["LATENT"]).toEqual([node.nodeId, 0]);
  });

  it("throws NodeNotFoundError for the outputs of an unknown node", () => {
    const workflow = new Workflow();

    expectError(() => workflow.getNodeOutput("Missing_0"), NodeNotFoundError, /Missing_0/);
  });

  it("throws node errors", () => {
    for (const type of [NodeNotFoundError, NodeHasDependentsError, InvalidNodeChangeError, InvalidLinkError]) {
      expect(type.prototype).toBeInstanceOf(NodeException);
    }
  });
});

describe("dependencies", () => {
  it("are none for a node without links", () => {
    const workflow = new Workflow();
    const checkpoint = workflow.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });

    expect(workflow.dependencies(checkpoint.nodeId)).toEqual(new Set());
  });

  it("are the nodes whose outputs a node uses", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const sampler = workflow.addNode(KSampler, samplerInputs(sources));

    expect(workflow.dependencies(sampler.nodeId)).toEqual(new Set(sourceNodes(sources).map((node) => node.nodeId)));
  });

  it("have the nodes that use them as dependents", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const sampler = workflow.addNode(KSampler, samplerInputs(sources));

    expect(workflow.dependents(sources.checkpoint.nodeId)).toEqual(
      new Set([sources.positive.nodeId, sources.negative.nodeId, sampler.nodeId]),
    );
  });

  it("come from links written by hand", () => {
    const workflow = new Workflow();
    const checkpoint = workflow.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });

    const text = workflow.addNode(TextNode, { text: "a cat", clip: [checkpoint.nodeId, 1] });

    expect(workflow.dependencies(text.nodeId)).toEqual(new Set([checkpoint.nodeId]));
  });

  it("must be in the workflow", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);

    expectError(
      () => workflow.addNode(KSampler, samplerInputs(sources, { latent_image: ["LatentNode_9", 0] })),
      NodeNotFoundError,
      /LatentNode_9/,
    );
  });

  it("cannot be the node itself", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);

    const error = expectError(
      () => workflow.addNode(KSampler, samplerInputs(sources, { latent_image: ["KSampler_4", 0] })),
      CycleError,
    );

    expect(error.cycle.map((node) => node.data)).toEqual(["KSampler_4", "KSampler_4"]);
  });

  it.each([
    ["missing-node", ["LatentNode_9", 0] as Link, NodeNotFoundError],
    ["own-output", ["KSampler_4", 0] as Link, CycleError],
  ] as const)("leave the workflow unchanged when a node is rejected (%s)", (_, latentImage, type) => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const before = state(workflow);

    expectError(() => workflow.addNode(KSampler, samplerInputs(sources, { latent_image: latentImage })), type);

    expectUnchanged(workflow, before);
    expectError(() => workflow.dependencies("KSampler_4"), NodeNotFoundError);
  });

  it("follow an input to its new link", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const sampler = workflow.addNode(KSampler, samplerInputs(sources));
    const otherLatent = workflow.addNode(LatentNode);

    sampler.latent_image = otherLatent.output.LATENT;

    expect(workflow.dependencies(sampler.nodeId).has(otherLatent.nodeId)).toBe(true);
    expect(workflow.dependents(sources.latent.nodeId)).toEqual(new Set());
  });

  it("stay while another input still uses the node", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const switchNode = workflow.addNode(ComfySwitchNode, {
      switch: true,
      on_true: sources.positive.output.CONDITIONING,
      on_false: sources.positive.output.CONDITIONING,
    });

    switchNode.on_false = sources.negative.output.CONDITIONING;

    expect(workflow.dependencies(switchNode.nodeId).has(sources.positive.nodeId)).toBe(true);
  });

  it("stay the same when a value input changes", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const sampler = workflow.addNode(KSampler, samplerInputs(sources));
    const before = workflow.dependencies(sampler.nodeId);

    sampler.steps = 30;

    expect(workflow.dependencies(sampler.nodeId)).toEqual(before);
  });

  it.each(["dependencies", "dependents"] as const)("throw NodeNotFoundError for an unknown node (%s)", (query) => {
    const workflow = new Workflow();

    expectError(() => workflow[query]("Missing_0"), NodeNotFoundError, /Missing_0/);
  });
});

describe("changing a link of a node in a workflow", () => {
  it("is rejected when it closes a cycle", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const first = workflow.addNode(KSampler, samplerInputs(sources));
    const second = workflow.addNode(KSampler, samplerInputs(sources, { latent_image: first.output.LATENT }));

    const error = expectError(
      () => {
        first.latent_image = second.output.LATENT;
      },
      InvalidNodeChangeError,
      "Cannot change 'latent_image' of node 'KSampler_4': node 'KSampler_5' depends on 'KSampler_4', " +
        "so the change would create the cycle 'KSampler_4' -> 'KSampler_5' -> 'KSampler_4'.",
    );

    expect(error.cause).toBeInstanceOf(CycleError);
    expect((error.cause as CycleError).cycle.map((node) => node.data)).toEqual([
      first.nodeId,
      second.nodeId,
      first.nodeId,
    ]);
  });

  it("is rejected when the node would use its own output", () => {
    const workflow = new Workflow();
    const sampler = workflow.addNode(KSampler, samplerInputs(addSources(workflow)));

    expectError(
      () => {
        sampler.latent_image = sampler.output.LATENT;
      },
      InvalidNodeChangeError,
      "Cannot change 'latent_image' of node 'KSampler_4': a node cannot use its own output.",
    );
  });

  it("is rejected when the node it links is not in the workflow", () => {
    const workflow = new Workflow();
    const sampler = workflow.addNode(KSampler, samplerInputs(addSources(workflow)));

    expectError(
      () => {
        sampler.latent_image = ["LatentNode_9", 0];
      },
      InvalidNodeChangeError,
      "Cannot change 'latent_image' of node 'KSampler_4': node 'LatentNode_9' is not in the workflow.",
    );
  });

  it.each([
    ["cycle", (second: KSampler): Link => second.output.LATENT],
    ["missing-node", (): Link => ["LatentNode_9", 0]],
  ] as const)("keeps the old input and the dependencies when rejected (%s)", (_, makeLink) => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const first = workflow.addNode(KSampler, samplerInputs(sources));
    const second = workflow.addNode(KSampler, samplerInputs(sources, { latent_image: first.output.LATENT }));
    const before = state(workflow);

    expectError(() => {
      first.latent_image = makeLink(second);
    }, InvalidNodeChangeError);

    expect(first.latent_image).toEqual(sources.latent.output.LATENT);
    expectUnchanged(workflow, before);
  });

  it("leaves an unset input unset when rejected", () => {
    const workflow = new Workflow();
    const node = workflow.addNode(MaskNode);

    expectError(() => {
      node.mask = ["LatentNode_9", 0];
    }, InvalidNodeChangeError);

    expect(node.mask).toBeUndefined();
    expect(node.inputs()).not.toHaveProperty("mask");
  });

  it("drops the dependency when the input is removed", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const node = workflow.addNode(MaskNode);
    node.mask = sources.latent.output.LATENT;

    node.mask = undefined;

    expect(workflow.dependencies(node.nodeId)).toEqual(new Set());
  });

  it("cannot remove an input that is not optional", () => {
    const workflow = new Workflow();
    const sampler = workflow.addNode(KSampler, samplerInputs(addSources(workflow)));
    const before = state(workflow);

    expectError(
      () => {
        sampler.latent_image = undefined as never;
      },
      InvalidNodeParameterValue,
      "latent_image cannot be removed because it is not optional",
    );

    expectUnchanged(workflow, before);
  });

  it("is rejected when the output has a type the input does not take", () => {
    const workflow = new Workflow();
    const { checkpoint, text } = typedSources(workflow);
    const before = state(workflow);

    expectError(
      () => workflow.updateNode(text.nodeId, { text: "a dog", clip: checkpoint.output.MODEL }),
      InvalidNodeChangeError,
      "Cannot change 'clip' of node 'CLIPTextEncode_1': it would use output 0 of node " +
        "'CheckpointLoaderSimple_0', which is MODEL, but the input takes CLIP.",
    );

    expectUnchanged(workflow, before);
  });

  it("is rejected when the source has no such output", () => {
    const workflow = new Workflow();
    const { checkpoint, text } = typedSources(workflow);

    expectError(
      () => {
        text.clip = [checkpoint.nodeId, 3];
      },
      InvalidNodeChangeError,
      "Cannot change 'clip' of node 'CLIPTextEncode_1': it would use output 3 of node " +
        "'CheckpointLoaderSimple_0', but CheckpointLoaderSimple has no output 3.",
    );

    expect(text.clip).toEqual(checkpoint.output.CLIP);
  });
});

describe("removeNode", () => {
  it("takes the node out of the workflow", () => {
    const workflow = new Workflow();
    const sampler = workflow.addNode(KSampler, samplerInputs(addSources(workflow)));

    workflow.removeNode(sampler.nodeId);

    expect(workflow.nodes.has(sampler.nodeId)).toBe(false);
    expect(workflow.getComfyUIWorkflowFormat()).not.toHaveProperty(sampler.nodeId);
  });

  it("takes the node out of the dependents of the nodes it used", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const sampler = workflow.addNode(KSampler, samplerInputs(sources));

    workflow.removeNode(sampler.nodeId);

    expect(
      Object.fromEntries(sourceNodes(sources).map((node) => [node.nodeId, workflow.dependents(node.nodeId)])),
    ).toEqual({
      [sources.checkpoint.nodeId]: new Set([sources.positive.nodeId, sources.negative.nodeId]),
      [sources.positive.nodeId]: new Set(),
      [sources.negative.nodeId]: new Set(),
      [sources.latent.nodeId]: new Set(),
    });
  });

  it.each(["getNodeOutput", "dependencies", "dependents", "removeNode"] as const)(
    "leaves an unknown node id (%s)",
    (query) => {
      const workflow = new Workflow();
      const node = workflow.addNode(LatentNode);
      workflow.removeNode(node.nodeId);

      expectError(() => workflow[query](node.nodeId), NodeNotFoundError, new RegExp(node.nodeId));
    },
  );

  it("leaves no node to link to", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    workflow.removeNode(sources.latent.nodeId);

    expectError(
      () => workflow.addNode(KSampler, samplerInputs(sources)),
      NodeNotFoundError,
      new RegExp(sources.latent.nodeId),
    );
  });

  it("keeps the changes of a removed node out of the workflow", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const sampler = workflow.addNode(KSampler, samplerInputs(sources));
    workflow.removeNode(sampler.nodeId);
    const before = state(workflow);

    sampler.latent_image = sources.latent.output.LATENT;

    expectUnchanged(workflow, before);
  });

  it("does not reuse the ids of removed nodes", () => {
    const workflow = new Workflow();
    const removed = workflow.addNode(LatentNode);
    workflow.removeNode(removed.nodeId);

    const added = workflow.addNode(LatentNode);

    expect(added.nodeId).not.toBe(removed.nodeId);
  });

  it("throws NodeNotFoundError for an unknown node", () => {
    const workflow = new Workflow();

    expectError(() => workflow.removeNode("Missing_0"), NodeNotFoundError, /Missing_0/);
  });

  it("rejects a node with dependents", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    workflow.addNode(KSampler, samplerInputs(sources));

    expectError(
      () => workflow.removeNode(sources.latent.nodeId),
      NodeHasDependentsError,
      "Cannot remove node 'LatentNode_3': node 'KSampler_4' depends on it.",
    );
  });

  it("lists every dependent in the order they were added", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    workflow.addNode(KSampler, samplerInputs(sources));

    const error = expectError(
      () => workflow.removeNode(sources.checkpoint.nodeId),
      NodeHasDependentsError,
      "Cannot remove node 'CheckpointLoaderSimple_0': nodes 'TextNode_1', 'TextNode_2' and 'KSampler_4' depend on it.",
    );

    expect(error.nodeId).toBe("CheckpointLoaderSimple_0");
    expect(error.dependents).toEqual(["TextNode_1", "TextNode_2", "KSampler_4"]);
  });

  it("leaves the workflow unchanged when rejected", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    workflow.addNode(KSampler, samplerInputs(sources));
    const before = state(workflow);

    expectError(() => workflow.removeNode(sources.checkpoint.nodeId), NodeHasDependentsError);

    expectUnchanged(workflow, before);
  });

  it("keeps checking a node it did not remove", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const first = workflow.addNode(KSampler, samplerInputs(sources));
    const second = workflow.addNode(KSampler, samplerInputs(sources, { latent_image: first.output.LATENT }));

    expectError(() => workflow.removeNode(first.nodeId), NodeHasDependentsError);

    expectError(() => {
      first.latent_image = second.output.LATENT;
    }, InvalidNodeChangeError);
  });
});

describe("the id of a node in a workflow", () => {
  it("cannot change while other nodes depend on it", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    workflow.addNode(KSampler, samplerInputs(sources));

    expectError(
      () => {
        sources.checkpoint.nodeId = "Loader";
      },
      InvalidNodeChangeError,
      "Cannot change 'nodeId' of node 'CheckpointLoaderSimple_0': " +
        "nodes 'TextNode_1', 'TextNode_2' and 'KSampler_4' depend on it.",
    );

    expect(sources.checkpoint.nodeId).toBe("CheckpointLoaderSimple_0");
  });

  it("cannot change", () => {
    const workflow = new Workflow();
    const node = workflow.addNode(LatentNode);
    const before = state(workflow);

    expectError(
      () => {
        node.nodeId = "Latent";
      },
      InvalidNodeChangeError,
      "Cannot change 'nodeId' of node 'LatentNode_0': a node keeps its id while it is in a workflow.",
    );

    expectUnchanged(workflow, before);
  });
});

describe("updateNode", () => {
  it("changes the given inputs", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const sampler = workflow.addNode(KSampler, samplerInputs(sources));
    const otherLatent = workflow.addNode(LatentNode);

    workflow.updateNode(sampler.nodeId, { steps: 30, latent_image: otherLatent.output.LATENT });

    const inputs = workflow.getComfyUIWorkflowFormat()[sampler.nodeId]!.inputs;
    expect([inputs["steps"], inputs["latent_image"]]).toEqual([30, ["LatentNode_5", 0]]);
    expect(workflow.dependents(otherLatent.nodeId)).toEqual(new Set([sampler.nodeId]));
  });

  it("leaves the later changes checked", () => {
    const workflow = new Workflow();
    const sampler = workflow.addNode(KSampler, samplerInputs(addSources(workflow)));
    workflow.updateNode(sampler.nodeId, { steps: 30 });

    expectError(() => {
      sampler.steps = 0;
    }, InvalidNodeParameterValue, /steps/);

    expect(sampler.steps).toBe(30);
  });

  it("can swap positive and negative", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const sampler = workflow.addNode(KSampler, samplerInputs(sources));
    const positive = sources.positive.output.CONDITIONING;
    const negative = sources.negative.output.CONDITIONING;

    workflow.updateNode(sampler.nodeId, { positive: negative, negative: positive });

    expect([sampler.positive, sampler.negative]).toEqual([negative, positive]);
  });

  it("can link an unset input", () => {
    const workflow = new Workflow();
    const latent = workflow.addNode(LatentNode);
    const node = workflow.addNode(MaskNode);

    workflow.updateNode(node.nodeId, { mask: latent.output.LATENT });

    expect(workflow.dependencies(node.nodeId)).toEqual(new Set([latent.nodeId]));
  });

  it.each([
    ["cycle", (second: KSampler): Link => second.output.LATENT],
    ["missing-node", (): Link => ["LatentNode_9", 0]],
  ] as const)("changes none of the inputs when rejected (%s)", (_, makeLink) => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const first = workflow.addNode(KSampler, samplerInputs(sources));
    const second = workflow.addNode(KSampler, samplerInputs(sources, { latent_image: first.output.LATENT }));
    const before = state(workflow);

    expectError(
      () =>
        workflow.updateNode(first.nodeId, {
          steps: 30,
          positive: sources.negative.output.CONDITIONING,
          latent_image: makeLink(second),
        }),
      InvalidNodeChangeError,
      /'latent_image'/,
    );

    expectUnchanged(workflow, before);
  });

  it("leaves unset inputs unset when rejected", () => {
    const workflow = new Workflow();
    const latent = workflow.addNode(LatentNode);
    const node = workflow.addNode(MaskNode);

    expectError(
      () => workflow.updateNode(node.nodeId, { mask: latent.output.LATENT, image: ["LatentNode_9", 0] }),
      InvalidNodeChangeError,
      /'image'/,
    );

    expect(node.mask).toBeUndefined();
    expect(workflow.dependents(latent.nodeId)).toEqual(new Set());
  });

  it.each(["stpes", "output", "nodeId", "constructor"])("rejects %j, which is not an input", (name) => {
    const workflow = new Workflow();
    const sampler = workflow.addNode(KSampler, samplerInputs(addSources(workflow)));
    const before = state(workflow);

    expectError(
      () => workflow.updateNode(sampler.nodeId, { steps: 30, [name]: null }),
      InvalidNodeChangeError,
      `Cannot change '${name}' of node 'KSampler_4': KSampler has no input '${name}'.`,
    );

    expectUnchanged(workflow, before);
  });

  it("keeps checking the node after a rejected update", () => {
    const workflow = new Workflow();
    const sampler = workflow.addNode(KSampler, samplerInputs(addSources(workflow)));

    expectError(() => workflow.updateNode(sampler.nodeId, { nodeId: null }), InvalidNodeChangeError);

    expectError(() => {
      sampler.latent_image = sampler.output.LATENT;
    }, InvalidNodeChangeError);
  });

  it("changes nothing when a value has another type", () => {
    const workflow = new Workflow();
    const sampler = workflow.addNode(KSampler, samplerInputs(addSources(workflow)));
    const before = state(workflow);

    expectError(
      () => workflow.updateNode(sampler.nodeId, { steps: 30, cfg: "high" }),
      InvalidNodeParameterValue,
      "cfg must be a number or a link, not string",
    );

    expectUnchanged(workflow, before);
  });

  it("changes nothing when a value is out of range", () => {
    const workflow = new Workflow();
    const sampler = workflow.addNode(KSampler, samplerInputs(addSources(workflow)));
    const before = state(workflow);

    expectError(
      () => workflow.updateNode(sampler.nodeId, { cfg: 7.0, steps: 0 }),
      InvalidNodeParameterValue,
      "steps must be in closed range: [1, 10000]",
    );

    expectUnchanged(workflow, before);
  });

  it("throws NodeNotFoundError for an unknown node", () => {
    const workflow = new Workflow();

    expectError(() => workflow.updateNode("Missing_0", { steps: 30 }), NodeNotFoundError, /Missing_0/);
  });

  it("checks the node once, with every change", () => {
    const workflow = new Workflow();
    const node = workflow.addNode(RangeNode);

    workflow.updateNode(node.nodeId, { low: 5, high: 10 });

    expect([node.low, node.high]).toEqual([5, 10]);
  });

  it("changes nothing when the result fails the checks of the node", () => {
    const workflow = new Workflow();
    const node = workflow.addNode(RangeNode);
    const before = state(workflow);

    expectError(
      () => workflow.updateNode(node.nodeId, { high: 10, low: 20 }),
      InvalidNodeParameterValue,
      "low must not be above high",
    );

    expectUnchanged(workflow, before);
  });
});

describe("positive and negative", () => {
  it("cannot be the same link in addNode", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const before = state(workflow);

    expectError(
      () =>
        workflow.addNode(KSampler, samplerInputs(sources, { negative: sources.positive.output.CONDITIONING })),
      InvalidNodeParameterValue,
    );

    expectUnchanged(workflow, before);
  });

  it("cannot be the same link in updateNode", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);
    const sampler = workflow.addNode(KSampler, samplerInputs(sources));
    const before = state(workflow);

    expectError(
      () => workflow.updateNode(sampler.nodeId, { negative: sources.positive.output.CONDITIONING }),
      InvalidNodeParameterValue,
    );

    expectUnchanged(workflow, before);
  });
});

function typedSources(workflow: Workflow) {
  const checkpoint = workflow.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
  const text = workflow.addNode(CLIPTextEncode, { text: "a cat", clip: checkpoint.output.CLIP });
  const latent = workflow.addNode(EmptyLatentImage);
  const negative = workflow.addNode(CLIPTextEncode, { text: "blurry", clip: checkpoint.output.CLIP });
  return { checkpoint, text, latent, negative };
}

function typedSamplerInputs(
  sources: ReturnType<typeof typedSources>,
  overrides: Partial<KSamplerInputs> = {},
): KSamplerInputs {
  return {
    model: sources.checkpoint.output.MODEL,
    sampler_name: "euler",
    scheduler: "normal",
    positive: sources.text.output.CONDITIONING,
    negative: sources.negative.output.CONDITIONING,
    latent_image: sources.latent.output.LATENT,
    ...overrides,
  };
}

function videoParts(workflow: Workflow): GetVideoComponents {
  const image = workflow.addNode(EmptyImage, { width: 8, height: 8 });
  const video = workflow.addNode(CreateVideo, { images: image.output.IMAGE });
  return workflow.addNode(GetVideoComponents, { video: video.output.VIDEO });
}

class SpacedSocketNode extends BaseNode<{ value: Link }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUT_SOCKET_TYPES = { value: "FLOAT, INT" };
  static override readonly INPUTS = { value: link() };
}

class RangeNode extends BaseNode<{ low?: number; high?: number }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUTS = { low: int({ default: 0 }), high: int({ default: 1 }) };

  static override validate(inputs: Readonly<Record<string, unknown>>): void {
    if ((inputs["low"] as number) > (inputs["high"] as number)) {
      throw new InvalidNodeParameterValue("low must not be above high");
    }
  }

  declare low: number;
  declare high: number;
}

describe("socket types", () => {
  it("reject a link to an output of a type the input does not take", () => {
    const workflow = new Workflow();
    const { checkpoint, text } = typedSources(workflow);
    const before = state(workflow);

    expectError(
      () => workflow.addNode(VAEDecode, { samples: text.output.CONDITIONING, vae: checkpoint.output.VAE }),
      InvalidLinkError,
      "Cannot add node 'VAEDecode_4': input 'samples' uses output 0 of node 'CLIPTextEncode_1', " +
        "which is CONDITIONING, but the input takes LATENT.",
    );

    expectUnchanged(workflow, before);
    expectError(() => workflow.dependencies("VAEDecode_4"), NodeNotFoundError);
  });

  it("reject a link to an output the source does not have", () => {
    const workflow = new Workflow();
    const { checkpoint } = typedSources(workflow);

    expectError(
      () => workflow.addNode(CLIPTextEncode, { text: "a dog", clip: [checkpoint.nodeId, 5] }),
      InvalidLinkError,
      "Cannot add node 'CLIPTextEncode_4': input 'clip' uses output 5 of node 'CheckpointLoaderSimple_0', " +
        "but CheckpointLoaderSimple has no output 5.",
    );
  });

  it("let a widget input take a link of its own type only", () => {
    const workflow = new Workflow();
    const sources = typedSources(workflow);
    const steps = workflow.addNode(PrimitiveInt, { value: 30 });
    const sampler = workflow.addNode(KSampler, typedSamplerInputs(sources, { steps: steps.output.INT }));

    expectError(
      () => workflow.addNode(KSampler, typedSamplerInputs(sources, { cfg: steps.output.INT })),
      InvalidLinkError,
      "Cannot add node 'KSampler_6': input 'cfg' uses output 0 of node 'PrimitiveInt_4', " +
        "which is INT, but the input takes FLOAT.",
    );
    expect(workflow.dependencies(sampler.nodeId).has(steps.nodeId)).toBe(true);
  });

  it("let any and match type sockets take every output", () => {
    const workflow = new Workflow();
    const sources = typedSources(workflow);

    const switchNode = workflow.addNode(ComfySwitchNode, {
      switch: true,
      on_true: sources.checkpoint.output.MODEL,
      on_false: sources.checkpoint.output.MODEL,
    });
    const sampler = workflow.addNode(KSampler, typedSamplerInputs(sources, { model: switchNode.output.output }));
    const preview = workflow.addNode(PreviewAny, { source: sampler.output.LATENT });

    expect(workflow.dependencies(preview.nodeId)).toEqual(new Set([sampler.nodeId]));
    expect(workflow.dependencies(sampler.nodeId).has(switchNode.nodeId)).toBe(true);
  });

  it("let a union socket take each of its types", () => {
    const workflow = new Workflow();
    const { checkpoint, text } = typedSources(workflow);
    const vae = checkpoint.output.VAE;
    const integer = workflow.addNode(PrimitiveInt, { value: 24 });
    const real = workflow.addNode(PrimitiveFloat, { value: 23.976 });

    workflow.addNode(LTXVEmptyLatentAudio, { audio_vae: vae, frame_rate: integer.output.INT });
    workflow.addNode(LTXVEmptyLatentAudio, { audio_vae: vae, frame_rate: real.output.FLOAT });
    const error = expectError(
      () => workflow.addNode(LTXVEmptyLatentAudio, { audio_vae: vae, frame_rate: text.output.CONDITIONING }),
      InvalidLinkError,
    );

    expect(error.message.endsWith("which is CONDITIONING, but the input takes FLOAT,INT.")).toBe(true);
  });

  it("do not let a V1 combo take a COMBO output, as ComfyUI does not", () => {
    const workflow = new Workflow();
    const sources = typedSources(workflow);
    const parts = videoParts(workflow);
    const before = state(workflow);

    expectError(
      () => workflow.addNode(KSampler, typedSamplerInputs(sources, { sampler_name: parts.output.bit_depth })),
      InvalidLinkError,
      "Cannot add node 'KSampler_7': input 'sampler_name' uses output 3 of node 'GetVideoComponents_6', " +
        "which is COMBO, but the input takes V1 COMBO.",
    );

    expectUnchanged(workflow, before);
  });

  it("let a V1 combo take any and match type outputs", () => {
    const workflow = new Workflow();
    const sources = typedSources(workflow);
    const name = workflow.addNode(PrimitiveString, { value: "euler" });
    const item = workflow.addNode(GetItemFromList, { list: name.output.STRING });
    const switchNode = workflow.addNode(ComfySwitchNode, {
      switch: true,
      on_true: name.output.STRING,
      on_false: name.output.STRING,
    });

    const byItem = workflow.addNode(KSampler, typedSamplerInputs(sources, { sampler_name: item.output["*"] }));
    const bySwitch = workflow.addNode(KSampler, typedSamplerInputs(sources, { scheduler: switchNode.output.output }));

    expect(workflow.dependencies(byItem.nodeId).has(item.nodeId)).toBe(true);
    expect(workflow.dependencies(bySwitch.nodeId).has(switchNode.nodeId)).toBe(true);
  });

  it("let a V3 combo take a COMBO output", () => {
    const workflow = new Workflow();
    const parts = videoParts(workflow);

    const video = workflow.addNode(CreateVideo, {
      images: parts.output.images,
      bit_depth: parts.output.bit_depth,
      color_space: parts.output.color_space,
    });

    expect(workflow.dependencies(video.nodeId)).toEqual(new Set([parts.nodeId]));
  });

  it("reject a COMBO output into a V1 combo in updateNode", () => {
    const workflow = new Workflow();
    const sources = typedSources(workflow);
    const sampler = workflow.addNode(KSampler, typedSamplerInputs(sources));
    const parts = videoParts(workflow);
    const before = state(workflow);

    expectError(
      () => workflow.updateNode(sampler.nodeId, { scheduler: parts.output.color_space }),
      InvalidNodeChangeError,
      /which is COMBO, but the input takes V1 COMBO/,
    );

    expectUnchanged(workflow, before);
  });

  it("are read without the spaces around commas", () => {
    const workflow = new Workflow();
    const integer = workflow.addNode(PrimitiveInt, { value: 24 });

    const node = workflow.addNode(SpacedSocketNode, { value: integer.output.INT });

    expect(workflow.dependencies(node.nodeId)).toEqual(new Set([integer.nodeId]));
  });

  it("are not checked for nodes without them", () => {
    const workflow = new Workflow();
    const sources = addSources(workflow);

    const node = workflow.addNode(TextNode, { text: "a dog", clip: sources.checkpoint.output.MODEL });

    expect(workflow.dependencies(node.nodeId)).toEqual(new Set([sources.checkpoint.nodeId]));
  });
});

