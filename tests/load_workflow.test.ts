// Port of the Python test_load_workflow.py. A JavaScript object only has
// string keys, so the case of a node id that is not a string has no
// counterpart; nor do the pickle and copy tests.
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { NODE_TYPES } from "../src/_internal/registry.js";
import { CycleError } from "../src/errors/data_structure.js";
import {
  InvalidNodeChangeError,
  InvalidNodeParameterValue,
  NodeHasDependentsError,
  NodeNotFoundError,
} from "../src/errors/node.js";
import { InvalidWorkflowError } from "../src/errors/workflow.js";
import { WorkflowException } from "../src/exceptions.js";
import { audio, image, model } from "../src/nodes.js";
import type { ComfyUIWorkflow } from "../src/nodes/base.js";
import { Workflow } from "../src/workflow.js";
import { expectError, expectUnchanged, workflowState } from "./helpers.js";

const { CLIPTextEncode } = model.conditioning;
const { EmptyLatentImage, VAEDecode } = model.latent;
const { CheckpointLoaderSimple } = model.loaders;
const { KSampler } = model.sampling;
const { EpsilonScaling } = model.patch.unet;
const { MoGeRender } = image.geometry_estimation;
const { SaveImage } = image;
const { RecordAudio } = audio;

const WAN_WORKFLOW = new URL("../video_wan2_2_14B_i2v.json", import.meta.url);

// The default workflow, as sent by ComfyUI's script_examples/basic_api_example.py.
const BASIC_API_EXAMPLE: ComfyUIWorkflow = {
  "3": {
    class_type: "KSampler",
    inputs: {
      cfg: 8,
      denoise: 1,
      latent_image: ["5", 0],
      model: ["4", 0],
      negative: ["7", 0],
      positive: ["6", 0],
      sampler_name: "euler",
      scheduler: "normal",
      seed: 8566257,
      steps: 20,
    },
  },
  "4": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "v1-5-pruned-emaonly.safetensors" } },
  "5": { class_type: "EmptyLatentImage", inputs: { batch_size: 1, height: 512, width: 512 } },
  "6": { class_type: "CLIPTextEncode", inputs: { clip: ["4", 1], text: "masterpiece best quality girl" } },
  "7": { class_type: "CLIPTextEncode", inputs: { clip: ["4", 1], text: "bad hands" } },
  "8": { class_type: "VAEDecode", inputs: { samples: ["3", 0], vae: ["4", 2] } },
  "9": { class_type: "SaveImage", inputs: { filename_prefix: "ComfyUI", images: ["8", 0] } },
};

const MOGE_RENDER: ComfyUIWorkflow = {
  "1": { class_type: "LoadMoGeModel", inputs: { model_name: "moge.safetensors" } },
  "2": { class_type: "LoadImage", inputs: { image: "photo.png" } },
  "3": { class_type: "MoGeInference", inputs: { moge_model: ["1", 0], image: ["2", 0] } },
  "4": { class_type: "MoGeRender", inputs: { moge_geometry: ["3", 0], output: "normal_opengl" } },
};

function withInputs(nodeId: string, inputs: Record<string, unknown>): ComfyUIWorkflow {
  const workflow = structuredClone(BASIC_API_EXAMPLE);
  Object.assign(workflow[nodeId]!.inputs, inputs);
  return workflow;
}

function withoutInputs(nodeId: string, ...names: string[]): ComfyUIWorkflow {
  const workflow = structuredClone(BASIC_API_EXAMPLE);
  for (const name of names) {
    delete workflow[nodeId]!.inputs[name];
  }
  return workflow;
}

const IMG2IMG_LOOP: ComfyUIWorkflow = {
  ...withInputs("3", { latent_image: ["10", 0] }),
  "10": { class_type: "VAEEncode", inputs: { pixels: ["8", 0], vae: ["4", 2] } },
};

function loaded(workflowFormat: unknown): Workflow {
  const workflow = new Workflow();
  workflow.loadWorkflow(workflowFormat);
  return workflow;
}

function exported(workflow: Workflow): unknown {
  return JSON.parse(JSON.stringify(workflow.getComfyUIWorkflowFormat()));
}

function linkedSources(workflowFormat: ComfyUIWorkflow): Record<string, Set<string>> {
  return Object.fromEntries(
    Object.entries(workflowFormat).map(([nodeId, node]) => [
      nodeId,
      new Set(
        Object.values(node.inputs)
          .filter((value): value is [string, number] => Array.isArray(value))
          .map((value) => value[0]),
      ),
    ]),
  );
}

function dependencies(workflow: Workflow): Record<string, Set<string>> {
  return Object.fromEntries([...workflow.nodes.keys()].map((nodeId) => [nodeId, workflow.dependencies(nodeId)]));
}

function links(workflow: Workflow) {
  return Object.fromEntries(
    [...workflow.nodes.keys()].map((nodeId) => [nodeId, [workflow.dependencies(nodeId), workflow.dependents(nodeId)]]),
  );
}

describe("loadWorkflow", () => {
  it("rebuilds the basic API example of ComfyUI", () => {
    expect(exported(loaded(BASIC_API_EXAMPLE))).toEqual(BASIC_API_EXAMPLE);
  });

  it("keeps the ids and class types of the workflow", () => {
    const workflow = loaded(BASIC_API_EXAMPLE);

    expect(Object.fromEntries([...workflow.nodes].map(([nodeId, node]) => [nodeId, node.constructor]))).toEqual({
      "3": KSampler,
      "4": CheckpointLoaderSimple,
      "5": EmptyLatentImage,
      "6": CLIPTextEncode,
      "7": CLIPTextEncode,
      "8": VAEDecode,
      "9": SaveImage,
    });
    expect([...workflow.nodes.keys()]).toEqual(Object.keys(BASIC_API_EXAMPLE));
    for (const [nodeId, node] of workflow.nodes) {
      expect(node.nodeId).toBe(nodeId);
    }
  });

  it("loads links as outputs of the loaded nodes", () => {
    const workflow = loaded(BASIC_API_EXAMPLE);
    const checkpoint = workflow.nodes.get("4") as InstanceType<typeof CheckpointLoaderSimple>;

    expect((workflow.nodes.get("3") as InstanceType<typeof KSampler>).model).toEqual(checkpoint.output.MODEL);
    expect((workflow.nodes.get("6") as InstanceType<typeof CLIPTextEncode>).clip).toEqual(checkpoint.output.CLIP);
    expect((workflow.nodes.get("8") as InstanceType<typeof VAEDecode>).vae).toEqual(checkpoint.output.VAE);
  });

  it("makes links dependencies, even on nodes listed after", () => {
    const workflow = loaded(BASIC_API_EXAMPLE);

    expect(dependencies(workflow)).toEqual(linkedSources(BASIC_API_EXAMPLE));
    expect(workflow.dependents("4")).toEqual(new Set(["3", "6", "7", "8"]));
    expect(workflow.dependents("9")).toEqual(new Set());
  });

  it("loads back a workflow built with addNode", () => {
    const built = new Workflow();
    const checkpoint = built.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
    const positive = built.addNode(CLIPTextEncode, { text: "a cat", clip: checkpoint.output.CLIP });
    const negative = built.addNode(CLIPTextEncode, { text: "blurry", clip: checkpoint.output.CLIP });
    const latent = built.addNode(EmptyLatentImage, { width: 512, height: 512, batch_size: 1 });
    built.addNode(KSampler, {
      model: checkpoint.output.MODEL,
      sampler_name: "euler",
      scheduler: "normal",
      positive: positive.output.CONDITIONING,
      negative: negative.output.CONDITIONING,
      latent_image: latent.output.LATENT,
    });

    const workflow = loaded(built.getComfyUIWorkflowFormat());

    expect(exported(workflow)).toEqual(exported(built));
    expect(links(workflow)).toEqual(links(built));
  });

  it("puts a node without links in the graph", () => {
    const workflow = loaded({ "4": BASIC_API_EXAMPLE["4"] });

    expect([workflow.dependencies("4"), workflow.dependents("4")]).toEqual([new Set(), new Set()]);
    workflow.removeNode("4");
    expect(workflow.nodes.size).toBe(0);
  });

  it("replaces the current workflow", () => {
    const workflow = new Workflow();
    workflow.addNode(CheckpointLoaderSimple, { ckpt_name: "old.safetensors" });

    workflow.loadWorkflow(BASIC_API_EXAMPLE);

    expect([...workflow.nodes.keys()]).toEqual(Object.keys(BASIC_API_EXAMPLE));
    expect(exported(workflow)).toEqual(BASIC_API_EXAMPLE);
    expectError(() => workflow.dependencies("CheckpointLoaderSimple_0"), NodeNotFoundError);
  });

  it("leaves the workflow empty when the workflow loaded is", () => {
    const workflow = loaded(BASIC_API_EXAMPLE);

    workflow.loadWorkflow({});

    expect(workflow.nodes.size).toBe(0);
    expect(workflow.getComfyUIWorkflowFormat()).toEqual({});
    expect(workflow.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" }).nodeId).toBe(
      "CheckpointLoaderSimple_0",
    );
  });

  it("lets the nodes of the replaced workflow no longer affect it", () => {
    const workflow = new Workflow();
    const checkpoint = workflow.addNode(CheckpointLoaderSimple, { ckpt_name: "old.safetensors" });
    const text = workflow.addNode(CLIPTextEncode, { text: "old", clip: checkpoint.output.CLIP });
    workflow.loadWorkflow(BASIC_API_EXAMPLE);

    text.clip = ["4", 1];
    checkpoint.ckpt_name = text.output.CONDITIONING;

    expect(workflow.dependents("4")).toEqual(new Set(["3", "6", "7", "8"]));
    expect(exported(workflow)).toEqual(BASIC_API_EXAMPLE);
  });

  it("does not let a loaded node used by others be removed", () => {
    const workflow = loaded(BASIC_API_EXAMPLE);

    expectError(
      () => workflow.removeNode("4"),
      NodeHasDependentsError,
      "Cannot remove node '4': nodes '3', '6', '7' and '8' depend on it.",
    );
  });

  it("rejects a change to a loaded node that creates a cycle", () => {
    const workflow = loaded({
      ...BASIC_API_EXAMPLE,
      "10": { class_type: "VAEEncode", inputs: { pixels: ["8", 0], vae: ["4", 2] } },
    });

    expectError(
      () => workflow.updateNode("3", { latent_image: ["10", 0] }),
      InvalidNodeChangeError,
      "Cannot change 'latent_image' of node '3': node '10' depends on '3', " +
        "so the change would create the cycle '3' -> '10' -> '8' -> '3'.",
    );
    expect((exported(workflow) as ComfyUIWorkflow)["3"]).toEqual(BASIC_API_EXAMPLE["3"]);
  });

  it("lets a widget take a link of its own type", () => {
    const workflow = loaded({
      ...withInputs("3", { steps: ["11", 0] }),
      "11": { class_type: "PrimitiveInt", inputs: { value: 30 } },
    });

    expect(workflow.dependencies("3")).toEqual(new Set(["4", "5", "6", "7", "11"]));
  });

  it("lets loaded nodes be linked again", () => {
    const workflow = loaded(BASIC_API_EXAMPLE);
    const text = workflow.addNode(CLIPTextEncode, { text: "blurry", clip: ["4", 1] });

    workflow.updateNode("3", { negative: text.output.CONDITIONING });

    expect(workflow.dependencies("3")).toEqual(new Set(["4", "5", "6", text.nodeId]));
    expect(workflow.dependents("7")).toEqual(new Set());
  });

  it("continues the numbering of addNode", () => {
    const built = new Workflow();
    const checkpoint = built.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
    built.addNode(CLIPTextEncode, { text: "a cat", clip: checkpoint.output.CLIP });
    const workflow = loaded(built.getComfyUIWorkflowFormat());

    const node = workflow.addNode(CLIPTextEncode, {
      text: "a dog",
      clip: workflow.getNodeOutput("CheckpointLoaderSimple_0")["CLIP"]!,
    });

    expect(node.nodeId).toBe("CLIPTextEncode_2");
  });

  it("lets addNode skip the ids of loaded nodes", () => {
    const workflow = loaded({
      CheckpointLoaderSimple_1: { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "model.safetensors" } },
    });

    const node = workflow.addNode(CheckpointLoaderSimple, { ckpt_name: "other.safetensors" });

    expect(node.nodeId).toBe("CheckpointLoaderSimple_2");
    expect([...workflow.nodes.keys()]).toEqual(["CheckpointLoaderSimple_1", "CheckpointLoaderSimple_2"]);
  });

  it("does not change the workflow it loads", () => {
    const given = structuredClone(BASIC_API_EXAMPLE);

    const workflow = loaded(given);
    workflow.updateNode("3", { steps: 30, positive: ["7", 0], negative: ["6", 0] });

    expect(given).toEqual(BASIC_API_EXAMPLE);
  });

  it("accepts the _meta of the nodes and leaves it out", () => {
    const withMeta = Object.fromEntries(
      Object.entries(BASIC_API_EXAMPLE).map(([nodeId, node]) => [nodeId, { ...node, _meta: { title: node.class_type } }]),
    );

    expect(exported(loaded(withMeta))).toEqual(BASIC_API_EXAMPLE);
  });

  it("unwraps wrapped values", () => {
    const workflow = loaded(withInputs("6", { text: { __value__: "a cat" } }));

    expect((workflow.nodes.get("6") as InstanceType<typeof CLIPTextEncode>).text).toBe("a cat");
  });

  it("reads wrapped arrays as values and not links", () => {
    const workflow = loaded({ "1": { class_type: "RecordAudio", inputs: { audio: { __value__: [0.25, 0.5] } } } });

    expect((workflow.nodes.get("1") as InstanceType<typeof RecordAudio>).audio).toEqual([0.25, 0.5]);
    expect(workflow.dependencies("1")).toEqual(new Set());
    expect((exported(workflow) as ComfyUIWorkflow)["1"]!.inputs["audio"]).toEqual({ __value__: [0.25, 0.5] });
  });

  it("reads the inputs by their ComfyUI names", () => {
    const workflow = loaded(MOGE_RENDER);

    expect((workflow.nodes.get("4") as InstanceType<typeof MoGeRender>).output_).toBe("normal_opengl");
    expect((exported(workflow) as ComfyUIWorkflow)["4"]!.inputs).toEqual(MOGE_RENDER["4"]!.inputs);
  });

  it("loads a class type that is not a class name", () => {
    const workflow = loaded({
      "4": BASIC_API_EXAMPLE["4"],
      "10": { class_type: "Epsilon Scaling", inputs: { model: ["4", 0], scaling_factor: 1.005 } },
    });

    expect(workflow.nodes.get("10")).toBeInstanceOf(EpsilonScaling);
    expect(workflow.dependencies("10")).toEqual(new Set(["4"]));
  });

  it("only loads the node classes of the library", () => {
    expectError(
      () => loaded({ "1": { class_type: "LocalNode", inputs: {} } }),
      InvalidWorkflowError,
      "Cannot load node '1': unknown class_type 'LocalNode'.",
    );
  });

  it("knows a single node class for each class type", () => {
    expect(NODE_TYPES.size).toBeGreaterThan(600);
    for (const [classType, type] of NODE_TYPES) {
      expect(type.CLASS_TYPE).toBe(classType);
    }
  });

  it.skipIf(!existsSync(WAN_WORKFLOW))("loads the Wan workflow exported from ComfyUI", () => {
    const exportedWan = JSON.parse(readFileSync(WAN_WORKFLOW, "utf8")) as ComfyUIWorkflow;

    const workflow = loaded(exportedWan);

    expect(exported(workflow)).toEqual(
      Object.fromEntries(
        Object.entries(exportedWan).map(([nodeId, node]) => [
          nodeId,
          { class_type: node.class_type, inputs: node.inputs },
        ]),
      ),
    );
    expect(dependencies(workflow)).toEqual(linkedSources(exportedWan));
  });
});

const INVALID: readonly (readonly [string, unknown, string])[] = [
  ["workflow-not-object", [], "Cannot load the workflow: expected an object of nodes, got array."],
  ["workflow-null", null, "Cannot load the workflow: expected an object of nodes, got null."],
  [
    "ui-format",
    {
      last_node_id: 9,
      last_link_id: 9,
      nodes: [{ id: 4, type: "CheckpointLoaderSimple" }],
      links: [[1, 4, 0, 3, 0, "MODEL"]],
      version: 0.4,
    },
    "Cannot load the workflow: it is in the ComfyUI UI format; export it with 'Export (API)' and load that file.",
  ],
  [
    "node-not-object",
    { "4": "CheckpointLoaderSimple" },
    "Cannot load node '4': expected an object with 'class_type' and 'inputs', got string.",
  ],
  [
    "no-class-type",
    { "4": { inputs: { ckpt_name: "model.safetensors" } } },
    "Cannot load node '4': 'class_type' is missing.",
  ],
  [
    "class-type-not-string",
    { "4": { class_type: 4, inputs: {} } },
    "Cannot load node '4': 'class_type' must be a string, got integer.",
  ],
  [
    "unknown-class-type",
    { "4": { class_type: "NoSuchNode", inputs: {} } },
    "Cannot load node '4': unknown class_type 'NoSuchNode'.",
  ],
  [
    "class-type-with-invisible-characters",
    { "4": { class_type: "KSampler​ \t\u{e0001}", inputs: {} } },
    "Cannot load node '4': unknown class_type 'KSampler\\u200b\\xa0\\t\\U000e0001'.",
  ],
  [
    "class-type-of-object-prototype",
    { "4": { class_type: "constructor", inputs: {} } },
    "Cannot load node '4': unknown class_type 'constructor'.",
  ],
  ["no-inputs", { "4": { class_type: "CheckpointLoaderSimple" } }, "Cannot load node '4': 'inputs' is missing."],
  [
    "inputs-not-object",
    { "4": { class_type: "CheckpointLoaderSimple", inputs: ["ckpt_name"] } },
    "Cannot load node '4': 'inputs' must be an object, got array.",
  ],
  ["unknown-input", withInputs("4", { foo: 1 }), "Cannot load node '4': CheckpointLoaderSimple has no input 'foo'."],
  [
    "name-of-a-renamed-input",
    { ...MOGE_RENDER, "4": { class_type: "MoGeRender", inputs: { moge_geometry: ["3", 0], output_: "depth" } } },
    "Cannot load node '4': MoGeRender has no input 'output_'.",
  ],
  ["missing-input", withoutInputs("3", "model"), "Cannot load node '3': KSampler is missing the input 'model'."],
  [
    "missing-inputs",
    withoutInputs("3", "model", "positive", "negative"),
    "Cannot load node '3': KSampler is missing the inputs 'model', 'positive' and 'negative'.",
  ],
  ["invalid-value", withInputs("3", { steps: 0 }), "Cannot load node '3': steps must be in closed range: [1, 10000]."],
  [
    "value-of-another-type",
    withInputs("3", { steps: "20" }),
    "Cannot load node '3': steps must be an integer or a link, not string.",
  ],
  [
    "wrapped-array-for-a-combo",
    withInputs("3", { sampler_name: { __value__: ["euler"] } }),
    "Cannot load node '3': sampler_name must be a string or a link, not array.",
  ],
  [
    "value-for-a-link-only-input",
    withInputs("3", { model: "v1-5-pruned-emaonly.safetensors" }),
    "Cannot load node '3': model must be a link, not string.",
  ],
  [
    "link-id-not-string",
    withInputs("3", { model: [4, 0] }),
    "Cannot load node '3': input 'model' is an array but not a link [node_id, output_index].",
  ],
  [
    "link-too-short",
    withInputs("3", { model: ["4"] }),
    "Cannot load node '3': input 'model' is an array but not a link [node_id, output_index].",
  ],
  [
    "link-too-long",
    withInputs("3", { model: ["4", 0, 1] }),
    "Cannot load node '3': input 'model' is an array but not a link [node_id, output_index].",
  ],
  [
    "boolean-output",
    withInputs("6", { clip: ["4", true] }),
    "Cannot load node '6': input 'clip' is an array but not a link [node_id, output_index].",
  ],
  [
    "link-to-missing-node",
    withInputs("3", { model: ["99", 0] }),
    "Cannot load node '3': input 'model' uses node '99', which is not in the workflow.",
  ],
  [
    "output-out-of-range",
    withInputs("6", { clip: ["4", 5] }),
    "Cannot load node '6': input 'clip' uses output 5 of node '4', but CheckpointLoaderSimple has no output 5.",
  ],
  [
    "negative-output",
    withInputs("6", { clip: ["4", -1] }),
    "Cannot load node '6': input 'clip' uses output -1 of node '4', but CheckpointLoaderSimple has no output -1.",
  ],
  [
    "output-type-mismatch",
    withInputs("3", { model: ["6", 0] }),
    "Cannot load node '3': input 'model' uses output 0 of node '6', which is CONDITIONING, but the input takes MODEL.",
  ],
  [
    "widget-link-type-mismatch",
    withInputs("3", { steps: ["4", 0] }),
    "Cannot load node '3': input 'steps' uses output 0 of node '4', which is MODEL, but the input takes INT.",
  ],
  [
    "type-mismatch-on-a-renamed-input",
    { ...MOGE_RENDER, "4": { class_type: "MoGeRender", inputs: { moge_geometry: ["3", 0], output: ["2", 0] } } },
    "Cannot load node '4': input 'output' uses output 0 of node '2', which is IMAGE, but the input takes COMBO.",
  ],
  [
    "combo-output-into-a-v1-combo",
    {
      "1": { class_type: "EmptyImage", inputs: { width: 8, height: 8 } },
      "2": { class_type: "CreateVideo", inputs: { images: ["1", 0] } },
      "3": { class_type: "GetVideoComponents", inputs: { video: ["2", 0] } },
      "4": {
        class_type: "ImageScale",
        inputs: { image: ["1", 0], upscale_method: ["3", 3], width: 8, height: 8, crop: "disabled" },
      },
    },
    "Cannot load node '4': input 'upscale_method' uses output 3 of node '3', which is COMBO, but the input takes V1 COMBO.",
  ],
  [
    "own-output",
    withInputs("8", { samples: ["8", 0] }),
    "Cannot load node '8': input 'samples' uses the node's own output.",
  ],
  [
    // Objects list integer keys in ascending order, so the nodes load in the
    // order of their ids: the link that closes the cycle is the one of node 3.
    "cycle-through-a-renamed-input",
    {
      "1": BASIC_API_EXAMPLE["4"],
      "2": { class_type: "ComfySwitchNode", inputs: { switch: true, on_true: ["3", 0] } },
      "3": {
        class_type: "ModelMergeSD1",
        inputs: { model1: ["1", 0], model2: ["1", 0], "time_embed.": ["2", 0] },
      },
    },
    "Cannot load node '3': input 'time_embed.' uses node '2', which depends on '3', " +
      "so the workflow has the cycle '3' -> '2' -> '3'.",
  ],
  [
    "cycle",
    IMG2IMG_LOOP,
    "Cannot load node '10': input 'pixels' uses node '8', which depends on '10', " +
      "so the workflow has the cycle '10' -> '8' -> '3' -> '10'.",
  ],
];

describe("an invalid workflow", () => {
  it.each(INVALID)("is rejected with the reason (%s)", (_, workflowFormat, message) => {
    expectError(() => loaded(workflowFormat), InvalidWorkflowError, message);
  });

  it.each([
    ["workflow-not-object", []],
    ["last-node-unknown", { ...BASIC_API_EXAMPLE, "99": { class_type: "NoSuchNode", inputs: {} } }],
    ["invalid-value", withInputs("3", { steps: 0 })],
    ["link-to-missing-node", withInputs("9", { images: ["99", 0] })],
    ["output-out-of-range", withInputs("9", { images: ["8", 1] })],
    ["cycle", IMG2IMG_LOOP],
  ] as const)("keeps the current workflow (%s)", (_, workflowFormat) => {
    const workflow = new Workflow();
    const checkpoint = workflow.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
    const text = workflow.addNode(CLIPTextEncode, { text: "a cat", clip: checkpoint.output.CLIP });
    const before = workflowState(workflow);

    expectError(() => workflow.loadWorkflow(workflowFormat), InvalidWorkflowError);

    expectUnchanged(workflow, before);
    expectError(() => {
      text.clip = text.output.CONDITIONING;
    }, InvalidNodeChangeError);
    expect(workflow.addNode(EmptyLatentImage).nodeId).toBe("EmptyLatentImage_2");
  });

  it("keeps the error of the node as the cause", () => {
    const error = expectError(() => loaded(withInputs("3", { steps: 0 })), InvalidWorkflowError);

    expect(error.cause).toBeInstanceOf(InvalidNodeParameterValue);
    expect(error.nodeId).toBe("3");
  });

  it("is rejected when positive and negative use the same link", () => {
    expectError(
      () => loaded(withInputs("3", { negative: ["6", 0] })),
      InvalidWorkflowError,
      "Cannot load node '3': positive and negative must not use the same input.",
    );
  });

  it("keeps the error of the graph as the cause of a cycle", () => {
    const error = expectError(() => loaded(IMG2IMG_LOOP), InvalidWorkflowError);

    expect(error.cause).toBeInstanceOf(CycleError);
  });

  it("throws a workflow error", () => {
    expect(InvalidWorkflowError.prototype).toBeInstanceOf(WorkflowException);
    const error = new InvalidWorkflowError(null, "a reason");
    expect([error.nodeId, error.reason, error.message]).toEqual([null, "a reason", "Cannot load the workflow: a reason."]);
  });
});
