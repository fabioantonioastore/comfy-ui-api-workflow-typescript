// Port of the Python test_comfy_ui_workflows.py: workflows ComfyUI made,
// rebuilt node by node. The Wan workflow needs video_wan2_2_14B_i2v.json,
// exported from ComfyUI, at the project root.
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { NODE_TYPES } from "../src/_internal/registry.js";
import type { BaseNode, ComfyUIWorkflow, NodeType } from "../src/nodes/base.js";
import { SaveImage } from "../src/nodes/image.js";
import { CLIPTextEncode } from "../src/nodes/model/conditioning.js";
import { EmptyLatentImage, VAEDecode } from "../src/nodes/model/latent.js";
import { CheckpointLoaderSimple } from "../src/nodes/model/loaders.js";
import { KSampler } from "../src/nodes/model/sampling.js";

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

function sent(...nodes: BaseNode[]): unknown {
  const prompt = Object.assign({}, ...nodes.map((node) => node.getComfyUIWorkflowFormat()));
  return JSON.parse(JSON.stringify(prompt));
}

// Dotted ComfyUI names, like "values.a", map to inputs through INPUT_NAMES.
function inputName(type: NodeType, name: string): string {
  return Object.entries(type.INPUT_NAMES).find(([, comfy]) => comfy === name)?.[0] ?? name;
}

describe("a ComfyUI workflow", () => {
  it("is the basic API example of ComfyUI, built from the default workflow", () => {
    const checkpoint = new CheckpointLoaderSimple("4", { ckpt_name: "v1-5-pruned-emaonly.safetensors" });
    const latent = new EmptyLatentImage("5", { width: 512, height: 512, batch_size: 1 });
    const positive = new CLIPTextEncode("6", { text: "masterpiece best quality girl", clip: checkpoint.output.CLIP });
    const negative = new CLIPTextEncode("7", { text: "bad hands", clip: checkpoint.output.CLIP });
    const sampler = new KSampler("3", {
      model: checkpoint.output.MODEL,
      sampler_name: "euler",
      scheduler: "normal",
      positive: positive.output.CONDITIONING,
      negative: negative.output.CONDITIONING,
      latent_image: latent.output.LATENT,
      seed: 8566257,
      steps: 20,
      cfg: 8,
      denoise: 1,
    });
    const decode = new VAEDecode("8", { samples: sampler.output.LATENT, vae: checkpoint.output.VAE });
    const save = new SaveImage("9", { images: decode.output.IMAGE, filename_prefix: "ComfyUI" });

    expect(sent(sampler, checkpoint, latent, positive, negative, decode, save)).toEqual(BASIC_API_EXAMPLE);
  });

  it.skipIf(!existsSync(WAN_WORKFLOW))("is the Wan workflow exported from ComfyUI, rebuilt node by node", () => {
    const exported = JSON.parse(readFileSync(WAN_WORKFLOW, "utf8")) as ComfyUIWorkflow;
    const expected = Object.fromEntries(
      Object.entries(exported).map(([nodeId, node]) => [nodeId, { class_type: node.class_type, inputs: node.inputs }]),
    );

    const rebuilt = Object.entries(exported).map(([nodeId, node]) => {
      const type = NODE_TYPES.get(node.class_type)!;
      const inputs = Object.fromEntries(
        Object.entries(node.inputs).map(([name, value]) => [inputName(type, name), value]),
      );
      return new type(nodeId, inputs);
    });

    expect(sent(...rebuilt)).toEqual(expected);
  });
});
