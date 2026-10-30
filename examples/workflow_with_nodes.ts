// A text to image workflow built node by node. Run it to print the workflow
// in the API format: npx tsx examples/workflow_with_nodes.ts
import { fileURLToPath } from "node:url";

import { Workflow } from "comfy-ui-api-workflow";
import { SaveImage } from "comfy-ui-api-workflow/nodes/image";
import { CLIPTextEncode } from "comfy-ui-api-workflow/nodes/model/conditioning";
import { EmptyLatentImage, VAEDecode } from "comfy-ui-api-workflow/nodes/model/latent";
import { CheckpointLoaderSimple } from "comfy-ui-api-workflow/nodes/model/loaders";
import { KSampler } from "comfy-ui-api-workflow/nodes/model/sampling";

export function build(): Workflow {
  const workflow = new Workflow();

  const model = workflow.addNode(CheckpointLoaderSimple, { ckpt_name: "sd_xl_base_1.0.safetensors" });
  const positive = workflow.addNode(CLIPTextEncode, {
    text: "a red apple on a wooden table, studio photo, soft light",
    clip: model.output.CLIP,
  });
  const negative = workflow.addNode(CLIPTextEncode, { text: "blurry, low quality", clip: model.output.CLIP });
  const latent = workflow.addNode(EmptyLatentImage, { width: 1024, height: 1024 });
  const sampler = workflow.addNode(KSampler, {
    model: model.output.MODEL,
    sampler_name: "dpmpp_2m",
    scheduler: "karras",
    positive: positive.output.CONDITIONING,
    negative: negative.output.CONDITIONING,
    latent_image: latent.output.LATENT,
    seed: 42,
    steps: 25,
    cfg: 7.0,
  });
  const image = workflow.addNode(VAEDecode, { samples: sampler.output.LATENT, vae: model.output.VAE });
  workflow.addNode(SaveImage, { images: image.output.IMAGE, filename_prefix: "apple" });

  // Inputs can still change after a node is added, with the same checks.
  workflow.updateNode(sampler.nodeId, { steps: 30 });
  return workflow;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(build().getComfyUIWorkflowFormat(), null, 2));
}
