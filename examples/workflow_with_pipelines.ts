// A workflow made of three pipelines, each feeding the next. Run it to print
// the workflow in the API format: npx tsx examples/workflow_with_pipelines.ts
import { fileURLToPath } from "node:url";

import { Workflow } from "comfy-ui-api-workflow";
import { SaveImages, TextToImage, Upscale } from "./pipelines.js";

export function build(): Workflow {
  const workflow = new Workflow();

  const image = workflow.addPipeline(TextToImage, { prompt: "a cozy cabin in a snowy forest, warm light" }).output
    .IMAGE;
  const upscaled = workflow.addPipeline(Upscale, { image, scale: 1.5 }).output.IMAGE;
  workflow.addPipeline(SaveImages, { images: upscaled, prefix: "cabin" });
  return workflow;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(build().getComfyUIWorkflowFormat(), null, 2));
}
