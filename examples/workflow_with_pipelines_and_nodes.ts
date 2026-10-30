// Pipelines and nodes linked both ways: a pipeline feeds nodes, which feed
// another pipeline. Run it to print the workflow in the API format:
// npx tsx examples/workflow_with_pipelines_and_nodes.ts
import { fileURLToPath } from "node:url";

import { Workflow } from "comfy-ui-api-workflow";
import { SaveImage } from "comfy-ui-api-workflow/nodes/image";
import { RepeatImageBatch } from "comfy-ui-api-workflow/nodes/image/batch";
import { ImageBlur } from "comfy-ui-api-workflow/nodes/image/filters";
import { ImagesToVideo, TextToImage } from "./pipelines.js";

export function build(): Workflow {
  const workflow = new Workflow();

  const image = workflow.addPipeline(TextToImage, { prompt: "a koi pond seen from above, clear water" }).output.IMAGE;

  workflow.addNode(SaveImage, { images: image, filename_prefix: "koi" });
  const blurred = workflow.addNode(ImageBlur, { image, blur_radius: 4, sigma: 2.0 });
  const frames = workflow.addNode(RepeatImageBatch, { image: blurred.output.IMAGE, amount: 16 });

  workflow.addPipeline(ImagesToVideo, { frames: frames.output.IMAGE, fps: 8.0, prefix: "video/koi" });
  return workflow;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(build().getComfyUIWorkflowFormat(), null, 2));
}
