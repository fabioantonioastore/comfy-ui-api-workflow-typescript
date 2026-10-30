// What Workflow.addPipeline checks. Run it to see a pipeline added and two
// rejected, which add no node: npx tsx examples/building_pipelines.ts
import { Workflow } from "comfy-ui-api-workflow";
import { PipelineError } from "comfy-ui-api-workflow/errors/pipeline";
import { TextToImage, Upscale } from "./pipelines.js";

function printRejection(add: () => unknown): void {
  try {
    add();
  } catch (error) {
    if (!(error instanceof PipelineError)) {
      throw error;
    }
    console.log(error.message);
  }
}

const workflow = new Workflow();
const pipeline = workflow.addPipeline(TextToImage, { prompt: "a lighthouse at dusk" });
console.log("nodes:", JSON.stringify([...pipeline.nodes.keys()]));
console.log("IMAGE output:", JSON.stringify(pipeline.output.IMAGE));
// @ts-expect-error TypeScript rejects the missing prompt first; JavaScript gets the PipelineError.
printRejection(() => workflow.addPipeline(TextToImage));
printRejection(() => workflow.addPipeline(Upscale, { image: pipeline.output.IMAGE, scale: 100.0 }));
console.log("nodes in the workflow:", workflow.nodes.size);
