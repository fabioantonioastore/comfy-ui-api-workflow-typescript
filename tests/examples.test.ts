// Port of the Python test_examples.py. Each example runs as a script, like a
// user would, so its imports of the package by name are checked too.
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import type { ComfyUIWorkflow } from "../src/nodes/base.js";
import { Workflow } from "../src/workflow.js";
import { readGolden } from "./helpers.js";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const TEXT_TO_IMAGE = [
  "CheckpointLoaderSimple",
  "CLIPTextEncode",
  "CLIPTextEncode",
  "EmptyLatentImage",
  "KSampler",
  "VAEDecode",
];
const printed = new Map<string, string>();

function runExample(name: string): string {
  let output = printed.get(name);
  if (output === undefined) {
    output = execFileSync(process.execPath, ["--import", "tsx", `examples/${name}.ts`], {
      cwd: ROOT,
      encoding: "utf8",
    });
    printed.set(name, output);
  }
  return output;
}

function workflowOf(name: string): ComfyUIWorkflow {
  return JSON.parse(runExample(name)) as ComfyUIWorkflow;
}

function classTypes(workflowFormat: ComfyUIWorkflow): string[] {
  return Object.values(workflowFormat)
    .map((node) => node.class_type)
    .sort();
}

const WORKFLOWS = ["workflow_with_nodes", "workflow_with_pipelines", "workflow_with_pipelines_and_nodes"];

describe("an example", { timeout: 30_000 }, () => {
  it.each(WORKFLOWS)("prints a workflow that loads back (%s)", (name) => {
    const workflowFormat = workflowOf(name);
    const workflow = new Workflow();

    workflow.loadWorkflow(workflowFormat);

    expect(JSON.parse(JSON.stringify(workflow.getComfyUIWorkflowFormat()))).toEqual(workflowFormat);
  });

  it.each(WORKFLOWS)("prints the workflow the Python example prints (%s)", (name) => {
    expect(workflowOf(name)).toEqual(readGolden().examples[name]);
  });

  it("workflow_with_nodes is text to image saved to a file", () => {
    const workflowFormat = workflowOf("workflow_with_nodes");

    expect(classTypes(workflowFormat)).toEqual([...TEXT_TO_IMAGE, "SaveImage"].sort());
    expect(workflowFormat["KSampler_4"]!.inputs["steps"]).toBe(30);
  });

  it("workflow_with_pipelines chains three pipelines", () => {
    const workflowFormat = workflowOf("workflow_with_pipelines");

    expect(classTypes(workflowFormat)).toEqual([...TEXT_TO_IMAGE, "ImageScaleBy", "SaveImage"].sort());
    expect(workflowFormat["ImageScaleBy_6"]!.inputs["image"]).toEqual(["VAEDecode_5", 0]);
    expect(workflowFormat["SaveImage_7"]!.inputs["images"]).toEqual(["ImageScaleBy_6", 0]);
  });

  it("workflow_with_pipelines_and_nodes links them both ways", () => {
    const workflowFormat = workflowOf("workflow_with_pipelines_and_nodes");

    expect(classTypes(workflowFormat)).toEqual(
      [...TEXT_TO_IMAGE, "SaveImage", "ImageBlur", "RepeatImageBatch", "CreateVideo", "SaveVideo"].sort(),
    );
    expect(workflowFormat["ImageBlur_7"]!.inputs["image"]).toEqual(["VAEDecode_5", 0]);
    expect(workflowFormat["CreateVideo_9"]!.inputs["images"]).toEqual(["RepeatImageBatch_8", 0]);
  });

  it("building_pipelines shows what addPipeline checks", () => {
    expect(runExample("building_pipelines").split("\n")).toEqual([
      'nodes: ["CheckpointLoaderSimple_0","CLIPTextEncode_1","CLIPTextEncode_2","EmptyLatentImage_3",' +
        '"KSampler_4","VAEDecode_5"]',
      'IMAGE output: ["VAEDecode_5",0]',
      "Cannot add pipeline 'TextToImage': it is missing the parameter 'prompt'. No node of the pipeline was added.",
      "Cannot add pipeline 'Upscale': node 1 of the pipeline (ImageScaleBy) was rejected: " +
        "scale_by must be in closed range: [0.01, 8]. No node of the pipeline was added.",
      "nodes in the workflow: 6",
      "",
    ]);
  });
});
