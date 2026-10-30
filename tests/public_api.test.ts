// Port of the Python test_public_api.py: what the package root gives. Every
// module is an ES module here, so the tests of Python's import machinery have
// no counterpart.
import { describe, expect, it } from "vitest";

import * as api from "../src/index.js";
import { NODE_TYPES } from "../src/_internal/registry.js";
import { PipelineError } from "../src/errors/pipeline.js";
import { BaseNode } from "../src/nodes/base.js";
import { EmptyImage } from "../src/nodes/image.js";
import { readSpec } from "../scripts/generate-nodes.js";

describe("the package root", () => {
  it("exports Workflow, Pipeline and the namespaces", () => {
    expect(Object.keys(api).sort()).toEqual(["Pipeline", "Workflow", "enums", "errors", "exceptions", "nodes"]);
  });

  it("reaches the node modules", () => {
    expect([
      api.nodes.image.EmptyImage.CLASS_TYPE,
      api.nodes.video.SaveVideo.CLASS_TYPE,
      api.nodes.model.sampling.KSampler.CLASS_TYPE,
      api.nodes.audio.LoadAudio.CLASS_TYPE,
      api.nodes.model.conditioning.wan.Wan22ImageToVideoLatent.CLASS_TYPE,
      api.nodes.partner.image.krea.Krea2StyleReferenceNode.CLASS_TYPE,
    ]).toEqual([
      "EmptyImage",
      "SaveVideo",
      "KSampler",
      "LoadAudio",
      "Wan22ImageToVideoLatent",
      "Krea2StyleReferenceNode",
    ]);
  });

  it("reaches the error modules", () => {
    expect([
      api.errors.pipeline.PipelineError.name,
      api.errors.output.OutputNotFoundError.name,
      api.errors.data_structure.CycleError.name,
      api.errors.node.InvalidLinkError.name,
    ]).toEqual(["PipelineError", "OutputNotFoundError", "CycleError", "InvalidLinkError"]);
  });

  it("reaches the classes the library uses", () => {
    expect(api.nodes.image.EmptyImage).toBe(EmptyImage);
    expect(api.errors.pipeline.PipelineError).toBe(PipelineError);
    expect(api.exceptions.NodeException).toBeDefined();
    expect(api.enums.OutputTypes.LATENT).toBe("LATENT");
  });

  it("builds a workflow alone", () => {
    class Canvas extends api.Pipeline {
      override build({ color }: { color: number }) {
        const image = this.addNode(api.nodes.image.EmptyImage, { width: 8, height: 8, color });
        return { IMAGE: image.output.IMAGE };
      }
    }

    const workflow = new api.Workflow();
    const canvas = workflow.addPipeline(Canvas, { color: 255 });
    workflow.addNode(api.nodes.image.SaveImage, { images: canvas.output.IMAGE, filename_prefix: "api" });
    const workflowFormat = workflow.getComfyUIWorkflowFormat();

    expect(Object.values(workflowFormat).map((node) => node.class_type).sort()).toEqual(["EmptyImage", "SaveImage"]);
    const save = Object.values(workflowFormat).find((node) => node.class_type === "SaveImage")!;
    expect(save.inputs["filename_prefix"]).toBe("api");
  });

  it("reaches every module of the Python library at the same path, with its node classes", () => {
    const missing: string[] = [];
    for (const module of readSpec().modules) {
      let namespace: unknown = api.nodes;
      for (const part of module.path === "" ? [] : module.path.split(".")) {
        namespace = (namespace as Record<string, unknown> | undefined)?.[part];
      }
      if (namespace === undefined) {
        missing.push(module.path);
        continue;
      }
      const found = namespace as Record<string, unknown>;
      for (const child of module.children) {
        if (found[child] === undefined) {
          missing.push(`${module.path}.${child}`);
        }
      }
      for (const node of module.nodes) {
        const type = found[node.name];
        if (typeof type !== "function" || !(type.prototype instanceof BaseNode) || NODE_TYPES.get(node.class_type) !== type) {
          missing.push(`${module.path}.${node.name}`);
        }
      }
    }

    expect(missing).toEqual([]);
  });
});
