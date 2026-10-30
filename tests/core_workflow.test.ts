// The workflow, pipeline and dynamic input machinery, on small nodes made for
// these tests. The generated ComfyUI nodes are tested in their own files.
import { describe, expect, it } from "vitest";

import { CycleError } from "../src/errors/data_structure.js";
import {
  InvalidLinkError,
  InvalidNodeChangeError,
  InvalidNodeParameterValue,
  NodeHasDependentsError,
  NodeNotFoundError,
} from "../src/errors/node.js";
import { OutputNotFoundError } from "../src/errors/output.js";
import { PipelineError } from "../src/errors/pipeline.js";
import { BaseNode, type Link } from "../src/nodes/base.js";
import * as dynamic from "../src/nodes/dynamic.js";
import { choice, float, int, link, literal } from "../src/nodes/schema.js";
import { Pipeline } from "../src/pipeline.js";
import { Workflow } from "../src/workflow.js";
import { expectError } from "./helpers.js";

class Loader extends BaseNode<{ name: string }, "MODEL" | "CLIP"> {
  static override readonly CATEGORY = "test";
  static override readonly OUTPUTS = { MODEL: 0, CLIP: 1 };
  static override readonly RETURN_TYPES = ["MODEL", "CLIP"];
  static override readonly INPUTS = { name: choice(["a.safetensors", "b.safetensors"]) };
}

interface SamplerInputs {
  model: Link;
  positive?: Link | null;
  negative?: Link | null;
  steps?: number | Link;
}

class Sampler extends BaseNode<SamplerInputs, "LATENT"> {
  static override readonly CATEGORY = "test";
  static override readonly OUTPUTS = { LATENT: 0 };
  static override readonly RETURN_TYPES = ["LATENT"];
  // steps has no socket type, so any output can be linked to it.
  static override readonly INPUT_SOCKET_TYPES = {
    model: "MODEL",
    positive: "CONDITIONING",
    negative: "CONDITIONING",
  };
  static override readonly INPUTS = {
    model: link(),
    positive: link({ optional: true }),
    negative: link({ optional: true }),
    steps: int({ link: true, default: 20, range: [1, 100] }),
  };
  declare model: Link;
  declare positive: Link | undefined;
  declare steps: number | Link;
}

interface ResizeInputs {
  image: Link;
  mode?: "scale" | "fit";
  mode_factor?: number | Link | null;
  mode_width?: number | Link | null;
}

class Resize extends dynamic.DynamicInputs<ResizeInputs, "IMAGE"> {
  static override readonly CATEGORY = "test";
  static override readonly OUTPUTS = { IMAGE: 0 };
  static override readonly INPUT_NAMES = { mode_factor: "mode.factor", mode_width: "mode.width" };
  static override readonly INPUTS = {
    image: link(),
    mode: literal(["scale", "fit"], { default: "scale" }),
    mode_factor: float({ link: true, optional: true }),
    mode_width: int({ link: true, optional: true }),
  };
  static override readonly DYNAMIC_INPUTS = {
    mode: new dynamic.Combo({
      scale: { mode_factor: dynamic.number({ default: 1.0, low: 0.1, high: 8.0 }) },
      fit: { mode_width: dynamic.number({ low: 1, high: 4096 }) },
    }),
  };
  declare mode: "scale" | "fit";
  declare mode_factor: number | Link | undefined;
  declare mode_width: number | Link | undefined;
}

class Loras extends dynamic.DynamicInputs<Record<string, unknown>, "MODEL"> {
  static override readonly CATEGORY = "test";
  static override readonly OUTPUTS = { MODEL: 0 };
  static override readonly INPUT_NAMES = {
    loras_0_name: "loras.lora_0.name",
    loras_0_strength: "loras.lora_0.strength",
    loras_1_name: "loras.lora_1.name",
    loras_1_strength: "loras.lora_1.strength",
  };
  static override readonly INPUTS = {
    loras_0_name: literal(["x"], { optional: true }),
    loras_0_strength: float({ optional: true }),
    loras_1_name: literal(["x"], { optional: true }),
    loras_1_strength: float({ optional: true }),
  };
  static override readonly DYNAMIC_INPUTS = {
    loras: new dynamic.Group(2, {
      name: dynamic.serverChoice(),
      strength: dynamic.number({ default: 1.0, low: -10, high: 10 }),
    }),
  };
}

function loader(workflow: Workflow): Loader {
  return workflow.addNode(Loader, { name: "a.safetensors" });
}

describe("Workflow", () => {
  it("names nodes after their class and sends them in the API format", () => {
    const workflow = new Workflow();
    const model = loader(workflow);
    const sampler = workflow.addNode(Sampler, { model: model.output.MODEL });

    expect(sampler.nodeId).toBe("Sampler_1");
    expect([...workflow.nodes.keys()]).toEqual(["Loader_0", "Sampler_1"]);
    expect(workflow.getComfyUIWorkflowFormat()).toEqual({
      Loader_0: { class_type: "Loader", inputs: { name: "a.safetensors" } },
      Sampler_1: { class_type: "Sampler", inputs: { model: ["Loader_0", 0], steps: 20 } },
    });
    expect(workflow.dependencies("Sampler_1")).toEqual(new Set(["Loader_0"]));
    expect(workflow.dependents("Loader_0")).toEqual(new Set(["Sampler_1"]));
  });

  it("rejects a link to an output of another type", () => {
    const workflow = new Workflow();
    const model = loader(workflow);

    const error = expectError(
      () => workflow.addNode(Sampler, { model: model.output.CLIP }),
      InvalidLinkError,
      "Cannot add node 'Sampler_1': input 'model' uses output 1 of node 'Loader_0', " +
        "which is CLIP, but the input takes MODEL.",
    );
    expect(error.inputName).toBe("model");
    expect(workflow.nodes.size).toBe(1);
  });

  it("rejects a link to a node it does not have", () => {
    const workflow = new Workflow();

    expectError(() => workflow.addNode(Sampler, { model: ["Loader_9", 0] }), NodeNotFoundError, "node: Loader_9 not found");
    expect(workflow.nodes.size).toBe(0);
  });

  it("rejects a node that uses its own output", () => {
    const workflow = new Workflow();

    expectError(() => workflow.addNode(Sampler, { model: ["Sampler_0", 0] }), CycleError);
    expect(workflow.nodes.size).toBe(0);
  });

  it("checks each change of a node in the workflow", () => {
    const workflow = new Workflow();
    const model = loader(workflow);
    const first = workflow.addNode(Sampler, { model: model.output.MODEL });
    const second = workflow.addNode(Sampler, { model: model.output.MODEL });

    expectError(
      () => {
        first.positive = second.output.LATENT;
      },
      InvalidNodeChangeError,
      "Cannot change 'positive' of node 'Sampler_1': it would use output 0 of node 'Sampler_2', " +
        "which is LATENT, but the input takes CONDITIONING.",
    );
    expect(first.positive).toBeUndefined();
  });

  it("rejects a change that would create a cycle", () => {
    const workflow = new Workflow();
    const model = loader(workflow);
    const first = workflow.addNode(Sampler, { model: model.output.MODEL });
    const second = workflow.addNode(Sampler, { model: model.output.MODEL, steps: first.output.LATENT });

    expectError(
      () => {
        first.steps = second.output.LATENT;
      },
      InvalidNodeChangeError,
      "Cannot change 'steps' of node 'Sampler_1': node 'Sampler_2' depends on 'Sampler_1', " +
        "so the change would create the cycle 'Sampler_1' -> 'Sampler_2' -> 'Sampler_1'.",
    );
    expect(first.steps).toBe(20);
    expect(workflow.dependencies("Sampler_1")).toEqual(new Set(["Loader_0"]));
  });

  it("keeps a node id while the node is in the workflow", () => {
    const workflow = new Workflow();
    const model = loader(workflow);
    workflow.addNode(Sampler, { model: model.output.MODEL });

    expectError(
      () => {
        model.nodeId = "Other";
      },
      InvalidNodeChangeError,
      "Cannot change 'nodeId' of node 'Loader_0': node 'Sampler_1' depends on it.",
    );
    expect(model.nodeId).toBe("Loader_0");
  });

  it("validates an update once, after every input changed", () => {
    const workflow = new Workflow();
    const model = loader(workflow);
    const resize = workflow.addNode(Resize, { image: model.output.MODEL, mode_factor: 2.0 });

    expectError(() => {
      resize.mode = "fit";
    }, InvalidNodeParameterValue);
    workflow.updateNode(resize.nodeId, { mode: "fit", mode_factor: undefined, mode_width: 640 });

    expect(resize.inputs()).toEqual({ image: ["Loader_0", 0], mode: "fit", mode_width: 640 });
  });

  it("restores every input of a rejected update", () => {
    const workflow = new Workflow();
    const model = loader(workflow);
    const sampler = workflow.addNode(Sampler, { model: model.output.MODEL });
    const before = sampler.inputs();

    expectError(
      () => workflow.updateNode(sampler.nodeId, { positive: ["Loader_0", 0], steps: 500 }),
      InvalidNodeChangeError,
    );
    expectError(() => workflow.updateNode(sampler.nodeId, { steps: 500 }), InvalidNodeParameterValue);
    expectError(() => workflow.updateNode(sampler.nodeId, { stpes: 5 }), InvalidNodeChangeError);
    expect(sampler.inputs()).toEqual(before);
    expect(workflow.dependencies(sampler.nodeId)).toEqual(new Set(["Loader_0"]));
  });

  it("removes a node only when no node depends on it", () => {
    const workflow = new Workflow();
    const model = loader(workflow);
    const sampler = workflow.addNode(Sampler, { model: model.output.MODEL });

    expectError(
      () => workflow.removeNode(model.nodeId),
      NodeHasDependentsError,
      "Cannot remove node 'Loader_0': node 'Sampler_1' depends on it.",
    );
    workflow.removeNode(sampler.nodeId);
    workflow.removeNode(model.nodeId);

    expect(workflow.nodes.size).toBe(0);
    model.nodeId = "Free";
    expect(model.nodeId).toBe("Free");
  });

  it("sends dynamic inputs under their ComfyUI names, as JSON", () => {
    const workflow = new Workflow();
    const model = loader(workflow);
    workflow.addNode(Resize, { image: model.output.MODEL, mode: "fit", mode_width: 64 });
    const format = JSON.parse(JSON.stringify(workflow.getComfyUIWorkflowFormat())) as unknown;

    expect(format).toEqual({
      Loader_0: { class_type: "Loader", inputs: { name: "a.safetensors" } },
      Resize_1: {
        class_type: "Resize",
        inputs: { image: ["Loader_0", 0], mode: "fit", "mode.width": 64 },
      },
    });
  });
});

describe("dynamic inputs", () => {
  it("send the defaults of the chosen branch", () => {
    const node = new Resize("Resize_0", { image: ["Loader_0", 0] });

    expect(node.getComfyUIWorkflowFormat()["Resize_0"]!.inputs).toEqual({
      image: ["Loader_0", 0],
      mode: "scale",
      "mode.factor": 1.0,
    });
    expect(node.inputs()).toEqual({ image: ["Loader_0", 0], mode: "scale" });
  });

  it("reject an input of another branch", () => {
    expectError(
      () => new Resize("Resize_0", { image: ["Loader_0", 0], mode_width: 64 }),
      InvalidNodeParameterValue,
      "mode_width only applies when mode is 'fit'",
    );
  });

  it("require the inputs of a branch without a default", () => {
    expectError(
      () => new Resize("Resize_0", { image: ["Loader_0", 0], mode: "fit" }),
      InvalidNodeParameterValue,
      "mode_width is required when mode is 'fit'",
    );
  });

  it("check the values of a branch", () => {
    expectError(
      () => new Resize("Resize_0", { image: ["Loader_0", 0], mode_factor: 9.0 }),
      InvalidNodeParameterValue,
      "mode_factor must be in closed range: [0.1, 8]",
    );
  });

  it("send the defaults of the used rows of a group", () => {
    const node = new Loras("Loras_0", { loras_1_name: "x" });

    expect(node.getComfyUIWorkflowFormat()["Loras_0"]!.inputs).toEqual({
      "loras.lora_1.name": "x",
      "loras.lora_1.strength": 1.0,
    });
  });

  it("require the fields of a used row without a default", () => {
    expectError(
      () => new Loras("Loras_0", { loras_0_strength: 0.5 }),
      InvalidNodeParameterValue,
      "loras_0_name is required in row 0 of loras",
    );
  });
});

class Checkpoint extends Pipeline {
  static override readonly PARAMETERS = { name: choice(["a.safetensors", "b.safetensors"]), steps: int({ default: 30 }) };

  override build({ name, steps }: { name: "a.safetensors" | "b.safetensors"; steps?: number }) {
    const model = this.addNode(Loader, { name });
    const sampler = this.addNode(Sampler, { model: model.output.MODEL, steps: steps ?? 30 });
    return { MODEL: model.output.MODEL, LATENT: sampler.output.LATENT };
  }
}

class Broken extends Pipeline {
  override build() {
    const model = this.addNode(Loader, { name: "a.safetensors" });
    this.addNode(Sampler, { model: model.output.CLIP });
  }
}

describe("Pipeline", () => {
  it("adds its nodes and exposes its outputs", () => {
    const workflow = new Workflow();

    const pipeline = workflow.addPipeline(Checkpoint, { name: "b.safetensors" });

    expect(pipeline.output.LATENT).toEqual(["Sampler_1", 0]);
    expect([...pipeline.nodes.keys()]).toEqual(["Loader_0", "Sampler_1"]);
    expect((pipeline.nodes.get("Sampler_1") as Sampler).steps).toBe(30);
    expectError(
      () => (pipeline.output as Record<string, Link>)["IMAGE"],
      OutputNotFoundError,
      "Pipeline 'Checkpoint' has no output 'IMAGE'. Its outputs are: 'MODEL' and 'LATENT'.",
    );
  });

  it("adds no node when one is rejected", () => {
    const workflow = new Workflow();

    const error = expectError(
      () => workflow.addPipeline(Broken),
      PipelineError,
      "Cannot add pipeline 'Broken': node 2 of the pipeline (Sampler) was rejected: " +
        "Cannot add node 'Sampler_1': input 'model' uses output 1 of node 'Loader_0', which is CLIP, " +
        "but the input takes MODEL. No node of the pipeline was added.",
    );
    expect(error.cause).toBeInstanceOf(InvalidLinkError);
    expect(workflow.nodes.size).toBe(0);
    expect(loader(workflow).nodeId).toBe("Loader_0");
  });

  it("checks the parameters before the build runs", () => {
    const workflow = new Workflow();

    expectError(
      () => workflow.addPipeline(Checkpoint, { name: "a.safetensors", seed: 1 } as never),
      PipelineError,
      "Cannot add pipeline 'Checkpoint': it has no parameter 'seed' (it takes 'name' and 'steps'). " +
        "No node of the pipeline was added.",
    );
    expectError(
      () => workflow.addPipeline(Checkpoint, {} as never),
      PipelineError,
      "Cannot add pipeline 'Checkpoint': it is missing the parameter 'name'. No node of the pipeline was added.",
    );
    expectError(
      () => workflow.addPipeline(Checkpoint, { name: "a.safetensors", steps: "30" } as never),
      PipelineError,
      "Cannot add pipeline 'Checkpoint': steps must be an integer, not string. No node of the pipeline was added.",
    );
  });

  it("takes a pipeline class that defines build", () => {
    const workflow = new Workflow();

    expectError(
      () => workflow.addPipeline(new Checkpoint() as unknown as typeof Checkpoint, { name: "a.safetensors" }),
      PipelineError,
      "Cannot add pipeline 'Checkpoint': addPipeline takes the pipeline class, not an instance. " +
        "No node of the pipeline was added.",
    );
    expectError(() => workflow.addPipeline(Pipeline), PipelineError, /it does not define build/);
    expectError(() => new Checkpoint().addNode(Loader, { name: "a.safetensors" }), PipelineError, /only works inside build/);
  });
});
