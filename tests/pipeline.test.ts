// Port of the Python test_pipeline.py. A TypeScript build takes its parameters
// in one object, which PARAMETERS describes for the checks Python reads from
// the signature of build. Object keys are always strings, so the case of an
// output name that is not one has no counterpart; nor do the pickle and copy
// tests.
import { describe, expect, it } from "vitest";

import { CycleError } from "../src/errors/data_structure.js";
import {
  InvalidLinkError,
  InvalidNodeChangeError,
  InvalidNodeParameterValue,
  NodeHasDependentsError,
} from "../src/errors/node.js";
import { OutputNotFoundError } from "../src/errors/output.js";
import { PipelineError } from "../src/errors/pipeline.js";
import { PipelineException } from "../src/exceptions.js";
import type { Link } from "../src/nodes/base.js";
import { LoadImage, SaveImage } from "../src/nodes/image.js";
import { ImageScaleBy } from "../src/nodes/image/upscaling.js";
import { CLIPTextEncode } from "../src/nodes/model/conditioning.js";
import { EmptyLatentImage, VAEDecode, type EmptyLatentImageInputs } from "../src/nodes/model/latent.js";
import { CheckpointLoaderSimple } from "../src/nodes/model/loaders.js";
import { KSampler } from "../src/nodes/model/sampling.js";
import { float, int, link, string } from "../src/nodes/schema.js";
import { ComfySwitchNode } from "../src/nodes/utilities/logic.js";
import { Pipeline, type PipelineOutputs } from "../src/pipeline.js";
import { Workflow } from "../src/workflow.js";
import { expectError, expectUnchanged, workflowState } from "./helpers.js";

interface TextToImageParameters {
  prompt: string;
  checkpoint: string;
  steps?: number;
  negative?: string;
}

class TextToImage extends Pipeline {
  static override readonly PARAMETERS = {
    prompt: string(),
    checkpoint: string(),
    steps: int({ default: 20 }),
    negative: string({ default: "blurry" }),
  };

  override build({ prompt, checkpoint, steps = 20, negative = "blurry" }: TextToImageParameters) {
    const model = this.addNode(CheckpointLoaderSimple, { ckpt_name: checkpoint });
    const positive = this.addNode(CLIPTextEncode, { text: prompt, clip: model.output.CLIP });
    const unwanted = this.addNode(CLIPTextEncode, { text: negative, clip: model.output.CLIP });
    const latent = this.addNode(EmptyLatentImage, { width: 512, height: 512, batch_size: 1 });
    const sampler = this.addNode(KSampler, {
      model: model.output.MODEL,
      sampler_name: "euler",
      scheduler: "normal",
      positive: positive.output.CONDITIONING,
      negative: unwanted.output.CONDITIONING,
      latent_image: latent.output.LATENT,
      steps,
    });
    const image = this.addNode(VAEDecode, { samples: sampler.output.LATENT, vae: model.output.VAE });
    return { IMAGE: image.output.IMAGE };
  }
}

class Upscale extends Pipeline {
  static override readonly PARAMETERS = { image: link(), scale: float({ default: 2.0 }) };

  override build({ image, scale = 2.0 }: { image: Link; scale?: number }) {
    const scaled = this.addNode(ImageScaleBy, { image, upscale_method: "lanczos", scale_by: scale });
    return { IMAGE: scaled.output.IMAGE };
  }
}

class SaveOnly extends Pipeline {
  static override readonly PARAMETERS = { image: link() };

  override build({ image }: { image: Link }) {
    this.addNode(SaveImage, { images: image, filename_prefix: "pipeline" });
  }
}

// Without PARAMETERS, build takes any parameters.
class AnyParameters extends Pipeline {
  override build(parameters: EmptyLatentImageInputs) {
    const latent = this.addNode(EmptyLatentImage, parameters);
    return { LATENT: latent.output.LATENT };
  }
}

class WrongLink extends Pipeline {
  static override readonly PARAMETERS = {};

  override build() {
    const model = this.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
    const text = this.addNode(CLIPTextEncode, { text: "a cat", clip: model.output.CLIP });
    this.addNode(VAEDecode, { samples: text.output.CONDITIONING, vae: model.output.VAE });
  }
}

class OwnOutput extends Pipeline {
  override build() {
    const model = this.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
    const index = Number(model.nodeId.split("_").at(-1)) + 1;
    this.addNode(CLIPTextEncode, { text: "a cat", clip: [`CLIPTextEncode_${index}`, 0] });
  }
}

class MissingSource extends Pipeline {
  override build() {
    this.addNode(CLIPTextEncode, { text: "a cat", clip: ["Missing_9", 1] });
  }
}

class FailingBuild extends Pipeline {
  static override readonly PARAMETERS = { message: string({ default: "no such style" }) };

  override build({ message = "no such style" }: { message?: string }) {
    this.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
    throw new RangeError(message);
  }
}

class ThrowingString extends Pipeline {
  override build() {
    this.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
    throw "no such style";
  }
}

class SwallowedRejection extends Pipeline {
  override build() {
    const model = this.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
    try {
      this.addNode(VAEDecode, { samples: model.output.MODEL, vae: model.output.VAE });
    } catch (error) {
      if (!(error instanceof InvalidLinkError)) {
        throw error;
      }
    }
    return { MODEL: model.output.MODEL };
  }
}

class RewiredCycle extends Pipeline {
  override build() {
    const first = this.addNode(ComfySwitchNode, { switch: true });
    const second = this.addNode(ComfySwitchNode, { switch: true, on_true: first.output.output });
    first.on_true = second.output.output;
  }
}

class RewiredToMissingNode extends Pipeline {
  override build() {
    const switchNode = this.addNode(ComfySwitchNode, { switch: true });
    switchNode.on_true = ["Missing_9", 0];
  }
}

class SwallowedTwice extends Pipeline {
  override build() {
    const model = this.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
    for (const source of [model.output.MODEL, model.output.CLIP]) {
      try {
        this.addNode(VAEDecode, { samples: source, vae: model.output.VAE });
      } catch (error) {
        if (!(error instanceof InvalidLinkError)) {
          throw error;
        }
      }
    }
  }
}

class BadOutput extends Pipeline {
  override build({ output }: { output: (model: CheckpointLoaderSimple) => unknown }) {
    const model = this.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
    return output(model) as PipelineOutputs;
  }
}

class SharedPrompt extends Pipeline {
  override build() {
    const model = this.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });
    const text = this.addNode(CLIPTextEncode, { text: "a cat", clip: model.output.CLIP });
    const latent = this.addNode(EmptyLatentImage, { width: 512, height: 512, batch_size: 1 });
    this.addNode(KSampler, {
      model: model.output.MODEL,
      sampler_name: "euler",
      scheduler: "normal",
      positive: text.output.CONDITIONING,
      negative: text.output.CONDITIONING,
      latent_image: latent.output.LATENT,
    });
  }
}

class OptionalImage extends Pipeline {
  static override readonly PARAMETERS = { image: link({ optional: true }) };

  override build({ image }: { image?: Link | null }) {
    const source = image ?? this.addNode(LoadImage, { image: "photo.png" }).output.IMAGE;
    this.addNode(SaveImage, { images: source, filename_prefix: "pipeline" });
  }
}

function textToImage(workflow: Workflow, parameters: Partial<TextToImageParameters> = {}): TextToImage {
  return workflow.addPipeline(TextToImage, { prompt: "a cat", checkpoint: "model.safetensors", ...parameters });
}

// addPipeline without its types, for what they would not let through.
function addAnyPipeline(workflow: Workflow, type: unknown, parameters: unknown): Pipeline {
  const addPipeline = workflow.addPipeline as unknown as (this: Workflow, type: unknown, parameters: unknown) => Pipeline;
  return addPipeline.call(workflow, type, parameters);
}

describe("addPipeline", () => {
  it("adds the nodes in the order build adds them", () => {
    const workflow = new Workflow();

    textToImage(workflow);

    expect(Object.fromEntries([...workflow.nodes].map(([nodeId, node]) => [nodeId, node.constructor]))).toEqual({
      CheckpointLoaderSimple_0: CheckpointLoaderSimple,
      CLIPTextEncode_1: CLIPTextEncode,
      CLIPTextEncode_2: CLIPTextEncode,
      EmptyLatentImage_3: EmptyLatentImage,
      KSampler_4: KSampler,
      VAEDecode_5: VAEDecode,
    });
    expect([...workflow.nodes.keys()].at(-1)).toBe("VAEDecode_5");
  });

  it("gives the nodes the parameters and links of the build", () => {
    const workflow = new Workflow();

    textToImage(workflow, { prompt: "a dog", steps: 30 });

    expect(workflow.dependencies("KSampler_4")).toEqual(
      new Set(["CheckpointLoaderSimple_0", "CLIPTextEncode_1", "CLIPTextEncode_2", "EmptyLatentImage_3"]),
    );
    expect((workflow.nodes.get("CLIPTextEncode_1") as CLIPTextEncode).text).toBe("a dog");
    expect((workflow.nodes.get("CLIPTextEncode_2") as CLIPTextEncode).text).toBe("blurry");
    expect((workflow.nodes.get("KSampler_4") as KSampler).steps).toBe(30);
  });

  it("returns the pipeline with its nodes and outputs", () => {
    const workflow = new Workflow();

    const pipeline = textToImage(workflow);

    expect(pipeline).toBeInstanceOf(TextToImage);
    expect([...pipeline.nodes.keys()]).toEqual([...workflow.nodes.keys()]);
    for (const [nodeId, node] of pipeline.nodes) {
      expect(node).toBe(workflow.nodes.get(nodeId));
    }
    expect(pipeline.output.IMAGE).toEqual(workflow.getNodeOutput("VAEDecode_5")["IMAGE"]);
  });

  it("takes outputs of nodes in the workflow as parameters", () => {
    const workflow = new Workflow();
    const photo = workflow.addNode(LoadImage, { image: "photo.png" });

    const pipeline = workflow.addPipeline(Upscale, { image: photo.output.IMAGE, scale: 1.5 });

    expect([...pipeline.nodes.keys()]).toEqual(["ImageScaleBy_1"]);
    expect(workflow.dependencies("ImageScaleBy_1")).toEqual(new Set([photo.nodeId]));
    expect((workflow.nodes.get("ImageScaleBy_1") as ImageScaleBy).scale_by).toBe(1.5);
  });

  it("lets the outputs of a pipeline feed nodes and other pipelines", () => {
    const workflow = new Workflow();
    const image = textToImage(workflow).output.IMAGE;

    const upscaled = workflow.addPipeline(Upscale, { image }).output.IMAGE;
    const save = workflow.addNode(SaveImage, { images: upscaled });

    expect(workflow.dependencies("ImageScaleBy_6")).toEqual(new Set(["VAEDecode_5"]));
    expect(workflow.dependencies(save.nodeId)).toEqual(new Set(["ImageScaleBy_6"]));
  });

  it("adds a pipeline more than once", () => {
    const workflow = new Workflow();

    const first = textToImage(workflow, { prompt: "a cat" });
    const second = textToImage(workflow, { prompt: "a dog" });

    expect(workflow.nodes.size).toBe(12);
    expect([...first.nodes.keys()].some((nodeId) => second.nodes.has(nodeId))).toBe(false);
    expect(second.output.IMAGE).toEqual(["VAEDecode_11", 0]);
  });

  it("lets an optional link parameter take null", () => {
    const workflow = new Workflow();

    const pipeline = workflow.addPipeline(OptionalImage, { image: null });

    expect([...pipeline.nodes.values()].map((node) => node.constructor.name)).toEqual(["LoadImage", "SaveImage"]);
  });

  it("passes any parameters to a build without PARAMETERS", () => {
    const workflow = new Workflow();

    const pipeline = workflow.addPipeline(AnyParameters, { width: 768, height: 512 });

    expect((workflow.nodes.get("EmptyLatentImage_0") as EmptyLatentImage).width).toBe(768);
    expect(pipeline.output.LATENT).toEqual(["EmptyLatentImage_0", 0]);
  });

  it("adds the nodes of a pipeline without outputs", () => {
    const workflow = new Workflow();
    const photo = workflow.addNode(LoadImage, { image: "photo.png" });

    const pipeline = workflow.addPipeline(SaveOnly, { image: photo.output.IMAGE });

    expect([...pipeline.nodes.keys()]).toEqual(["SaveImage_1"]);
    expectError(
      () => (pipeline.output as Record<string, Link>)["IMAGE"],
      OutputNotFoundError,
      "Pipeline 'SaveOnly' has no output 'IMAGE'.",
    );
  });

  it("lists the outputs of a pipeline when one is unknown", () => {
    const pipeline = textToImage(new Workflow());

    expectError(
      () => (pipeline.output as Record<string, Link>)["LATENT"],
      OutputNotFoundError,
      "Pipeline 'TextToImage' has no output 'LATENT'. Its outputs are: 'IMAGE'.",
    );
  });

  it("protects the nodes of a pipeline like any node", () => {
    const workflow = new Workflow();
    const pipeline = textToImage(workflow);
    const text = pipeline.nodes.get("CLIPTextEncode_1") as CLIPTextEncode;

    expectError(() => workflow.removeNode("CheckpointLoaderSimple_0"), NodeHasDependentsError);
    expectError(
      () => workflow.updateNode("VAEDecode_5", { samples: text.output.CONDITIONING }),
      InvalidNodeChangeError,
    );
  });
});

const REJECTED: readonly (readonly [string, unknown, unknown, string])[] = [
  [
    "unknown-parameter",
    TextToImage,
    { prompt: "a cat", checkpoint: "model.safetensors", stepz: 30 },
    "Cannot add pipeline 'TextToImage': it has no parameter 'stepz' " +
      "(it takes 'prompt', 'checkpoint', 'steps' and 'negative'). No node of the pipeline was added.",
  ],
  [
    "unknown-parameter-of-a-single-parameter-pipeline",
    SaveOnly,
    { images: ["LoadImage_0", 0] },
    "Cannot add pipeline 'SaveOnly': it has no parameter 'images' (it takes 'image'). " +
      "No node of the pipeline was added.",
  ],
  [
    "parameter-for-a-pipeline-without-parameters",
    WrongLink,
    { model: "model.safetensors" },
    "Cannot add pipeline 'WrongLink': it has no parameter 'model' (it takes no parameters). " +
      "No node of the pipeline was added.",
  ],
  [
    "parameters-not-an-object",
    TextToImage,
    "a cat",
    "Cannot add pipeline 'TextToImage': its parameters must be an object, not string. " +
      "No node of the pipeline was added.",
  ],
  [
    "missing-parameters",
    TextToImage,
    {},
    "Cannot add pipeline 'TextToImage': it is missing the parameters 'prompt' and 'checkpoint'. " +
      "No node of the pipeline was added.",
  ],
  [
    "missing-parameter",
    TextToImage,
    { prompt: "a cat" },
    "Cannot add pipeline 'TextToImage': it is missing the parameter 'checkpoint'. No node of the pipeline was added.",
  ],
  [
    "parameter-of-another-type",
    TextToImage,
    { prompt: "a cat", checkpoint: "model.safetensors", steps: "30" },
    "Cannot add pipeline 'TextToImage': steps must be an integer, not string. No node of the pipeline was added.",
  ],
  [
    "value-for-a-link-parameter",
    Upscale,
    { image: "photo.png" },
    "Cannot add pipeline 'Upscale': image must be a link, not string. No node of the pipeline was added.",
  ],
  [
    "parameter-link-to-missing-node",
    Upscale,
    { image: ["Missing_9", 0] },
    "Cannot add pipeline 'Upscale': parameter 'image' uses node 'Missing_9', which is not in the workflow. " +
      "No node of the pipeline was added.",
  ],
  [
    "parameter-link-to-missing-output",
    Upscale,
    { image: ["LoadImage_0", 3] },
    "Cannot add pipeline 'Upscale': parameter 'image' uses output 3 of node 'LoadImage_0', " +
      "but LoadImage has no output 3. No node of the pipeline was added.",
  ],
  [
    "invalid-value",
    TextToImage,
    { prompt: "a cat", checkpoint: "model.safetensors", steps: 0 },
    "Cannot add pipeline 'TextToImage': node 5 of the pipeline (KSampler) was rejected: " +
      "steps must be in closed range: [1, 10000]. No node of the pipeline was added.",
  ],
  [
    "incompatible-link",
    WrongLink,
    {},
    "Cannot add pipeline 'WrongLink': node 3 of the pipeline (VAEDecode) was rejected: " +
      "Cannot add node 'VAEDecode_3': input 'samples' uses output 0 of node 'CLIPTextEncode_2', " +
      "which is CONDITIONING, but the input takes LATENT. No node of the pipeline was added.",
  ],
  [
    "cycle",
    OwnOutput,
    {},
    "Cannot add pipeline 'OwnOutput': node 2 of the pipeline (CLIPTextEncode) was rejected: " +
      "Cannot add edge 'CLIPTextEncode_2' -> 'CLIPTextEncode_2': a node cannot have an edge to itself. " +
      "No node of the pipeline was added.",
  ],
  [
    "cycle-made-by-a-change-in-build",
    RewiredCycle,
    {},
    "Cannot add pipeline 'RewiredCycle': a change to node 1 of the pipeline (ComfySwitchNode) was rejected: " +
      "Cannot change 'on_true' of node 'ComfySwitchNode_1': node 'ComfySwitchNode_2' depends on " +
      "'ComfySwitchNode_1', so the change would create the cycle " +
      "'ComfySwitchNode_1' -> 'ComfySwitchNode_2' -> 'ComfySwitchNode_1'. No node of the pipeline was added.",
  ],
  [
    "missing-node-linked-by-a-change-in-build",
    RewiredToMissingNode,
    {},
    "Cannot add pipeline 'RewiredToMissingNode': a change to node 1 of the pipeline (ComfySwitchNode) " +
      "was rejected: Cannot change 'on_true' of node 'ComfySwitchNode_1': node 'Missing_9' is not in the " +
      "workflow. No node of the pipeline was added.",
  ],
  [
    "link-to-missing-node",
    MissingSource,
    {},
    "Cannot add pipeline 'MissingSource': node 1 of the pipeline (CLIPTextEncode) was rejected: " +
      "node: Missing_9 not found. No node of the pipeline was added.",
  ],
  [
    "same-link-for-positive-and-negative",
    SharedPrompt,
    {},
    "Cannot add pipeline 'SharedPrompt': node 4 of the pipeline (KSampler) was rejected: " +
      "positive and negative must not use the same input. No node of the pipeline was added.",
  ],
  [
    "rejection-caught-by-the-build",
    SwallowedRejection,
    {},
    "Cannot add pipeline 'SwallowedRejection': node 2 of the pipeline (VAEDecode) was rejected: " +
      "Cannot add node 'VAEDecode_2': input 'samples' uses output 0 of node 'CheckpointLoaderSimple_1', " +
      "which is MODEL, but the input takes LATENT. No node of the pipeline was added.",
  ],
  [
    "first-of-two-caught-rejections",
    SwallowedTwice,
    {},
    "Cannot add pipeline 'SwallowedTwice': node 2 of the pipeline (VAEDecode) was rejected: " +
      "Cannot add node 'VAEDecode_2': input 'samples' uses output 0 of node 'CheckpointLoaderSimple_1', " +
      "which is MODEL, but the input takes LATENT. No node of the pipeline was added.",
  ],
  [
    "build-error",
    FailingBuild,
    {},
    "Cannot add pipeline 'FailingBuild': its build failed with RangeError: no such style. " +
      "No node of the pipeline was added.",
  ],
  [
    "build-error-without-message",
    FailingBuild,
    { message: "" },
    "Cannot add pipeline 'FailingBuild': its build failed with RangeError. No node of the pipeline was added.",
  ],
  [
    "build-throws-a-string",
    ThrowingString,
    {},
    "Cannot add pipeline 'ThrowingString': its build failed with string: no such style. " +
      "No node of the pipeline was added.",
  ],
  [
    "output-not-a-link",
    BadOutput,
    { output: (model: CheckpointLoaderSimple) => ({ MODEL: model }) },
    "Cannot add pipeline 'BadOutput': its output 'MODEL' must be a link [nodeId, outputIndex], " +
      "not CheckpointLoaderSimple. No node of the pipeline was added.",
  ],
  [
    "output-the-node-does-not-have",
    BadOutput,
    { output: (model: CheckpointLoaderSimple) => ({ MODEL: [model.nodeId, 3] }) },
    "Cannot add pipeline 'BadOutput': its output 'MODEL' uses output 3 of node 'CheckpointLoaderSimple_1', " +
      "but CheckpointLoaderSimple has no output 3. No node of the pipeline was added.",
  ],
  [
    "output-of-a-missing-node",
    BadOutput,
    { output: () => ({ MODEL: ["Missing_9", 0] }) },
    "Cannot add pipeline 'BadOutput': its output 'MODEL' uses node 'Missing_9', which is not in the workflow. " +
      "No node of the pipeline was added.",
  ],
  [
    "build-result-not-an-object",
    BadOutput,
    { output: (model: CheckpointLoaderSimple) => model },
    "Cannot add pipeline 'BadOutput': its build must return an object of outputs or undefined, " +
      "not CheckpointLoaderSimple. No node of the pipeline was added.",
  ],
  [
    "not-a-pipeline",
    KSampler,
    {},
    "Cannot add pipeline 'KSampler': addPipeline takes a Pipeline subclass. No node of the pipeline was added.",
  ],
  [
    "not-a-class",
    "TextToImage",
    {},
    "Cannot add pipeline 'string': addPipeline takes a Pipeline subclass. No node of the pipeline was added.",
  ],
  [
    "pipeline-instance",
    new TextToImage(),
    {},
    "Cannot add pipeline 'TextToImage': addPipeline takes the pipeline class, not an instance. " +
      "No node of the pipeline was added.",
  ],
  [
    "pipeline-without-build",
    Pipeline,
    {},
    "Cannot add pipeline 'Pipeline': it does not define build. No node of the pipeline was added.",
  ],
];

describe("a rejected pipeline", () => {
  it.each(REJECTED)("adds no node and says why (%s)", (_, type, parameters, message) => {
    const workflow = new Workflow();
    workflow.addNode(LoadImage, { image: "photo.png" });
    const before = workflowState(workflow);

    expectError(() => addAnyPipeline(workflow, type, parameters), PipelineError, message);

    expectUnchanged(workflow, before);
    expect(workflow.addNode(EmptyLatentImage).nodeId).toBe("EmptyLatentImage_1");
  });

  it.each([
    ["parameter-type", TextToImage, { prompt: "a cat", checkpoint: "m", steps: "30" }, InvalidNodeParameterValue],
    ["node-value", TextToImage, { prompt: "a cat", checkpoint: "m", steps: 0 }, InvalidNodeParameterValue],
    ["node-link", WrongLink, {}, InvalidLinkError],
    ["node-cycle", OwnOutput, {}, CycleError],
    ["build-error", FailingBuild, {}, RangeError],
  ] as const)("keeps the error it comes from as the cause (%s)", (_, type, parameters, cause) => {
    const error = expectError(() => addAnyPipeline(new Workflow(), type, parameters), PipelineError);

    expect(error.cause).toBeInstanceOf(cause);
  });
});

describe("Pipeline", () => {
  it("cannot add nodes outside addPipeline", () => {
    expectError(
      () => new TextToImage().build({ prompt: "a cat", checkpoint: "model.safetensors" }),
      PipelineError,
      "Cannot add pipeline 'TextToImage': addNode only works inside build, while Workflow.addPipeline adds " +
        "the pipeline. No node of the pipeline was added.",
    );
  });

  it("cannot add nodes after it was added", () => {
    const workflow = new Workflow();
    const pipeline = textToImage(workflow);

    expectError(() => pipeline.addNode(EmptyLatentImage), PipelineError);

    expect(workflow.nodes.size).toBe(6);
  });

  it("keeps the nodes it uses from being removed", () => {
    const workflow = new Workflow();
    const photo = workflow.addNode(LoadImage, { image: "photo.png" });
    workflow.addPipeline(Upscale, { image: photo.output.IMAGE });

    expectError(
      () => workflow.removeNode(photo.nodeId),
      NodeHasDependentsError,
      "Cannot remove node 'LoadImage_0': node 'ImageScaleBy_1' depends on it.",
    );
  });

  it("throws pipeline errors", () => {
    expect(PipelineError.prototype).toBeInstanceOf(PipelineException);
    const error = new PipelineError("TextToImage", "a reason");
    expect([error.pipeline, error.reason]).toEqual(["TextToImage", "a reason"]);
  });

  it("leaves build to the subclasses", () => {
    expect(() => new Pipeline().build()).toThrow("Pipeline does not define build");
  });

  it("only works when its constructor made it", () => {
    const pipeline = Object.create(TextToImage.prototype) as TextToImage;

    expectError(() => pipeline.nodes, TypeError, "This object is not a pipeline made by its constructor.");
  });
});
