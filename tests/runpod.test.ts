// Port of the Python test_runpod.py. Every test here sends a real job to the
// endpoint in .env, so they only run with `RUNPOD=1 npm test`. The checkpoint
// is the one on the endpoint's volume.
import { inflateSync } from "node:zlib";
import { describe, expect, it as base } from "vitest";

import { InvalidWorkflowError } from "../src/errors/workflow.js";
import { EmptyImage, SaveImage, type EmptyImageInputs } from "../src/nodes/image.js";
import { CLIPTextEncode } from "../src/nodes/model/conditioning.js";
import { EmptyLatentImage, VAEDecode } from "../src/nodes/model/latent.js";
import { CheckpointLoaderSimple } from "../src/nodes/model/loaders.js";
import { KSampler } from "../src/nodes/model/sampling.js";
import { int, string } from "../src/nodes/schema.js";
import { ComfyMathExpression } from "../src/nodes/utilities.js";
import { PrimitiveInt } from "../src/nodes/utilities/primitive.js";
import { CreateVideo, SaveVideo } from "../src/nodes/video.js";
import { Pipeline } from "../src/pipeline.js";
import { Workflow } from "../src/workflow.js";
import { expectError } from "./helpers.js";
import { RUNPOD, runpodJobs, type RunPodJob, type RunPodJobs } from "./runpod.js";

const CHECKPOINT = "sd_xl_base_1.0.safetensors";
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

const it = base.extend<{ runpod: RunPodJobs }>({
  runpod: async ({ task }, use) => {
    await use(runpodJobs(task.name));
  },
});

class TextToImage extends Pipeline {
  static override readonly PARAMETERS = {
    prompt: string(),
    size: int({ optional: true }),
    steps: int({ optional: true }),
  };

  // SDXL was trained at 1024x1024; with fewer steps or a smaller size it only
  // produces blurred shapes.
  override build({ prompt, size = 1024, steps = 25 }: { prompt: string; size?: number; steps?: number }) {
    const model = this.addNode(CheckpointLoaderSimple, { ckpt_name: CHECKPOINT });
    const positive = this.addNode(CLIPTextEncode, { text: prompt, clip: model.output.CLIP });
    const negative = this.addNode(CLIPTextEncode, {
      text: "blurry, low quality, distorted, deformed",
      clip: model.output.CLIP,
    });
    const latent = this.addNode(EmptyLatentImage, { width: size, height: size, batch_size: 1 });
    const sampler = this.addNode(KSampler, {
      model: model.output.MODEL,
      sampler_name: "dpmpp_2m",
      scheduler: "karras",
      positive: positive.output.CONDITIONING,
      negative: negative.output.CONDITIONING,
      latent_image: latent.output.LATENT,
      seed: 42,
      steps,
      cfg: 7.0,
    });
    const image = this.addNode(VAEDecode, { samples: sampler.output.LATENT, vae: model.output.VAE });
    return { IMAGE: image.output.IMAGE };
  }
}

function sent(workflow: Workflow): unknown {
  return JSON.parse(JSON.stringify(workflow.getComfyUIWorkflowFormat()));
}

function onlyFile(job: RunPodJob): Buffer {
  expect(job.status, job.error).toBe("COMPLETED");
  const files = (job.output?.images ?? []).map((image) => Buffer.from(image.data, "base64"));
  expect(files).toHaveLength(1);
  return files[0]!;
}

function pngSize(png: Buffer): [number, number] {
  expect(png.subarray(0, 8)).toEqual(PNG_SIGNATURE);
  return [png.readUInt32BE(16), png.readUInt32BE(20)];
}

function firstPixel(png: Buffer): [number, number, number] {
  // The first pixel of the first row is stored as is, whatever the row filter.
  const chunks: Buffer[] = [];
  let position = 8;
  while (position < png.length) {
    const length = png.readUInt32BE(position);
    if (png.toString("latin1", position + 4, position + 8) === "IDAT") {
      chunks.push(png.subarray(position + 8, position + 8 + length));
    }
    position += 12 + length;
  }
  const data = inflateSync(Buffer.concat(chunks));
  return [data[1]!, data[2]!, data[3]!];
}

function solidImage(workflow: Workflow, inputs: EmptyImageInputs): EmptyImage {
  const image = workflow.addNode(EmptyImage, { batch_size: 1, ...inputs });
  workflow.addNode(SaveImage, { images: image.output.IMAGE, filename_prefix: "test" });
  return image;
}

describe.skipIf(!RUNPOD)("a workflow sent to ComfyUI on RunPod", { timeout: 1_000_000 }, () => {
  it("runs when the library built it", async ({ runpod }) => {
    const workflow = new Workflow();
    solidImage(workflow, { width: 64, height: 32, color: 0xff0000 });

    const png = onlyFile(await runpod.run(sent(workflow)));

    expect(pngSize(png)).toEqual([64, 32]);
    expect(firstPixel(png)).toEqual([255, 0, 0]);
  });

  it("runs when the library loaded it", async ({ runpod }) => {
    const built = new Workflow();
    solidImage(built, { width: 48, height: 16, color: 0x00ff00 });
    const workflow = new Workflow();

    workflow.loadWorkflow(sent(built));
    const png = onlyFile(await runpod.run(sent(workflow)));

    expect(pngSize(png)).toEqual([48, 16]);
    expect(firstPixel(png)).toEqual([0, 255, 0]);
  });

  it("gets the values of a math expression", async ({ runpod }) => {
    const workflow = new Workflow();
    const number = workflow.addNode(PrimitiveInt, { value: 16 });
    const math = workflow.addNode(ComfyMathExpression, { expression: "a * 4", values_a: number.output.INT });
    solidImage(workflow, { width: math.output.INT, height: 8, color: 0 });

    const png = onlyFile(await runpod.run(sent(workflow)));

    expect(pngSize(png)).toEqual([64, 8]);
  });

  it("gets the branch of SaveVideo the library sends", async ({ runpod }) => {
    const workflow = new Workflow();
    const frames = workflow.addNode(EmptyImage, { width: 128, height: 128, batch_size: 16, color: 0x2060ff });
    const video = workflow.addNode(CreateVideo, { images: frames.output.IMAGE, fps: 8.0 });
    workflow.addNode(SaveVideo, {
      video: video.output.VIDEO,
      filename_prefix: "test",
      format: "mp4",
      format_codec: "av1",
      format_codec_encoding: "re-encode",
      format_codec_encoding_crf: 40.0,
    });

    const job = await runpod.run(sent(workflow));
    const clip = onlyFile(job);

    // AV1 is not the mp4 default (h264), so it shows that "format.codec" took effect.
    expect(job.output!.images![0]!.filename.endsWith(".mp4")).toBe(true);
    expect([clip.includes("ftyp"), clip.includes("av01"), clip.includes("avc1")]).toEqual([true, true, false]);
  });

  it("generates with SDXL from the text to image pipeline", async ({ runpod }) => {
    const workflow = new Workflow();
    const pipeline = workflow.addPipeline(TextToImage, {
      prompt: "a red apple on a wooden table, studio photo, soft light, detailed",
    });
    workflow.addNode(SaveImage, { images: pipeline.output.IMAGE, filename_prefix: "test" });

    const png = onlyFile(await runpod.run(sent(workflow), 900));

    expect(pngSize(png)).toEqual([1024, 1024]);
  });

  it("fails on a link the library refuses too", async ({ runpod }) => {
    const floatIntoInt = {
      "1": { class_type: "PrimitiveFloat", inputs: { value: 64.0 } },
      "2": { class_type: "EmptyImage", inputs: { width: ["1", 0], height: 64, batch_size: 1, color: 0 } },
      "3": { class_type: "SaveImage", inputs: { images: ["2", 0], filename_prefix: "test" } },
    };

    const error = expectError(() => new Workflow().loadWorkflow(floatIntoInt), InvalidWorkflowError);
    const job = await runpod.run(floatIntoInt);

    expect(error.message).toContain("which is FLOAT, but the input takes INT");
    expect(job.status).toBe("FAILED");
    expect(job.error).toContain("received_type(FLOAT) mismatch input_type(INT)");
  });

  it("fails on a COMBO output into a V1 combo, which the library refuses too", async ({ runpod }) => {
    // A V1 combo's type is its list of options, so ComfyUI links no COMBO output to it.
    const comboIntoV1Combo = {
      "1": { class_type: "EmptyImage", inputs: { width: 8, height: 8, batch_size: 1, color: 0 } },
      "2": { class_type: "CreateVideo", inputs: { images: ["1", 0], fps: 1.0 } },
      "3": { class_type: "GetVideoComponents", inputs: { video: ["2", 0] } },
      "4": {
        class_type: "ImageScale",
        inputs: { image: ["1", 0], upscale_method: ["3", 3], width: 8, height: 8, crop: "disabled" },
      },
      "5": { class_type: "SaveImage", inputs: { images: ["4", 0], filename_prefix: "test" } },
    };

    const error = expectError(() => new Workflow().loadWorkflow(comboIntoV1Combo), InvalidWorkflowError);
    const job = await runpod.run(comboIntoV1Combo);

    expect(error.message).toContain("which is COMBO, but the input takes V1 COMBO");
    expect(job.status).toBe("FAILED");
    expect(job.error).toContain("received_type(COMBO) mismatch input_type([");
  });

  it("runs the shared conditioning the library refuses", async ({ runpod }) => {
    // ComfyUI has no such rule: only the library refuses it.
    const sharedConditioning = {
      "1": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: CHECKPOINT } },
      "2": { class_type: "CLIPTextEncode", inputs: { text: "a red apple on a wooden table", clip: ["1", 1] } },
      "3": { class_type: "EmptyLatentImage", inputs: { width: 512, height: 512, batch_size: 1 } },
      "4": {
        class_type: "KSampler",
        inputs: {
          model: ["1", 0],
          positive: ["2", 0],
          negative: ["2", 0],
          latent_image: ["3", 0],
          sampler_name: "euler",
          scheduler: "normal",
          seed: 42,
          steps: 4,
          cfg: 7.0,
          denoise: 1.0,
        },
      },
      "5": { class_type: "VAEDecode", inputs: { samples: ["4", 0], vae: ["1", 2] } },
      "6": { class_type: "SaveImage", inputs: { images: ["5", 0], filename_prefix: "test" } },
    };

    expectError(
      () => new Workflow().loadWorkflow(sharedConditioning),
      InvalidWorkflowError,
      "Cannot load node '4': positive and negative must not use the same input.",
    );
    const png = onlyFile(await runpod.run(sharedConditioning));

    expect(pngSize(png)).toEqual([512, 512]);
  });
});
