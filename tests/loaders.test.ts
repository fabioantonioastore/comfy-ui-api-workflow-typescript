// Port of the Python test_loaders.py.
import { describe, expect, it } from "vitest";

import { InvalidNodeParameterValue } from "../src/errors/node.js";
import { CheckpointLoaderSimple } from "../src/nodes/model/loaders.js";
import { expectError } from "./helpers.js";

function makeCheckpoint(ckptName = "sd_xl_base_1.0.safetensors"): CheckpointLoaderSimple {
  return new CheckpointLoaderSimple("CheckpointLoaderSimple_0", { ckpt_name: ckptName });
}

describe("CheckpointLoaderSimple", () => {
  it("sends the checkpoint in the ComfyUI format", () => {
    expect(makeCheckpoint().getComfyUIWorkflowFormat()).toEqual({
      CheckpointLoaderSimple_0: {
        class_type: "CheckpointLoaderSimple",
        inputs: { ckpt_name: "sd_xl_base_1.0.safetensors" },
      },
    });
  });

  it.each([
    ["MODEL", 0],
    ["CLIP", 1],
    ["VAE", 2],
  ] as const)("has the output %s at the index ComfyUI gives it", (name, index) => {
    expect(makeCheckpoint().output[name]).toEqual(["CheckpointLoaderSimple_0", index]);
  });

  it("is in the category of ComfyUI", () => {
    expect(CheckpointLoaderSimple.CATEGORY).toBe("model/loaders");
  });

  it("rejects an empty checkpoint name", () => {
    expectError(() => makeCheckpoint(""), InvalidNodeParameterValue, /ckpt_name/);
  });
});
