import { describe, expect, it } from "vitest";

import { OutputTypes } from "../src/enums.js";
import { OutputNotFoundError } from "../src/errors/output.js";
import { ComfyUIAPIWorkflowException, OutputException } from "../src/exceptions.js";
import { NodeOutput } from "../src/nodes/base.js";
import { expectError } from "./helpers.js";

describe("OutputTypes", () => {
  it("are ComfyUI type strings", () => {
    expect(OutputTypes.MODEL).toBe("MODEL");
    expect(OutputTypes.CONDITIONING).toBe("CONDITIONING");
    expect(OutputTypes.LATENT).toBe("LATENT");
    expect(`${OutputTypes.LATENT}`).toBe("LATENT");
  });
});

describe("NodeOutput", () => {
  it("returns the ComfyUI link of an output by name", () => {
    const output = new NodeOutput("Loader_0", { MODEL: 0, CLIP: 1 });

    expect(output.MODEL).toEqual(["Loader_0", 0]);
    expect(output["CLIP"]).toEqual(["Loader_0", 1]);
  });

  it("tells outputs of the same type apart by name", () => {
    const output = new NodeOutput("ControlNet_0", { positive: 0, negative: 1 });

    expect(output.positive).toEqual(["ControlNet_0", 0]);
    expect(output.negative).toEqual(["ControlNet_0", 1]);
  });

  it("reads names that are not identifiers", () => {
    const output = new NodeOutput("LTXVAudioVAELoader_0", { "Audio VAE": 0 });

    expect(output["Audio VAE"]).toEqual(["LTXVAudioVAELoader_0", 0]);
    expect("Audio VAE" in output).toBe(true);
  });

  it("throws for a missing output, naming the node and the output", () => {
    const output: NodeOutput = new NodeOutput("Loader_0", { MODEL: 0 });

    expectError(() => output["LATENT"], OutputNotFoundError, "node: Loader_0 don't have the output: LATENT");
  });

  it("puts an output before a member of the same name", () => {
    const output: NodeOutput = new NodeOutput("Batch_0", { outputs: 0 });

    expect(output["outputs"]).toEqual(["Batch_0", 0]);
  });

  it("is an output error of the library", () => {
    expect(OutputNotFoundError.prototype).toBeInstanceOf(OutputException);
    expect(OutputException.prototype).toBeInstanceOf(ComfyUIAPIWorkflowException);
  });
});
