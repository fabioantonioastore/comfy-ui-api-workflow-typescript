// Port of the Python test_ksampler.py. The seed of KSampler goes up to
// 2^64 - 1 in Python, and here up to Number.MAX_SAFE_INTEGER, the widest
// integer a number holds exactly.
import { describe, expect, it } from "vitest";

import { InvalidNodeParameterValue } from "../src/errors/node.js";
import { NodeException } from "../src/exceptions.js";
import type { Link } from "../src/nodes/base.js";
import {
  KSAMPLER_SAMPLER_NAMES,
  KSAMPLER_SCHEDULERS,
  KSampler,
  type KSamplerInputs,
} from "../src/nodes/model/sampling.js";
import { expectError } from "./helpers.js";

const KSAMPLER_INPUTS: KSamplerInputs = {
  model: ["CheckpointLoaderSimple_0", 0],
  sampler_name: "euler",
  scheduler: "normal",
  positive: ["CLIPTextEncode_1", 0],
  negative: ["CLIPTextEncode_2", 0],
  latent_image: ["EmptyLatentImage_3", 0],
};

function makeKSampler(overrides: Record<string, unknown> = {}): KSampler {
  return new KSampler("KSampler_0", { ...KSAMPLER_INPUTS, ...overrides } as KSamplerInputs);
}

function inputsOf(node: KSampler): Record<string, unknown> {
  return node.getComfyUIWorkflowFormat()[node.nodeId]!.inputs;
}

describe("KSampler", () => {
  it("sends its defaults in the ComfyUI format", () => {
    expect(makeKSampler().getComfyUIWorkflowFormat()).toEqual({
      KSampler_0: {
        class_type: "KSampler",
        inputs: {
          model: ["CheckpointLoaderSimple_0", 0],
          sampler_name: "euler",
          scheduler: "normal",
          positive: ["CLIPTextEncode_1", 0],
          negative: ["CLIPTextEncode_2", 0],
          latent_image: ["EmptyLatentImage_3", 0],
          seed: 0,
          steps: 20,
          cfg: 8.0,
          denoise: 1.0,
        },
      },
    });
  });

  it("has a LATENT output", () => {
    expect(makeKSampler().output.LATENT).toEqual(["KSampler_0", 0]);
  });

  it("is in the category of ComfyUI", () => {
    expect(KSampler.CATEGORY).toBe("model/sampling");
  });

  it("names the scheduler input scheduler", () => {
    expect(makeKSampler({ scheduler: "karras" }).scheduler).toBe("karras");
  });

  it.each([...KSAMPLER_SAMPLER_NAMES].sort())("takes the sampler %s", (samplerName) => {
    expect(inputsOf(makeKSampler({ sampler_name: samplerName }))["sampler_name"]).toBe(samplerName);
  });

  it.each([...KSAMPLER_SCHEDULERS].sort())("takes the scheduler %s", (scheduler) => {
    expect(inputsOf(makeKSampler({ scheduler }))["scheduler"]).toBe(scheduler);
  });

  it.each([
    ["sampler_name", KSAMPLER_SAMPLER_NAMES],
    ["scheduler", KSAMPLER_SCHEDULERS],
  ] as const)("rejects an unknown %s, listing the sorted options", (name, choices) => {
    const options = [...choices].sort().map((choice) => `'${choice}'`);
    expectError(
      () => makeKSampler({ [name]: "not_an_option" }),
      InvalidNodeParameterValue,
      `Invalid ${name}, must be one of the following values: [${options.join(", ")}]`,
    );
  });

  it.each([
    ["seed", 0],
    ["seed", Number.MAX_SAFE_INTEGER],
    ["steps", 1],
    ["steps", 10_000],
    ["cfg", 0.0],
    ["cfg", 100.0],
    ["denoise", 0.0],
    ["denoise", 1.0],
  ] as const)("takes %s at the boundary %d", (name, value) => {
    expect(inputsOf(makeKSampler({ [name]: value }))[name]).toBe(value);
  });

  it.each([
    ["seed", -1],
    ["steps", 0],
    ["steps", 10_001],
    ["cfg", -0.1],
    ["cfg", 100.1],
    ["denoise", -0.01],
    ["denoise", 1.01],
  ] as const)("rejects %s out of range at %d", (name, value) => {
    expectError(() => makeKSampler({ [name]: value }), InvalidNodeParameterValue, new RegExp(name));
  });

  it.each([2 ** 53, 2 ** 64 - 1])("rejects a seed of %d, which is no safe integer", (seed) => {
    expectError(
      () => makeKSampler({ seed }),
      InvalidNodeParameterValue,
      "seed must be an integer or a link, not unsafe integer",
    );
  });

  it.each(["cfg", "denoise"])("rejects NaN for %s", (name) => {
    expectError(() => makeKSampler({ [name]: Number.NaN }), InvalidNodeParameterValue, `${name} must not be NaN`);
  });

  it.each(["seed", "steps", "cfg", "denoise"])("rejects a boolean for %s", (name) => {
    expectError(() => makeKSampler({ [name]: true }), InvalidNodeParameterValue, new RegExp(name));
  });

  it.each(["seed", "steps", "cfg", "denoise", "sampler_name", "scheduler"])(
    "takes a link from another node for the widget %s",
    (name) => {
      const link: Link = ["PrimitiveInt_9", 0];

      expect(inputsOf(makeKSampler({ [name]: link }))[name]).toEqual(link);
    },
  );

  it.each([
    ["steps", "20", "steps must be an integer or a link, not string"],
    ["seed", 1.5, "seed must be an integer or a link, not float"],
    ["cfg", null, "cfg must be a number or a link, not null"],
    ["sampler_name", ["euler"], "sampler_name must be a string or a link, not array"],
    ["model", "v1-5-pruned-emaonly.safetensors", "model must be a link, not string"],
  ] as const)("rejects a value of another type for %s", (name, value, message) => {
    expectError(() => makeKSampler({ [name]: value }), InvalidNodeParameterValue, message);
  });

  it.each([
    ["steps", 0, "steps must be in closed range: [1, 10000]"],
    ["cfg", 100.5, "cfg must be in closed range: [0, 100]"],
    ["scheduler", "not_a_scheduler", "Invalid scheduler"],
  ] as const)("keeps the previous value when %s changes to an invalid one", (name, value, message) => {
    const node = makeKSampler();
    const before = inputsOf(node);

    const error = expectError(() => {
      (node as unknown as Record<string, unknown>)[name] = value;
    }, InvalidNodeParameterValue);

    expect(error.message.startsWith(message)).toBe(true);
    expect(inputsOf(node)).toEqual(before);
  });

  it("throws node errors for invalid values", () => {
    expect(InvalidNodeParameterValue.prototype).toBeInstanceOf(NodeException);
  });

  it("rejects the same link for positive and negative", () => {
    expectError(
      () => makeKSampler({ positive: ["CLIPTextEncode_1", 0], negative: ["CLIPTextEncode_1", 0] }),
      InvalidNodeParameterValue,
      "positive and negative must not use the same input",
    );
  });

  it("takes two outputs of the same node for positive and negative", () => {
    const node = makeKSampler({ positive: ["Node_1", 0], negative: ["Node_1", 1] });

    expect([node.positive, node.negative]).toEqual([
      ["Node_1", 0],
      ["Node_1", 1],
    ]);
  });

  it("undoes a change to the same link for positive and negative", () => {
    const node = makeKSampler();
    const before = inputsOf(node);

    expectError(() => {
      node.negative = node.positive;
    }, InvalidNodeParameterValue);

    expect(inputsOf(node)).toEqual(before);
  });
});
