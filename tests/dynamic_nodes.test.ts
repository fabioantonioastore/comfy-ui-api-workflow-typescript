// Port of the Python test_dynamic_nodes.py: ComfyMathExpression and SaveVideo,
// whose inputs ComfyUI sends with dotted names.
import { describe, expect, it } from "vitest";

import { InvalidLinkError, InvalidNodeParameterValue } from "../src/errors/node.js";
import type { BaseNode, Link } from "../src/nodes/base.js";
import { CheckpointLoaderSimple } from "../src/nodes/model/loaders.js";
import { ComfyMathExpression, type ComfyMathExpressionInputs } from "../src/nodes/utilities.js";
import { PrimitiveInt } from "../src/nodes/utilities/primitive.js";
import {
  LoadVideo,
  SAVEVIDEO_CODECS,
  SAVEVIDEO_CRF_RANGES,
  SAVEVIDEO_FORMAT_CODECS,
  SAVEVIDEO_FORMATS,
  SaveVideo,
  type SaveVideoInputs,
} from "../src/nodes/video.js";
import { Workflow } from "../src/workflow.js";
import { expectError } from "./helpers.js";

const VIDEO: Link = ["CreateVideo_0", 0];
const FPS: Link = ["PrimitiveFloat_1", 0];
const SECONDS: Link = ["PrimitiveFloat_2", 0];
const LETTERS = [..."abcdefghijklmnopqrstuvwxyz"];

function sent(node: BaseNode) {
  return node.getComfyUIWorkflowFormat()[node.nodeId]!;
}

describe("ComfyMathExpression", () => {
  it("sends each value as values.<name>", () => {
    const node = new ComfyMathExpression("129:163", {
      expression: "floor (a * b + 1)",
      values_a: FPS,
      values_b: SECONDS,
    });

    expect(sent(node)).toEqual({
      class_type: "ComfyMathExpression",
      inputs: { expression: "floor (a * b + 1)", "values.a": FPS, "values.b": SECONDS },
    });
  });

  it("defaults to a + b and needs only values_a", () => {
    expect(sent(new ComfyMathExpression("1", { values_a: FPS })).inputs).toEqual({
      expression: "a + b",
      "values.a": FPS,
    });
  });

  it("takes a value for each letter", () => {
    const node = new ComfyMathExpression("1", { values_a: FPS, values_z: SECONDS });

    expect(sent(node).inputs["values.z"]).toEqual(SECONDS);
    expect(ComfyMathExpression.INPUT_NAMES).toEqual(
      Object.fromEntries(LETTERS.map((letter) => [`values_${letter}`, `values.${letter}`])),
    );
  });

  it("takes links as values", () => {
    expectError(
      () => new ComfyMathExpression("1", { values_a: FPS, values_b: 2.5 as never }),
      InvalidNodeParameterValue,
      "values_b must be a link or undefined, not float",
    );
  });

  it("outputs a float, an int and a boolean", () => {
    const node = new ComfyMathExpression("1", { values_a: FPS });

    expect([node.output.FLOAT, node.output.INT, node.output.BOOL]).toEqual([
      ["1", 0],
      ["1", 1],
      ["1", 2],
    ]);
    expect(ComfyMathExpression.RETURN_TYPES).toEqual(["FLOAT", "INT", "BOOLEAN"]);
  });

  it("takes numbers but not other sockets", () => {
    const workflow = new Workflow();
    const number = workflow.addNode(PrimitiveInt, { value: 16 });
    const math = workflow.addNode(ComfyMathExpression, { values_a: number.output.INT });
    const model = workflow.addNode(CheckpointLoaderSimple, { ckpt_name: "model.safetensors" });

    expectError(
      () => workflow.addNode(ComfyMathExpression, { values_a: model.output.MODEL }),
      InvalidLinkError,
      "Cannot add node 'ComfyMathExpression_3': input 'values_a' uses output 0 of node " +
        "'CheckpointLoaderSimple_2', which is MODEL, but the input takes FLOAT,INT,BOOLEAN.",
    );
    expect(workflow.dependencies(math.nodeId)).toEqual(new Set([number.nodeId]));
  });

  it("takes every letter", () => {
    const links = Object.fromEntries(
      LETTERS.map((letter) => [`values_${letter}`, [`PrimitiveFloat_${letter}`, 0] as Link]),
    ) as unknown as ComfyMathExpressionInputs;

    expect(sent(new ComfyMathExpression("1", links)).inputs).toEqual({
      expression: "a + b",
      ...Object.fromEntries(LETTERS.map((letter) => [`values.${letter}`, [`PrimitiveFloat_${letter}`, 0]])),
    });
  });
});

describe("SaveVideo", () => {
  it("has the defaults of ComfyUI", () => {
    expect(sent(new SaveVideo("108", { video: VIDEO }))).toEqual({
      class_type: "SaveVideo",
      inputs: { video: VIDEO, filename_prefix: "video/ComfyUI", format: "auto", "format.codec": "auto" },
    });
  });

  it("sends the chosen branch with dotted names", () => {
    const node = new SaveVideo("108", {
      video: VIDEO,
      format: "mp4",
      format_codec: "h264",
      format_codec_encoding: "re-encode",
      format_codec_encoding_crf: 18.0,
    });

    expect(sent(node).inputs).toEqual({
      video: VIDEO,
      filename_prefix: "video/ComfyUI",
      format: "mp4",
      "format.codec": "h264",
      "format.codec.encoding": "re-encode",
      "format.codec.encoding.crf": 18.0,
    });
  });

  it("sends the flat codec group with its names", () => {
    const node = new SaveVideo("108", {
      video: VIDEO,
      codec: "av1",
      codec_encoding: "re-encode",
      codec_encoding_crf: 60.0,
    });

    expect(sent(node).inputs).toEqual({
      video: VIDEO,
      filename_prefix: "video/ComfyUI",
      format: "auto",
      "format.codec": "auto",
      codec: "av1",
      "codec.encoding": "re-encode",
      "codec.encoding.crf": 60.0,
    });
  });

  it("checks crf against the range of the codec", () => {
    const av1 = new SaveVideo("108", {
      video: VIDEO,
      format_codec: "av1",
      format_codec_encoding: "re-encode",
      format_codec_encoding_crf: 63.0,
    });
    const linked = new SaveVideo("109", {
      video: VIDEO,
      format_codec: "h264",
      format_codec_encoding: "re-encode",
      format_codec_encoding_crf: ["PrimitiveFloat_3", 0],
    });

    expect(av1.format_codec_encoding_crf).toBe(63.0);
    expect(linked.format_codec_encoding_crf).toEqual(["PrimitiveFloat_3", 0]);
  });

  it("has options that agree with its checks", () => {
    expect(Object.keys(SAVEVIDEO_FORMAT_CODECS).sort()).toEqual([...SAVEVIDEO_FORMATS].sort());
    for (const codecs of Object.values(SAVEVIDEO_FORMAT_CODECS)) {
      expect(codecs.every((codec) => (SAVEVIDEO_CODECS as readonly string[]).includes(codec))).toBe(true);
    }
    expect(Object.keys(SAVEVIDEO_CRF_RANGES).sort()).toEqual(["av1", "h264"]);
  });

  it.each([
    ["unknown-format", { format: "avi" }, "Invalid format, must be one of the following values: ['auto', 'mkv', 'mp4', 'webm']"],
    [
      "codec-of-another-format",
      { format: "webm", format_codec: "h264" },
      "format_codec 'h264' is not available for format 'webm'; use one of ['auto', 'av1']",
    ],
    [
      "unknown-codec",
      { format_codec: "vp9" },
      "format_codec 'vp9' is not available for format 'auto'; use one of ['auto', 'av1', 'h264']",
    ],
    [
      "encoding-without-a-codec-that-takes-it",
      { format_codec_encoding: "re-encode" },
      "format_codec_encoding only applies when format_codec is h264 or av1, not 'auto'",
    ],
    [
      "unknown-encoding",
      { format_codec: "h264", format_codec_encoding: "fast" },
      "Invalid format_codec_encoding, must be one of the following values: ['auto', 're-encode']",
    ],
    [
      "re-encode-without-crf",
      { format_codec: "h264", format_codec_encoding: "re-encode" },
      "format_codec_encoding_crf is required when format_codec_encoding is 're-encode'",
    ],
    [
      "crf-without-re-encode",
      { format_codec: "h264", format_codec_encoding: "auto", format_codec_encoding_crf: 18.0 },
      "format_codec_encoding_crf only applies when format_codec_encoding is 're-encode'",
    ],
    [
      "crf-without-encoding",
      { format_codec: "h264", format_codec_encoding_crf: 18.0 },
      "format_codec_encoding_crf only applies when format_codec_encoding is 're-encode'",
    ],
    [
      "crf-above-the-h264-maximum",
      { format_codec: "h264", format_codec_encoding: "re-encode", format_codec_encoding_crf: 52.0 },
      "format_codec_encoding_crf must be in closed range: [0, 51]",
    ],
    [
      "crf-above-the-av1-maximum",
      { format_codec: "av1", format_codec_encoding: "re-encode", format_codec_encoding_crf: 64.0 },
      "format_codec_encoding_crf must be in closed range: [0, 63]",
    ],
    [
      "flat-encoding-without-codec",
      { codec_encoding: "auto" },
      "codec_encoding only applies when codec is h264 or av1, not undefined",
    ],
    [
      "flat-re-encode-without-crf",
      { codec: "h264", codec_encoding: "re-encode" },
      "codec_encoding_crf is required when codec_encoding is 're-encode'",
    ],
    [
      "unknown-flat-codec",
      { codec: "vp9" },
      "Invalid codec, must be one of the following values: ['auto', 'av1', 'h264']",
    ],
    ["link-for-a-branch-choice", { format: ["PrimitiveString_0", 0] }, "format must be a string, not array"],
  ] as const)("rejects inputs outside the chosen branch (%s)", (_, inputs, message) => {
    expectError(
      () => new SaveVideo("108", { video: VIDEO, ...inputs } as unknown as SaveVideoInputs),
      InvalidNodeParameterValue,
      message,
    );
  });

  it("changes codec and crf together with updateNode", () => {
    const workflow = new Workflow();
    const clip = workflow.addNode(LoadVideo, { file: "clip.mp4" });
    const node = workflow.addNode(SaveVideo, {
      video: clip.output.VIDEO,
      format_codec: "av1",
      format_codec_encoding: "re-encode",
      format_codec_encoding_crf: 60.0,
    });

    expectError(() => {
      node.format_codec = "h264";
    }, InvalidNodeParameterValue);
    workflow.updateNode(node.nodeId, { format_codec: "h264", format_codec_encoding_crf: 40.0 });

    expect([node.format_codec, node.format_codec_encoding_crf]).toEqual(["h264", 40.0]);
  });

  it("keeps crf while the encoding is re-encode", () => {
    const node = reEncodedVideo(new Workflow());

    expectError(
      () => {
        node.format_codec_encoding_crf = undefined;
      },
      InvalidNodeParameterValue,
      "format_codec_encoding_crf is required when format_codec_encoding is 're-encode'",
    );

    expect(node.format_codec_encoding_crf).toBe(60.0);
  });

  it("restores every input when an update is rejected", () => {
    const workflow = new Workflow();
    const node = reEncodedVideo(workflow);
    const before = node.inputs();

    expectError(
      () => workflow.updateNode(node.nodeId, { format_codec: "h264", format_codec_encoding_crf: 55.0 }),
      InvalidNodeParameterValue,
      "format_codec_encoding_crf must be in closed range: [0, 51]",
    );

    expect(node.inputs()).toEqual(before);
  });
});

function reEncodedVideo(workflow: Workflow): SaveVideo {
  const clip = workflow.addNode(LoadVideo, { file: "clip.mp4" });
  return workflow.addNode(SaveVideo, {
    video: clip.output.VIDEO,
    format_codec: "av1",
    format_codec_encoding: "re-encode",
    format_codec_encoding_crf: 60.0,
  });
}

describe("dynamic nodes", () => {
  it("are loaded from the ComfyUI format", () => {
    const workflowFormat = {
      "1": { class_type: "PrimitiveFloat", inputs: { value: 16 } },
      "2": { class_type: "PrimitiveFloat", inputs: { value: 5 } },
      "3": {
        class_type: "ComfyMathExpression",
        inputs: { expression: "floor (a * b + 1)", "values.a": ["1", 0], "values.b": ["2", 0] },
      },
      "4": { class_type: "LoadVideo", inputs: { file: "clip.mp4" } },
      "5": {
        class_type: "SaveVideo",
        inputs: {
          video: ["4", 0],
          filename_prefix: "video/Wan2.2_i2v",
          format: "auto",
          "format.codec": "auto",
          codec: "auto",
        },
      },
    };
    const workflow = new Workflow();

    workflow.loadWorkflow(workflowFormat);

    expect(workflow.dependencies("3")).toEqual(new Set(["1", "2"]));
    expect(workflow.dependencies("5")).toEqual(new Set(["4"]));
    expect(JSON.parse(JSON.stringify(workflow.getComfyUIWorkflowFormat()))).toEqual(workflowFormat);
  });
});
