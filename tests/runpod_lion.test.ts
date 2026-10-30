// Port of the Python test_runpod_lion.py. The twelve WAN 2.2 clips of a lion running
// in the savanna must send what the Python suite sends and make up one minute; with
// `RUNPOD=1 npm test` they render on the RunPod endpoint in .env, the four scenes at
// once, and ffmpeg joins them in runpod_outputs/workflows/000-especial-leao-na-savana.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { ComfyUIWorkflow } from "../src/nodes/base.js";
import { Workflow } from "../src/workflow.js";
import { RUNPOD, runpodEndpoint } from "./runpod.js";
import { expectUpload, readRunPodGolden } from "./runpod_golden.js";
import {
  CLIP_FRAMES,
  CLIPS,
  clipWorkflow,
  FPS,
  HEIGHT,
  MINUTE,
  PARTS,
  SCENES,
  SHOWN,
  uploadsOf,
  WAN_MODEL,
  WAN_TEXT_ENCODER,
  WIDTH,
  wind,
  WIND_RATE,
  windOf,
} from "./runpod_lion/clips.js";
import { alreadyRendered, assemble, filmTheLion, sceneBoundaries } from "./runpod_lion/film.js";
import { FFMPEG, Media, TOOLS } from "./runpod_videos/media.js";
import { wav } from "./runpod_videos/sound.js";
import { writeJson } from "./runpod_workflows/gallery.js";
import { GPU_TIMEOUT, sent } from "./runpod_workflows/registry.js";

const golden = readRunPodGolden();

function nodesOf(payload: ComfyUIWorkflow, classType: string): Record<string, unknown>[] {
  return Object.values(payload)
    .filter((node) => node.class_type === classType)
    .map((node) => node.inputs);
}

/** The inputs of the one node of that type. */
function onlyNode(payload: ComfyUIWorkflow, classType: string): Record<string, unknown> {
  const nodes = nodesOf(payload, classType);
  expect(nodes, classType).toHaveLength(1);
  return nodes[0]!;
}

describe("the lion clips", () => {
  it("are the clips of the Python suite, in its order", () => {
    expect(CLIPS.map((clip) => clip.label)).toEqual(golden.lion.map((expected) => expected.label));
  });

  it.each(golden.lion)("$label sends what the Python clip sends", (expected) => {
    const clip = CLIPS.find((candidate) => candidate.label === expected.label);
    expect(clip, `no clip labelled ${expected.label}`).toBeDefined();

    expect(sent(clipWorkflow(clip!))).toEqual(expected.payload);
    expectUpload(clip!.sound, windOf(clip!), expected.wind);
    expect({
      folder: clip!.folder,
      description: clip!.description,
      seed: clip!.seed,
      start: clip!.start,
      sound: clip!.sound,
    }).toEqual({
      folder: expected.folder,
      description: expected.description,
      seed: expected.seed,
      start: expected.start,
      sound: expected.sound,
    });
  });

  it("follow the WAN 2.2 TI2V 5B template", () => {
    const payload = sent(clipWorkflow(CLIPS[0]!));

    expect(nodesOf(payload, "UNETLoader")).toEqual([{ unet_name: WAN_MODEL, weight_dtype: "default" }]);
    const textEncoder = onlyNode(payload, "CLIPLoader");
    expect([textEncoder["clip_name"], textEncoder["type"]]).toEqual([WAN_TEXT_ENCODER, "wan"]);
    expect(nodesOf(payload, "ModelSamplingSD3")[0]!["shift"]).toBe(8.0);
    const sampler = onlyNode(payload, "KSampler");
    expect([sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]]).toEqual([
      20,
      5.0,
      "uni_pc",
      "simple",
    ]);
    const latent = onlyNode(payload, "Wan22ImageToVideoLatent");
    expect([latent["width"], latent["height"], latent["length"]]).toEqual([WIDTH, HEIGHT, CLIP_FRAMES]);
    expect(latent).not.toHaveProperty("start_image");
    expect(nodesOf(payload, "CreateVideo")[0]!["fps"]).toBe(FPS);
    const batches = nodesOf(payload, "ImageFromBatch").map((node) => [node["batch_index"], node["length"]] as number[]);
    expect(batches.sort((a, b) => a[0]! - b[0]! || a[1]! - b[1]!)).toEqual([
      [0, SHOWN],
      [SHOWN, 1],
    ]);
  });

  it("go on from the frame they are sent", () => {
    const clip = CLIPS[1]!;
    const payload = sent(clipWorkflow(clip));
    const loaders = Object.keys(payload).filter((nodeId) => payload[nodeId]!.class_type === "LoadImage");
    expect(loaders).toHaveLength(1);
    const loaderId = loaders[0]!;

    expect(clip.start).toBe("inicio_02.png");
    expect(payload[loaderId]!.inputs["image"]).toBe(clip.start);
    expect(onlyNode(payload, "Wan22ImageToVideoLatent")["start_image"]).toEqual([loaderId, 0]);
    expect(Object.keys(uploadsOf(clip, Buffer.from("png"))).sort()).toEqual(["inicio_02.png", "vento_02.wav"]);
    expect(() => uploadsOf(clip, null)).toThrow("needs the frame before it");
  });

  it("make a minute of twelve clips of five seconds", () => {
    expect(CLIPS).toHaveLength(12);
    expect(SCENES.length * PARTS).toBe(12);
    expect(CLIPS.length * SHOWN).toBe(MINUTE * FPS);
    expect(CLIP_FRAMES).toBe(SHOWN + 1);
    expect((CLIP_FRAMES - 1) % 4).toBe(0);
    expect(CLIPS.map((clip) => clip.start === null)).toEqual(SCENES.flatMap(() => [true, false, false]));
    expect(new Set(CLIPS.map((clip) => clip.seed)).size).toBe(CLIPS.length);
    expect(sceneBoundaries()).toEqual([1, 2, 4, 5, 7, 8, 10, 11]);
  });

  it("cut one minute of wind into the clips", () => {
    const header = 44;
    const chunks = CLIPS.map(windOf);
    const whole = wav(WIND_RATE, wind());
    const level = Math.sqrt(wind().reduce((total, value) => total + value * value, 0) / wind().length);

    expect(wind()).toHaveLength(MINUTE * WIND_RATE);
    expect(Buffer.concat(chunks.map((chunk) => chunk.subarray(header))).equals(whole.subarray(header))).toBe(true);
    expect(level).toBeGreaterThan(0.05);
    expect(level).toBeLessThan(0.3);
  });

  it.each(CLIPS)("$label survives a reload", (clip) => {
    const payload = sent(clipWorkflow(clip));
    const workflow = new Workflow();

    workflow.loadWorkflow(JSON.parse(JSON.stringify(payload)));

    expect(sent(workflow)).toEqual(payload);
  });
});

describe("filming the lion", () => {
  let folder: string;

  beforeEach(() => {
    folder = mkdtempSync(join(tmpdir(), "lion-"));
  });

  afterEach(() => {
    rmSync(folder, { recursive: true, force: true });
  });

  it("reuses a clip only when the same job passed", () => {
    const payload: ComfyUIWorkflow = { "1": { class_type: "EmptyImage", inputs: {} } };
    const uploads = { "vento_01.wav": Buffer.from("wind") };
    writeJson(join(folder, "result.json"), { status: "passed" });
    writeJson(join(folder, "workflow.json"), payload);
    writeFileSync(join(folder, "entrada-vento_01.wav"), "wind");
    writeFileSync(join(folder, "leao_01_00001_.mp4"), "");
    writeFileSync(join(folder, "proximo_01_00001_.png"), "");

    expect(alreadyRendered(folder, payload, uploads)).toBe(true);
    expect(alreadyRendered(folder, { "2": { class_type: "EmptyImage", inputs: {} } }, uploads)).toBe(false);
    expect(alreadyRendered(folder, payload, { "vento_01.wav": Buffer.from("other") })).toBe(false);
    writeJson(join(folder, "result.json"), { status: "failed" });
    expect(alreadyRendered(folder, payload, uploads)).toBe(false);
  });

  it.skipIf(!TOOLS)("joins the clips in order", () => {
    const clips = ["red", "lime", "blue"].map((color, index) => {
      const path = join(folder, `clip_${index}.mp4`);
      execFileSync(FFMPEG!, [
        ...["-v", "error"],
        ...["-f", "lavfi", "-i", `color=c=${color}:s=64x48:r=24:d=1`],
        ...["-f", "lavfi", "-i", "sine=frequency=300:sample_rate=22050:duration=1"],
        ...["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "1", path],
      ]);
      return path;
    });
    const target = join(folder, "joined.mp4");

    assemble(clips, target);

    const media = new Media(readFileSync(target), ".mp4");
    const sound = media.sound();
    expect(media.frames).toBe(72);
    expect(Math.abs(media.seconds - 3.0)).toBeLessThan(0.1);
    expect([0, 1, 2].map((index) => media.frame(index + 0.5).pixel(32, 24)[1]! > 200)).toEqual([false, true, false]);
    expect(media.frame(0.5).pixel(32, 24)[0]).toBeGreaterThan(200);
    expect(media.frame(2.5).pixel(32, 24)[2]).toBeGreaterThan(200);
    expect([0, 1, 2].every((second) => sound.rms(second + 0.1, second + 0.9) > 0.05)).toBe(true);
    expect(readdirSync(folder).filter((name) => name.endsWith("-lista.txt"))).toEqual([]);
  });
});

// The clips of a scene render one after the other, each with the GPU timeout from
// when it is sent; then ffmpeg joins and measures the minute.
const MINUTE_TIMEOUT = (PARTS * (GPU_TIMEOUT + 120) + 600) * 1000;

// Without ffmpeg and ffprobe the clips cannot be joined and measured.
describe.skipIf(!RUNPOD || !TOOLS)("the lion on ComfyUI", () => {
  it(
    "runs for a minute in the savanna",
    async () => {
      await filmTheLion(runpodEndpoint());
    },
    MINUTE_TIMEOUT,
  );
});
