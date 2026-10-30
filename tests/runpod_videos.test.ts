// Port of the Python test_runpod_videos.py. Ten 30-second videos with sound must
// send what the Python suite sends and survive a reload, and the helpers that
// measure them must measure what ffmpeg makes; with `RUNPOD=1 npm test` each video
// also runs on the RunPod endpoint in .env, next to the workflows in
// runpod_outputs/workflows.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { Link } from "../src/nodes/base.js";
import { Workflow } from "../src/workflow.js";
import { canonical } from "./helpers.js";
import { RUNPOD, runpodEndpoint } from "./runpod.js";
import { expectUploads, readRunPodGolden, readWav } from "./runpod_golden.js";
import { NIGHT, RED, WHITE } from "./runpod_videos/blocks.js";
import { FFMPEG, Media, SECONDS, streamSeconds, TOOLS } from "./runpod_videos/media.js";
import { decibels, frange, hiss, PythonRandom, sine, Sound, wav } from "./runpod_videos/sound.js";
import { VIDEOS } from "./runpod_videos/videos.js";
import { mediaHtml, runAndRecord } from "./runpod_workflows/gallery.js";
import { assertPixel } from "./runpod_workflows/media.js";
import { GPU_TIMEOUT, sent } from "./runpod_workflows/registry.js";

const golden = readRunPodGolden();

describe("the videos with sound", () => {
  it("are the videos of the Python suite, in its order", () => {
    expect(VIDEOS.map((video) => video.name)).toEqual(golden.videos.map((expected) => expected.name));
  });

  it.each(golden.videos)("$name sends what the Python video sends", (expected) => {
    const video = VIDEOS.find((candidate) => candidate.name === expected.name);
    expect(video, `no video named ${expected.name}`).toBeDefined();

    expect(video!.payload()[0]).toEqual(expected.payload);
    expectUploads(video!.uploads, expected.uploads);
    expect({ description: video!.description, fps: video!.fps, gpu: video!.gpu }).toEqual({
      description: expected.description,
      fps: expected.fps,
      gpu: expected.gpu,
    });
  });

  it("are ten different videos, each saved with its sound", () => {
    const payloads = VIDEOS.map((video) => video.payload()[0]);

    expect(VIDEOS).toHaveLength(10);
    expect(new Set(VIDEOS.map((video) => video.name)).size).toBe(10);
    expect(new Set(payloads.map(canonical)).size).toBe(10);
    VIDEOS.forEach((video, index) => {
      const payload = payloads[index]!;
      const saves = Object.values(payload).filter((node) => node.class_type === "SaveVideo");
      expect(saves, video.name).toHaveLength(1);
      const source = payload[(saves[0]!.inputs["video"] as Link)[0]]!;
      expect(source.class_type, video.name).toBe("CreateVideo");
      expect(source.inputs, video.name).toHaveProperty("audio");
    });
  });

  it.each(VIDEOS)("$name survives a reload", (video) => {
    const [payload] = video.payload();
    const workflow = new Workflow();

    workflow.loadWorkflow(JSON.parse(JSON.stringify(payload)));

    expect(sent(workflow)).toEqual(payload);
  });

  it.each(VIDEOS)("$name sends every sound it loads", (video) => {
    const loaded = Object.values(video.payload()[0])
      .filter((node) => node.class_type === "LoadAudio")
      .map((node) => node.inputs["audio"]);
    const size = Object.values(video.uploads).reduce((total, data) => total + data.length, 0);

    expect([...new Set(loaded)].sort()).toEqual(Object.keys(video.uploads).sort());
    expect(size).toBeLessThan(1_500_000);
  });

  it.each(VIDEOS)("$name lasts thirty seconds at its frame rate", (video) => {
    const [payload] = video.payload();
    const saves = Object.values(payload).filter((node) => node.class_type === "SaveVideo");
    expect(saves).toHaveLength(1);
    const fps = payload[(saves[0]!.inputs["video"] as Link)[0]]!.inputs["fps"];

    expect(Number.isInteger(video.fps * SECONDS)).toBe(true);
    expect(fps === video.fps || Array.isArray(fps)).toBe(true);
  });
});

describe("the sound helpers", () => {
  it("draw the numbers Python's random.Random draws", () => {
    // Printed by CPython 3.14: random.Random(seed).random() and .uniform(-1.0, 1.0).
    const seven = new PythonRandom(7);
    expect([seven.random(), seven.random(), seven.random()]).toEqual([
      0.32383276483316237, 0.15084917392450192, 0.6509344730398537,
    ]);
    for (let draw = 0; draw < 996; draw++) {
      seven.random();
    }
    expect(seven.random()).toBe(0.37786262968738116);
    const eleven = new PythonRandom(11);
    expect([eleven.uniform(-1.0, 1.0), eleven.uniform(-1.0, 1.0)]).toEqual([
      -0.09524089298036276, 0.11954477216099191,
    ]);
    expect(new PythonRandom(0).random()).toBe(0.8444218515250481);
    expect(new PythonRandom(2 ** 40 + 5).random()).toBe(0.5043802970418443);
  });

  it("write the WAV Python's wave module writes", () => {
    const data = wav(8000, [0.0, 0.5, -0.5, 1.0], [0.25, -0.25, 0.0, -1.0]);
    const file = readWav("test.wav", data);

    expect([file.rate, file.channels, file.frames]).toEqual([8000, 2, 4]);
    expect(Array.from({ length: 8 }, (_, index) => file.data.readInt16LE(2 * index))).toEqual([
      0, 8192, 16384, -8192, -16384, 0, 32767, -32767,
    ]);
    expect(data.toString("hex")).toBe(
      "524946463400000057415645666d74201000000001000200401f0000007d0000040010006461746110000000" +
        "00000020004000e000c00000ff7f0180",
    );
  });

  it("measure a tone's level and frequency", () => {
    const tone = new Sound(8000, [sine(440, 1.0, 8000, 0.5)]);

    expect(Math.abs(tone.amplitude(440, 0.0, 1.0) - 0.5)).toBeLessThan(0.005);
    expect(tone.amplitude(660, 0.0, 1.0)).toBeLessThan(0.005);
    expect(tone.dominant([330, 440, 550], 0.1, 0.9)).toBe(440);
    expect(Math.abs(tone.rms(0.0, 1.0) - 0.5 / Math.SQRT2)).toBeLessThan(0.002);
    expect(Math.abs(decibels(tone.rms(0.0, 1.0)) - decibels(0.5 / Math.SQRT2))).toBeLessThan(0.05);
  });

  it("follow the signal in a band and in an envelope", () => {
    const rate = 8000;
    const noise = new Sound(rate, [hiss(2.0, rate, 0.5, 1)]);
    const high = sine(206, 1.0, rate, 0.5);
    const beating = new Sound(rate, [sine(200, 1.0, rate, 0.5).map((value, index) => value + high[index]!)]);

    const band = noise.band([500, 1000], 0.0, 2.0);
    const envelope = beating.envelope(0.0, 1.0, 0.01);

    expect(band).toBeGreaterThan(0);
    expect(band).toBeLessThan(0.05);
    expect(Math.min(...envelope)).toBeLessThan(0.1 * Math.max(...envelope));
    expect(frange(0.0, 1.0, 0.25)).toEqual([0.0, 0.25, 0.5, 0.75, 1.0]);
  });
});

describe("measuring the videos", () => {
  let folder: string;

  beforeEach(() => {
    folder = mkdtempSync(join(tmpdir(), "videos-"));
  });

  afterEach(() => {
    rmSync(folder, { recursive: true, force: true });
  });

  function ffmpeg(name: string, ...args: string[]): Media {
    const path = join(folder, name);
    execFileSync(FFMPEG!, ["-v", "error", ...args, path]);
    return new Media(readFileSync(path), ".mp4");
  }

  it.skipIf(!TOOLS)("Media measures a video made by ffmpeg", () => {
    const media = ffmpeg(
      "sample.mp4",
      ...["-f", "lavfi", "-i", "color=c=0xE02020:s=64x48:r=4:d=2"],
      ...["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2"],
      ...["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"],
    );

    expect([media.size, media.fps, media.frames]).toEqual([[64, 48], 4, 8]);
    expect(Math.abs(media.seconds - 2.0)).toBeLessThan(0.1);
    expect([media.audio.codec_name, Number(media.audio.sample_rate)]).toEqual(["aac", 16000]);
    expect(Math.abs(streamSeconds(media.audio) - 2.0)).toBeLessThan(0.1);
    expect(Math.abs(media.sound().amplitude(440, 0.2, 1.8) - 0.125)).toBeLessThan(0.02);
    assertPixel(media.frame(1.1), 32, 24, RED, 16);
  });

  it.skipIf(!TOOLS)("Media gives the frame shown at that time", () => {
    const media = ffmpeg(
      "flash.mp4",
      ...["-f", "lavfi", "-i", "color=c=white:s=64x48:r=4:d=0.25"],
      ...["-f", "lavfi", "-i", "color=c=0x101828:s=64x48:r=4:d=0.75"],
      ...["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=1"],
      ...["-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a"],
      ...["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"],
    );

    expect(media.frames).toBe(4);
    assertPixel(media.frame(0.0), 32, 24, WHITE, 16);
    assertPixel(media.frame(0.125), 32, 24, WHITE, 16);
    assertPixel(media.frame(0.25), 32, 24, NIGHT, 16);
    assertPixel(media.frame(0.875), 32, 24, NIGHT, 16);
  });

  it("the gallery plays the videos and the sent sounds with audio", () => {
    for (const name of ["lago_00001_.mp4", "batimento_00001_.webm", "entrada-clique.wav"]) {
      writeFileSync(join(folder, name), "");
    }

    const videos = ["lago_00001_.mp4", "batimento_00001_.webm"].map((name) => mediaHtml(folder, name));
    const sound = mediaHtml(folder, "entrada-clique.wav");

    for (const html of videos) {
      expect(html).toContain("<video");
      expect(html).not.toContain("muted");
    }
    expect(sound).toContain("<audio");
    expect(sound).toContain("entrada-clique.wav");
  });

  it("a Matroska stream's length comes from its tag", () => {
    expect(streamSeconds({ tags: { DURATION: "00:00:30.004000000" } })).toBeCloseTo(30.004, 5);
    expect(streamSeconds({ duration: "29.97" })).toBeCloseTo(29.97, 5);
  });
});

// The timeouts of the videos count from when the job is sent, so give vitest a bit more.
const TEST_TIMEOUT = (GPU_TIMEOUT + 120) * 1000;

// Without ffmpeg and ffprobe the returned videos cannot be measured.
describe.skipIf(!RUNPOD || !TOOLS)("the videos with sound on ComfyUI", () => {
  const numbered = VIDEOS.map((video, index) => ({ name: video.name, video, number: index + 1 }));

  it.each(numbered)(
    "$name runs on ComfyUI",
    async ({ video, number }) => {
      const [payload, check] = video.payload();

      await runAndRecord(
        runpodEndpoint(),
        `v${String(number).padStart(2, "0")}-${video.name}`,
        video.name,
        video.description,
        payload,
        { timeout: video.timeout, uploads: video.uploads, check },
      );
    },
    TEST_TIMEOUT,
  );
});
