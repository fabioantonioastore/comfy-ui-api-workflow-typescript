// What the Python RunPod suites send, as scripts/export_runpod_golden.py records
// it in tests/fixtures/runpod_golden.json. The TypeScript suites must build the
// same jobs, so both libraries send RunPod the same workflows and uploads.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { expect } from "vitest";

import { decodePng, PNG_SIGNATURE } from "./runpod_workflows/media.js";

// The export script keeps every FRAME_STEP-th sample of a WAV upload.
const FRAME_STEP = 997;

/** An upload as the golden file records it: a PNG by its pixels, a WAV by its samples. */
export type UploadSummary =
  | { readonly png: readonly [number, number, number]; readonly pixels: string }
  | { readonly wav: readonly [number, number, number, number]; readonly samples: readonly number[] };

type Payload = Readonly<Record<string, unknown>>;
type Summaries = Readonly<Record<string, UploadSummary>>;

export interface WorkflowGolden {
  readonly name: string;
  readonly description: string;
  readonly gpu: boolean;
  readonly fails_with: readonly string[];
  readonly library_says: string | null;
  readonly uploads: Summaries;
  readonly payload: Payload;
}

export interface VideoGolden {
  readonly name: string;
  readonly description: string;
  readonly fps: number;
  readonly gpu: boolean;
  readonly uploads: Summaries;
  readonly payload: Payload;
}

export interface LionGolden {
  readonly label: string;
  readonly folder: string;
  readonly description: string;
  readonly seed: number;
  readonly start: string | null;
  readonly sound: string;
  readonly wind: UploadSummary;
  readonly payload: Payload;
}

export interface RunPodGolden {
  readonly source: string;
  readonly workflows: readonly WorkflowGolden[];
  readonly special: Payload;
  readonly videos: readonly VideoGolden[];
  readonly lion: readonly LionGolden[];
}

export function readRunPodGolden(): RunPodGolden {
  return JSON.parse(readFileSync(new URL("./fixtures/runpod_golden.json", import.meta.url), "utf8")) as RunPodGolden;
}

export interface Wav {
  readonly rate: number;
  readonly channels: number;
  readonly sampleWidth: number;
  readonly frames: number;
  readonly data: Buffer;
}

/** The format and the samples of a WAV file. */
export function readWav(name: string, data: Buffer): Wav {
  let format: Buffer | undefined;
  let samples: Buffer | undefined;
  let position = 12;
  while (position + 8 <= data.length) {
    const length = data.readUInt32LE(position + 4);
    const body = data.subarray(position + 8, position + 8 + length);
    const kind = data.toString("latin1", position, position + 4);
    if (kind === "fmt ") {
      format = body;
    } else if (kind === "data") {
      samples = body;
    }
    position += 8 + length + (length & 1);
  }
  if (format === undefined || samples === undefined) {
    throw new Error(`${name} has no fmt or data chunk`);
  }
  const channels = format.readUInt16LE(2);
  const sampleWidth = (format.readUInt16LE(14) + 7) >> 3;
  return {
    rate: format.readUInt32LE(4),
    channels,
    sampleWidth,
    frames: Math.floor(samples.length / (channels * sampleWidth)),
    data: samples,
  };
}

/** What the golden file records of an upload. */
export function summarize(name: string, data: Uint8Array): UploadSummary {
  const bytes = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  if (bytes.subarray(0, 8).equals(PNG_SIGNATURE)) {
    const picture = decodePng(bytes);
    const hash = createHash("sha256");
    for (const row of picture.rows) {
      hash.update(row);
    }
    return { png: [picture.width, picture.height, picture.channels], pixels: hash.digest("hex") };
  }
  if (bytes.toString("latin1", 0, 4) === "RIFF" && bytes.toString("latin1", 8, 12) === "WAVE") {
    const wav = readWav(name, bytes);
    const samples: number[] = [];
    for (let index = 0; index < wav.data.length >> 1; index += FRAME_STEP) {
      samples.push(wav.data.readInt16LE(2 * index));
    }
    return { wav: [wav.rate, wav.channels, wav.sampleWidth, wav.frames], samples };
  }
  throw new Error(`${name} is neither a PNG nor a WAV file`);
}

/** Expects an upload to hold what the Python one holds. */
export function expectUpload(name: string, data: Uint8Array, expected: UploadSummary): void {
  const actual = summarize(name, data);
  if ("samples" in actual && "samples" in expected) {
    // Another sin() can put a sample one step off; anything more is a real difference.
    expect(actual.wav, name).toEqual(expected.wav);
    expect(actual.samples, name).toHaveLength(expected.samples.length);
    const off = actual.samples.flatMap((sample, index) =>
      Math.abs(sample - expected.samples[index]!) > 1 ? [`sample ${index * FRAME_STEP}: ${sample}`] : [],
    );
    expect(off, `samples of ${name} more than one step off`).toEqual([]);
  } else {
    expect(actual, name).toEqual(expected);
  }
}

/** Expects the uploads of a job to be the Python ones, by name. */
export function expectUploads(uploads: Readonly<Record<string, Uint8Array>>, expected: Summaries): void {
  expect(Object.keys(uploads).sort()).toEqual(Object.keys(expected).sort());
  for (const [name, data] of Object.entries(uploads)) {
    expectUpload(name, data, expected[name]!);
  }
}
