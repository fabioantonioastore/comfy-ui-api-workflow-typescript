// How the Python test_runpod_videos.py measures the videos that come back: ffprobe
// reads the streams, ffmpeg decodes the sound and the frames.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { accessSync, constants, mkdtempSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, extname, join } from "node:path";

import { Picture, tuple, type Outputs, type Size } from "../runpod_workflows/media.js";
import { Sound } from "./sound.js";

export const SECONDS = 30;
export const ANALYSIS_RATE = 16_000;

/** Python's shutil.which: where the program is on the PATH, if it is. */
function which(program: string): string | null {
  for (const folder of (process.env["PATH"] ?? "").split(delimiter).filter(Boolean)) {
    const path = join(folder, program);
    try {
      accessSync(path, constants.X_OK);
      if (statSync(path).isFile()) {
        return path;
      }
    } catch {
      // Not in this folder.
    }
  }
  return null;
}

export const FFMPEG = which("ffmpeg");
export const FFPROBE = which("ffprobe");
export const TOOLS = FFMPEG !== null && FFPROBE !== null;

/** What the program prints; it throws, with what the program said, when the program fails. */
export function runTool(program: string | null, ...args: string[]): Buffer {
  assert(program !== null, "ffmpeg and ffprobe are needed to measure the videos");
  return execFileSync(program, args, { stdio: ["ignore", "pipe", "pipe"], maxBuffer: 2 ** 30 });
}

/** A stream as ffprobe prints it; it prints some of the numbers as text. */
export interface Stream {
  readonly codec_type?: string;
  readonly codec_name?: string;
  readonly width?: number;
  readonly height?: number;
  readonly avg_frame_rate?: string;
  readonly nb_read_frames?: string;
  readonly sample_rate?: string;
  readonly channels?: number;
  readonly duration?: string;
  readonly tags?: Readonly<Record<string, string>>;
}

export interface Format {
  readonly format_name: string;
  readonly duration: string;
}

export function streamSeconds(stream: Stream): number {
  if (stream.duration !== undefined) {
    return Number(stream.duration);
  }
  // Matroska keeps each stream's length in a DURATION tag, as HH:MM:SS.fraction.
  const [hours, minutes, seconds] = stream.tags!["DURATION"]!.split(":") as [string, string, string];
  return Number.parseInt(hours, 10) * 3600 + Number.parseInt(minutes, 10) * 60 + Number(seconds);
}

/** A video file, measured with ffprobe and decoded with ffmpeg. */
export class Media {
  readonly format: Format;
  readonly video: Stream;
  readonly audio: Stream;
  readonly #data: Buffer;
  readonly #suffix: string;

  constructor(data: Buffer, suffix: string) {
    this.#data = data;
    this.#suffix = suffix;
    const probe = ["-v", "error", "-count_frames", "-print_format", "json", "-show_format", "-show_streams"];
    const info = JSON.parse(this.#run(FFPROBE, probe, []).toString("utf8")) as { format: Format; streams: Stream[] };
    this.format = info.format;
    const videos = info.streams.filter((stream) => stream.codec_type === "video");
    assert(videos.length === 1, `the file has ${videos.length} video streams`);
    this.video = videos[0]!;
    const audio = info.streams.filter((stream) => stream.codec_type === "audio");
    assert(audio.length === 1, `the video has ${audio.length} audio tracks`);
    this.audio = audio[0]!;
  }

  static returned(out: Outputs, label: string): Media {
    return new Media(out.one(label), extname(out.name(label)));
  }

  get seconds(): number {
    return Number(this.format.duration);
  }

  get size(): [number, number] {
    return [this.video.width!, this.video.height!];
  }

  get fps(): number {
    const [numerator, denominator = "1"] = this.video.avg_frame_rate!.split("/");
    return Number(numerator) / Number(denominator);
  }

  get frames(): number {
    return Number.parseInt(this.video.nb_read_frames!, 10);
  }

  sound(channels = 1): Sound {
    const decode = ["-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ac", String(channels)];
    const raw = this.#run(FFMPEG, ["-v", "error", "-i"], [...decode, "-ar", String(ANALYSIS_RATE), "-"]);
    const samples = raw.length >> 1;
    return new Sound(
      ANALYSIS_RATE,
      Array.from({ length: channels }, (_, channel) => {
        const values: number[] = [];
        for (let index = channel; index < samples; index += channels) {
          values.push(raw.readInt16LE(2 * index) / 32768);
        }
        return values;
      }),
    );
  }

  frame(seconds: number): Picture {
    // ffmpeg's -ss gives the first frame at or after that time, so the seek goes
    // just before the start of the frame shown at `seconds`.
    const fps = this.fps;
    const shown = Math.floor(seconds * fps + 1e-6);
    const seek = Math.max(0.0, (shown - 0.25) / fps).toFixed(4);
    const raw = this.#run(
      FFMPEG,
      ["-v", "error", "-ss", seek, "-i"],
      ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
    );
    const [width, height] = this.size;
    assert(raw.length === width * height * 3, `frame at ${seconds}s has ${raw.length} bytes`);
    const line = width * 3;
    return new Picture(
      width,
      height,
      3,
      Array.from({ length: height }, (_, y) => raw.subarray(y * line, (y + 1) * line)),
    );
  }

  // The tools read a copy of the file that lives only while they run: a test
  // worker never gets to clean up at exit, so a longer-lived copy would stay.
  #run(program: string | null, before: readonly string[], after: readonly string[]): Buffer {
    const folder = mkdtempSync(join(tmpdir(), "video-"));
    try {
      const path = join(folder, `media${this.#suffix}`);
      writeFileSync(path, this.#data);
      return runTool(program, ...before, path, ...after);
    } finally {
      rmSync(folder, { recursive: true, force: true });
    }
  }
}

export interface Expected {
  readonly fps: number;
  readonly size: Size;
  readonly video: string;
  readonly audio: string;
  readonly rate: number;
  readonly channels: number;
  readonly frameSlack?: number;
}

export function assertThirtySeconds(media: Media, expected: Expected): void {
  const { fps, size, video, audio, rate, channels, frameSlack = 0 } = expected;
  // One frame of slack also lets the length run one frame over.
  const slack = 0.1 + frameSlack / fps;
  assert(Math.abs(media.seconds - SECONDS) <= slack, `the file lasts ${media.seconds}s`);
  assert(media.video.codec_name === video, String(media.video.codec_name));
  const actual = media.size;
  assert(actual[0] === size[0] && actual[1] === size[1], `the video is ${tuple(actual)}, expected ${tuple(size)}`);
  // Matroska stores times in milliseconds, so its average rate can be a near fraction.
  assert(Math.abs(media.fps - fps) < 0.01, `the video runs at ${media.fps} fps`);
  assert(Math.abs(media.frames - fps * SECONDS) <= frameSlack, `the video has ${media.frames} frames`);
  assert(Math.abs(streamSeconds(media.video) - SECONDS) <= slack);
  assert(media.audio.codec_name === audio, String(media.audio.codec_name));
  assert(Number(media.audio.sample_rate) === rate, String(media.audio.sample_rate));
  assert(media.audio.channels === channels, String(media.audio.channels));
  const seconds = streamSeconds(media.audio);
  assert(Math.abs(seconds - SECONDS) <= 0.1, `the sound lasts ${seconds}s`);
}

export function luminance(picture: Picture, step = 4): number[] {
  const values: number[] = [];
  for (let y = 0; y < picture.height; y += step) {
    for (let x = 0; x < picture.width; x += step) {
      const [r, g, b] = picture.pixel(x, y) as [number, number, number];
      values.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
  }
  return values;
}

export function spread(values: readonly number[]): number {
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  return Math.sqrt(values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length);
}

export function pictureDifference(first: Picture, second: Picture): number {
  const [a, b] = [luminance(first), luminance(second)];
  let total = 0;
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    total += Math.abs(a[index]! - b[index]!);
  }
  return total / a.length;
}

export function inverted(color: number): number {
  return color ^ 0xffffff;
}
