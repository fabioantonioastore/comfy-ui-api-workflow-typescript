// How the Python test_runpod_lion.py films the minute: each scene renders its clips
// on RunPod one after the other, the four scenes at once, and ffmpeg joins the
// twelve clips without re-encoding them.
import assert from "node:assert/strict";
import { existsSync, mkdirSync, readdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, join, resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

import type { ComfyUIWorkflow } from "../../src/nodes/base.js";
import { FFMPEG, Media, pictureDifference, runTool, streamSeconds } from "../runpod_videos/media.js";
import { GALLERY, runAndRecord, writeGallery, writeJson, type Endpoint } from "../runpod_workflows/gallery.js";
import { tuple } from "../runpod_workflows/media.js";
import { GPU_TIMEOUT, sent, type Uploads } from "../runpod_workflows/registry.js";
import {
  checkClip,
  Clip,
  CLIP_SECONDS,
  CLIPS,
  clipWorkflow,
  FINAL_FOLDER,
  FPS,
  HEIGHT,
  MINUTE,
  PARTS,
  SCENES,
  uploadsOf,
  WIDTH,
} from "./clips.js";

/** The paths of the folder's files that match the pattern, where * stands for any text, sorted. */
function glob(folder: string, pattern: string): string[] {
  if (!existsSync(folder)) {
    return [];
  }
  const parts = pattern.split("*").map((part) => part.replace(/[.+?^${}()|[\]\\]/g, "\\$&"));
  const matches = new RegExp(`^${parts.join(".*")}$`);
  return readdirSync(folder)
    .filter((name) => matches.test(name))
    .sort()
    .map((name) => join(folder, name));
}

export function onlyFile(folder: string, pattern: string): string {
  const found = glob(folder, pattern);
  const names = found.map((path) => `'${basename(path)}'`).join(", ");
  assert(found.length === 1, `expected one ${pattern} in ${basename(folder)}, got [${names}]`);
  return found[0]!;
}

export function alreadyRendered(folder: string, payload: ComfyUIWorkflow, uploads: Uploads): boolean {
  // A clip is only reused when the same job, with the same files, passed before.
  let result: { readonly status?: unknown };
  let saved: unknown;
  let sentFiles: [string, Buffer][];
  try {
    result = JSON.parse(readFileSync(join(folder, "result.json"), "utf8")) as { readonly status?: unknown };
    saved = JSON.parse(readFileSync(join(folder, "workflow.json"), "utf8"));
    sentFiles = Object.keys(uploads).map((name) => [name, readFileSync(join(folder, `entrada-${name}`))]);
  } catch (error) {
    // A file that is missing or is not JSON, as Python's OSError and ValueError.
    if (error instanceof SyntaxError || (error instanceof Error && "code" in error)) {
      return false;
    }
    throw error;
  }
  return (
    result.status === "passed" &&
    isDeepStrictEqual(saved, payload) &&
    sentFiles.every(([name, data]) => data.equals(uploads[name]!)) &&
    glob(folder, "leao_*.mp4").length === 1 &&
    glob(folder, "proximo_*.png").length === 1
  );
}

export async function renderScene(endpoint: Endpoint, scene: number): Promise<string[]> {
  const videos: string[] = [];
  let startPng: Buffer | null = null;
  for (let part = 0; part < PARTS; part++) {
    const clip = new Clip(scene, part);
    const payload = sent(clipWorkflow(clip));
    const uploads = uploadsOf(clip, startPng);
    const folder = join(GALLERY, clip.folder);
    if (!alreadyRendered(folder, payload, uploads)) {
      await runAndRecord(
        endpoint,
        clip.folder,
        `leao-${String(clip.number).padStart(2, "0")}`,
        clip.description,
        payload,
        { timeout: GPU_TIMEOUT, uploads, check: checkClip(clip) },
      );
    }
    videos.push(onlyFile(folder, `${clip.label}_*.mp4`));
    startPng = readFileSync(onlyFile(folder, `${clip.following}_*.png`));
  }
  return videos;
}

export function assemble(videos: readonly string[], target: string): void {
  // The clips share every encoder setting, so the video is copied as is; the sound
  // is decoded and encoded again, which joins it without gaps.
  const listing = join(dirname(target), `${basename(target, extname(target))}-lista.txt`);
  writeFileSync(listing, videos.map((path) => `file '${resolve(path)}'\n`).join(""), "utf8");
  try {
    runTool(
      FFMPEG,
      ...["-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing],
      ...["-c:v", "copy", "-c:a", "aac", "-b:a", "96k", target],
    );
  } finally {
    unlinkSync(listing);
  }
}

export function contactSheet(video: string, target: string, frames: readonly number[]): void {
  const chosen = frames.map((frame) => `eq(n\\,${frame})`).join("+");
  runTool(
    FFMPEG,
    ...["-v", "error", "-y", "-i", video],
    ...["-vf", `select='${chosen}',scale=480:-2,tile=${frames.length}x1`, "-frames:v", "1", target],
  );
}

export function sceneBoundaries(): number[] {
  // Where one clip goes on from the one before, inside a scene (not the cuts).
  return CLIPS.filter((clip) => clip.part > 0).map((clip) => clip.number - 1);
}

export function checkMinute(media: Media): void {
  assert(Math.abs(media.seconds - MINUTE) <= 0.1, `the video lasts ${media.seconds}s`);
  const [width, height] = media.size;
  assert(width === WIDTH && height === HEIGHT, `the video is ${tuple(media.size)}`);
  assert(media.fps === FPS, `the video runs at ${media.fps} fps`);
  assert(media.frames === MINUTE * FPS, `the video has ${media.frames} frames`);
  assert.deepEqual([media.audio.codec_name, media.audio.channels], ["aac", 1]);
  const seconds = streamSeconds(media.audio);
  assert(Math.abs(seconds - MINUTE) <= 0.15, `the wind lasts ${seconds}s`);
  const sound = media.sound();
  for (let second = 0; second < MINUTE; second++) {
    assert(sound.rms(second, second + 1) > 0.005, `the wind stops at ${second}s`);
  }
  const step = 1 / FPS;
  for (const index of sceneBoundaries()) {
    const at = index * CLIP_SECONDS;
    const motion = [0, 1, 2].map((back) =>
      pictureDifference(media.frame(at - (back + 2) * step), media.frame(at - (back + 1) * step)),
    );
    const across = pictureDifference(media.frame(at - step), media.frame(at));
    const most = Math.max(...motion);
    assert(
      across <= 2.5 * most + 6,
      `the clips jump at ${at}s: ${across.toFixed(1)} against a motion of ${most.toFixed(1)}`,
    );
  }
}

interface ClipResult {
  readonly nodes: number;
  readonly class_types: readonly string[];
  readonly execution_ms?: number | null;
  readonly queue_ms?: number | null;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

/** Renders the twelve clips, the four scenes at once, and joins them into the minute. */
export async function filmTheLion(endpoint: Endpoint): Promise<void> {
  const started = performance.now();
  // As Python's thread pool, every scene finishes before the first failure is raised.
  const scenes = await Promise.allSettled(SCENES.map((_, scene) => renderScene(endpoint, scene)));
  const videos = scenes.flatMap((scene) => {
    if (scene.status === "rejected") {
      throw scene.reason;
    }
    return scene.value;
  });
  const folder = join(GALLERY, FINAL_FOLDER);
  mkdirSync(folder, { recursive: true });
  const parts = videos.map(
    (video) => JSON.parse(readFileSync(join(dirname(video), "result.json"), "utf8")) as ClipResult,
  );
  const result: Record<string, unknown> = {
    title: "leao-na-savana",
    description:
      "Especial: um leão correndo na savana por um minuto. Doze trechos de 5 s do WAN " +
      "2.2 TI2V 5B em quatro cenas (manhã, tarde rente ao chão, meio-dia de frente e " +
      "pôr do sol); dentro de cada cena um trecho continua o anterior a partir do " +
      "último quadro dele. Os trechos foram gerados no RunPod e emendados com ffmpeg.",
    status: "running",
    nodes: sum(parts.map((part) => part.nodes)),
    class_types: [...new Set(parts.flatMap((part) => part.class_types))].sort(),
    execution_ms: sum(parts.map((part) => part.execution_ms ?? 0)),
    queue_ms: sum(parts.map((part) => part.queue_ms ?? 0)),
  };
  writeJson(join(folder, "workflow.json"), { trechos: videos.map((video) => basename(dirname(video))) });
  const target = join(folder, "leao_na_savana_1_minuto.mp4");
  try {
    assemble(videos, target);
    contactSheet(
      target,
      join(folder, "cenas.png"),
      SCENES.map((_, scene) => 180 + 360 * scene),
    );
    checkMinute(new Media(readFileSync(target), ".mp4"));
    result["status"] = "passed";
  } catch (error) {
    result["status"] = "failed";
    result["message"] = (error instanceof Error ? error.message : String(error)).slice(0, 6000);
    throw error;
  } finally {
    result["seconds"] = Math.round((performance.now() - started) / 100) / 10;
    writeJson(join(folder, "result.json"), result);
    writeGallery();
  }
}
