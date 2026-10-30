// The clips of the Python test_runpod_lion.py: a one-minute video of a lion running
// in the savanna, made with WAN 2.2 TI2V 5B on the endpoint's volume. WAN renders
// about 5 s (121 frames) per job, so the minute is four scenes of three clips; inside
// a scene each clip starts from the last frame of the one before.
import assert from "node:assert/strict";

import { AudioEqualizer3Band, LoadAudio } from "../../src/nodes/audio.js";
import type { Link } from "../../src/nodes/base.js";
import { LoadImage, SaveImage } from "../../src/nodes/image.js";
import { ImageFromBatch } from "../../src/nodes/image/batch.js";
import { CLIPTextEncode } from "../../src/nodes/model/conditioning.js";
import { Wan22ImageToVideoLatent } from "../../src/nodes/model/conditioning/wan.js";
import { VAEDecode } from "../../src/nodes/model/latent.js";
import { CLIPLoader, UNETLoader, VAELoader } from "../../src/nodes/model/loaders.js";
import { ModelSamplingSD3 } from "../../src/nodes/model/patch/stable_diffusion.js";
import { KSampler } from "../../src/nodes/model/sampling.js";
import { int, link, string } from "../../src/nodes/schema.js";
import { CreateVideo, SaveVideo } from "../../src/nodes/video.js";
import { Pipeline } from "../../src/pipeline.js";
import { Workflow } from "../../src/workflow.js";
import { luminance, Media, pictureDifference, spread, streamSeconds } from "../runpod_videos/media.js";
import { PythonRandom, wav } from "../runpod_videos/sound.js";
import { pngSize, roundHalfEven, tuple } from "../runpod_workflows/media.js";
import type { Check } from "../runpod_workflows/registry.js";

export const WAN_MODEL = "wan2.2_ti2v_5B_fp16.safetensors";
export const WAN_TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors";
export const WAN_VAE = "wan2.2_vae.safetensors";
// WAN's own negative prompt, plus five legs, extra limbs, deformed animal, text, watermark.
export const WAN_NEGATIVE =
  "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，" +
  "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，" +
  "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，" +
  "杂乱的背景，三条腿，背景人很多，倒着走，五条腿，多余的四肢，变形的动物，文字，水印";
export const FPS = 24;
export const WIDTH = 1280;
export const HEIGHT = 704;
export const CLIP_FRAMES = 121;
// The 121st frame starts the next clip, so each clip shows 120 frames: 5 s.
export const SHOWN = CLIP_FRAMES - 1;
export const CLIP_SECONDS = SHOWN / FPS;
export const PARTS = 3;
export const MINUTE = 60;
export const WIND_RATE = 22_050;

const LION = "a large adult male lion with a thick golden-brown mane";
const DOCUMENTARY = "Cinematic wildlife documentary footage";
const MOTION = "Realistic animal anatomy and natural running motion, sharp detail.";
export const SCENES = [
  `${DOCUMENTARY}. Wide tracking shot from a moving vehicle: ${LION} sprints at full ` +
    "speed from left to right across the vast golden grass of the African savanna, its " +
    "paws pounding the dry ground and kicking up dust, scattered acacia trees and " +
    "distant hills behind it, warm early-morning sunlight, the camera smoothly follows " +
    `the lion. ${MOTION}`,
  `${DOCUMENTARY}. Low-angle side tracking shot at ground level: ${LION} gallops ` +
    "through tall dry savanna grass from left to right, shoulder muscles rippling, the " +
    "mane bouncing with every stride, dust and grass seeds flying, golden afternoon " +
    "light, shallow depth of field, the camera moves alongside the lion at the same " +
    `speed. ${MOTION}`,
  `${DOCUMENTARY}. Frontal tracking shot: ${LION} runs straight toward the camera ` +
    "across the open savanna, intense amber eyes, the mane flowing in the wind, front " +
    "paws reaching forward with each bound, a dust cloud behind it, bright midday sun, " +
    `the camera moves backward to keep the lion in frame. ${MOTION}`,
  `${DOCUMENTARY} at sunset. Side tracking shot: the silhouette of ${LION} runs from ` +
    "left to right along a savanna ridge in front of a huge orange setting sun, acacia " +
    "tree silhouettes, an orange and purple sky, glowing dust trailing behind it, the " +
    `camera follows the lion smoothly. ${MOTION}`,
];
export const SCENE_NAMES = ["manhã, plano aberto", "tarde, rente ao chão", "meio-dia, de frente", "pôr do sol"];
export const FINAL_FOLDER = "000-especial-leao-na-savana";

/** A WAN 2.2 TI2V 5B clip of 121 frames, from the prompt or from a start frame. */
export class WanClip extends Pipeline {
  static override readonly PARAMETERS = { prompt: string(), seed: int(), start_image: link({ optional: true }) };

  override build({ prompt, seed, start_image }: { prompt: string; seed: number; start_image?: Link }) {
    const unet = this.addNode(UNETLoader, { unet_name: WAN_MODEL, weight_dtype: "default" });
    const model = this.addNode(ModelSamplingSD3, { model: unet.output.MODEL, shift: 8.0 });
    const textEncoder = this.addNode(CLIPLoader, { clip_name: WAN_TEXT_ENCODER, type: "wan", device: "default" });
    const clip = textEncoder.output.CLIP;
    const vae = this.addNode(VAELoader, { vae_name: WAN_VAE }).output.VAE;
    const start = start_image === undefined ? {} : { start_image };
    const latent = this.addNode(Wan22ImageToVideoLatent, {
      vae,
      width: WIDTH,
      height: HEIGHT,
      length: CLIP_FRAMES,
      batch_size: 1,
      ...start,
    });
    const sampled = this.addNode(KSampler, {
      model: model.output.MODEL,
      positive: this.addNode(CLIPTextEncode, { text: prompt, clip }).output.CONDITIONING,
      negative: this.addNode(CLIPTextEncode, { text: WAN_NEGATIVE, clip }).output.CONDITIONING,
      latent_image: latent.output.LATENT,
      seed,
      steps: 20,
      cfg: 5.0,
      sampler_name: "uni_pc",
      scheduler: "simple",
      denoise: 1.0,
    });
    const frames = this.addNode(VAEDecode, { samples: sampled.output.LATENT, vae });
    return { IMAGE: frames.output.IMAGE };
  }
}

/** Two digits, as the Python suite numbers its clips. */
function two(number: number): string {
  return String(number).padStart(2, "0");
}

export class Clip {
  constructor(
    readonly scene: number,
    readonly part: number,
  ) {}

  get number(): number {
    return this.scene * PARTS + this.part + 1;
  }

  get label(): string {
    return `leao_${two(this.number)}`;
  }

  get following(): string {
    return `proximo_${two(this.number)}`;
  }

  get folder(): string {
    return `000-especial-leao-parte-${two(this.number)}`;
  }

  get start(): string | null {
    return this.part === 0 ? null : `inicio_${two(this.number)}.png`;
  }

  get sound(): string {
    return `vento_${two(this.number)}.wav`;
  }

  get seed(): number {
    return 2026 + 100 * this.scene + this.part;
  }

  get description(): string {
    const origin = this.start === null ? "do texto" : "do último quadro do trecho anterior";
    return (
      `Leão na savana, trecho ${this.number} de ${CLIPS.length} (cena ${this.scene + 1}: ` +
      `${SCENE_NAMES[this.scene]}, parte ${this.part + 1} de ${PARTS}): WAN 2.2 TI2V 5B ` +
      `em ${WIDTH}x${HEIGHT}, ${CLIP_FRAMES} quadros a ${FPS} fps, partindo ${origin}; ` +
      "mostra 120 quadros e devolve o 121º para o trecho seguinte, com o vento " +
      "equalizado no ComfyUI."
    );
  }
}

export const CLIPS = SCENES.flatMap((_, scene) => Array.from({ length: PARTS }, (_, part) => new Clip(scene, part)));

let theWind: readonly number[] | undefined;

export function wind(): readonly number[] {
  if (theWind === undefined) {
    // A minute of brown noise (integrated white noise) under two slow gusts.
    const generator = new PythonRandom(7);
    let level = 0.0;
    const samples: number[] = [];
    for (let n = 0; n < MINUTE * WIND_RATE; n++) {
      level = 0.996 * level + 0.06 * generator.uniform(-1.0, 1.0);
      const t = n / WIND_RATE;
      let gust = 0.55 + 0.3 * Math.sin(2 * Math.PI * 0.11 * t);
      gust += 0.15 * Math.sin(2 * Math.PI * 0.037 * t + 1.3);
      samples.push(level * gust);
    }
    const peak = samples.reduce((loudest, value) => Math.max(loudest, Math.abs(value)), 0);
    theWind = samples.map((value) => (0.5 * value) / peak);
  }
  return theWind;
}

export function windOf(clip: Clip): Buffer {
  const size = roundHalfEven(CLIP_SECONDS * WIND_RATE);
  const start = (clip.number - 1) * size;
  return wav(WIND_RATE, wind().slice(start, start + size));
}

export function clipWorkflow(clip: Clip): Workflow {
  const w = new Workflow();
  const start: { start_image?: Link } = {};
  if (clip.start !== null) {
    start.start_image = w.addNode(LoadImage, { image: clip.start }).output.IMAGE;
  }
  const frames = w.addPipeline(WanClip, { prompt: SCENES[clip.scene]!, seed: clip.seed, ...start }).output.IMAGE;
  const shown = w.addNode(ImageFromBatch, { image: frames, batch_index: 0, length: SHOWN });
  const following = w.addNode(ImageFromBatch, { image: frames, batch_index: SHOWN, length: 1 });
  const sound = w.addNode(AudioEqualizer3Band, {
    audio: w.addNode(LoadAudio, { audio: clip.sound }).output.AUDIO,
    low_gain_dB: 3.0,
    low_freq: 200,
    mid_gain_dB: -6.0,
    mid_freq: 1000,
    mid_q: 0.7,
    high_gain_dB: -18.0,
    high_freq: 3000,
  }).output.AUDIO;
  const video = w.addNode(CreateVideo, { images: shown.output.IMAGE, fps: FPS, audio: sound }).output.VIDEO;
  w.addNode(SaveVideo, { video, filename_prefix: clip.label, format: "mp4", format_codec: "h264" });
  w.addNode(SaveImage, { images: following.output.IMAGE, filename_prefix: clip.following });
  return w;
}

export function uploadsOf(clip: Clip, startPng: Buffer | null): Record<string, Buffer> {
  const uploads: Record<string, Buffer> = { [clip.sound]: windOf(clip) };
  if (clip.start !== null) {
    assert(startPng !== null, `clip ${clip.number} needs the frame before it`);
    uploads[clip.start] = startPng;
  }
  return uploads;
}

export function checkClip(clip: Clip): Check {
  return (out) => {
    const media = Media.returned(out, clip.label);
    const [width, height] = media.size;
    assert(width === WIDTH && height === HEIGHT, `the clip is ${tuple(media.size)}`);
    assert(media.fps === FPS, `the clip runs at ${media.fps} fps`);
    assert(media.frames === SHOWN, `the clip has ${media.frames} frames`);
    assert(Math.abs(media.seconds - CLIP_SECONDS) <= 0.1, `the clip lasts ${media.seconds}s`);
    assert.deepEqual([media.audio.codec_name, media.audio.channels], ["aac", 1]);
    assert(Number(media.audio.sample_rate) === WIND_RATE);
    assert(Math.abs(streamSeconds(media.audio) - CLIP_SECONDS) <= 0.1);
    assert.deepEqual(pngSize(out.one(clip.following)), [WIDTH, HEIGHT]);
    const [first, last] = [media.frame(0.5), media.frame(4.5)];
    assert(spread(luminance(first)) > 15, "the clip is flat");
    assert(pictureDifference(first, last) > 2, "nothing moves in the clip");
  };
}
