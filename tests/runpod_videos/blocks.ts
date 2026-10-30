// The colors and building blocks of the Python test_runpod_videos.py: sounds loaded,
// joined and cut with the audio nodes, and frames laid side by side in a strip that
// a timeline cuts back into one batch.
import {
  AudioAdjustVolume,
  AudioConcat,
  AudioMerge,
  EmptyAudio,
  JoinAudioChannels,
  LoadAudio,
  SplitAudioChannels,
  TrimAudioDuration,
  type AudioConcat_Directions,
  type AudioMerge_Merge_Methods,
} from "../../src/nodes/audio.js";
import type { Link } from "../../src/nodes/base.js";
import { RebatchImages, RepeatImageBatch, SplitImageToTileList } from "../../src/nodes/image/batch.js";
import { ImageBlend } from "../../src/nodes/image/filters.js";
import { CreateVideo, SaveVideo, type SaveVideo_Codecs, type SaveVideo_Formats } from "../../src/nodes/video.js";
import type { Workflow } from "../../src/workflow.js";
import { solid, stitch } from "../runpod_workflows/blocks.js";
import type { Size } from "../runpod_workflows/media.js";

export const WHITE = 0xffffff;
export const BLACK = 0x000000;
export const NIGHT = 0x101828;
export const YELLOW = 0xffd000;
export const NAVY = 0x14205a;
export const RED = 0xe02020;
export const GREEN = 0x20b040;
export const BLUE = 0x2050e0;

export function load(w: Workflow, name: string): Link {
  return w.addNode(LoadAudio, { audio: name }).output.AUDIO;
}

export function concat(w: Workflow, first: Link, second: Link, direction: AudioConcat_Directions = "after"): Link {
  return w.addNode(AudioConcat, { audio1: first, audio2: second, direction }).output.AUDIO;
}

export function doubled(w: Workflow, audio: Link, times: number): Link {
  for (let time = 0; time < times; time++) {
    audio = concat(w, audio, audio);
  }
  return audio;
}

export function trim(w: Workflow, audio: Link, start: number, duration: number): Link {
  return w.addNode(TrimAudioDuration, { audio, start_index: start, duration }).output.AUDIO;
}

export function louder(w: Workflow, audio: Link, decibels: number): Link {
  return w.addNode(AudioAdjustVolume, { audio, volume: decibels }).output.AUDIO;
}

export function merge(w: Workflow, first: Link, second: Link, method: AudioMerge_Merge_Methods): Link {
  return w.addNode(AudioMerge, { audio1: first, audio2: second, merge_method: method }).output.AUDIO;
}

export function stereo(w: Workflow, left: Link, right: Link): Link {
  return w.addNode(JoinAudioChannels, { audio_left: left, audio_right: right }).output.audio;
}

export function leftChannel(w: Workflow, audio: Link): Link {
  return w.addNode(SplitAudioChannels, { audio }).output.left;
}

export function pause(w: Workflow, seconds: number, rate: number): Link {
  return w.addNode(EmptyAudio, { duration: seconds, sample_rate: rate, channels: 1 }).output.AUDIO;
}

export function sideBySide(w: Workflow, images: readonly Link[]): Link {
  let strip = images[0]!;
  for (const image of images.slice(1)) {
    strip = stitch(w, strip, image);
  }
  return strip;
}

export function colorStrip(w: Workflow, colors: readonly number[], size: Size): Link {
  return sideBySide(
    w,
    colors.map((color) => solid(w, size[0], size[1], color)),
  );
}

export function timeline(w: Workflow, strip: Link, size: Size, total: number, hold = 1): Link {
  // The strip is cut into frame-sized tiles, a list in left-to-right order; each
  // tile (a frame or a whole batch) is held for `hold` frames and the list is
  // rebatched into one batch of `total` frames.
  let tiles = w.addNode(SplitImageToTileList, {
    image: strip,
    tile_width: size[0],
    tile_height: size[1],
    overlap: 0,
  }).output.IMAGE;
  if (hold > 1) {
    tiles = w.addNode(RepeatImageBatch, { image: tiles, amount: hold }).output.IMAGE;
  }
  return w.addNode(RebatchImages, { images: tiles, batch_size: total }).output.IMAGE;
}

export function repeated(w: Workflow, images: Link, amount: number): Link {
  return w.addNode(RepeatImageBatch, { image: images, amount }).output.IMAGE;
}

export function dimmed(w: Workflow, image: Link, size: Size, darkness: number): Link {
  const black = solid(w, size[0], size[1], BLACK);
  return w.addNode(ImageBlend, { image1: image, image2: black, blend_factor: darkness, blend_mode: "normal" }).output
    .IMAGE;
}

export function movie(
  w: Workflow,
  images: Link,
  fps: number,
  audio: Link,
  label: string,
  container: SaveVideo_Formats = "mp4",
  codec: SaveVideo_Codecs = "h264",
): void {
  const clip = w.addNode(CreateVideo, { images, fps, audio }).output.VIDEO;
  w.addNode(SaveVideo, { video: clip, filename_prefix: label, format: container, format_codec: codec });
}
