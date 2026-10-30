// The building blocks of the Python test_runpod_workflows.py: small groups of
// nodes the cases put together, and the pipelines they use.
import assert from "node:assert/strict";

import type { Link } from "../../src/nodes/base.js";
import { EmptyImage, GetImageSize, SaveImage } from "../../src/nodes/image.js";
import { RebatchImages, SplitImageToTileList } from "../../src/nodes/image/batch.js";
import { ImageCompositeMasked } from "../../src/nodes/image/compositing.js";
import { GrowMask, MaskComposite, MaskToImage, SolidMask, type MaskComposite_Operations } from "../../src/nodes/image/mask.js";
import {
  ImageCropV2,
  ImageFlip,
  ImageRotate,
  ImageStitch,
  type ImageFlip_Flip_Methods,
  type ImageRotate_Rotations,
  type ImageStitch_Directions,
  type ImageStitch_Spacing_Colors,
} from "../../src/nodes/image/transform.js";
import {
  ImageScale,
  ImageScaleBy,
  type ImageScale_Crops,
  type ImageScale_Upscale_Methods,
  type ImageScaleBy_Upscale_Methods,
} from "../../src/nodes/image/upscaling.js";
import { CLIPTextEncode } from "../../src/nodes/model/conditioning.js";
import { EmptyLatentImage, VAEDecode } from "../../src/nodes/model/latent.js";
import { CheckpointLoaderSimple } from "../../src/nodes/model/loaders.js";
import { KSampler, type KSampler_Sampler_Names, type KSampler_Schedulers } from "../../src/nodes/model/sampling.js";
import { int, link, string } from "../../src/nodes/schema.js";
import { StringCompare, StringLength } from "../../src/nodes/text.js";
import { ComfyMathExpression, type ComfyMathExpressionInputs } from "../../src/nodes/utilities.js";
import { ComfyNotNode, ComfySwitchNode } from "../../src/nodes/utilities/logic.js";
import { PrimitiveBoolean, PrimitiveBoundingBox, PrimitiveFloat, PrimitiveInt } from "../../src/nodes/utilities/primitive.js";
import { Pipeline } from "../../src/pipeline.js";
import type { Workflow } from "../../src/workflow.js";
import { BLACK, GREEN, QUADRANTS, RED, rgb, tuple, type Outputs } from "./media.js";
import type { Check } from "./registry.js";

export const CHECKPOINT = "sd_xl_base_1.0.safetensors";
export const NEGATIVE = "blurry, low quality, deformed, watermark, text, nsfw, nude";

/** An input that takes a value or a link to one. */
type Value<T> = T | Link;

export function solid(w: Workflow, width: Value<number>, height: Value<number>, color: Value<number>, batch: Value<number> = 1): Link {
  return w.addNode(EmptyImage, { width, height, batch_size: batch, color }).output.IMAGE;
}

export function save(w: Workflow, images: Link, label: Value<string>): void {
  w.addNode(SaveImage, { images, filename_prefix: label });
}

export function paste(
  w: Workflow,
  destination: Link,
  source: Link,
  x: Value<number> = 0,
  y: Value<number> = 0,
  mask: Link | null = null,
  resize = false,
): Link {
  const extra = mask === null ? {} : { mask };
  return w.addNode(ImageCompositeMasked, { destination, source, x, y, resize_source: resize, ...extra }).output.IMAGE;
}

export function stitch(
  w: Workflow,
  first: Link,
  second: Link,
  direction: ImageStitch_Directions = "right",
  match = false,
  spacing = 0,
  color: ImageStitch_Spacing_Colors = "white",
): Link {
  return w.addNode(ImageStitch, {
    image1: first,
    image2: second,
    direction,
    match_image_size: match,
    spacing_width: spacing,
    spacing_color: color,
  }).output.IMAGE;
}

export function strip(w: Workflow, colors: readonly number[], size: number): Link {
  let image = solid(w, size, size, colors[0]!);
  for (const color of colors.slice(1)) {
    image = stitch(w, image, solid(w, size, size, color));
  }
  return image;
}

export function tilesOf(w: Workflow, image: Link, size: number, overlap = 0): Link {
  return w.addNode(SplitImageToTileList, { image, tile_width: size, tile_height: size, overlap }).output.IMAGE;
}

export function batchOf(w: Workflow, colors: readonly number[], size = 64): Link {
  // A real batch of frames with different colors: a strip cut into a tile
  // list, then rebatched into a single batch.
  const tiles = tilesOf(w, strip(w, colors, size), size);
  return w.addNode(RebatchImages, { images: tiles, batch_size: colors.length }).output.IMAGE;
}

export function quadrants(w: Workflow, size: number, colors: readonly number[] = QUADRANTS): Link {
  const half = Math.floor(size / 2);
  let image = solid(w, size, size, colors[0]!);
  const corners = [
    [half, 0],
    [0, half],
    [half, half],
  ] as const;
  colors.slice(1, 4).forEach((color, index) => {
    const [x, y] = corners[index]!;
    image = paste(w, image, solid(w, half, half, color), x, y);
  });
  return image;
}

export function rotate(w: Workflow, image: Link, rotation: ImageRotate_Rotations): Link {
  return w.addNode(ImageRotate, { image, rotation }).output.IMAGE;
}

export function flip(w: Workflow, image: Link, method: ImageFlip_Flip_Methods): Link {
  return w.addNode(ImageFlip, { image, flip_method: method }).output.IMAGE;
}

export function scale(
  w: Workflow,
  image: Link,
  method: ImageScale_Upscale_Methods,
  width: Value<number>,
  height: Value<number>,
  crop: ImageScale_Crops = "disabled",
): Link {
  return w.addNode(ImageScale, { image, upscale_method: method, width, height, crop }).output.IMAGE;
}

export function scaleBy(w: Workflow, image: Link, method: ImageScaleBy_Upscale_Methods, factor: Value<number>): Link {
  return w.addNode(ImageScaleBy, { image, upscale_method: method, scale_by: factor }).output.IMAGE;
}

export function crop(w: Workflow, image: Link, x: number, y: number, width: number, height: number): Link {
  const region = w.addNode(PrimitiveBoundingBox, { x, y, width, height });
  return w.addNode(ImageCropV2, { image, crop_region: region.output.BOUNDING_BOX }).output.IMAGE;
}

export function solidMask(w: Workflow, value: Value<number>, width: Value<number>, height: Value<number>): Link {
  return w.addNode(SolidMask, { value, width, height }).output.MASK;
}

export function maskImage(w: Workflow, mask: Link): Link {
  return w.addNode(MaskToImage, { mask }).output.IMAGE;
}

export function combineMasks(
  w: Workflow,
  destination: Link,
  source: Link,
  operation: MaskComposite_Operations,
  x: Value<number> = 0,
  y: Value<number> = 0,
): Link {
  return w.addNode(MaskComposite, { destination, source, x, y, operation }).output.MASK;
}

export function constant(w: Workflow, value: number | boolean): Link {
  // The math node's values are sockets without widgets, so numbers come from
  // primitives. Python tells 2 from 2.0; the cases only pass floats with a
  // fraction, so an integer number is an int here too.
  if (typeof value === "boolean") {
    return w.addNode(PrimitiveBoolean, { value }).output.BOOLEAN;
  }
  if (Number.isInteger(value)) {
    return w.addNode(PrimitiveInt, { value }).output.INT;
  }
  return w.addNode(PrimitiveFloat, { value }).output.FLOAT;
}

export function calc(w: Workflow, expression: string, values: Readonly<Record<string, Link | number | boolean>>) {
  const named = Object.fromEntries(
    Object.entries(values).map(([name, value]) => [
      `values_${name}`,
      Array.isArray(value) ? value : constant(w, value as number | boolean),
    ]),
  );
  return w.addNode(ComfyMathExpression, { expression, ...named } as ComfyMathExpressionInputs).output;
}

export function length(w: Workflow, text: Value<string>): Link {
  return w.addNode(StringLength, { string: text }).output.length;
}

export function equals(w: Workflow, text: Value<string>, expected: Value<string>): Link {
  return w.addNode(StringCompare, { string_a: text, string_b: expected, mode: "Equal", case_sensitive: true }).output
    .BOOLEAN;
}

export function negate(w: Workflow, value: Link): Link {
  return w.addNode(ComfyNotNode, { value }).output.BOOLEAN;
}

export function board(w: Workflow, checks: Readonly<Record<string, Link>>, label = "board"): Check {
  // Each boolean picks a green (true) or red (false) tile; the tiles are
  // stitched in order, so a red tile names the check that failed on ComfyUI.
  const [green, red] = [solid(w, 8, 8, GREEN), solid(w, 8, 8, RED)];
  let image: Link | null = null;
  for (const flag of Object.values(checks)) {
    const tile = w.addNode(ComfySwitchNode, { switch: flag, on_true: green, on_false: red });
    image = image === null ? tile.output.output : stitch(w, image, tile.output.output);
  }
  save(w, image!, label);
  const names = Object.keys(checks);

  return (out: Outputs) => {
    const picture = out.picture(label);
    assert.deepEqual(picture.size, [8 * names.length, 8]);
    const failed = names.filter((_, index) => tuple(picture.pixel(8 * index + 4, 4)) !== tuple(rgb(GREEN)));
    assert.deepEqual(failed, [], `checks that came back false on ComfyUI: [${failed.map((name) => `'${name}'`).join(", ")}]`);
  };
}

export function sdxl(w: Workflow): [model: Link, clip: Link, vae: Link] {
  const loader = w.addNode(CheckpointLoaderSimple, { ckpt_name: CHECKPOINT });
  return [loader.output.MODEL, loader.output.CLIP, loader.output.VAE];
}

export function prompt(w: Workflow, clip: Link, text: Value<string>): Link {
  return w.addNode(CLIPTextEncode, { text, clip }).output.CONDITIONING;
}

export function emptyLatent(w: Workflow, width: Value<number>, height: Value<number>, batch: Value<number> = 1): Link {
  return w.addNode(EmptyLatentImage, { width, height, batch_size: batch }).output.LATENT;
}

export interface Sampling {
  readonly seed?: Value<number>;
  readonly steps?: Value<number>;
  readonly cfg?: Value<number>;
  readonly sampler?: KSampler_Sampler_Names;
  readonly scheduler?: KSampler_Schedulers;
  readonly denoise?: Value<number>;
}

export function ksample(w: Workflow, model: Link, positive: Link, negative: Link, latent: Link, sampling: Sampling = {}): Link {
  const { seed = 7, steps = 14, cfg = 6.5, sampler = "dpmpp_2m", scheduler = "karras", denoise = 1.0 } = sampling;
  return w.addNode(KSampler, {
    model,
    positive,
    negative,
    latent_image: latent,
    seed,
    steps,
    cfg,
    sampler_name: sampler,
    scheduler,
    denoise,
  }).output.LATENT;
}

export function decode(w: Workflow, latent: Link, vae: Link): Link {
  return w.addNode(VAEDecode, { samples: latent, vae }).output.IMAGE;
}

export function thumbnail(w: Workflow, image: Link, label: string, size = 16): void {
  save(w, scale(w, image, "area", size, size), label);
}

export class Framed extends Pipeline {
  static override readonly PARAMETERS = { image: link(), border: int(), color: int() };

  override build({ image, border, color }: { image: Link; border: number; color: number }) {
    const size = this.addNode(GetImageSize, { image });
    const width = this.addNode(ComfyMathExpression, { expression: `a + ${2 * border}`, values_a: size.output.width });
    const height = this.addNode(ComfyMathExpression, { expression: `a + ${2 * border}`, values_a: size.output.height });
    const frame = this.addNode(EmptyImage, { width: width.output.INT, height: height.output.INT, color });
    const framed = this.addNode(ImageCompositeMasked, {
      destination: frame.output.IMAGE,
      source: image,
      x: border,
      y: border,
      resize_source: false,
    });
    return { IMAGE: framed.output.IMAGE };
  }
}

export class Spotlight extends Pipeline {
  static override readonly PARAMETERS = { image: link(), size: int(), grow: int() };

  override build({ image, size, grow }: { image: Link; size: number; grow: number }) {
    const dims = this.addNode(GetImageSize, { image });
    const dark = this.addNode(EmptyImage, { width: dims.output.width, height: dims.output.height, color: BLACK });
    const base = this.addNode(SolidMask, { value: 0.0, width: dims.output.width, height: dims.output.height });
    const spot = this.addNode(SolidMask, { value: 1.0, width: size, height: size });
    const offset = this.addNode(ComfyMathExpression, { expression: `(a - ${size}) / 2`, values_a: dims.output.width });
    const placed = this.addNode(MaskComposite, {
      destination: base.output.MASK,
      source: spot.output.MASK,
      x: offset.output.INT,
      y: offset.output.INT,
      operation: "add",
    });
    const grown = this.addNode(GrowMask, { mask: placed.output.MASK, expand: grow, tapered_corners: false });
    const lit = this.addNode(ImageCompositeMasked, {
      destination: dark.output.IMAGE,
      source: image,
      x: 0,
      y: 0,
      resize_source: false,
      mask: grown.output.MASK,
    });
    return { IMAGE: lit.output.IMAGE, MASK: grown.output.MASK };
  }
}

export class TextToImage extends Pipeline {
  static override readonly PARAMETERS = {
    text: string(),
    width: int({ optional: true }),
    height: int({ optional: true }),
    steps: int({ optional: true }),
    seed: int({ optional: true }),
  };

  override build({
    text,
    width = 768,
    height = 768,
    steps = 14,
    seed = 42,
  }: {
    text: string;
    width?: number;
    height?: number;
    steps?: number;
    seed?: number;
  }) {
    const loader = this.addNode(CheckpointLoaderSimple, { ckpt_name: CHECKPOINT });
    const positive = this.addNode(CLIPTextEncode, { text, clip: loader.output.CLIP });
    const negative = this.addNode(CLIPTextEncode, { text: NEGATIVE, clip: loader.output.CLIP });
    const latent = this.addNode(EmptyLatentImage, { width, height, batch_size: 1 });
    const sampled = this.addNode(KSampler, {
      model: loader.output.MODEL,
      positive: positive.output.CONDITIONING,
      negative: negative.output.CONDITIONING,
      latent_image: latent.output.LATENT,
      sampler_name: "dpmpp_2m",
      scheduler: "karras",
      seed,
      steps,
      cfg: 6.5,
    });
    const image = this.addNode(VAEDecode, { samples: sampled.output.LATENT, vae: loader.output.VAE });
    return { IMAGE: image.output.IMAGE };
  }
}
