// The cases of the Python test_runpod_workflows.py that upload images to the
// worker before the job.
import assert from "node:assert/strict";

import { LoadImage, LoadImageMask } from "../../src/nodes/image.js";
import { CenterCropImages } from "../../src/nodes/image/transform.js";
import { crop, flip, maskImage, paste, rotate, save, scaleBy, solid } from "./blocks.js";
import {
  assertPixel,
  BLACK,
  BLUE,
  colorsOf,
  FLIP_LABELS,
  flippedSource,
  FLIPS,
  pattern,
  PATTERN_PNG,
  rgb,
  ROTATION_LABELS,
  rotatedSource,
  ROTATIONS,
  STRIPE_PNG,
  tuple,
  WHITE,
  YELLOW,
  type Outputs,
  type Size,
} from "./media.js";
import { addCase } from "./registry.js";

/** Expects every pixel of the output to be the pattern's pixel it comes from. */
function expectPattern(out: Outputs, label: string, size: Size, source: (x: number, y: number) => [number, number]): void {
  const picture = out.picture(label);
  assert.deepEqual(picture.size, size);
  for (let y = 0; y < size[1]; y++) {
    for (let x = 0; x < size[0]; x++) {
      assertPixel(picture, x, y, pattern(...source(x, y)), 0);
    }
  }
}

addCase(
  "uploaded_image_round_trip",
  "PNG com um padrão diferente em cada pixel enviado ao worker e carregado com " +
    "LoadImage: salvo de volta, tem de ser idêntico pixel a pixel.",
  { uploads: { "probe_pattern.png": PATTERN_PNG } },
  (w) => {
    const loaded = w.addNode(LoadImage, { image: "probe_pattern.png" });
    save(w, loaded.output.IMAGE, "loaded");
    save(w, maskImage(w, loaded.output.MASK), "mask");

    return (out) => {
      expectPattern(out, "loaded", [24, 16], (x, y) => [x, y]);
      assert.deepEqual([...colorsOf(out.picture("mask"))], [tuple(rgb(BLACK))]);
    };
  },
);

addCase(
  "uploaded_image_geometry",
  "O padrão enviado é girado e espelhado no ComfyUI; todos os pixels de todas as saídas " +
    "são conferidos contra a transformação exata.",
  { uploads: { "probe_pattern.png": PATTERN_PNG } },
  (w) => {
    const loaded = w.addNode(LoadImage, { image: "probe_pattern.png" }).output.IMAGE;
    for (const rotation of ROTATIONS) {
      save(w, rotate(w, loaded, rotation), ROTATION_LABELS[rotation]);
    }
    for (const method of FLIPS) {
      save(w, flip(w, loaded, method), FLIP_LABELS[method]);
    }

    return (out) => {
      for (const rotation of ROTATIONS) {
        const size: Size = rotation === "180 degrees" ? [24, 16] : [16, 24];
        expectPattern(out, ROTATION_LABELS[rotation], size, (x, y) => rotatedSource(rotation, x, y, 24, 16));
      }
      for (const method of FLIPS) {
        expectPattern(out, FLIP_LABELS[method], [24, 16], (x, y) => flippedSource(method, x, y, 24, 16));
      }
    };
  },
);

addCase(
  "uploaded_image_scale_and_crop_exact",
  "O padrão enviado é ampliado 3x com nearest-exact e recortado com ImageCropV2 e " +
    "CenterCropImages; cada pixel tem de vir do lugar certo.",
  { uploads: { "probe_pattern.png": PATTERN_PNG } },
  (w) => {
    const loaded = w.addNode(LoadImage, { image: "probe_pattern.png" }).output.IMAGE;
    save(w, scaleBy(w, loaded, "nearest-exact", 3.0), "tripled");
    save(w, crop(w, loaded, 5, 3, 8, 6), "cropped");
    const centered = w.addNode(CenterCropImages, { images: loaded, width: 12, height: 8 });
    save(w, centered.output.images, "centered");

    return (out) => {
      expectPattern(out, "tripled", [72, 48], (x, y) => [Math.floor(x / 3), Math.floor(y / 3)]);
      expectPattern(out, "cropped", [8, 6], (x, y) => [x + 5, y + 3]);
      expectPattern(out, "centered", [12, 8], (x, y) => [x + 6, y + 4]);
    };
  },
);

addCase(
  "uploaded_mask_drives_composite",
  "Faixa vermelha enviada ao worker vira máscara com LoadImageMask (canal red) e " +
    "decide onde o amarelo cobre o azul.",
  { uploads: { "probe_stripe.png": STRIPE_PNG } },
  (w) => {
    const mask = w.addNode(LoadImageMask, { image: "probe_stripe.png", channel: "red" }).output.MASK;
    save(w, paste(w, solid(w, 24, 16, BLUE), solid(w, 24, 16, YELLOW), 0, 0, mask), "composite");
    save(w, maskImage(w, mask), "mask");

    return (out) => {
      const composite = out.picture("composite");
      assertPixel(composite, 5, 8, YELLOW, 0);
      assertPixel(composite, 18, 8, BLUE, 0);
      assertPixel(out.picture("mask"), 5, 8, WHITE, 0);
    };
  },
);
