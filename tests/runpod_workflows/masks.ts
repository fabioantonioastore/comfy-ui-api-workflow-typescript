// The mask cases of the Python test_runpod_workflows.py.
import assert from "node:assert/strict";

import { LoadImage, LoadImageMask } from "../../src/nodes/image.js";
import {
  CropMask,
  FeatherMask,
  GrowMask,
  ImageColorToMask,
  ImageToMask,
  InvertMask,
  ThresholdMask,
} from "../../src/nodes/image/mask.js";
import { combineMasks, maskImage, paste, quadrants, save, solid, solidMask } from "./blocks.js";
import {
  assertPixel,
  assertQuadrants,
  BLACK,
  BLUE,
  colorsOf,
  gray,
  GREEN,
  QUADRANTS,
  quarterPoints,
  RED,
  rgb,
  RGBA_PNG,
  tuple,
  WHITE,
  YELLOW,
  type Color,
} from "./media.js";
import { addCase } from "./registry.js";

addCase(
  "mask_algebra_board",
  "Álgebra de máscaras com MaskComposite (add, subtract, multiply, xor, or, and) com " +
    "deslocamentos, inclusive saindo da borda; cada região conferida.",
  {},
  (w) => {
    let mask = solidMask(w, 0.5, 64, 64);
    mask = combineMasks(w, mask, solidMask(w, 0.25, 32, 32), "add");
    mask = combineMasks(w, mask, solidMask(w, 1.0, 32, 32), "subtract", 32);
    mask = combineMasks(w, mask, solidMask(w, 0.5, 32, 32), "multiply", 0, 32);
    const full = solidMask(w, 1.0, 64, 64);
    const results = {
      mixed: mask,
      xor: combineMasks(w, mask, full, "xor"),
      or_corner: combineMasks(w, mask, solidMask(w, 1.0, 16, 16), "or", 56, 56),
      and: combineMasks(w, mask, full, "and"),
    };
    for (const [label, result] of Object.entries(results)) {
      save(w, maskImage(w, result), label);
    }

    return (out) => {
      const expected = {
        mixed: [0.75, 0.0, 0.25, 0.5],
        xor: [0.0, 1.0, 1.0, 1.0],
        and: [1.0, 0.0, 0.0, 0.0],
      };
      for (const [label, values] of Object.entries(expected)) {
        const picture = out.picture(label);
        quarterPoints(64, 64).forEach(([x, y], index) => assertPixel(picture, x, y, gray(values[index]!), 1));
      }
      const corner = out.picture("or_corner");
      assertPixel(corner, 60, 60, WHITE, 1);
      assertPixel(corner, 40, 40, gray(0.5), 1);
    };
  },
);

addCase(
  "mask_feather_crop_threshold",
  "FeatherMask em três lados, CropMask e ThresholdMask; o degradê linear do feather " +
    "conferido coluna a coluna.",
  {},
  (w) => {
    const feathered = w.addNode(FeatherMask, { mask: solidMask(w, 1.0, 32, 32), left: 8, top: 0, right: 4, bottom: 2 })
      .output.MASK;
    const cropped = w.addNode(CropMask, { mask: feathered, x: 0, y: 8, width: 16, height: 8 });
    const threshold = w.addNode(ThresholdMask, { mask: feathered, value: 0.3 });
    save(w, maskImage(w, feathered), "feathered");
    save(w, maskImage(w, cropped.output.MASK), "cropped");
    save(w, maskImage(w, threshold.output.MASK), "threshold");

    return (out) => {
      const ramp = out.picture("feathered");
      const expected = [
        [0, 10, 1 / 8],
        [3, 10, 4 / 8],
        [7, 10, 1.0],
        [31, 10, 1 / 4],
        [29, 10, 3 / 4],
        [15, 31, 1 / 2],
        [0, 31, 1 / 16],
      ] as const;
      for (const [x, y, value] of expected) {
        assertPixel(ramp, x, y, gray(value), 1);
      }
      const piece = out.picture("cropped");
      assert.deepEqual(piece.size, [16, 8]);
      assertPixel(piece, 3, 2, gray(0.5), 1);
      const cut = out.picture("threshold");
      assertPixel(cut, 0, 10, BLACK);
      assertPixel(cut, 3, 10, WHITE);
      assertPixel(cut, 31, 10, BLACK);
    };
  },
);

addCase(
  "grow_mask_square_vs_tapered",
  "GrowMask com kernel quadrado e com cantos arredondados, e encolhendo; as bordas do " +
    "quadrado conferidas pixel a pixel.",
  {},
  (w) => {
    const square = combineMasks(w, solidMask(w, 0.0, 32, 32), solidMask(w, 1.0, 8, 8), "add", 12, 12);
    const variants = { square: [2, false], tapered: [2, true], shrunk: [-2, false] } as const;
    for (const [label, [expand, tapered]] of Object.entries(variants)) {
      const grown = w.addNode(GrowMask, { mask: square, expand, tapered_corners: tapered });
      save(w, maskImage(w, grown.output.MASK), label);
    }

    return (out) => {
      const expected: Readonly<Record<string, readonly (readonly [number, number, Color])[]>> = {
        square: [
          [10, 10, WHITE],
          [21, 21, WHITE],
          [9, 9, BLACK],
        ],
        tapered: [
          [10, 12, WHITE],
          [11, 11, WHITE],
          [10, 10, BLACK],
        ],
        shrunk: [
          [14, 14, WHITE],
          [17, 17, WHITE],
          [13, 13, BLACK],
        ],
      };
      for (const [label, points] of Object.entries(expected)) {
        const picture = out.picture(label);
        for (const [x, y, color] of points) {
          assertPixel(picture, x, y, color);
        }
      }
    };
  },
);

addCase(
  "image_channels_to_masks",
  "ImageToMask nos canais vermelho, verde e azul de quadrantes coloridos: cada máscara " +
    "acende só onde o canal existe.",
  {},
  (w) => {
    const image = quadrants(w, 32);
    for (const channel of ["red", "green", "blue"] as const) {
      const mask = w.addNode(ImageToMask, { image, channel }).output.MASK;
      save(w, maskImage(w, mask), `mask_${channel}`);
    }

    return (out) => {
      ["red", "green", "blue"].forEach((channel, index) => {
        const picture = out.picture(`mask_${channel}`);
        quarterPoints(32, 32).forEach(([x, y], point) => {
          assertPixel(picture, x, y, rgb(QUADRANTS[point]!)[index] ? WHITE : BLACK);
        });
      });
    };
  },
);

addCase(
  "color_to_mask_picks_quadrant",
  "ImageColorToMask isola o quadrante verde; a máscara pinta esse quadrante de amarelo " +
    "com ImageCompositeMasked; uma cor ausente gera máscara vazia.",
  {},
  (w) => {
    const image = quadrants(w, 32);
    const green = w.addNode(ImageColorToMask, { image, color: GREEN }).output.MASK;
    const absent = w.addNode(ImageColorToMask, { image, color: 0x123456 }).output.MASK;
    const inverted = w.addNode(InvertMask, { mask: absent }).output.MASK;
    save(w, maskImage(w, green), "green_mask");
    save(w, maskImage(w, inverted), "inverted_absent");
    save(w, paste(w, image, solid(w, 32, 32, YELLOW), 0, 0, green), "painted");

    return (out) => {
      const mask = out.picture("green_mask");
      quarterPoints(32, 32).forEach(([x, y], index) => {
        assertPixel(mask, x, y, QUADRANTS[index] === GREEN ? WHITE : BLACK);
      });
      assert.deepEqual([...colorsOf(out.picture("inverted_absent"))], [tuple(rgb(WHITE))]);
      assertQuadrants(out.picture("painted"), [RED, YELLOW, BLUE, WHITE]);
    };
  },
);

addCase(
  "uploaded_rgba_masks",
  "Upload de um PNG RGBA pelo worker: LoadImage devolve RGB intacto e máscara 1 - alfa, " +
    "LoadImageMask lê os canais alfa, vermelho e verde.",
  { uploads: { "probe_rgba.png": RGBA_PNG } },
  (w) => {
    const loaded = w.addNode(LoadImage, { image: "probe_rgba.png" });
    save(w, loaded.output.IMAGE, "image");
    save(w, maskImage(w, loaded.output.MASK), "mask");
    for (const channel of ["alpha", "red", "green"] as const) {
      const mask = w.addNode(LoadImageMask, { image: "probe_rgba.png", channel });
      save(w, maskImage(w, mask.output.MASK), `channel_${channel}`);
    }

    return (out) => {
      const image = out.picture("image");
      assert.deepEqual(image.size, [32, 32]);
      assertPixel(image, 4, 4, [200, 100, 50], 0);
      assertPixel(image, 28, 4, [200, 100, 50], 0);
      for (const label of ["mask", "channel_alpha"]) {
        const mask = out.picture(label);
        assertPixel(mask, 4, 4, BLACK, 1);
        assertPixel(mask, 28, 4, gray(1 - 64 / 255), 1);
      }
      assertPixel(out.picture("channel_red"), 10, 10, [200, 200, 200], 1);
      assertPixel(out.picture("channel_green"), 10, 10, [100, 100, 100], 1);
    };
  },
);

addCase(
  "feathered_mask_blends_composite",
  "Máscara com FeatherMask usada no ImageCompositeMasked: a mistura preto/branco " +
    "segue a rampa da máscara coluna por coluna.",
  {},
  (w) => {
    const mask = w.addNode(FeatherMask, { mask: solidMask(w, 1.0, 32, 32), left: 8 }).output.MASK;
    save(w, paste(w, solid(w, 32, 32, BLACK), solid(w, 32, 32, WHITE), 0, 0, mask), "blend");

    return (out) => {
      const picture = out.picture("blend");
      for (const column of [0, 3, 6, 10, 25]) {
        assertPixel(picture, column, 16, gray(Math.min((column + 1) / 8, 1.0)), 1);
      }
    };
  },
);
