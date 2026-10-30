// The color and filter cases of the Python test_runpod_workflows.py.
import assert from "node:assert/strict";

import type { Link } from "../../src/nodes/base.js";
import { AdjustBrightness, AdjustContrast } from "../../src/nodes/image/adjustments.js";
import { ImageColorSpace, ImageInvert, ImageRGBToYUV, ImageYUVToRGB, NormalizeImages } from "../../src/nodes/image/color.js";
import { JoinImageWithAlpha, PorterDuffImageComposite, SplitImageWithAlpha } from "../../src/nodes/image/compositing.js";
import { Canny, ImageAddNoise, ImageBlend, ImageBlur, ImageQuantize, ImageSharpen, Morphology } from "../../src/nodes/image/filters.js";
import type { Workflow } from "../../src/workflow.js";
import { maskImage, paste, quadrants, save, scale, scaleBy, solid, solidMask, stitch } from "./blocks.js";
import {
  assertPixel,
  assertQuadrants,
  BLACK,
  BLUE,
  colorsOf,
  CYAN,
  gray,
  GRAY,
  pixelOf,
  QUADRANTS,
  quarterPoints,
  RED,
  rgb,
  sameRows,
  WHITE,
  type Color,
} from "./media.js";
import { addCase } from "./registry.js";

const BLEND_MODES = ["normal", "multiply", "screen", "overlay", "soft_light", "difference"] as const;

function blended(mode: (typeof BLEND_MODES)[number], first: number, second: number, factor: number): number[] {
  const g = (x: number) => (x <= 0.25 ? ((16 * x - 12) * x + 4) * x : Math.sqrt(x));
  const one = (a: number, b: number) => {
    let value: number;
    if (mode === "normal") {
      value = b;
    } else if (mode === "multiply") {
      value = a * b;
    } else if (mode === "screen") {
      value = 1 - (1 - a) * (1 - b);
    } else if (mode === "overlay") {
      value = a <= 0.5 ? 2 * a * b : 1 - 2 * (1 - a) * (1 - b);
    } else if (mode === "soft_light") {
      value = b <= 0.5 ? a - (1 - 2 * b) * a * (1 - a) : a + (2 * b - 1) * (g(a) - a);
    } else {
      value = a - b;
    }
    return a * (1 - factor) + value * factor;
  };
  const seconds = rgb(second);
  return pixelOf(rgb(first).map((a, index) => one(a / 255, seconds[index]! / 255)));
}

addCase(
  "blend_modes_board",
  "Os seis modos do ImageBlend com cores intermediárias, conferidos contra as fórmulas " +
    "do ComfyUI, e o blend que redimensiona a segunda imagem.",
  {},
  (w) => {
    const [first, second] = [solid(w, 32, 32, 0xcc6633), solid(w, 32, 32, 0x66cc99)];
    const tiles: Link[] = [];
    for (const mode of BLEND_MODES) {
      const factor = mode === "normal" ? 0.25 : 1.0;
      const tile = w.addNode(ImageBlend, { image1: first, image2: second, blend_mode: mode, blend_factor: factor }).output
        .IMAGE;
      save(w, tile, `blend_${mode}`);
      tiles.push(tile);
    }
    let sheet = tiles[0]!;
    for (const tile of tiles.slice(1)) {
      sheet = stitch(w, sheet, tile);
    }
    save(w, sheet, "all_modes");
    const resized = w.addNode(ImageBlend, {
      image1: solid(w, 64, 64, RED),
      image2: solid(w, 8, 8, BLUE),
      blend_mode: "normal",
      blend_factor: 0.5,
    });
    save(w, resized.output.IMAGE, "resized");

    return (out) => {
      for (const mode of BLEND_MODES) {
        const factor = mode === "normal" ? 0.25 : 1.0;
        assertPixel(out.picture(`blend_${mode}`), 16, 16, blended(mode, 0xcc6633, 0x66cc99, factor));
      }
      assert.deepEqual(out.picture("all_modes").size, [192, 32]);
      const larger = out.picture("resized");
      assert.deepEqual(larger.size, [64, 64]);
      assertPixel(larger, 32, 32, [127, 0, 127]);
    };
  },
);

addCase(
  "brightness_contrast_normalize_invert_chain",
  "Cadeia AdjustBrightness -> AdjustContrast -> NormalizeImages -> ImageInvert com " +
    "cada etapa salva e conferida contra a conta feita em Python.",
  {},
  (w) => {
    const base = solid(w, 16, 16, 0x663399);
    const brighter = w.addNode(AdjustBrightness, { images: base, factor: 1.5 }).output.images;
    const contrast = w.addNode(AdjustContrast, { images: brighter, factor: 2.0 }).output.images;
    const normalized = w.addNode(NormalizeImages, { images: contrast, mean: 0.2, std: 0.5 });
    const inverted = w.addNode(ImageInvert, { image: normalized.output.images });
    save(w, brighter, "brighter");
    save(w, contrast, "contrast");
    save(w, normalized.output.images, "normalized");
    save(w, inverted.output.IMAGE, "inverted");

    return (out) => {
      const clip = (value: number) => Math.min(Math.max(value, 0), 1);
      let values = rgb(0x663399).map((channel) => channel / 255);
      values = values.map((value) => clip(value * 1.5));
      assertPixel(out.picture("brighter"), 8, 8, pixelOf(values));
      values = values.map((value) => clip((value - 0.5) * 2 + 0.5));
      assertPixel(out.picture("contrast"), 8, 8, pixelOf(values));
      values = values.map((value) => (value - 0.2) / 0.5);
      assertPixel(out.picture("normalized"), 8, 8, pixelOf(values));
      assertPixel(out.picture("inverted"), 8, 8, pixelOf(values.map((value) => 1 - value)));
    };
  },
);

const SOLID_FILTERS: Readonly<Record<string, (w: Workflow, image: Link) => Link>> = {
  blur: (w, image) => w.addNode(ImageBlur, { image, blur_radius: 3, sigma: 2.0 }).output.IMAGE,
  sharpen: (w, image) => w.addNode(ImageSharpen, { image, sharpen_radius: 2, sigma: 1.0, alpha: 0.5 }).output.IMAGE,
  erode: (w, image) => w.addNode(Morphology, { image, operation: "erode", kernel_size: 5 }).output.IMAGE,
  dilate: (w, image) => w.addNode(Morphology, { image, operation: "dilate", kernel_size: 5 }).output.IMAGE,
  open: (w, image) => w.addNode(Morphology, { image, operation: "open", kernel_size: 5 }).output.IMAGE,
  close: (w, image) => w.addNode(Morphology, { image, operation: "close", kernel_size: 5 }).output.IMAGE,
  quantize: (w, image) => w.addNode(ImageQuantize, { image, colors: 8, dither: "none" }).output.IMAGE,
  no_noise: (w, image) => w.addNode(ImageAddNoise, { image, seed: 1, strength: 0.0 }).output.IMAGE,
  same_space: (w, image) => w.addNode(ImageColorSpace, { image, source: "sRGB", destination: "sRGB" }).output.IMAGE,
  brightness_1: (w, image) => w.addNode(AdjustBrightness, { images: image, factor: 1.0 }).output.images,
  contrast_1: (w, image) => w.addNode(AdjustContrast, { images: image, factor: 1.0 }).output.images,
  scale_1: (w, image) => scaleBy(w, image, "bicubic", 1.0),
};

addCase(
  "filters_keep_solid_colors",
  "Doze filtros (blur, sharpen, morfologia, quantização, ruído zero, etc.) aplicados a " +
    "uma cor sólida: nenhum pode alterar a cor, nem nas bordas.",
  {},
  (w) => {
    const base = solid(w, 48, 48, 0x3366cc);
    for (const [label, apply] of Object.entries(SOLID_FILTERS)) {
      save(w, apply(w, base), label);
    }

    return (out) => {
      for (const label of Object.keys(SOLID_FILTERS)) {
        const picture = out.picture(label);
        assert.deepEqual(picture.size, [48, 48], label);
        for (const [x, y] of [
          [24, 24],
          [0, 0],
          [47, 47],
        ] as const) {
          assertPixel(picture, x, y, 0x3366cc);
        }
      }
    };
  },
);

addCase(
  "edge_filters_cancel_on_solid",
  "Canny e as morfologias gradient/top_hat/bottom_hat sobre cor sólida: sem bordas, " +
    "o resultado tem de ser preto.",
  {},
  (w) => {
    const base = solid(w, 48, 48, 0x3366cc);
    const canny = w.addNode(Canny, { image: base, low_threshold: 0.4, high_threshold: 0.8 });
    save(w, canny.output.IMAGE, "canny");
    for (const operation of ["gradient", "top_hat", "bottom_hat"] as const) {
      const node = w.addNode(Morphology, { image: base, operation, kernel_size: 3 });
      save(w, node.output.IMAGE, operation);
    }

    return (out) => {
      for (const label of ["canny", "gradient", "top_hat", "bottom_hat"]) {
        const picture = out.picture(label);
        for (const [x, y] of [
          [24, 24],
          [1, 1],
          [46, 46],
        ] as const) {
          assertPixel(picture, x, y, BLACK);
        }
      }
    };
  },
);

addCase(
  "seeded_image_noise",
  "ImageAddNoise com a mesma semente duas vezes e uma semente diferente: mesmas " +
    "sementes dão imagens idênticas, sementes diferentes não.",
  {},
  (w) => {
    const base = solid(w, 32, 32, GRAY);
    for (const [label, seed] of [
      ["seed_7", 7],
      ["seed_7_again", 7],
      ["seed_8", 8],
    ] as const) {
      const noisy = w.addNode(ImageAddNoise, { image: base, seed, strength: 0.3 });
      save(w, noisy.output.IMAGE, label);
    }

    return (out) => {
      const first = out.picture("seed_7");
      const again = out.picture("seed_7_again");
      const other = out.picture("seed_8");
      assert(sameRows(first, again), "the same seed gave two different images");
      assert(!sameRows(first, other), "two seeds gave the same image");
      assert(colorsOf(first).size > 50);
    };
  },
);

addCase(
  "quantize_gradient_with_dithers",
  "Degradê criado escalando um 2x1 preto/branco com bilinear, quantizado em 2 cores " +
    "sem dithering, com Floyd-Steinberg e com Bayer.",
  {},
  (w) => {
    const ramp = scale(w, stitch(w, solid(w, 1, 1, BLACK), solid(w, 1, 1, WHITE)), "bilinear", 64, 16);
    save(w, ramp, "ramp");
    const dithers = ["none", "floyd-steinberg", "bayer-4"] as const;
    for (const dither of dithers) {
      const quantized = w.addNode(ImageQuantize, { image: ramp, colors: 2, dither });
      save(w, quantized.output.IMAGE, `quantized_${dither}`);
    }

    return (out) => {
      const gradient = out.picture("ramp");
      assert.deepEqual(gradient.size, [64, 16]);
      const [left, middle, right] = [gradient.pixel(4, 8)[0]!, gradient.pixel(32, 8)[0]!, gradient.pixel(60, 8)[0]!];
      assert(left < middle && middle < right);
      for (const dither of dithers) {
        const picture = out.picture(`quantized_${dither}`);
        const colors = colorsOf(picture);
        assert(colors.size <= 2, `${dither} produced ${colors.size} colors`);
        if (dither === "none") {
          assert(picture.pixel(2, 8)[0]! < 128 && 128 < picture.pixel(61, 8)[0]!);
        } else {
          const dark = new Set([...picture.pixels([24, 0, 40, 16])].map((value) => value[0]! < 128));
          assert.equal(dark.size, 2, `${dither} did not mix the two colors`);
        }
      }
    };
  },
);

addCase(
  "yuv_round_trip_and_luma",
  "ImageRGBToYUV e ImageYUVToRGB: o Y/Cb/Cr de vermelho, verde, azul e branco bate com " +
    "a matriz BT.601 e a volta para RGB recupera os quadrantes.",
  {},
  (w) => {
    const image = quadrants(w, 32);
    const yuv = w.addNode(ImageRGBToYUV, { image });
    const back = w.addNode(ImageYUVToRGB, { Y: yuv.output.Y, U: yuv.output.U, V: yuv.output.V });
    for (const channel of ["Y", "U", "V"] as const) {
      save(w, yuv.output[channel], `channel_${channel}`);
    }
    save(w, back.output.IMAGE, "back");

    return (out) => {
      assertQuadrants(out.picture("back"), QUADRANTS, 3);
      for (const channel of ["Y", "U", "V"] as const) {
        const picture = out.picture(`channel_${channel}`);
        quarterPoints(32, 32).forEach(([x, y], index) => {
          const [r, g, b] = rgb(QUADRANTS[index]!).map((value) => value / 255) as [number, number, number];
          const luma = 0.299 * r + 0.587 * g + 0.114 * b;
          const value = { Y: luma, U: (b - luma) * 0.564 + 0.5, V: (r - luma) * 0.713 + 0.5 }[channel];
          assertPixel(picture, x, y, gray(value), 3);
        });
      }
    };
  },
);

function srgbToLinear(value: number): number {
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

addCase(
  "color_space_round_trip",
  "ImageColorSpace sRGB -> linear -> sRGB em quatro cores: o cinza linear escurece " +
    "como a curva sRGB manda e a volta recupera as cores.",
  {},
  (w) => {
    const colors = [GRAY, 0x40c080, 0xc04020, 0x204060];
    const image = quadrants(w, 32, colors);
    const linear = w.addNode(ImageColorSpace, { image, source: "sRGB", destination: "linear" });
    const back = w.addNode(ImageColorSpace, { image: linear.output.IMAGE, source: "linear", destination: "sRGB" });
    save(w, linear.output.IMAGE, "linear");
    save(w, back.output.IMAGE, "back");

    return (out) => {
      assertQuadrants(out.picture("back"), colors, 3);
      assertPixel(out.picture("linear"), 8, 8, gray(srgbToLinear(128 / 255)), 3);
    };
  },
);

addCase(
  "canny_finds_quadrant_borders",
  "Canny num tabuleiro preto/branco: bordas brancas exatamente nas linhas que separam " +
    "os quadrantes e preto longe delas.",
  {},
  (w) => {
    const boardImage = quadrants(w, 64, [BLACK, WHITE, WHITE, BLACK]);
    const edges = w.addNode(Canny, { image: boardImage, low_threshold: 0.1, high_threshold: 0.3 });
    save(w, edges.output.IMAGE, "edges");

    return (out) => {
      const picture = out.picture("edges");
      assert.deepEqual(picture.size, [64, 64]);
      for (const [x, y] of quarterPoints(64, 64)) {
        assertPixel(picture, x, y, BLACK);
      }
      // Kornia keeps the whole vertical border, but where the four quadrants
      // meet it drops half of the horizontal one, so only part of it is required.
      const bright = (x: number, y: number) => picture.pixel(x, y)[0]! > 128;
      const near = [29, 30, 31, 32, 33, 34];
      const all = Array.from({ length: 64 }, (_, index) => index);
      const rows = all.filter((y) => near.some((x) => bright(x, y))).length;
      const columns = all.filter((x) => near.some((y) => bright(x, y))).length;
      assert(rows >= 58, `the vertical border shows up in ${rows} of 64 rows`);
      assert(columns >= 16, `the horizontal border shows up in ${columns} of 64 columns`);
    };
  },
);

addCase(
  "morphology_grows_and_shrinks_square",
  "Morphology dilate/erode/open/close num quadrado branco 8x8: cresce e encolhe " +
    "exatamente 1 pixel por lado, abrir e fechar preservam o quadrado.",
  {},
  (w) => {
    const square = paste(w, solid(w, 32, 32, BLACK), solid(w, 8, 8, WHITE), 12, 12);
    for (const operation of ["dilate", "erode", "open", "close"] as const) {
      const node = w.addNode(Morphology, { image: square, operation, kernel_size: 3 });
      save(w, node.output.IMAGE, operation);
    }

    return (out) => {
      const expected: Readonly<Record<string, readonly (readonly [number, number, Color])[]>> = {
        dilate: [
          [11, 11, WHITE],
          [20, 20, WHITE],
          [10, 10, BLACK],
          [21, 21, BLACK],
        ],
        erode: [
          [13, 13, WHITE],
          [18, 18, WHITE],
          [12, 12, BLACK],
          [19, 19, BLACK],
        ],
        open: [
          [12, 12, WHITE],
          [19, 19, WHITE],
          [11, 11, BLACK],
        ],
        close: [
          [12, 12, WHITE],
          [19, 19, WHITE],
          [20, 20, BLACK],
        ],
      };
      for (const [operation, points] of Object.entries(expected)) {
        const picture = out.picture(operation);
        for (const [x, y, color] of points) {
          assertPixel(picture, x, y, color);
        }
      }
    };
  },
);

const PORTER_DUFF = {
  SRC_OVER: [[127, 0, 127], 1.0],
  DST_OVER: [[0, 0, 255], 1.0],
  SRC_IN: [[255, 0, 0], 0.5],
  XOR: [[0, 0, 255], 0.5],
  MULTIPLY: [[0, 0, 127], 1.0],
  SCREEN: [[127, 0, 255], 1.0],
  CLEAR: [[0, 0, 0], 0.0],
} as const;

addCase(
  "porter_duff_modes",
  "PorterDuffImageComposite em sete modos com fonte vermelha 50% e destino azul opaco; " +
    "imagem e máscara (1 - alfa) resultantes conferidas com as equações Porter-Duff.",
  {},
  (w) => {
    const [source, destination] = [solid(w, 32, 32, RED), solid(w, 32, 32, BLUE)];
    const [half, opaque] = [solidMask(w, 0.5, 32, 32), solidMask(w, 0.0, 32, 32)];
    for (const mode of Object.keys(PORTER_DUFF) as (keyof typeof PORTER_DUFF)[]) {
      const node = w.addNode(PorterDuffImageComposite, {
        source,
        source_alpha: half,
        destination,
        destination_alpha: opaque,
        mode,
      });
      save(w, node.output.IMAGE, `image_${mode}`);
      save(w, maskImage(w, node.output.MASK), `mask_${mode}`);
    }

    return (out) => {
      // Like its mask inputs, the node's MASK output is 1 - alpha.
      for (const [mode, [color, alpha]] of Object.entries(PORTER_DUFF)) {
        assertPixel(out.picture(`image_${mode}`), 16, 16, color);
        assertPixel(out.picture(`mask_${mode}`), 16, 16, gray(1 - alpha));
      }
    };
  },
);

addCase(
  "alpha_join_and_split",
  "JoinImageWithAlpha gera PNG RGBA, ImageInvert preserva o alfa e SplitImageWithAlpha " +
    "devolve a máscara invertida (1 - alfa).",
  {},
  (w) => {
    const joined = w.addNode(JoinImageWithAlpha, { image: solid(w, 32, 32, RED), alpha: solidMask(w, 0.25, 32, 32) });
    const inverted = w.addNode(ImageInvert, { image: joined.output.IMAGE });
    const split = w.addNode(SplitImageWithAlpha, { image: joined.output.IMAGE });
    save(w, joined.output.IMAGE, "rgba");
    save(w, inverted.output.IMAGE, "inverted");
    save(w, split.output.IMAGE, "rgb");
    save(w, maskImage(w, split.output.MASK), "mask");

    return (out) => {
      const rgba = out.picture("rgba");
      assert.equal(rgba.channels, 4);
      assertPixel(rgba, 5, 5, RED);
      assert(Math.abs(rgba.alpha(5, 5) - 191) <= 1);
      const negative = out.picture("inverted");
      assertPixel(negative, 5, 5, CYAN);
      assert(Math.abs(negative.alpha(5, 5) - 191) <= 1);
      const rgbPart = out.picture("rgb");
      assert.equal(rgbPart.channels, 3);
      assertPixel(rgbPart, 5, 5, RED);
      assertPixel(out.picture("mask"), 5, 5, gray(0.25));
    };
  },
);
