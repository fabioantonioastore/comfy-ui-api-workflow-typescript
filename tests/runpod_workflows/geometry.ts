// The geometry cases of the Python test_runpod_workflows.py.
import assert from "node:assert/strict";

import type { Link } from "../../src/nodes/base.js";
import { ImageFromBatch, ImageGrid, ImageMergeTileList, RepeatImageBatch, ShuffleDataset } from "../../src/nodes/image/batch.js";
import {
  CenterCropImages,
  ImagePadForOutpaint,
  RandomCropImages,
  ResizeAndPadImage,
  type ResizeAndPadImage_Interpolations,
  type ResizeAndPadImage_Padding_Colors,
} from "../../src/nodes/image/transform.js";
import { ImageScaleToMaxDimension, ImageScaleToTotalPixels } from "../../src/nodes/image/upscaling.js";
import {
  batchOf,
  combineMasks,
  crop,
  flip,
  maskImage,
  paste,
  quadrants,
  rotate,
  save,
  scale,
  scaleBy,
  solid,
  solidMask,
  stitch,
  strip,
  tilesOf,
} from "./blocks.js";
import {
  assertPixel,
  assertQuadrants,
  assertSameRows,
  BLACK,
  BLUE,
  CYAN,
  FLIP_LABELS,
  flippedSource,
  FLIPS,
  gray,
  GRAY,
  GREEN,
  MAGENTA,
  nearest,
  ORANGE,
  quadrantColor,
  QUADRANTS,
  quarterPoints,
  RED,
  rgb,
  ROTATION_LABELS,
  rotatedSource,
  ROTATIONS,
  roundHalfEven,
  SIX,
  type Size,
  WHITE,
  YELLOW,
  type Outputs,
} from "./media.js";
import { addCase } from "./registry.js";

addCase(
  "quadrant_rotations_and_flips",
  "Quatro quadrantes montados com ImageCompositeMasked, girados em 90/180/270 graus " +
    "e espelhados nos dois eixos; cada cor tem de cair no quadrante certo.",
  {},
  (w) => {
    const image = quadrants(w, 64);
    save(w, image, "original");
    for (const rotation of ROTATIONS) {
      save(w, rotate(w, image, rotation), ROTATION_LABELS[rotation]);
    }
    for (const method of FLIPS) {
      save(w, flip(w, image, method), FLIP_LABELS[method]);
    }
    save(w, flip(w, rotate(w, image, "90 degrees"), FLIPS[0]), "rot90_flip_x");

    return (out) => {
      const expect = (label: string, source: (x: number, y: number) => [number, number]) => {
        const picture = out.picture(label);
        assert.deepEqual(picture.size, [64, 64]);
        for (const [x, y] of quarterPoints(64, 64)) {
          assertPixel(picture, x, y, quadrantColor(...source(x, y), 64, 64));
        }
      };

      expect("original", (x, y) => [x, y]);
      for (const rotation of ROTATIONS) {
        expect(ROTATION_LABELS[rotation], (x, y) => rotatedSource(rotation, x, y, 64, 64));
      }
      for (const method of FLIPS) {
        expect(FLIP_LABELS[method], (x, y) => flippedSource(method, x, y, 64, 64));
      }
      expect("rot90_flip_x", (x, y) => rotatedSource("90 degrees", ...flippedSource(FLIPS[0], x, y, 64, 64), 64, 64));
    };
  },
);

addCase(
  "stitch_in_every_direction",
  "ImageStitch nas quatro direções, com ajuste de tamanho, espaçamento colorido e o " +
    "preenchimento que o ComfyUI faz quando as alturas/larguras não batem.",
  {},
  (w) => {
    const right = stitch(w, solid(w, 32, 32, RED), solid(w, 32, 32, GREEN), "right", true, 4);
    const down = stitch(w, right, solid(w, 32, 16, BLUE), "down", true);
    const left = stitch(w, down, solid(w, 16, 16, YELLOW), "left");
    const up = stitch(w, left, solid(w, 20, 10, MAGENTA), "up", false, 2, "red");
    for (const [label, image] of [
      ["right", right],
      ["down", down],
      ["left", left],
      ["up", up],
    ] as const) {
      save(w, image, label);
    }

    return (out) => {
      assert.deepEqual(out.picture("right").size, [68, 32]);
      assert.deepEqual(out.picture("down").size, [68, 66]);
      assert.deepEqual(out.picture("left").size, [84, 66]);
      const final = out.picture("up");
      assert.deepEqual(final.size, [84, 78]);
      const expected = [
        [42, 5, MAGENTA],
        [5, 5, BLACK],
        [70, 5, BLACK],
        [40, 10, RED],
        [40, 11, RED],
        [8, 45, YELLOW],
        [8, 17, WHITE],
        [8, 72, WHITE],
        [26, 22, RED],
        [49, 22, WHITE],
        [66, 22, GREEN],
        [46, 62, BLUE],
      ] as const;
      for (const [x, y, color] of expected) {
        assertPixel(final, x, y, color);
      }
    };
  },
);

function outpaintFeather(i: number, j: number, feathering = 8, height = 32, width = 32, top = 8, bottom = 24, left = 16, right = 0) {
  // Mirrors ImagePadForOutpaint's mask for pixel (row i, column j) of the original.
  const distance = Math.min(top ? i : height, bottom ? height - i : height, left ? j : width, right ? width - j : width);
  return distance >= feathering ? 0.0 : ((feathering - distance) / feathering) ** 2;
}

addCase(
  "outpaint_canvas_and_mask",
  "ImagePadForOutpaint com e sem feathering: borda nova cinza 50%, original no lugar " +
    "e a máscara com o degradê quadrático do ComfyUI.",
  {},
  (w) => {
    const image = solid(w, 32, 32, RED);
    const padding = { left: 16, top: 8, right: 0, bottom: 24 };
    const plain = w.addNode(ImagePadForOutpaint, { image, feathering: 0, ...padding });
    const soft = w.addNode(ImagePadForOutpaint, { image, feathering: 8, ...padding });
    save(w, plain.output.IMAGE, "canvas");
    save(w, maskImage(w, plain.output.MASK), "mask");
    save(w, maskImage(w, soft.output.MASK), "feathered");

    return (out) => {
      const canvas = out.picture("canvas");
      assert.deepEqual(canvas.size, [48, 64]);
      assertPixel(canvas, 5, 5, gray(0.5));
      assertPixel(canvas, 21, 13, RED);
      assertPixel(canvas, 30, 50, gray(0.5));
      const mask = out.picture("mask");
      assertPixel(mask, 5, 5, WHITE);
      assertPixel(mask, 21, 13, BLACK);
      const feathered = out.picture("feathered");
      for (const [i, j] of [
        [0, 10],
        [4, 10],
        [16, 16],
        [16, 2],
        [31, 20],
      ] as const) {
        assertPixel(feathered, 16 + j, 8 + i, gray(outpaintFeather(i, j)));
      }
    };
  },
);

addCase(
  "crop_chain_back_to_quadrants",
  "Recorte com ImageCropV2 + PrimitiveBoundingBox, CenterCropImages e RandomCropImages; " +
    "ampliado de volta com nearest-exact, tem de ser idêntico ao original.",
  {},
  (w) => {
    const image = quadrants(w, 64);
    const cropped = crop(w, image, 16, 16, 32, 32);
    const centered = w.addNode(CenterCropImages, { images: cropped, width: 16, height: 16 });
    const untouched = w.addNode(RandomCropImages, { images: centered.output.images, width: 64, height: 64, seed: 5 });
    const restored = scale(w, untouched.output.images, "nearest-exact", 64, 64);
    save(w, image, "original");
    save(w, cropped, "cropped");
    save(w, centered.output.images, "centered");
    save(w, restored, "restored");

    return (out) => {
      for (const [label, size] of [
        ["cropped", 32],
        ["centered", 16],
        ["restored", 64],
      ] as const) {
        const picture = out.picture(label);
        assert.deepEqual(picture.size, [size, size]);
        assertQuadrants(picture);
      }
      assertSameRows(out.picture("restored"), out.picture("original"));
    };
  },
);

function totalPixelsSize(width: number, height: number, megapixels: number, steps: number): Size {
  const factor = Math.sqrt((megapixels * 1024 * 1024) / (width * height));
  return [roundHalfEven((width * factor) / steps) * steps, roundHalfEven((height * factor) / steps) * steps];
}

addCase(
  "scale_methods_keep_halves",
  "Imagem metade vermelha/metade azul redimensionada com todos os métodos e nós de " +
    "escala; tamanhos calculados como o ComfyUI calcula e as metades preservadas.",
  {},
  (w) => {
    const halves = stitch(w, solid(w, 20, 20, RED), solid(w, 20, 20, BLUE));
    const methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"] as const;
    for (const method of methods) {
      save(w, scale(w, halves, method, 80, 40), `scale_${method}`);
    }
    save(w, scale(w, halves, "bilinear", 0, 30), "keep_aspect");
    save(w, scale(w, halves, "bilinear", 20, 20, "center"), "center_crop");
    save(w, scaleBy(w, halves, "area", 0.5), "half");
    save(w, scaleBy(w, halves, "lanczos", 1.75), "bigger");
    const largest = w.addNode(ImageScaleToMaxDimension, { image: halves, upscale_method: "lanczos", largest_size: 30 });
    save(w, largest.output.IMAGE, "max_dimension");
    const total = w.addNode(ImageScaleToTotalPixels, {
      image: halves,
      upscale_method: "bicubic",
      megapixels: 0.01,
      resolution_steps: 8,
    });
    save(w, total.output.IMAGE, "total_pixels");

    return (out) => {
      const sizes = new Map<string, Size>(methods.map((method) => [`scale_${method}`, [80, 40]]));
      sizes.set("keep_aspect", [60, 30]);
      sizes.set("center_crop", [20, 20]);
      sizes.set("half", [20, 10]);
      sizes.set("bigger", [70, 35]);
      sizes.set("max_dimension", [30, 15]);
      sizes.set("total_pixels", totalPixelsSize(40, 20, 0.01, 8));
      for (const [label, size] of sizes) {
        const picture = out.picture(label);
        assert.deepEqual(picture.size, size, `${label} is (${picture.size.join(", ")}), expected (${size.join(", ")})`);
        const middle = Math.floor(size[1] / 2);
        assertPixel(picture, Math.floor((size[0] * 15) / 100), middle, RED, 3);
        assertPixel(picture, Math.floor((size[0] * 85) / 100), middle, BLUE, 3);
      }
    };
  },
);

addCase(
  "tile_split_and_merge",
  "SplitImageToTileList corta 128x128 em 9 tiles com sobreposição e " +
    "ImageMergeTileList remonta a imagem; tiles e mosaico conferidos.",
  {},
  (w) => {
    const tiles = tilesOf(w, quadrants(w, 128), 64, 16);
    save(w, tiles, "tile");
    const merged = w.addNode(ImageMergeTileList, { image_list: tiles, final_width: 128, final_height: 128, overlap: 16 });
    save(w, merged.output.IMAGE, "merged");

    return (out) => {
      const pieces = out.pictures("tile");
      assert.equal(pieces.length, 9);
      assert(pieces.every((tile) => tile.size[0] === 64 && tile.size[1] === 64));
      for (const [index, color] of [
        [0, RED],
        [2, GREEN],
        [6, BLUE],
        [8, WHITE],
      ] as const) {
        assertPixel(pieces[index]!, 32, 32, color);
      }
      assertPixel(pieces[4]!, 4, 4, RED);
      assertPixel(pieces[4]!, 60, 60, WHITE);
      const whole = out.picture("merged");
      assert.deepEqual(whole.size, [128, 128]);
      assertQuadrants(whole);
    };
  },
);

const byValue = (a: number, b: number) => a - b;

addCase(
  "grid_collage_and_shuffle",
  "Seis cores viram uma lista de tiles, um ImageGrid 3x2 com espaçamento preto e um " +
    "ShuffleDataset embaralhado (mesmas cores em outra ordem).",
  {},
  (w) => {
    const tiles = tilesOf(w, strip(w, SIX, 64), 64);
    const grid = w.addNode(ImageGrid, { images: tiles, columns: 3, cell_width: 32, cell_height: 32, padding: 4 });
    const shuffled = w.addNode(ShuffleDataset, { images: tiles, seed: 3 });
    const row = w.addNode(ImageGrid, {
      images: shuffled.output.images,
      columns: 6,
      cell_width: 32,
      cell_height: 32,
      padding: 0,
    });
    save(w, tiles, "tile");
    save(w, grid.output.images, "grid");
    save(w, row.output.images, "shuffled");

    return (out) => {
      assert.deepEqual(
        out.pictures("tile").map((tile) => tile.pixel(32, 32)),
        SIX.map((color) => rgb(color)),
      );
      const collage = out.picture("grid");
      assert.deepEqual(collage.size, [104, 68]);
      SIX.forEach((color, index) => {
        assertPixel(collage, 16 + 36 * (index % 3), 16 + 36 * Math.floor(index / 3), color);
      });
      assertPixel(collage, 33, 10, BLACK);
      assertPixel(collage, 10, 33, BLACK);
      const line = out.picture("shuffled");
      assert.deepEqual(line.size, [192, 32]);
      const found = Array.from({ length: 6 }, (_, index) => nearest(line.pixel(16 + 32 * index, 16), SIX));
      assert.deepEqual(found.sort(byValue), [...SIX].sort(byValue));
    };
  },
);

addCase(
  "rebatch_pick_and_repeat",
  "Lista de seis cores reagrupada com RebatchImages; ImageFromBatch com índice positivo " +
    "e negativo e RepeatImageBatch; a ordem das cores tem de ser exata.",
  {},
  (w) => {
    const frames = batchOf(w, SIX);
    const picked = w.addNode(ImageFromBatch, { image: frames, batch_index: 2, length: 3 });
    const repeated = w.addNode(RepeatImageBatch, { image: picked.output.IMAGE, amount: 2 });
    const last = w.addNode(ImageFromBatch, { image: frames, batch_index: -1, length: 1 });
    save(w, picked.output.IMAGE, "picked");
    save(w, repeated.output.IMAGE, "repeated");
    save(w, last.output.IMAGE, "last");

    return (out) => {
      const colors = (label: string) => out.pictures(label).map((picture) => picture.pixel(10, 10));
      const middle = SIX.slice(2, 5);

      assert.deepEqual(colors("picked"), middle.map((color) => rgb(color)));
      assert.deepEqual(colors("repeated"), [...middle, ...middle].map((color) => rgb(color)));
      assert.deepEqual(colors("last"), [rgb(MAGENTA)]);
    };
  },
);

addCase(
  "masked_composite_with_offsets",
  "ImageCompositeMasked com máscara recortada por MaskComposite, com origem saindo da " +
    "borda e com resize_source.",
  {},
  (w) => {
    const [background, patch] = [solid(w, 64, 64, BLUE), solid(w, 32, 32, YELLOW)];
    const leftHalf = combineMasks(w, solidMask(w, 1.0, 32, 32), solidMask(w, 1.0, 16, 32), "subtract", 16);
    save(w, paste(w, background, patch, 16, 16, leftHalf), "masked");
    save(w, paste(w, background, patch, 48, 48), "corner");
    save(w, paste(w, background, patch, 0, 0, leftHalf, true), "resized");

    return (out) => {
      const masked = out.picture("masked");
      assertPixel(masked, 20, 30, YELLOW);
      assertPixel(masked, 40, 30, BLUE);
      assertPixel(masked, 8, 8, BLUE);
      const corner = out.picture("corner");
      assertPixel(corner, 56, 56, YELLOW);
      assertPixel(corner, 40, 56, BLUE);
      const resized = out.picture("resized");
      assertPixel(resized, 8, 32, YELLOW);
      assertPixel(resized, 56, 32, BLUE);
    };
  },
);

addCase(
  "letterbox_portrait_and_landscape",
  "ResizeAndPadImage em paisagem, retrato e quadrantes, com preenchimento preto e " +
    "branco, e o resultado costurado lado a lado.",
  {},
  (w) => {
    const letterbox = (
      image: Link,
      width: number,
      height: number,
      color: ResizeAndPadImage_Padding_Colors,
      method: ResizeAndPadImage_Interpolations,
    ) =>
      w.addNode(ResizeAndPadImage, {
        image,
        target_width: width,
        target_height: height,
        padding_color: color,
        interpolation: method,
      }).output.IMAGE;

    const landscape = letterbox(solid(w, 64, 32, RED), 48, 48, "black", "nearest-exact");
    const portrait = letterbox(solid(w, 32, 64, GREEN), 48, 48, "white", "bilinear");
    const squeezed = letterbox(quadrants(w, 64), 40, 24, "black", "area");
    save(w, landscape, "landscape");
    save(w, portrait, "portrait");
    save(w, stitch(w, landscape, portrait), "both");
    save(w, squeezed, "squeezed");

    return (out) => {
      const wide = out.picture("landscape");
      for (const [x, y, color] of [
        [24, 6, BLACK],
        [24, 24, RED],
        [24, 41, BLACK],
      ] as const) {
        assertPixel(wide, x, y, color);
      }
      const tall = out.picture("portrait");
      for (const [x, y, color] of [
        [6, 24, WHITE],
        [24, 24, GREEN],
        [41, 24, WHITE],
      ] as const) {
        assertPixel(tall, x, y, color);
      }
      const both = out.picture("both");
      assert.deepEqual(both.size, [96, 48]);
      for (const [x, y, color] of [
        [24, 24, RED],
        [72, 24, GREEN],
        [54, 24, WHITE],
        [24, 4, BLACK],
      ] as const) {
        assertPixel(both, x, y, color);
      }
      const small = out.picture("squeezed");
      assert.deepEqual(small.size, [40, 24]);
      assertPixel(small, 2, 12, BLACK);
      assertPixel(small, 37, 12, BLACK);
      const points = [
        [14, 6],
        [26, 6],
        [14, 18],
        [26, 18],
      ] as const;
      points.forEach(([x, y], index) => assertPixel(small, x, y, QUADRANTS[index]!, 3));
    };
  },
);

addCase(
  "non_square_rotations",
  "Rotações e espelhamentos de uma imagem 96x48 (não quadrada): as dimensões trocam e " +
    "cada pixel amostrado vem exatamente da posição esperada.",
  {},
  (w) => {
    const halves = stitch(w, solid(w, 48, 48, RED), solid(w, 48, 48, BLUE));
    for (const rotation of ROTATIONS) {
      save(w, rotate(w, halves, rotation), ROTATION_LABELS[rotation]);
    }
    for (const method of FLIPS) {
      save(w, flip(w, halves, method), FLIP_LABELS[method]);
    }

    return (out) => {
      const color = (sourceX: number) => (sourceX < 48 ? RED : BLUE);
      const expect = (label: string, size: Size, source: (x: number, y: number) => [number, number]) => {
        const picture = out.picture(label);
        assert.deepEqual(picture.size, size);
        for (let x = 3; x < size[0]; x += 9) {
          for (let y = 3; y < size[1]; y += 9) {
            assertPixel(picture, x, y, color(source(x, y)[0]), 0);
          }
        }
      };

      for (const rotation of ROTATIONS) {
        const size: Size = rotation === "180 degrees" ? [96, 48] : [48, 96];
        expect(ROTATION_LABELS[rotation], size, (x, y) => rotatedSource(rotation, x, y, 96, 48));
      }
      for (const method of FLIPS) {
        expect(FLIP_LABELS[method], [96, 48], (x, y) => flippedSource(method, x, y, 96, 48));
      }
    };
  },
);

addCase(
  "nearest_up_area_down_is_lossless",
  "Ampliação 4x com nearest-exact seguida de redução 4x com area: o resultado tem de " +
    "ser idêntico, pixel a pixel, à imagem de partida.",
  {},
  (w) => {
    const colors = [ORANGE, CYAN, MAGENTA, GRAY];
    const image = quadrants(w, 32, colors);
    const up = scaleBy(w, image, "nearest-exact", 4.0);
    const down = scaleBy(w, up, "area", 0.25);
    save(w, image, "original");
    save(w, up, "up");
    save(w, down, "down");

    return (out) => {
      const larger = out.picture("up");
      assert.deepEqual(larger.size, [128, 128]);
      assertQuadrants(larger, colors, 0);
      assertSameRows(out.picture("down"), out.picture("original"));
    };
  },
);

addCase(
  "outpaint_all_sides_then_crop_back",
  "Borda nova nos quatro lados com ImagePadForOutpaint e recorte de volta com " +
    "ImageCropV2: a área original volta intacta.",
  {},
  (w) => {
    const image = quadrants(w, 32);
    const padded = w.addNode(ImagePadForOutpaint, { image, left: 8, top: 16, right: 24, bottom: 8, feathering: 0 }).output
      .IMAGE;
    save(w, image, "original");
    save(w, padded, "padded");
    save(w, crop(w, padded, 8, 16, 32, 32), "back");

    return (out: Outputs) => {
      const larger = out.picture("padded");
      assert.deepEqual(larger.size, [64, 56]);
      assertPixel(larger, 2, 2, gray(0.5));
      assertPixel(larger, 60, 50, gray(0.5));
      assertPixel(larger, 16, 24, RED);
      assertPixel(larger, 32, 40, WHITE);
      assertSameRows(out.picture("back"), out.picture("original"));
    };
  },
);
