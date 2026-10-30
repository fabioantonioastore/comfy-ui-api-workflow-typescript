// The text, math and logic cases of the Python test_runpod_workflows.py.
import assert from "node:assert/strict";

import type { Link } from "../../src/nodes/base.js";
import { GetImageSize } from "../../src/nodes/image.js";
import { RepeatImageBatch } from "../../src/nodes/image/batch.js";
import { ImageAddNoise } from "../../src/nodes/image/filters.js";
import {
  CaseConverter,
  JsonExtractString,
  RegexExtract,
  RegexMatch,
  RegexReplace,
  StringCompare,
  StringConcatenate,
  StringContains,
  StringReplace,
  StringSubstring,
  StringTrim,
  TextOverlay,
  type CaseConverter_Modes,
  type RegexExtract_Modes,
  type StringCompare_Modes,
  type StringTrim_Modes,
  type TextOverlay_Aligns,
  type TextOverlay_Positions,
} from "../../src/nodes/text.js";
import {
  ColorToRGBInt,
  ComfyNumberConvert,
  ResolutionSelector,
  SeedNode,
  type ResolutionSelector_Aspect_Ratios,
} from "../../src/nodes/utilities.js";
import { ComfySwitchNode } from "../../src/nodes/utilities/logic.js";
import {
  PrimitiveBoolean,
  PrimitiveFloat,
  PrimitiveInt,
  PrimitiveString,
  PrimitiveStringMultiline,
} from "../../src/nodes/utilities/primitive.js";
import { board, calc, crop, equals, length, negate, save, scale, scaleBy, solid } from "./blocks.js";
import {
  assertPixel,
  BLACK,
  BLUE,
  colorsOf,
  CYAN,
  GRAY,
  GREEN,
  hues,
  ORANGE,
  RED,
  rgb,
  roundHalfEven,
  tuple,
  WHITE,
  YELLOW,
  type Box,
  type Picture,
  type Size,
} from "./media.js";
import { addCase } from "./registry.js";

/** Whether every pixel of the box is the color. */
function only(picture: Picture, box: Box, color: number): boolean {
  const colors = colorsOf(picture, box);
  return colors.size === 1 && colors.has(tuple(rgb(color)));
}

addCase(
  "string_basics_board",
  "Quadro de 14 verificações de texto (concatenar, substring com índice negativo, " +
    "maiúsculas/minúsculas, trim, replace, unicode) feitas no ComfyUI.",
  {},
  (w) => {
    const substring = (text: string, start: number, end: number) =>
      w.addNode(StringSubstring, { string: text, start, end }).output.STRING;
    const caseOf = (text: string, mode: CaseConverter_Modes) => w.addNode(CaseConverter, { string: text, mode }).output.STRING;
    const trim = (text: string, mode: StringTrim_Modes) => w.addNode(StringTrim, { string: text, mode }).output.STRING;
    const concatenate = (first: string | Link, second: string | Link, delimiter = "") =>
      w.addNode(StringConcatenate, { string_a: first, string_b: second, delimiter }).output.STRING;

    const chained = concatenate(trim("  ab ", "Both"), caseOf("cd", "UPPERCASE"));
    const replaced = w.addNode(StringReplace, { string: "a-b-c", find: "-", replace: "+" });
    const checks = {
      concatenate: equals(w, concatenate("comfy", "ui", "-"), "comfy-ui"),
      substring: equals(w, substring("workflow", 4, 8), "flow"),
      negative_start: equals(w, substring("workflow", -4, 8), "flow"),
      upper: equals(w, caseOf("hello world", "UPPERCASE"), "HELLO WORLD"),
      lower: equals(w, caseOf("HeLLo", "lowercase"), "hello"),
      capitalize: equals(w, caseOf("hello world", "Capitalize"), "Hello world"),
      title: equals(w, caseOf("hello world", "Title Case"), "Hello World"),
      trim_both: equals(w, trim("  pad  ", "Both"), "pad"),
      trim_left: equals(w, trim("  pad  ", "Left"), "pad  "),
      trim_right: equals(w, trim("  pad  ", "Right"), "  pad"),
      replace: equals(w, replaced.output.STRING, "a+b+c"),
      unicode_length: calc(w, "a == 4", { a: length(w, "ação") }).BOOL,
      chained: equals(w, chained, "abCD"),
      chained_length: calc(w, "a == 4", { a: length(w, chained) }).BOOL,
    };
    return board(w, checks);
  },
);

addCase(
  "string_compare_contains_board",
  "Quadro de comparações: StringCompare (começa/termina/igual, com e sem caixa), " +
    "StringContains e a veracidade Python do ComfyNotNode.",
  {},
  (w) => {
    const compare = (first: string, second: string, mode: StringCompare_Modes, sensitive = true) =>
      w.addNode(StringCompare, { string_a: first, string_b: second, mode, case_sensitive: sensitive }).output.BOOLEAN;
    const contains = (text: string, part: string, sensitive = true) =>
      w.addNode(StringContains, { string: text, substring: part, case_sensitive: sensitive }).output.contains;

    const empty = w.addNode(StringSubstring, { string: "abc", start: 0, end: 0 }).output.STRING;
    const zero = w.addNode(PrimitiveInt, { value: 0 }).output.INT;
    const half = w.addNode(PrimitiveFloat, { value: 0.5 }).output.FLOAT;
    const checks = {
      starts_any_case: compare("ComfyUI", "comfy", "Starts With", false),
      starts_exact_case: negate(w, compare("ComfyUI", "comfy", "Starts With")),
      ends: compare("pipeline.json", ".json", "Ends With"),
      equal_any_case: compare("abc", "ABC", "Equal", false),
      equal_exact_case: negate(w, compare("abc", "ABC", "Equal")),
      contains: contains("stable diffusion xl", "diffusion"),
      contains_exact_case: negate(w, contains("Stable", "stable")),
      contains_any_case: contains("Stable", "stable", false),
      empty_is_falsy: negate(w, empty),
      zero_is_falsy: negate(w, zero),
      half_is_truthy: negate(w, negate(w, half)),
    };
    return board(w, checks);
  },
);

interface RegexFlags {
  readonly case_insensitive?: boolean;
  readonly multiline?: boolean;
  readonly dotall?: boolean;
}

addCase(
  "regex_board",
  "Quadro de regex: RegexMatch com flags (ignorecase, multiline, dotall), RegexExtract " +
    "nos quatro modos, RegexReplace com grupos e padrão inválido.",
  {},
  (w) => {
    const match = (text: string, regex: string, flags: RegexFlags = {}) =>
      w.addNode(RegexMatch, { string: text, regex_pattern: regex, ...flags }).output.matches;
    const extract = (text: string, regex: string, mode: RegexExtract_Modes, options: { group_index?: number } = {}) =>
      w.addNode(RegexExtract, { string: text, regex_pattern: regex, mode, ...options }).output.STRING;
    const replace = (text: string, regex: string, replacement: string, flags: RegexFlags = {}) =>
      w.addNode(RegexReplace, { string: text, regex_pattern: regex, replace: replacement, ...flags }).output.STRING;

    const invalid = extract("anything", "(", "First Match");
    const checks = {
      match: match("Order #1234 shipped", String.raw`#\d+`),
      no_match: negate(w, match("abc", String.raw`\d`)),
      ignore_case: match("HELLO", "hello", { case_insensitive: true }),
      exact_case: negate(w, match("HELLO", "hello", { case_insensitive: false })),
      multiline: match("line1\nline2", "^line2$", { multiline: true }),
      single_line: negate(w, match("line1\nline2", "^line2$", { multiline: false })),
      dotall: match("a\nb", "a.b", { dotall: true }),
      no_dotall: negate(w, match("a\nb", "a.b", { dotall: false })),
      first_match: equals(w, extract("size=24px; scale=2x", String.raw`\d+`, "First Match"), "24"),
      all_matches: equals(w, extract("size=24px; scale=2x", String.raw`\d+`, "All Matches"), "24\n2"),
      first_group: equals(w, extract("key=value", String.raw`(\w+)=(\w+)`, "First Group", { group_index: 2 }), "value"),
      all_groups: equals(w, extract("a=1 b=2 c=3", String.raw`(\w)=(\d)`, "All Groups", { group_index: 1 }), "a\nb\nc"),
      tuple_matches: equals(w, extract("a=1 b=2", String.raw`(\w)=(\d)`, "All Matches"), "a\nb"),
      replace_groups: equals(w, replace("2026-10-16", String.raw`(\d+)-(\d+)-(\d+)`, String.raw`\3/\2/\1`), "16/10/2026"),
      replace_case: equals(w, replace("aAaA", "a", "b", { case_insensitive: false }), "bAbA"),
      invalid_pattern: calc(w, "a == 0", { a: length(w, invalid) }).BOOL,
    };
    return board(w, checks);
  },
);

addCase(
  "json_and_number_conversion_board",
  "Quadro de JSON e números: JsonExtractString (texto solto, bloco Markdown, chave " +
    "ausente) e ComfyNumberConvert de texto, float negativo e booleano.",
  {},
  (w) => {
    const fromJson = (text: string, key: string) => w.addNode(JsonExtractString, { json_string: text, key }).output.STRING;
    const convert = (value: Link) => w.addNode(ComfyNumberConvert, { value }).output;

    const width = fromJson('noise {"w": 40} trailing', "w");
    const decimal = convert(w.addNode(PrimitiveString, { value: " 2.75 " }).output.STRING);
    const flag = convert(w.addNode(PrimitiveBoolean, { value: true }).output.BOOLEAN);
    const negative = convert(w.addNode(PrimitiveFloat, { value: -3.9 }).output.FLOAT);
    const seven = convert(w.addNode(PrimitiveInt, { value: 7 }).output.INT);
    const checks = {
      name: equals(w, fromJson('{"w": 40, "name": "comfy"}', "name"), "comfy"),
      loose_text: equals(w, width, "40"),
      markdown: equals(w, fromJson('```json\n{"steps": 30}\n```', "steps"), "30"),
      missing_key: equals(w, fromJson('{"a": 1}', "missing"), ""),
      text_to_int: calc(w, "a == 40", { a: convert(width).INT }).BOOL,
      decimal_float: calc(w, "a == 2.75", { a: decimal.FLOAT }).BOOL,
      decimal_int: calc(w, "a == 2", { a: decimal.INT }).BOOL,
      bool_to_float: calc(w, "a == 1", { a: flag.FLOAT }).BOOL,
      negative_truncates: calc(w, "a == -3", { a: negative.INT }).BOOL,
      int_to_float: calc(w, "a == 7.0", { a: seven.FLOAT }).BOOL,
    };
    return board(w, checks);
  },
);

addCase(
  "math_expression_board",
  "Quadro de ComfyMathExpression: sum/min/max/abs/round/pow/sqrt/ceil/floor/log2/int, " +
    "ternário, módulo e Pitágoras; saídas INT e FLOAT viram tamanhos de imagem.",
  {},
  (w) => {
    const truth = w.addNode(PrimitiveBoolean, { value: true }).output.BOOLEAN;
    const checks = {
      sum: calc(w, "sum(values) == 10", { a: 1, b: 2, c: 3, d: 4 }).BOOL,
      min: calc(w, "min(a, b, c) == -2", { a: 5, b: -2, c: 3 }).BOOL,
      max: calc(w, "max(values) == 9.5", { a: 1, b: 9.5, c: 3 }).BOOL,
      abs: calc(w, "abs(a) == 7", { a: -7 }).BOOL,
      round: calc(w, "round(a, 1) == 3.1", { a: 3.14159 }).BOOL,
      pow: calc(w, "pow(a, b) == 1024", { a: 2, b: 10 }).BOOL,
      sqrt: calc(w, "sqrt(a) == 12", { a: 144 }).BOOL,
      ceil_floor: calc(w, "ceil(a) == 3 and floor(a) == 2", { a: 2.5 }).BOOL,
      log2: calc(w, "log2(a) == 8", { a: 256 }).BOOL,
      int: calc(w, "int(a / b) == 3", { a: 10, b: 3 }).BOOL,
      ternary: calc(w, "(a if b else c) == 42", { a: 42, b: truth, c: 0 }).BOOL,
      modulo: calc(w, "a % b == 1", { a: 10, b: 3 }).BOOL,
      pythagoras: calc(w, "a ** 2 + b ** 2 == c ** 2", { a: 3, b: 4, c: 5 }).BOOL,
    };
    const checkBoard = board(w, checks);
    const width = calc(w, "a * 2.6", { a: 10 }).INT;
    const factor = calc(w, "a / 4", { a: 2 }).FLOAT;
    save(w, solid(w, width, 8, ORANGE), "from_int");
    save(w, scaleBy(w, solid(w, 64, 64, ORANGE), "nearest-exact", factor), "from_float");

    return (out) => {
      checkBoard(out);
      assert.deepEqual(out.picture("from_int").size, [26, 8]);
      assert.deepEqual(out.picture("from_float").size, [32, 32]);
    };
  },
);

addCase(
  "switch_selects_lazily",
  "ComfySwitchNode é preguiçoso: o ramo não escolhido divide por zero e mesmo assim o " +
    "job passa; switches escolhem tamanhos e imagens, com ComfyNotNode.",
  {},
  (w) => {
    const broken = calc(w, "a / b", { a: 1, b: 0 }).INT;
    const fortyEight = w.addNode(PrimitiveInt, { value: 48 }).output.INT;
    const thirtyTwo = w.addNode(PrimitiveInt, { value: 32 }).output.INT;
    const off = w.addNode(ComfySwitchNode, { switch: false, on_true: broken, on_false: fortyEight });
    const on = w.addNode(ComfySwitchNode, { switch: true, on_true: thirtyTwo, on_false: broken });
    save(w, solid(w, off.output.output, on.output.output, GREEN), "sized");
    const found = w.addNode(StringContains, { string: "stable diffusion", substring: "diffusion" });
    const picked = w.addNode(ComfySwitchNode, {
      switch: found.output.contains,
      on_true: solid(w, 8, 8, GREEN),
      on_false: solid(w, 8, 8, RED),
    });
    save(w, picked.output.output, "picked");
    const flipped = negate(w, w.addNode(PrimitiveBoolean, { value: true }).output.BOOLEAN);
    const other = w.addNode(ComfySwitchNode, {
      switch: flipped,
      on_true: solid(w, 8, 8, RED),
      on_false: solid(w, 8, 8, BLUE),
    });
    save(w, other.output.output, "negated");

    return (out) => {
      const sized = out.picture("sized");
      assert.deepEqual(sized.size, [48, 32]);
      assertPixel(sized, 10, 10, GREEN);
      assertPixel(out.picture("picked"), 4, 4, GREEN);
      assertPixel(out.picture("negated"), 4, 4, BLUE);
    };
  },
);

addCase(
  "math_drives_image_size",
  "GetImageSize alimenta ComfyMathExpression e EmptyImage: largura, altura e batch " +
    "derivados no servidor, inclusive ImageScale com altura 0 (proporcional).",
  {},
  (w) => {
    const base = solid(w, 40, 24, ORANGE, 3);
    const size = w.addNode(GetImageSize, { image: base }).output;
    const width = calc(w, "a * 2 + 8", { a: size.width }).INT;
    const height = calc(w, "a * 3", { a: size.height }).INT;
    save(w, solid(w, width, height, CYAN, size.batch_size), "canvas");
    save(w, scale(w, base, "bilinear", width, 0), "proportional");

    return (out) => {
      const canvases = out.pictures("canvas");
      assert.equal(canvases.length, 3);
      assert(canvases.every((canvas) => canvas.size[0] === 88 && canvas.size[1] === 72));
      const proportional = out.pictures("proportional");
      assert.equal(proportional.length, 3);
      assert.deepEqual(proportional[0]!.size, [88, 53]);
    };
  },
);

const ASPECT_RATIOS: Readonly<Record<string, Size>> = {
  "1:1 (Square)": [1, 1],
  "16:9 (Widescreen)": [16, 9],
  "2:3 (Portrait Photo)": [2, 3],
  "9:16 (Portrait Widescreen)": [9, 16],
  "4:3 (Standard)": [4, 3],
};
const RESOLUTIONS: readonly (readonly [ResolutionSelector_Aspect_Ratios, number, number])[] = [
  ["1:1 (Square)", 0.1, 8],
  ["16:9 (Widescreen)", 0.25, 16],
  ["2:3 (Portrait Photo)", 0.2, 32],
  ["9:16 (Portrait Widescreen)", 0.15, 8],
  ["4:3 (Standard)", 0.3, 64],
];

function resolution(ratio: string, megapixels: number, multiple: number): Size {
  const [widthRatio, heightRatio] = ASPECT_RATIOS[ratio]!;
  const factor = Math.sqrt((megapixels * 1024 * 1024) / (widthRatio * heightRatio));
  return [roundHalfEven((widthRatio * factor) / multiple) * multiple, roundHalfEven((heightRatio * factor) / multiple) * multiple];
}

addCase(
  "resolution_selector_aspect_ratios",
  "ResolutionSelector em cinco proporções, megapixels e múltiplos: cada EmptyImage tem " +
    "exatamente o tamanho que a fórmula do ComfyUI produz.",
  {},
  (w) => {
    RESOLUTIONS.forEach(([ratio, megapixels, multiple], index) => {
      const size = w.addNode(ResolutionSelector, { aspect_ratio: ratio, megapixels, multiple }).output;
      save(w, solid(w, size.width, size.height, hues(5)[index]!), `ratio_${index}`);
    });

    return (out) => {
      RESOLUTIONS.forEach(([ratio, megapixels, multiple], index) => {
        const picture = out.picture(`ratio_${index}`);
        assert.deepEqual(picture.size, resolution(ratio, megapixels, multiple), ratio);
        assert(picture.width % multiple === 0 && picture.height % multiple === 0);
      });
    };
  },
);

addCase(
  "color_to_rgb_int_paints",
  "ColorToRGBInt: o inteiro pinta um EmptyImage, o alfa de #RRGGBBAA vira largura via " +
    "matemática e o hex colore o texto do TextOverlay.",
  {},
  (w) => {
    const spring = w.addNode(ColorToRGBInt, { color: "#00ff7f" }).output;
    save(w, solid(w, 16, 16, spring.rgb_int), "painted");
    const steel = w.addNode(ColorToRGBInt, { color: "#336699cc" }).output;
    save(w, solid(w, calc(w, "round(a * 100)", { a: steel.alpha }).INT, 8, WHITE), "alpha_width");
    const text = w.addNode(TextOverlay, {
      images: solid(w, 128, 64, WHITE),
      text: "Cor",
      font_size: 40.0,
      color: steel.hex,
      position: "top",
      align: "left",
      outline: false,
    });
    save(w, text.output.images, "text");

    return (out) => {
      assertPixel(out.picture("painted"), 8, 8, [0, 255, 127], 0);
      assert.deepEqual(out.picture("alpha_width").size, [80, 8]);
      const target = rgb(0x336699);
      const shows = [...out.picture("text").pixels()].some((value) =>
        value.every((channel, index) => Math.abs(channel - target[index]!) <= 12),
      );
      assert(shows, "the text color never shows up");
    };
  },
);

addCase(
  "primitives_feed_everything",
  "Todos os primitivos (Int, Float, Boolean, String, StringMultiline, BoundingBox, " +
    "Seed) alimentam um mesmo fluxo de imagem com tamanho calculado.",
  {},
  (w) => {
    const width = w.addNode(PrimitiveInt, { value: 40 }).output.INT;
    const factor = w.addNode(PrimitiveFloat, { value: 1.5 }).output.FLOAT;
    const useBlue = w.addNode(PrimitiveBoolean, { value: true }).output.BOOLEAN;
    const word = w.addNode(PrimitiveString, { value: "comfy" }).output.STRING;
    const lines = w.addNode(PrimitiveStringMultiline, { value: "a\nb\nc" }).output.STRING;
    const height = calc(w, "a * b", { a: length(w, word), b: length(w, lines) }).INT;
    const color = w.addNode(ComfySwitchNode, {
      switch: useBlue,
      on_true: w.addNode(PrimitiveInt, { value: BLUE }).output.INT,
      on_false: w.addNode(PrimitiveInt, { value: RED }).output.INT,
    }).output.output;
    const base = solid(w, width, height, color);
    const bigger = scaleBy(w, base, "nearest-exact", factor);
    const cropped = crop(w, bigger, 10, 5, 20, 10);
    const seed = w.addNode(SeedNode, { seed: 1234 }).output.seed;
    const noisy = w.addNode(ImageAddNoise, { image: cropped, seed, strength: 0.0 });
    save(w, base, "base");
    save(w, bigger, "bigger");
    save(w, noisy.output.IMAGE, "cropped");

    return (out) => {
      assert.deepEqual(out.picture("base").size, [40, 25]);
      assert.deepEqual(out.picture("bigger").size, [60, 38]);
      const piece = out.picture("cropped");
      assert.deepEqual(piece.size, [20, 10]);
      assertPixel(piece, 10, 5, BLUE);
    };
  },
);

addCase(
  "text_overlay_variants",
  "TextOverlay no topo/esquerda e embaixo/direita com e sem contorno: o texto aparece " +
    "só na faixa esperada e o resto da imagem fica intacto.",
  {},
  (w) => {
    const overlay = (background: Link, position: TextOverlay_Positions, align: TextOverlay_Aligns, outline: boolean) =>
      w.addNode(TextOverlay, {
        images: background,
        text: "ComfyUI",
        font_size: 20.0,
        color: "#ffffff",
        position,
        align,
        outline,
      }).output.images;

    save(w, overlay(solid(w, 256, 128, BLACK), "top", "left", false), "top_left");
    save(w, overlay(solid(w, 256, 128, GRAY), "bottom", "right", true), "bottom_right");

    return (out) => {
      const top = out.picture("top_left");
      assert.deepEqual(top.size, [256, 128]);
      assert([...top.pixels([0, 0, 128, 64])].some((value) => value[0]! > 200));
      assert(only(top, [0, 100, 256, 128], BLACK));
      const bottom = out.picture("bottom_right");
      assert([...bottom.pixels([128, 64, 256, 128])].some((value) => value[0]! > 200));
      assert(only(bottom, [0, 0, 128, 40], GRAY));
    };
  },
);

addCase(
  "string_length_drives_batch",
  "StringLength vira batch_size do EmptyImage e amount do RepeatImageBatch: o número de " +
    "arquivos devolvidos prova a conta.",
  {},
  (w) => {
    const frames = solid(w, 16, 16, YELLOW, length(w, "abc"));
    const repeated = w.addNode(RepeatImageBatch, { image: frames, amount: length(w, "ab") });
    save(w, frames, "frames");
    save(w, repeated.output.IMAGE, "repeated");

    return (out) => {
      assert.equal(out.of("frames").length, 3);
      assert.equal(out.of("repeated").length, 6);
    };
  },
);

addCase(
  "string_ops_build_filename_prefix",
  "O filename_prefix do SaveImage vem de StringConcatenate + CaseConverter + " +
    "StringReplace calculados no servidor: o nome do arquivo prova o resultado.",
  {},
  (w) => {
    const lower = w.addNode(CaseConverter, { string: "GALERIA", mode: "lowercase" }).output.STRING;
    const snake = w.addNode(StringReplace, { string: "caso 01", find: " ", replace: "_" }).output.STRING;
    const prefix = w.addNode(StringConcatenate, { string_a: lower, string_b: snake, delimiter: "-" });
    save(w, solid(w, 16, 16, ORANGE), prefix.output.STRING);

    return (out) => {
      assert.equal(out.of("galeria-caso_01").length, 1, out.names().join(", "));
      assertPixel(out.picture("galeria-caso_01"), 8, 8, ORANGE);
    };
  },
);
