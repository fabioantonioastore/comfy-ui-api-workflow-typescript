// The media part of the Python test_runpod_workflows.py: colors, and readers for
// the PNG, WebP, MP4 and Matroska files the RunPod jobs return.
import assert from "node:assert/strict";
import { deflateSync, inflateSync } from "node:zlib";

export const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
export const MATROSKA = Buffer.from([0x1a, 0x45, 0xdf, 0xa3]);

export const RED = 0xff0000;
export const GREEN = 0x00ff00;
export const BLUE = 0x0000ff;
export const YELLOW = 0xffff00;
export const CYAN = 0x00ffff;
export const MAGENTA = 0xff00ff;
export const WHITE = 0xffffff;
export const BLACK = 0x000000;
export const GRAY = 0x808080;
export const ORANGE = 0xff8000;
export const QUADRANTS = [RED, GREEN, BLUE, WHITE] as const;
export const SIX = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA] as const;
export const ROTATIONS = ["90 degrees", "180 degrees", "270 degrees"] as const;
export const ROTATION_LABELS = { "90 degrees": "rot90", "180 degrees": "rot180", "270 degrees": "rot270" } as const;
export const FLIPS = ["x-axis: vertically", "y-axis: horizontally"] as const;
export const FLIP_LABELS = { "x-axis: vertically": "flip_x", "y-axis: horizontally": "flip_y" } as const;

/** A color as 0xRRGGBB, or as its channel values. */
export type Color = number | readonly number[];
/** The pixel at (x, y) of an image being drawn, as its channel values. */
export type PixelSource = (x: number, y: number) => readonly number[];
/** left, top, right and bottom, the last two excluded. */
export type Box = readonly [number, number, number, number];
export type Size = readonly [number, number];

/** Python's round(): a half goes to the even neighbour. */
export function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction !== 0.5) {
    return fraction < 0.5 ? floor : floor + 1;
  }
  return floor % 2 === 0 ? floor : floor + 1;
}

/** How Python prints a tuple of numbers, so messages read as the Python suite's. */
export function tuple(values: readonly number[]): string {
  return `(${values.join(", ")})`;
}

export function rgb(color: Color): number[] {
  if (typeof color !== "number") {
    return [...color];
  }
  return [(color >> 16) & 0xff, (color >> 8) & 0xff, color & 0xff];
}

// colorsys.hsv_to_rgb, step by step, so the colors come out as in Python.
function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  if (s === 0) {
    return [v, v, v];
  }
  const i = Math.trunc(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - s * f);
  const t = v * (1 - s * (1 - f));
  switch (i % 6) {
    case 0:
      return [v, t, p];
    case 1:
      return [q, v, p];
    case 2:
      return [p, v, t];
    case 3:
      return [p, q, v];
    case 4:
      return [t, p, v];
    default:
      return [v, p, q];
  }
}

export function hues(count: number): number[] {
  return Array.from({ length: count }, (_, index) => {
    const [r, g, b] = hsvToRgb(index / count, 1, 1).map((value) => roundHalfEven(value * 255)) as [
      number,
      number,
      number,
    ];
    return (r << 16) | (g << 8) | b;
  });
}

export function pixelOf(values: readonly number[]): number[] {
  // SaveImage clips 255 * value and truncates it to uint8.
  return values.map((value) => Math.trunc(255 * Math.min(Math.max(value, 0), 1)));
}

export function gray(value: number): number[] {
  return pixelOf([value, value, value]);
}

export class Picture {
  constructor(
    readonly width: number,
    readonly height: number,
    readonly channels: number,
    readonly rows: readonly Buffer[],
  ) {}

  get size(): [number, number] {
    return [this.width, this.height];
  }

  pixel(x: number, y: number): number[] {
    const start = x * this.channels;
    const values = this.#row(y).subarray(start, start + this.channels);
    assert(values.length > 0, `(${x}, ${y}) is outside the ${this.width}x${this.height} image`);
    return this.channels >= 3 ? [values[0]!, values[1]!, values[2]!] : [values[0]!, values[0]!, values[0]!];
  }

  alpha(x: number, y: number): number {
    return this.#row(y)[x * this.channels + 3]!;
  }

  *pixels(box?: Box): Generator<number[]> {
    const [left, top, right, bottom] = box ?? [0, 0, this.width, this.height];
    for (let y = top; y < bottom; y++) {
      for (let x = left; x < right; x++) {
        yield this.pixel(x, y);
      }
    }
  }

  mean(box?: Box): [number, number, number] {
    const totals: [number, number, number] = [0, 0, 0];
    let count = 0;
    for (const value of this.pixels(box)) {
      count += 1;
      for (let channel = 0; channel < 3; channel++) {
        totals[channel]! += value[channel]!;
      }
    }
    return [totals[0] / count, totals[1] / count, totals[2] / count];
  }

  mirrored(): Picture {
    const step = this.channels;
    const rows = this.rows.map((row) => {
      const pixels = [];
      for (let x = this.width - 1; x >= 0; x--) {
        pixels.push(row.subarray(x * step, (x + 1) * step));
      }
      return Buffer.concat(pixels);
    });
    return new Picture(this.width, this.height, this.channels, rows);
  }

  #row(y: number): Buffer {
    // Python indexes from the end with a negative y.
    const row = this.rows[y < 0 ? this.rows.length + y : y];
    assert(row !== undefined, `row ${y} is outside the ${this.width}x${this.height} image`);
    return row;
  }
}

const CRC_TABLE = Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit++) {
    value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  }
  return value >>> 0;
});

/** zlib's CRC-32, which PNG chunks end with. */
export function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of data) {
    crc = CRC_TABLE[(crc ^ byte) & 0xff]! ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function* pngChunks(data: Buffer): Generator<[string, Buffer]> {
  assert(data.subarray(0, 8).equals(PNG_SIGNATURE), "not a PNG file");
  let position = 8;
  while (position < data.length) {
    const length = data.readUInt32BE(position);
    yield [data.toString("latin1", position + 4, position + 8), data.subarray(position + 8, position + 8 + length)];
    position += 12 + length;
  }
}

export function pngSize(data: Buffer): [number, number] {
  assert(data.subarray(0, 8).equals(PNG_SIGNATURE), "not a PNG file");
  return [data.readUInt32BE(16), data.readUInt32BE(20)];
}

export function apngFrames(data: Buffer): number {
  for (const [kind, body] of pngChunks(data)) {
    if (kind === "acTL") {
      return body.readUInt32BE(0);
    }
  }
  return 1;
}

function unfilter(kind: number, line: Buffer, previous: Buffer, step: number): void {
  if (kind === 1) {
    for (let i = step; i < line.length; i++) {
      line[i] = (line[i]! + line[i - step]!) & 0xff;
    }
  } else if (kind === 2) {
    for (let i = 0; i < line.length; i++) {
      line[i] = (line[i]! + previous[i]!) & 0xff;
    }
  } else if (kind === 3) {
    for (let i = 0; i < line.length; i++) {
      const left = i >= step ? line[i - step]! : 0;
      line[i] = (line[i]! + ((left + previous[i]!) >> 1)) & 0xff;
    }
  } else if (kind === 4) {
    for (let i = 0; i < line.length; i++) {
      const a = i >= step ? line[i - step]! : 0;
      const b = previous[i]!;
      const c = i >= step ? previous[i - step]! : 0;
      const p = a + b - c;
      const [pa, pb, pc] = [Math.abs(p - a), Math.abs(p - b), Math.abs(p - c)];
      const predictor = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      line[i] = (line[i]! + predictor) & 0xff;
    }
  }
}

const PNG_CHANNELS: Readonly<Record<number, number>> = { 0: 1, 2: 3, 4: 2, 6: 4 };

export function decodePng(data: Buffer): Picture {
  let header: Buffer = Buffer.alloc(0);
  const idat: Buffer[] = [];
  for (const [kind, body] of pngChunks(data)) {
    if (kind === "IHDR") {
      header = body;
    } else if (kind === "IDAT") {
      idat.push(body);
    }
  }
  const [width, height] = [header.readUInt32BE(0), header.readUInt32BE(4)];
  const [depth, colorType, interlace] = [header[8]!, header[9]!, header[12]!];
  assert(depth === 8 && interlace === 0, `unsupported PNG: depth ${depth}, interlace ${interlace}`);
  const channels = PNG_CHANNELS[colorType];
  assert(channels !== undefined, `unsupported PNG: color type ${colorType}`);
  const stride = width * channels;
  const raw = inflateSync(Buffer.concat(idat));
  const rows: Buffer[] = [];
  let previous: Buffer = Buffer.alloc(stride);
  for (let y = 0; y < height; y++) {
    const offset = y * (stride + 1);
    const line = Buffer.from(raw.subarray(offset + 1, offset + 1 + stride));
    unfilter(raw[offset]!, line, previous, channels);
    rows.push(line);
    previous = line;
  }
  return new Picture(width, height, channels, rows);
}

function chunk(kind: string, body: Buffer): Buffer {
  const head = Buffer.alloc(8);
  head.writeUInt32BE(body.length);
  head.write(kind, 4, "latin1");
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), body])));
  return Buffer.concat([head, body, crc]);
}

export function encodePng(width: number, height: number, pixel: PixelSource, alpha = false): Buffer {
  const rows: number[] = [];
  for (let y = 0; y < height; y++) {
    rows.push(0);
    for (let x = 0; x < width; x++) {
      rows.push(...pixel(x, y));
    }
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header.set([8, alpha ? 6 : 2, 0, 0, 0], 8);
  return Buffer.concat([
    PNG_SIGNATURE,
    chunk("IHDR", header),
    chunk("IDAT", deflateSync(Buffer.from(rows))),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

/** The size of a WebP file, if it says, and how many frames it has. */
export function webpInfo(data: Buffer): [Size | null, number] {
  assert(data.toString("latin1", 0, 4) === "RIFF" && data.toString("latin1", 8, 12) === "WEBP", "not a WEBP file");
  let size: Size | null = null;
  let frames = 0;
  let position = 12;
  while (position + 8 <= data.length) {
    const kind = data.toString("latin1", position, position + 4);
    const length = data.readUInt32LE(position + 4);
    const body = data.subarray(position + 8, position + 8 + length);
    if (kind === "VP8X") {
      size = [1 + body.readUIntLE(4, 3), 1 + body.readUIntLE(7, 3)];
    } else if (kind === "ANMF") {
      frames += 1;
    }
    position += 8 + length + (length & 1);
  }
  return [size, Math.max(frames, 1)];
}

export function assertMp4(data: Buffer, ...codecs: string[]): void {
  assert(data.toString("latin1", 4, 8) === "ftyp", "not an MP4 file");
  for (const codec of codecs) {
    assert(data.includes(codec, 0, "latin1"), `'${codec}' is not in the MP4 file`);
  }
}

export function assertMatroska(data: Buffer, ...codecs: string[]): void {
  assert(data.subarray(0, 4).equals(MATROSKA), "not a Matroska/WebM file");
  for (const codec of codecs) {
    assert(data.includes(codec, 0, "latin1"), `'${codec}' is not in the Matroska file`);
  }
}

function filesOf(label: string): RegExp {
  // The files of a save node: <filename_prefix>_<counter>_.<extension>.
  return new RegExp(`^${label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}_\\d+_\\.\\w+$`);
}

/** The files a job returned, found by the filename_prefix of each save node. */
export class Outputs {
  readonly files: readonly (readonly [string, Buffer])[];

  constructor(files: readonly (readonly [string, Buffer])[]) {
    this.files = [...files].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  }

  names(): string[] {
    return this.files.map(([name]) => name);
  }

  of(label: string): Buffer[] {
    const pattern = filesOf(label);
    return this.files.filter(([name]) => pattern.test(name)).map(([, data]) => data);
  }

  one(label: string): Buffer {
    const found = this.of(label);
    assert(found.length === 1, `expected one '${label}' file, got [${this.names().map((name) => `'${name}'`).join(", ")}]`);
    return found[0]!;
  }

  name(label: string): string {
    const pattern = filesOf(label);
    const names = this.names().filter((name) => pattern.test(name));
    assert(names.length === 1, `expected one '${label}' file, got [${this.names().map((name) => `'${name}'`).join(", ")}]`);
    return names[0]!;
  }

  picture(label: string): Picture {
    return decodePng(this.one(label));
  }

  pictures(label: string): Picture[] {
    return this.of(label).map(decodePng);
  }
}

export function assertPixel(picture: Picture, x: number, y: number, expected: Color, tolerance = 2): void {
  const actual = picture.pixel(x, y);
  const want = rgb(expected);
  assert(
    actual.every((value, index) => index >= want.length || Math.abs(value - want[index]!) <= tolerance),
    `pixel (${x}, ${y}) of the ${picture.width}x${picture.height} image is ${tuple(actual)}, expected ${tuple(want)}`,
  );
}

export function quarterPoints(width: number, height: number): [number, number][] {
  const [left, right] = [Math.floor(width / 4), Math.floor((3 * width) / 4)];
  const [top, bottom] = [Math.floor(height / 4), Math.floor((3 * height) / 4)];
  return [
    [left, top],
    [right, top],
    [left, bottom],
    [right, bottom],
  ];
}

export function quadrantColor(
  x: number,
  y: number,
  width: number,
  height: number,
  colors: readonly Color[] = QUADRANTS,
): Color {
  return colors[(y >= Math.floor(height / 2) ? 2 : 0) + (x >= Math.floor(width / 2) ? 1 : 0)]!;
}

export function assertQuadrants(picture: Picture, colors: readonly Color[] = QUADRANTS, tolerance = 2): void {
  quarterPoints(...picture.size).forEach(([x, y], index) => {
    if (index < colors.length) {
      assertPixel(picture, x, y, colors[index]!, tolerance);
    }
  });
}

export function assertGenerated(data: Buffer, size: Size, minimum = 40_000): void {
  const actual = pngSize(data);
  assert(actual[0] === size[0] && actual[1] === size[1], `the image is ${tuple(actual)}, expected ${tuple(size)}`);
  assert(data.length > minimum, `the ${size[0]}x${size[1]} image has only ${data.length} bytes, so it is nearly flat`);
}

/** The distinct colors of the image, or of a box of it, each as a tuple. */
export function colorsOf(picture: Picture, box?: Box): Set<string> {
  return new Set([...picture.pixels(box)].map(tuple));
}

/** Whether the two images have the same pixels, byte for byte. */
export function sameRows(first: Picture, second: Picture): boolean {
  return first.rows.length === second.rows.length && first.rows.every((row, index) => row.equals(second.rows[index]!));
}

export function assertSameRows(actual: Picture, expected: Picture): void {
  assert(
    sameRows(actual, expected),
    `the ${actual.width}x${actual.height} image is not the ${expected.width}x${expected.height} one`,
  );
}

export function meanDifference(first: Picture, second: Picture): number {
  assert.deepEqual(first.size, second.size, `${tuple(first.size)} != ${tuple(second.size)}`);
  let total = 0;
  const others = second.pixels();
  for (const a of first.pixels()) {
    const b = others.next().value as number[];
    total += a.reduce((sum, value, index) => sum + Math.abs(value - b[index]!), 0);
  }
  return total / (first.width * first.height * 3);
}

export function nearest<C extends Color>(value: readonly number[], palette: readonly C[], tolerance = 3): C {
  for (const candidate of palette) {
    const channels = rgb(candidate);
    if (value.every((a, index) => index >= channels.length || Math.abs(a - channels[index]!) <= tolerance)) {
      return candidate;
    }
  }
  assert.fail(`${tuple(value)} is none of [${palette.map((color) => tuple(rgb(color))).join(", ")}]`);
}

export function rotatedSource(rotation: string, x: number, y: number, width: number, height: number): [number, number] {
  // (width, height) is the size before the rotation; ImageRotate turns clockwise.
  if (rotation === "90 degrees") {
    return [y, height - 1 - x];
  }
  if (rotation === "180 degrees") {
    return [width - 1 - x, height - 1 - y];
  }
  if (rotation === "270 degrees") {
    return [width - 1 - y, x];
  }
  return [x, y];
}

export function flippedSource(method: string, x: number, y: number, width: number, height: number): [number, number] {
  if (method.startsWith("x")) {
    return [x, height - 1 - y];
  }
  return [width - 1 - x, y];
}

export function pattern(x: number, y: number): [number, number, number] {
  return [(x * 10) % 256, (y * 15) % 256, (x * y * 3 + 40) % 256];
}

export const PATTERN_PNG = encodePng(24, 16, pattern);
export const RGBA_PNG = encodePng(32, 32, (x) => [200, 100, 50, x < 16 ? 255 : 64], true);
export const STRIPE_PNG = encodePng(24, 16, (x) => [x < 12 ? 255 : 0, 0, 0]);
