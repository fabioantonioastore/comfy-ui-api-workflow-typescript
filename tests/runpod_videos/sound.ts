// The sound of the Python test_runpod_videos.py: the short WAV files made here and
// sent with the jobs, and the measures of the sound that comes back.
import assert from "node:assert/strict";

import { roundHalfEven } from "../runpod_workflows/media.js";

// CPython's Mersenne Twister (Modules/_randommodule.c), so the noise is Python's.
const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

/** Python's random.Random, for the draws the suites make. */
export class PythonRandom {
  readonly #state = new Uint32Array(N);
  #index = N;

  constructor(seed: number) {
    assert(Number.isSafeInteger(seed) && seed >= 0, `the seed ${seed} is not a non-negative integer`);
    // random.seed() takes an integer's 32-bit words, the lowest first.
    const key: number[] = [];
    for (let rest = seed; rest > 0; rest = Math.floor(rest / 2 ** 32)) {
      key.push(rest % 2 ** 32);
    }
    this.#seed(key.length > 0 ? key : [0]);
  }

  /** A float in [0, 1) with 53 random bits. */
  random(): number {
    const high = this.#next() >>> 5;
    const low = this.#next() >>> 6;
    return (high * 67108864 + low) / 9007199254740992;
  }

  uniform(a: number, b: number): number {
    return a + (b - a) * this.random();
  }

  // init_by_array; the Uint32Array keeps each sum modulo 2**32.
  #seed(key: readonly number[]): void {
    const mt = this.#state;
    mt[0] = 19650218;
    for (let i = 1; i < N; i++) {
      mt[i] = Math.imul(1812433253, mt[i - 1]! ^ (mt[i - 1]! >>> 30)) + i;
    }
    let i = 1;
    let j = 0;
    for (let k = Math.max(N, key.length); k > 0; k--) {
      mt[i] = (mt[i]! ^ Math.imul(mt[i - 1]! ^ (mt[i - 1]! >>> 30), 1664525)) + key[j]! + j;
      i++;
      j++;
      if (i >= N) {
        mt[0] = mt[N - 1]!;
        i = 1;
      }
      if (j >= key.length) {
        j = 0;
      }
    }
    for (let k = N - 1; k > 0; k--) {
      mt[i] = (mt[i]! ^ Math.imul(mt[i - 1]! ^ (mt[i - 1]! >>> 30), 1566083941)) - i;
      i++;
      if (i >= N) {
        mt[0] = mt[N - 1]!;
        i = 1;
      }
    }
    mt[0] = 0x80000000;
  }

  // genrand_uint32
  #next(): number {
    const mt = this.#state;
    if (this.#index >= N) {
      for (let k = 0; k < N; k++) {
        const y = (mt[k]! & UPPER_MASK) | (mt[(k + 1) % N]! & LOWER_MASK);
        mt[k] = mt[(k + M) % N]! ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      this.#index = 0;
    }
    let y = mt[this.#index++]!;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
}

/** A 16-bit PCM WAV file, byte for byte as Python's wave module writes it. */
export function wav(rate: number, ...channels: (readonly number[])[]): Buffer {
  const frames = Math.min(...channels.map((channel) => channel.length));
  const data = Buffer.alloc(2 * frames * channels.length);
  let offset = 0;
  for (let frame = 0; frame < frames; frame++) {
    for (const channel of channels) {
      offset = data.writeInt16LE(roundHalfEven(Math.max(-1.0, Math.min(1.0, channel[frame]!)) * 32767), offset);
    }
  }
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "latin1");
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVEfmt ", 8, "latin1");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(channels.length, 22);
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * channels.length * 2, 28);
  header.writeUInt16LE(channels.length * 2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "latin1");
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}

export function sine(frequency: number, seconds: number, rate: number, amplitude: number): number[] {
  const step = (2 * Math.PI * frequency) / rate;
  return Array.from({ length: roundHalfEven(seconds * rate) }, (_, n) => amplitude * Math.sin(step * n));
}

export function chord(frequencies: readonly number[], seconds: number, rate: number, amplitude: number): number[] {
  const tones = frequencies.map((frequency) => sine(frequency, seconds, rate, amplitude));
  return tones[0]!.map((_, index) => tones.reduce((total, tone) => total + tone[index]!, 0));
}

export function hiss(seconds: number, rate: number, amplitude: number, seed: number): number[] {
  const generator = new PythonRandom(seed);
  return Array.from({ length: roundHalfEven(seconds * rate) }, () => amplitude * generator.uniform(-1.0, 1.0));
}

export function quiet(seconds: number, rate: number): number[] {
  return new Array<number>(roundHalfEven(seconds * rate)).fill(0.0);
}

export function faded(samples: readonly number[], rate: number, seconds = 0.005): number[] {
  // A short ramp at both ends, so the click starts and stops without a pop.
  const ramp = roundHalfEven(seconds * rate);
  return samples.map((value, index) => value * Math.min(1.0, (index + 1) / ramp, (samples.length - index) / ramp));
}

/** Sound as samples between -1 and 1, one list per channel. */
export class Sound {
  constructor(
    readonly rate: number,
    readonly channels: readonly (readonly number[])[],
  ) {}

  part(start: number, end: number, channel = 0): readonly number[] {
    return this.channels[channel]!.slice(roundHalfEven(start * this.rate), roundHalfEven(end * this.rate));
  }

  rms(start: number, end: number, channel = 0): number {
    const samples = this.part(start, end, channel);
    return Math.sqrt(samples.reduce((total, value) => total + value * value, 0) / samples.length);
  }

  amplitude(frequency: number, start: number, end: number, channel = 0): number {
    // Goertzel: the amplitude of the sine at this exact frequency.
    const samples = this.part(start, end, channel);
    const coefficient = 2 * Math.cos((2 * Math.PI * frequency) / this.rate);
    let previous = 0.0;
    let before = 0.0;
    for (const value of samples) {
      const next = value + coefficient * previous - before;
      before = previous;
      previous = next;
    }
    const power = previous * previous + before * before - coefficient * previous * before;
    return (2 * Math.sqrt(Math.max(power, 0.0))) / samples.length;
  }

  /** The loudest of the frequencies; the first of them on a tie, as Python's max() gives. */
  dominant(frequencies: readonly number[], start: number, end: number, channel = 0): number {
    const levels = frequencies.map((frequency) => this.amplitude(frequency, start, end, channel));
    return frequencies[levels.indexOf(Math.max(...levels))]!;
  }

  band(frequencies: readonly number[], start: number, end: number, window = 0.2): number {
    // The mean level of noise around these frequencies, over many short windows.
    const powers = frange(start, end - window, window).flatMap((at) =>
      frequencies.map((frequency) => this.amplitude(frequency, at, at + window) ** 2),
    );
    return Math.sqrt(powers.reduce((total, power) => total + power, 0) / powers.length);
  }

  envelope(start: number, end: number, window: number, channel = 0): number[] {
    return frange(start, end - window, window).map((at) => this.rms(at, at + window, channel));
  }
}

export function frange(start: number, stop: number, step: number): number[] {
  return Array.from({ length: Math.trunc((stop - start) / step + 1e-9) + 1 }, (_, index) => start + index * step);
}

export function decibels(value: number): number {
  return 20 * Math.log10(value);
}
