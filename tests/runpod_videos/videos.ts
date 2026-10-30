// The ten videos of the Python test_runpod_videos.py, each 30 seconds long and built
// and checked differently. The sound starts as short WAV files made here and sent
// with the job; the workflow builds the soundtrack from them. The returned files
// are measured with ffprobe and ffmpeg.
import assert from "node:assert/strict";

import { AudioEqualizer3Band } from "../../src/nodes/audio.js";
import type { ComfyUIWorkflow, Link } from "../../src/nodes/base.js";
import { ImageInvert } from "../../src/nodes/image/color.js";
import { ImageAddNoise } from "../../src/nodes/image/filters.js";
import { CreateVideo, GetVideoComponents, SaveVideo, VideoSlice } from "../../src/nodes/video.js";
import { Workflow } from "../../src/workflow.js";
import { decode, emptyLatent, ksample, NEGATIVE, prompt, scale, sdxl, solid, stitch } from "../runpod_workflows/blocks.js";
import { assertPixel, hues, rgb, roundHalfEven, type Size } from "../runpod_workflows/media.js";
import { GPU_TIMEOUT, sent, TIMEOUT, type Check, type Uploads } from "../runpod_workflows/registry.js";
import {
  BLUE,
  colorStrip,
  concat,
  dimmed,
  doubled,
  GREEN,
  leftChannel,
  load,
  louder,
  merge,
  movie,
  NAVY,
  NIGHT,
  pause,
  RED,
  repeated,
  sideBySide,
  stereo,
  timeline,
  trim,
  WHITE,
  YELLOW,
} from "./blocks.js";
import { assertThirtySeconds, inverted, luminance, Media, pictureDifference, SECONDS, spread } from "./media.js";
import { chord, decibels, faded, hiss, quiet, sine, Sound, wav } from "./sound.js";

/** A 30-second video with sound: the workflow it builds and the check of the file it returns. */
export class Video {
  constructor(
    readonly name: string,
    readonly description: string,
    readonly build: (w: Workflow) => Check,
    readonly fps: number,
    readonly uploads: Uploads = {},
    readonly gpu = false,
  ) {}

  get timeout(): number {
    return this.gpu ? GPU_TIMEOUT : TIMEOUT;
  }

  payload(): [ComfyUIWorkflow, Check] {
    const workflow = new Workflow();
    const check = this.build(workflow);
    return [sent(workflow), check];
  }
}

export const VIDEOS: Video[] = [];

interface VideoOptions {
  readonly fps: number;
  readonly uploads: Uploads;
  readonly gpu?: boolean;
}

function addVideo(name: string, description: string, options: VideoOptions, build: (w: Workflow) => Check): void {
  VIDEOS.push(new Video(name, description, build, options.fps, options.uploads, options.gpu ?? false));
}

/** How Python prints a list of numbers. */
function list(values: readonly number[]): string {
  return `[${values.join(", ")}]`;
}

// ---------------------------------------------------------------- 1. metronome

const CLICK_RATE = 22_050;

addVideo(
  "metronome_flashes_on_every_beat",
  "Metrônomo: um clique de 0,1 s a 1 kHz (WAV enviado) é dobrado cinco vezes com " +
    "AudioConcat, que o deixa estéreo, e cortado em 30 s com TrimAudioDuration; a tela " +
    "pisca branca no primeiro quarto de cada segundo, junto com a batida.",
  {
    fps: 4.0,
    uploads: {
      "clique.wav": wav(CLICK_RATE, [
        ...faded(sine(1000, 0.1, CLICK_RATE, 0.6), CLICK_RATE),
        ...quiet(0.9, CLICK_RATE),
      ]),
    },
  },
  (w) => {
    const beats = trim(w, doubled(w, load(w, "clique.wav"), 5), 0.0, 30.0);
    const second = timeline(w, colorStrip(w, [WHITE, NIGHT, NIGHT, NIGHT], [256, 144]), [256, 144], 4);
    movie(w, repeated(w, second, 30), 4.0, beats, "metronomo");

    return (out) => {
      const media = Media.returned(out, "metronomo");
      assertThirtySeconds(media, { fps: 4, size: [256, 144], video: "h264", audio: "aac", rate: CLICK_RATE, channels: 2 });
      const sound = media.sound();
      for (let beat = 0; beat < SECONDS; beat++) {
        assert(sound.amplitude(1000, beat + 0.02, beat + 0.08) > 0.3, `no click at ${beat}s`);
        assert(sound.rms(beat + 0.3, beat + 0.9) < 0.01, `noise after the click at ${beat}s`);
      }
      for (const beat of [0, 14, 29]) {
        assertPixel(media.frame(beat + 0.125), 128, 72, WHITE, 16);
        assertPixel(media.frame(beat + 0.625), 128, 72, NIGHT, 16);
      }
    };
  },
);

// ---------------------------------------------------------------- 2. stereo ping-pong

const PING_RATE = 16_000;

addVideo(
  "stereo_ping_pong_between_speakers",
  "Pingue-pongue estéreo: um lá de 440 Hz (WAV) e 5 s de EmptyAudio são emendados com " +
    "AudioConcat (after e before), reduzidos a um canal com SplitAudioChannels e juntados " +
    "com JoinAudioChannels; o som troca de caixa a cada 5 s e acende a metade da tela do " +
    "lado que toca.",
  { fps: 6.0, uploads: { "la_440.wav": wav(PING_RATE, sine(440, 5.0, PING_RATE, 0.5)) } },
  (w) => {
    const [tone, rest] = [load(w, "la_440.wav"), pause(w, 5.0, PING_RATE)];
    const [toneFirst, restFirst] = [concat(w, tone, rest), concat(w, tone, rest, "before")];
    const left = leftChannel(w, concat(w, concat(w, toneFirst, toneFirst), toneFirst));
    const right = leftChannel(w, concat(w, concat(w, restFirst, restFirst), restFirst));
    const onLeft = colorStrip(w, [YELLOW, NAVY], [160, 180]);
    const onRight = colorStrip(w, [NAVY, YELLOW], [160, 180]);
    const tenSeconds = timeline(w, stitch(w, onLeft, onRight), [320, 180], 60, 30);
    movie(w, repeated(w, tenSeconds, 3), 6.0, stereo(w, left, right), "pingue_pongue");

    return (out) => {
      const media = Media.returned(out, "pingue_pongue");
      assertThirtySeconds(media, { fps: 6, size: [320, 180], video: "h264", audio: "aac", rate: PING_RATE, channels: 2 });
      const sound = media.sound(2);
      for (let part = 0; part < 6; part++) {
        const [playing, silent] = part % 2 === 0 ? [0, 1] : [1, 0];
        const [start, end] = [5 * part + 0.5, 5 * part + 4.5];
        assert(sound.amplitude(440, start, end, playing) > 0.35, `part ${part} is not playing`);
        assert(sound.rms(start, end, silent) < 0.01, `part ${part} leaks to the other side`);
        const frame = media.frame(5 * part + 2.5);
        const [lit, dark] = playing === 0 ? [80, 240] : [240, 80];
        assertPixel(frame, lit, 90, YELLOW, 16);
        assertPixel(frame, dark, 90, NAVY, 16);
      }
    };
  },
);

// ---------------------------------------------------------------- 3. volume staircase

const STEP_RATE = 24_000;
const STEPS = [0, -6, -12, -18, -24, -30];
const GRAYS = [0xf0f0f0, 0xb4b4b4, 0x878787, 0x5a5a5a, 0x373737, 0x1e1e1e];

addVideo(
  "volume_staircase_dims_with_the_screen",
  "Escada de volume: o mesmo mi de 660 Hz passa por seis AudioAdjustVolume (0 a -30 dB, " +
    "de 6 em 6) e os trechos de 5 s são emendados; a tela escurece no mesmo passo.",
  { fps: 10.0, uploads: { "mi_660.wav": wav(STEP_RATE, sine(660, 5.0, STEP_RATE, 0.8)) } },
  (w) => {
    const tone = load(w, "mi_660.wav");
    const steps = STEPS.map((step) => louder(w, tone, step));
    let sound = steps[0]!;
    for (const step of steps.slice(1)) {
      sound = concat(w, sound, step);
    }
    const frames = timeline(w, colorStrip(w, GRAYS, [256, 144]), [256, 144], 300, 50);
    movie(w, frames, 10.0, sound, "escada_de_volume");

    return (out) => {
      const media = Media.returned(out, "escada_de_volume");
      assertThirtySeconds(media, { fps: 10, size: [256, 144], video: "h264", audio: "aac", rate: STEP_RATE, channels: 2 });
      const sound = media.sound();
      const first = sound.amplitude(660, 0.5, 4.5);
      assert(Math.abs(first - 0.8) < 0.1, `the first step plays at ${first.toFixed(3)}, not 0.8`);
      const levels = Array.from({ length: 6 }, (_, part) => decibels(sound.rms(5 * part + 0.5, 5 * part + 4.5)));
      const drops = levels.slice(1).map((level, index) => Number((levels[index]! - level).toFixed(2)));
      assert(
        drops.every((drop) => 5.0 <= drop && drop <= 7.0),
        `the steps drop ${list(drops)} dB`,
      );
      GRAYS.forEach((gray, part) => assertPixel(media.frame(5 * part + 2.5), 128, 72, gray, 10));
    };
  },
);

// ---------------------------------------------------------------- 4. chord and tremolo

const CHORD_RATE = 16_000;
const C_MAJOR = [262, 330, 392];

addVideo(
  "chord_then_tremolo",
  "Acorde e tremolo: dó, mi e sol (WAVs de 1 s) somados com AudioMerge add viram um " +
    "acorde; multiplicado (AudioMerge multiply) por um LFO de 2 Hz, vira tremolo. Os dois " +
    "são dobrados e cortados em 15 s; as listras da tela ficam firmes e depois pulsam no " +
    "ritmo do tremolo.",
  {
    fps: 8.0,
    uploads: {
      "do_262.wav": wav(CHORD_RATE, sine(262, 1.0, CHORD_RATE, 0.3)),
      "mi_330.wav": wav(CHORD_RATE, sine(330, 1.0, CHORD_RATE, 0.3)),
      "sol_392.wav": wav(CHORD_RATE, sine(392, 1.0, CHORD_RATE, 0.3)),
      "lfo_2hz.wav": wav(
        CHORD_RATE,
        Array.from({ length: CHORD_RATE }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * 2 * n) / CHORD_RATE)),
      ),
    },
  },
  (w) => {
    const notes = ["do_262.wav", "mi_330.wav", "sol_392.wav"].map((name) => load(w, name));
    const triad = merge(w, merge(w, notes[0]!, notes[1]!, "add"), notes[2]!, "add");
    const tremolo = merge(w, triad, load(w, "lfo_2hz.wav"), "multiply");
    const plain = trim(w, doubled(w, triad, 4), 0.0, 15.0);
    const pulsing = trim(w, doubled(w, tremolo, 4), 0.0, 15.0);
    const size: Size = [288, 160];
    const stripes = colorStrip(w, [RED, GREEN, BLUE], [96, 160]);
    const half = dimmed(w, stripes, size, 0.5);
    const pulse = sideBySide(w, [dimmed(w, stripes, size, 0.85), half, stripes, half]);
    const steady = repeated(w, stripes, 120);
    const beating = repeated(w, timeline(w, pulse, size, 4), 30);
    const frames = timeline(w, stitch(w, steady, beating), size, 240);
    movie(w, frames, 8.0, concat(w, plain, pulsing), "acorde_e_tremolo");

    return (out) => {
      const media = Media.returned(out, "acorde_e_tremolo");
      assertThirtySeconds(media, { fps: 8, size, video: "h264", audio: "aac", rate: CHORD_RATE, channels: 2 });
      const sound = media.sound();
      for (const frequency of C_MAJOR) {
        assert(sound.amplitude(frequency, 1.0, 14.0) > 0.2, `${frequency} Hz is missing`);
      }
      for (const frequency of [300, 440]) {
        assert(sound.amplitude(frequency, 1.0, 14.0) < 0.03, `${frequency} Hz is not in the chord`);
      }
      const steadyEnvelope = sound.envelope(2.0, 4.0, 0.05);
      assert(Math.min(...steadyEnvelope) > 0.75 * Math.max(...steadyEnvelope), "the chord is not steady");
      // The LFO starts at zero, so the tremolo peaks 0.25 s into each half second.
      for (let cycle = 4; cycle < 28; cycle++) {
        const peak = sound.rms(15.2 + cycle / 2, 15.3 + cycle / 2);
        const trough = sound.rms(14.95 + cycle / 2, 15.05 + cycle / 2);
        assert(trough < 0.3 * peak, `no tremolo around ${15 + cycle / 2}s`);
      }
      assertPixel(media.frame(5.0625), 48, 80, RED, 16);
      assertPixel(media.frame(16.3125), 48, 80, RED, 16);
      assertPixel(
        media.frame(16.0625),
        48,
        80,
        rgb(RED).map((channel) => roundHalfEven(channel * 0.15)),
        16,
      );
    };
  },
);

// ---------------------------------------------------------------- 5. off the air

const TV_RATE = 22_050;
const BARS = [0xc0c0c0, 0xc0c000, 0x00c0c0, 0x00c000, 0xc000c0, 0xc00000, 0x0000c0, 0x101010];

addVideo(
  "tv_static_then_test_pattern",
  "TV fora do ar: 10 s de chiado (ruído branco em WAV, dobrado e cortado) com estática " +
    "animada do ImageAddNoise, depois 20 s de barras de cor com um tom de 1 kHz a -6 dB. " +
    "Os dois lotes de quadros são emendados lado a lado e recortados de volta em sequência.",
  {
    fps: 10.0,
    uploads: {
      "chiado.wav": wav(TV_RATE, hiss(1.0, TV_RATE, 0.5, 11)),
      "tom_1khz.wav": wav(TV_RATE, sine(1000, 1.0, TV_RATE, 0.5)),
    },
  },
  (w) => {
    const noise = trim(w, doubled(w, load(w, "chiado.wav"), 4), 0.0, 10.0);
    const tone = louder(w, trim(w, doubled(w, load(w, "tom_1khz.wav"), 5), 0.0, 20.0), -6);
    const snow = w.addNode(ImageAddNoise, { image: solid(w, 320, 180, 0x808080, 100), seed: 3, strength: 1.0 }).output
      .IMAGE;
    const bars = repeated(w, colorStrip(w, BARS, [40, 180]), 100);
    const frames = timeline(w, sideBySide(w, [snow, bars, bars]), [320, 180], 300);
    movie(w, frames, 10.0, concat(w, noise, tone), "fora_do_ar");

    return (out) => {
      const media = Media.returned(out, "fora_do_ar");
      assertThirtySeconds(media, { fps: 10, size: [320, 180], video: "h264", audio: "aac", rate: TV_RATE, channels: 2 });
      const sound = media.sound();
      assert(sound.rms(0.5, 9.5) > 0.15, "the static is too quiet");
      assert(sound.amplitude(1000, 0.5, 9.5) < 0.05, "the static has a tone");
      assert(sound.rms(9.0, 9.9) > 0.15, "the static stops before 10s");
      assert(Math.abs(sound.amplitude(1000, 10.1, 11.9) - 0.25) < 0.05, "no tone at 10s");
      assert(Math.abs(sound.amplitude(1000, 10.5, 29.5) - 0.25) < 0.05);
      assert(Math.abs(sound.rms(10.5, 29.5) - 0.25 / Math.SQRT2) < 0.03, "noise under the tone");
      const [first, second] = [media.frame(5.05), media.frame(5.15)];
      assert(spread(luminance(first)) > 25, "the static is flat");
      assert(pictureDifference(first, second) > 15, "the static does not move");
      const barsFrame = media.frame(20.05);
      BARS.forEach((color, index) => assertPixel(barsFrame, 40 * index + 20, 90, color, 16));
    };
  },
);

// ---------------------------------------------------------------- 6. equalizer

const EQ_RATE = 32_000;
const EQ_NOISE = hiss(1.0, EQ_RATE, 0.1, 5);
const BANDS: Readonly<Record<string, readonly number[]>> = {
  graves: [100, 150],
  medios: [950, 1000, 1050],
  agudos: [5000, 6000, 7000],
};
const EQ_SETTINGS = [
  [
    "graves",
    RED,
    {
      low_gain_dB: 12.0,
      low_freq: 250,
      mid_gain_dB: -24.0,
      mid_freq: 1000,
      mid_q: 1.0,
      high_gain_dB: -24.0,
      high_freq: 2000,
    },
  ],
  [
    "medios",
    GREEN,
    {
      low_gain_dB: -24.0,
      low_freq: 250,
      mid_gain_dB: 12.0,
      mid_freq: 1000,
      mid_q: 2.0,
      high_gain_dB: -24.0,
      high_freq: 4000,
    },
  ],
  [
    "agudos",
    BLUE,
    {
      low_gain_dB: -24.0,
      low_freq: 250,
      mid_gain_dB: -24.0,
      mid_freq: 1000,
      mid_q: 1.0,
      high_gain_dB: 12.0,
      high_freq: 4000,
    },
  ],
] as const;

addVideo(
  "equalizer_moves_the_noise_between_bands",
  "Equalizador: o mesmo ruído branco (WAV de 1 s, dobrado e cortado em 10 s) passa por " +
    "três AudioEqualizer3Band, que realçam graves, médios e agudos e cortam o resto; a " +
    "tela fica vermelha, verde e azul, uma cor por faixa.",
  { fps: 12.0, uploads: { "ruido_branco.wav": wav(EQ_RATE, EQ_NOISE) } },
  (w) => {
    const noise = trim(w, doubled(w, load(w, "ruido_branco.wav"), 4), 0.0, 10.0);
    const parts = EQ_SETTINGS.map(
      ([, , gains]) => w.addNode(AudioEqualizer3Band, { audio: noise, ...gains }).output.AUDIO,
    );
    const colors = EQ_SETTINGS.map(([, color]) => color);
    const frames = timeline(w, colorStrip(w, colors, [256, 144]), [256, 144], 360, 120);
    movie(w, frames, 12.0, concat(w, concat(w, parts[0]!, parts[1]!), parts[2]!), "equalizador");

    return (out) => {
      const media = Media.returned(out, "equalizador");
      assertThirtySeconds(media, { fps: 12, size: [256, 144], video: "h264", audio: "aac", rate: EQ_RATE, channels: 2 });
      const sound = media.sound();
      // The gains are measured against the noise that was sent, looped to 10 s.
      const source = new Sound(EQ_RATE, [Array.from({ length: 10 }, () => EQ_NOISE).flat()]);
      const flat = new Map(Object.entries(BANDS).map(([band, hz]) => [band, source.band(hz, 0.5, 9.5)]));
      EQ_SETTINGS.forEach(([name, color], part) => {
        const [start, end] = [10 * part + 0.5, 10 * part + 9.5];
        const gains = new Map(
          Object.entries(BANDS).map(([band, hz]) => [
            band,
            decibels(Math.max(sound.band(hz, start, end), 1e-9) / flat.get(band)!),
          ]),
        );
        const shown = `{${[...gains].map(([band, gain]) => `'${band}': ${Number(gain.toFixed(1))}`).join(", ")}}`;
        const boosted = gains.get(name)!;
        assert(5.0 < boosted && boosted < 15.0, `${name} is not boosted 12 dB: ${shown}`);
        assert(
          [...gains].every(([band, gain]) => band === name || gain < -12.0),
          `${name}: the other bands are not cut: ${shown}`,
        );
        assertPixel(media.frame(10 * part + 5.04), 128, 72, color, 16);
      });
    };
  },
);

// ---------------------------------------------------------------- 7. musical scale

const NOTES: Readonly<Record<string, readonly [frequency: number, rate: number]>> = {
  do4: [262, 16_000],
  re4: [294, 22_050],
  mi4: [330, 32_000],
  fa4: [349, 44_100],
  sol4: [392, 48_000],
  la4: [440, 24_000],
  si4: [494, 11_025],
  do5: [523, 8_000],
};
const MELODY = ["do4", "re4", "mi4", "fa4", "sol4", "la4", "si4", "do5", "sol4", "do4"];

addVideo(
  "musical_scale_across_sample_rates",
  "Escala musical com taxas misturadas: oito notas de 1 s gravadas de 8 kHz a 48 kHz; " +
    "cada uma é esticada para 3 s com AudioConcat e a melodia de dez notas é emendada, o " +
    "que reamostra tudo para a maior taxa (48 kHz) sem mudar a afinação; cada nota tem a " +
    "sua cor na tela.",
  {
    fps: 5.0,
    uploads: Object.fromEntries(
      Object.entries(NOTES).map(([name, [frequency, rate]]) => [`${name}.wav`, wav(rate, sine(frequency, 1.0, rate, 0.5))]),
    ),
  },
  (w) => {
    const held = new Map<string, Link>();
    for (const name of new Set(MELODY)) {
      const note = load(w, `${name}.wav`);
      held.set(name, concat(w, concat(w, note, note), note));
    }
    let melody = held.get(MELODY[0]!)!;
    for (const name of MELODY.slice(1)) {
      melody = concat(w, melody, held.get(name)!);
    }
    const colors = hues(MELODY.length);
    const frames = timeline(w, colorStrip(w, colors, [320, 180]), [320, 180], 150, 15);
    movie(w, frames, 5.0, melody, "escala");

    return (out) => {
      const media = Media.returned(out, "escala");
      assertThirtySeconds(media, { fps: 5, size: [320, 180], video: "h264", audio: "aac", rate: 48_000, channels: 2 });
      const sound = media.sound();
      const frequencies = Object.values(NOTES).map(([frequency]) => frequency);
      MELODY.forEach((name, index) => {
        const [start, end] = [3 * index + 0.3, 3 * index + 2.7];
        const expected = NOTES[name]![0];
        assert(sound.dominant(frequencies, start, end) === expected, `note ${index} is not ${name}`);
        assert(sound.amplitude(expected, start, end) > 0.3, `note ${index} is too quiet`);
        assertPixel(media.frame(3 * index + 1.5), 160, 90, colors[index]!, 20);
      });
    };
  },
);

// ---------------------------------------------------------------- 8. remix

const CODE_RATE = 16_000;
const TIME_CODE = [300, 400, 500, 600, 700, 800, 900, 1000];

addVideo(
  "video_slice_remix_shifts_the_time_code",
  "Remix com Video Slice: um vídeo de 40 s com um código de tempo sonoro (um tom por " +
    "trecho de 5 s, num WAV) é cortado de 5 s a 35 s; GetVideoComponents tira os quadros, " +
    "o fps e o áudio, os quadros são invertidos e o vídeo é recriado: som e cores chegam " +
    "deslocados 5 s.",
  {
    fps: 10.0,
    uploads: {
      "codigo_de_tempo.wav": wav(
        CODE_RATE,
        TIME_CODE.flatMap((tone) => sine(tone, 5.0, CODE_RATE, 0.5)),
      ),
    },
  },
  (w) => {
    const colors = hues(TIME_CODE.length);
    const frames = timeline(w, colorStrip(w, colors, [256, 144]), [256, 144], 400, 50);
    const source = w.addNode(CreateVideo, { images: frames, fps: 10.0, audio: load(w, "codigo_de_tempo.wav") }).output
      .VIDEO;
    const sliced = w.addNode(VideoSlice, { video: source, start_time: 5.0, duration: 30.0, strict_duration: true })
      .output.VIDEO;
    const parts = w.addNode(GetVideoComponents, { video: sliced }).output;
    const negative = w.addNode(ImageInvert, { image: parts.images }).output.IMAGE;
    const remix = w.addNode(CreateVideo, { images: negative, fps: parts.fps, audio: parts.audio }).output.VIDEO;
    w.addNode(SaveVideo, { video: remix, filename_prefix: "remix", format: "mp4", format_codec: "h264" });

    return (out) => {
      const media = Media.returned(out, "remix");
      assertThirtySeconds(media, {
        fps: 10,
        size: [256, 144],
        video: "h264",
        audio: "aac",
        rate: CODE_RATE,
        channels: 1,
        frameSlack: 1,
      });
      const sound = media.sound();
      for (let part = 0; part < 6; part++) {
        const [start, end] = [5 * part + 0.5, 5 * part + 4.5];
        const expected = TIME_CODE[part + 1]!;
        assert(sound.dominant(TIME_CODE, start, end) === expected, `part ${part} is not ${expected} Hz`);
        assert(sound.amplitude(expected, start, end) > 0.35, `${expected} Hz does not fill part ${part}`);
        assertPixel(media.frame(5 * part + 2.55), 128, 72, inverted(colors[part + 1]!), 20);
      }
    };
  },
);

// ---------------------------------------------------------------- 9. binaural beat

const BEAT_RATE = 44_100;
const SUNSET = [0xffb347, 0xff7f50, 0xe0558a, 0x9b4dca, 0x4b3f9e, 0x1b1f4b];

addVideo(
  "binaural_beat_in_webm",
  "Batimento binaural num WebM: 200 Hz à esquerda e 206 Hz à direita (dois WAVs mono) " +
    "juntados com JoinAudioChannels, dobrados e cortados em 30 s; SaveVideo grava WebM com " +
    "AV1 e Opus, que reamostra o som para 48 kHz. Somados, os canais batem 6 vezes por " +
    "segundo; a tela passa pelas cores de um pôr do sol.",
  {
    fps: 12.0,
    uploads: {
      "esquerda_200.wav": wav(BEAT_RATE, sine(200, 1.0, BEAT_RATE, 0.5)),
      "direita_206.wav": wav(BEAT_RATE, sine(206, 1.0, BEAT_RATE, 0.5)),
    },
  },
  (w) => {
    const pair = stereo(w, load(w, "esquerda_200.wav"), load(w, "direita_206.wav"));
    const frames = timeline(w, colorStrip(w, SUNSET, [256, 144]), [256, 144], 360, 60);
    movie(w, frames, 12.0, trim(w, doubled(w, pair, 5), 0.0, 30.0), "batimento", "webm", "av1");

    return (out) => {
      assert(out.name("batimento").endsWith(".webm"));
      const media = Media.returned(out, "batimento");
      assert(media.format.format_name.includes("webm"));
      assertThirtySeconds(media, { fps: 12, size: [256, 144], video: "av1", audio: "opus", rate: 48_000, channels: 2 });
      const sound = media.sound(2);
      for (const second of [2, 15, 28]) {
        for (const [channel, own, other] of [
          [0, 200, 206],
          [1, 206, 200],
        ] as const) {
          const side = channel === 0 ? "left" : "right";
          assert(
            sound.amplitude(own, second, second + 1, channel) > 0.35,
            `${own} Hz is missing on the ${side} at ${second}s`,
          );
          assert(
            sound.amplitude(other, second, second + 1, channel) < 0.05,
            `${other} Hz leaks to the ${side} at ${second}s`,
          );
        }
      }
      const mixed = media.sound().envelope(10.0, 11.0, 0.01);
      assert(Math.min(...mixed) < 0.25 * Math.max(...mixed), "the channels do not beat together");
      SUNSET.forEach((color, part) => assertPixel(media.frame(5 * part + 2.54), 128, 72, color, 20));
    };
  },
);

// ---------------------------------------------------------------- 10. SDXL day at the lake

const LAKE_RATE = 22_050;
const LAKE: readonly (readonly [name: string, text: string, notes: readonly number[]])[] = [
  [
    "amanhecer",
    "sunrise over a calm mountain lake, soft pink light, morning mist, landscape photograph",
    [262, 330, 392],
  ],
  [
    "meio_dia",
    "bright midday over a calm mountain lake, deep blue sky, white clouds, landscape photograph",
    [349, 440, 523],
  ],
  [
    "por_do_sol",
    "golden sunset over a calm mountain lake, orange sky, long shadows, landscape photograph",
    [220, 262, 330],
  ],
  ["noite", "night over a calm mountain lake, starry sky, full moon, dark blue, landscape photograph", [196, 247, 294]],
];
const LAKE_SIZE: Size = [512, 288];

addVideo(
  "sdxl_day_at_the_lake",
  "Um dia no lago com SDXL: quatro cenas (amanhecer, meio-dia, pôr do sol, noite) a " +
    "1344x768 com 25 passos, reduzidas e seguradas 7,5 s cada; a trilha toca um acorde " +
    "por cena (WAVs de 1 s, dobrados e cortados), a noite 6 dB mais baixa, com vento " +
    "(ruído suave) somado por AudioMerge.",
  {
    fps: 8.0,
    gpu: true,
    uploads: {
      ...Object.fromEntries(
        LAKE.map(([name, , notes]) => [`acorde_${name}.wav`, wav(LAKE_RATE, chord(notes, 1.0, LAKE_RATE, 0.22))]),
      ),
      "vento.wav": wav(LAKE_RATE, hiss(1.0, LAKE_RATE, 0.04, 9)),
    },
  },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const negative = prompt(w, clip, `${NEGATIVE}, people, person, text`);
    const scenes: Link[] = [];
    let music: Link | null = null;
    for (const [index, [name, text]] of LAKE.entries()) {
      const latent = ksample(w, model, prompt(w, clip, text), negative, emptyLatent(w, 1344, 768), {
        seed: 21 + index,
        steps: 25,
        cfg: 7.0,
      });
      scenes.push(scale(w, decode(w, latent, vae), "area", ...LAKE_SIZE));
      let part = trim(w, doubled(w, load(w, `acorde_${name}.wav`), 3), 0.0, 7.5);
      if (name === "noite") {
        part = louder(w, part, -6);
      }
      music = music === null ? part : concat(w, music, part);
    }
    const wind = trim(w, doubled(w, load(w, "vento.wav"), 5), 0.0, 30.0);
    const frames = timeline(w, sideBySide(w, scenes), LAKE_SIZE, 240, 60);
    movie(w, frames, 8.0, merge(w, music!, wind, "add"), "lago");

    return (out) => {
      const media = Media.returned(out, "lago");
      assertThirtySeconds(media, { fps: 8, size: LAKE_SIZE, video: "h264", audio: "aac", rate: LAKE_RATE, channels: 2 });
      const sound = media.sound();
      const everyNote = new Set(LAKE.flatMap(([, , notes]) => notes));
      LAKE.forEach(([name, , notes], index) => {
        const [start, end] = [7.5 * index + 0.5, 7.5 * index + 7.0];
        const floor = name === "noite" ? 0.07 : 0.15;
        for (const note of notes) {
          assert(sound.amplitude(note, start, end) > floor, `${name}: ${note} Hz is missing`);
        }
        for (const note of [...everyNote].filter((other) => !notes.includes(other))) {
          assert(sound.amplitude(note, start, end) < 0.03, `${name}: ${note} Hz is extra`);
        }
      });
      assert(sound.rms(23.0, 29.5) < 0.7 * sound.rms(8.0, 14.5), "the night is not quieter");
      const pictures = LAKE.map((_, index) => media.frame(7.5 * index + 3.76));
      LAKE.forEach(([name], index) => assert(spread(luminance(pictures[index]!)) > 15, `the ${name} scene is flat`));
      for (let index = 1; index < pictures.length; index++) {
        assert(pictureDifference(pictures[index - 1]!, pictures[index]!) > 10, "two scenes are the same");
      }
      const mean = pictures.map((picture) => {
        const values = luminance(picture);
        return values.reduce((total, value) => total + value, 0) / values.length;
      });
      assert(
        mean[3]! < mean[1]!,
        `the night (${mean[3]!.toFixed(0)}) is not darker than noon (${mean[1]!.toFixed(0)})`,
      );
    };
  },
);
