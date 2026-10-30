// The animation and video cases of the Python test_runpod_workflows.py.
import assert from "node:assert/strict";

import {
  AudioAdjustVolume,
  AudioConcat,
  AudioEqualizer3Band,
  AudioMerge,
  EmptyAudio,
  JoinAudioChannels,
  SplitAudioChannels,
  TrimAudioDuration,
} from "../../src/nodes/audio.js";
import { SaveAnimatedPNG, SaveAnimatedWEBP } from "../../src/nodes/image.js";
import { ImageFromBatch, RebatchImages } from "../../src/nodes/image/batch.js";
import { CreateVideo, GetVideoComponents, SaveVideo, SaveWEBM, VideoFrameSample, VideoSlice } from "../../src/nodes/video.js";
import { VideoTemporalCrop } from "../../src/nodes/video/transform.js";
import { batchOf, calc, quadrants, rotate, save, solid, stitch, tilesOf } from "./blocks.js";
import {
  apngFrames,
  assertMatroska,
  assertMp4,
  assertPixel,
  decodePng,
  hues,
  pngSize,
  quadrantColor,
  quarterPoints,
  RED,
  rgb,
  rotatedSource,
  ROTATIONS,
  SIX,
  webpInfo,
  WHITE,
} from "./media.js";
import { addCase } from "./registry.js";

addCase(
  "animated_png_frames",
  "Seis cores viram um lote e um PNG animado (SaveAnimatedPNG): o arquivo tem 6 quadros " +
    "no acTL e o primeiro quadro é vermelho.",
  {},
  (w) => {
    w.addNode(SaveAnimatedPNG, { images: batchOf(w, SIX), filename_prefix: "loop", fps: 3.0, compress_level: 6 });

    return (out) => {
      const data = out.one("loop");
      assert.deepEqual(pngSize(data), [64, 64]);
      assert.equal(apngFrames(data), 6);
      assertPixel(decodePng(data), 10, 10, RED, 0);
    };
  },
);

addCase(
  "animated_webp_frames",
  "O mesmo lote vira WEBP animado sem perdas (SaveAnimatedWEBP): canvas 64x64 e um " +
    "quadro ANMF por cor.",
  {},
  (w) => {
    w.addNode(SaveAnimatedWEBP, {
      images: batchOf(w, SIX),
      method: "default",
      filename_prefix: "loop",
      fps: 3.0,
      lossless: true,
      quality: 100,
    });

    return (out) => {
      const [size, frames] = webpInfo(out.one("loop"));
      assert.deepEqual(size, [64, 64]);
      assert.equal(frames, 6);
    };
  },
);

addCase(
  "webm_vp9_and_av1",
  "SaveWEBM com VP9 e com AV1 a partir de 8 quadros 128x128: dois WebM com os codecs " +
    "certos no cabeçalho Matroska.",
  {},
  (w) => {
    const frames = batchOf(w, hues(8), 128);
    for (const codec of ["vp9", "av1"] as const) {
      w.addNode(SaveWEBM, { images: frames, codec, filename_prefix: codec, fps: 8.0, crf: 32.0 });
    }

    return (out) => {
      assert(out.name("vp9").endsWith(".webm"));
      assertMatroska(out.one("vp9"), "V_VP9");
      assert(out.name("av1").endsWith(".webm"));
      assertMatroska(out.one("av1"), "V_AV1");
    };
  },
);

const VIDEO_TARGETS = {
  mp4_h264: ["mp4", "h264", ".mp4"],
  mp4_av1: ["mp4", "av1", ".mp4"],
  mkv_h264: ["mkv", "h264", ".mkv"],
  webm_av1: ["webm", "av1", ".webm"],
  auto: ["auto", "auto", ".mp4"],
} as const;

addCase(
  "save_video_containers",
  "SaveVideo em todas as combinações de contêiner e codec (mp4/mkv/webm, h264/av1, " +
    "auto): extensão, contêiner e codec conferidos em cada arquivo.",
  {},
  (w) => {
    const video = w.addNode(CreateVideo, { images: batchOf(w, hues(8), 128), fps: 8.0 }).output.VIDEO;
    for (const [label, [container, codec]] of Object.entries(VIDEO_TARGETS)) {
      w.addNode(SaveVideo, { video, filename_prefix: label, format: container, format_codec: codec });
    }

    return (out) => {
      for (const [label, [, , extension]] of Object.entries(VIDEO_TARGETS)) {
        assert(out.name(label).endsWith(extension), out.names().join(", "));
      }
      assertMp4(out.one("mp4_h264"), "avc1");
      assertMp4(out.one("mp4_av1"), "av01");
      assertMatroska(out.one("mkv_h264"), "V_MPEG4/ISO/AVC");
      assertMatroska(out.one("webm_av1"), "V_AV1");
      assertMp4(out.one("auto"));
    };
  },
);

addCase(
  "video_round_trip_frames",
  "Vídeo -> GetVideoComponents -> CreateVideo de novo (fps, bit_depth e color_space " +
    "ligados como combos) -> componentes: os seis quadros voltam idênticos e o fps vira " +
    "largura via matemática.",
  {},
  (w) => {
    const video = w.addNode(CreateVideo, { images: batchOf(w, SIX), fps: 6.0 }).output.VIDEO;
    const parts = w.addNode(GetVideoComponents, { video }).output;
    const rebuilt = w.addNode(CreateVideo, {
      images: parts.images,
      fps: parts.fps,
      bit_depth: parts.bit_depth,
      color_space: parts.color_space,
    }).output.VIDEO;
    const again = w.addNode(GetVideoComponents, { video: rebuilt }).output;
    save(w, again.images, "frame");
    save(w, solid(w, calc(w, "a * 10", { a: again.fps }).INT, 8, WHITE), "fps_width");

    return (out) => {
      assert.deepEqual(
        out.pictures("frame").map((frame) => frame.pixel(20, 20)),
        SIX.map((color) => rgb(color)),
      );
      assert.deepEqual(out.picture("fps_width").size, [60, 8]);
    };
  },
);

function h264Profile(data: Buffer): number {
  // avcC starts with configurationVersion, then AVCProfileIndication.
  const position = data.indexOf("avcC", 0, "latin1");
  assert(position >= 0, "no avcC box in the MP4 file");
  return data[position + 5]!;
}

addCase(
  "temporal_crop_and_frame_sample",
  "VideoTemporalCrop e VideoFrameSample (uniform e head) sobre 16 quadros de cores " +
    "diferentes: contagem de quadros e cores amostradas conferidas.",
  {},
  (w) => {
    const colors = hues(16);
    const video = w.addNode(CreateVideo, { images: batchOf(w, colors), fps: 8.0 }).output.VIDEO;
    const cropped = w.addNode(VideoTemporalCrop, { video, start_frame: 4, length: 6 });
    const uniform = w.addNode(VideoFrameSample, { video, num_frames: 4, strategy: "uniform" });
    const head = w.addNode(VideoFrameSample, { video, num_frames: 3, strategy: "head" });
    for (const [label, node] of [
      ["cropped", cropped],
      ["uniform", uniform],
      ["head", head],
    ] as const) {
      const frames = w.addNode(GetVideoComponents, { video: node.output.video }).output.images;
      save(w, frames, label);
    }

    return (out) => {
      const count = (label: string) => out.of(label).length;
      assert(5 <= count("cropped") && count("cropped") <= 7, out.names().join(", "));
      assert(2 <= count("head") && count("head") <= 4, out.names().join(", "));
      const sampled = out.pictures("uniform");
      assert.equal(sampled.length, 4);
      [0, 5, 10, 15].forEach((index, frame) => assertPixel(sampled[frame]!, 32, 32, colors[index]!, 48));
    };
  },
);

addCase(
  "video_slice_node",
  "O nó 'Video Slice' (class_type com espaço) corta 1 s a partir de 0,5 s de um vídeo " +
    "de 2 s; a fatia é salva em MP4 e decomposta em quadros.",
  {},
  (w) => {
    const video = w.addNode(CreateVideo, { images: batchOf(w, hues(16)), fps: 8.0 }).output.VIDEO;
    const sliced = w.addNode(VideoSlice, { video, start_time: 0.5, duration: 1.0 }).output.VIDEO;
    w.addNode(SaveVideo, { video: sliced, filename_prefix: "slice", format: "mp4", format_codec: "h264" });
    save(w, w.addNode(GetVideoComponents, { video: sliced }).output.images, "frame");

    return (out) => {
      assertMp4(out.one("slice"), "avc1");
      const frames = out.of("frame").length;
      assert(7 <= frames && frames <= 9, out.names().join(", "));
    };
  },
);

addCase(
  "video_with_audio_tracks",
  "Áudio silencioso com AudioAdjustVolume e TrimAudioDuration entra no CreateVideo; o " +
    "MP4 sai com faixa AAC e o WebM com Opus.",
  {},
  (w) => {
    let audio = w.addNode(EmptyAudio, { duration: 1.0, sample_rate: 44100, channels: 2 }).output.AUDIO;
    audio = w.addNode(AudioAdjustVolume, { audio, volume: -6 }).output.AUDIO;
    audio = w.addNode(TrimAudioDuration, { audio, start_index: 0.0, duration: 1.0 }).output.AUDIO;
    const video = w.addNode(CreateVideo, { images: batchOf(w, hues(8), 128), fps: 8.0, audio });
    for (const [label, container, codec] of [
      ["with_aac", "mp4", "h264"],
      ["with_opus", "webm", "av1"],
    ] as const) {
      w.addNode(SaveVideo, { video: video.output.VIDEO, filename_prefix: label, format: container, format_codec: codec });
    }

    return (out) => {
      assertMp4(out.one("with_aac"), "avc1", "mp4a");
      assertMatroska(out.one("with_opus"), "V_AV1", "A_OPUS");
    };
  },
);

addCase(
  "audio_channel_ops_into_video",
  "Cadeia de áudio com SplitAudioChannels, JoinAudioChannels, AudioConcat, AudioMerge, " +
    "AudioEqualizer3Band e TrimAudioDuration, gravada num MKV com H.264 + AAC.",
  {},
  (w) => {
    const first = w.addNode(EmptyAudio, { duration: 2.0, sample_rate: 48000, channels: 2 }).output.AUDIO;
    const channels = w.addNode(SplitAudioChannels, { audio: first }).output;
    const joined = w.addNode(JoinAudioChannels, { audio_left: channels.left, audio_right: channels.right });
    const second = w.addNode(EmptyAudio, { duration: 1.0, sample_rate: 48000, channels: 2 }).output.AUDIO;
    const longer = w.addNode(AudioConcat, { audio1: joined.output.audio, audio2: second, direction: "after" });
    const merged = w.addNode(AudioMerge, {
      audio1: longer.output.AUDIO,
      audio2: longer.output.AUDIO,
      merge_method: "mean",
    });
    const shaped = w.addNode(AudioEqualizer3Band, { audio: merged.output.AUDIO, low_gain_dB: 3.0, high_gain_dB: -3.0 });
    const trimmed = w.addNode(TrimAudioDuration, { audio: shaped.output.AUDIO, start_index: 0.5, duration: 1.5 });
    const video = w.addNode(CreateVideo, { images: batchOf(w, hues(12), 128), fps: 8.0, audio: trimmed.output.AUDIO });
    w.addNode(SaveVideo, { video: video.output.VIDEO, filename_prefix: "mixed", format: "mkv", format_codec: "h264" });

    return (out) => {
      assert(out.name("mixed").endsWith(".mkv"));
      assertMatroska(out.one("mixed"), "V_MPEG4/ISO/AVC", "A_AAC");
    };
  },
);

addCase(
  "rotation_frames_through_video",
  "Quatro rotações dos quadrantes viram os quadros de um vídeo; GetVideoComponents + " +
    "ImageFromBatch recuperam o 2º e o último quadro exatos, e um WEBP animado.",
  {},
  (w) => {
    const image = quadrants(w, 64);
    const rotated = [image, ...ROTATIONS.map((rotation) => rotate(w, image, rotation))];
    let sheet = rotated[0]!;
    for (const frame of rotated.slice(1)) {
      sheet = stitch(w, sheet, frame);
    }
    const frames = w.addNode(RebatchImages, { images: tilesOf(w, sheet, 64), batch_size: 4 }).output.IMAGE;
    const video = w.addNode(CreateVideo, { images: frames, fps: 4.0 }).output.VIDEO;
    const parts = w.addNode(GetVideoComponents, { video }).output.images;
    const second = w.addNode(ImageFromBatch, { image: parts, batch_index: 1, length: 1 });
    const last = w.addNode(ImageFromBatch, { image: parts, batch_index: -1, length: 1 });
    save(w, second.output.IMAGE, "second");
    save(w, last.output.IMAGE, "last");
    w.addNode(SaveAnimatedWEBP, {
      images: parts,
      method: "fastest",
      filename_prefix: "spin",
      fps: 4.0,
      lossless: true,
      quality: 90,
    });

    return (out) => {
      for (const [label, rotation] of [
        ["second", "90 degrees"],
        ["last", "270 degrees"],
      ] as const) {
        const picture = out.picture(label);
        for (const [x, y] of quarterPoints(64, 64)) {
          assertPixel(picture, x, y, quadrantColor(...rotatedSource(rotation, x, y, 64, 64), 64, 64), 0);
        }
      }
      assert.deepEqual(webpInfo(out.one("spin")), [[64, 64], 4]);
    };
  },
);

addCase(
  "video_bit_depth_and_color_space",
  "CreateVideo com bit_depth 10, decomposto e recriado com os combos bit_depth e " +
    "color_space do GetVideoComponents: o MP4 sai em H.264 High 10, o de 8 bits não.",
  {},
  (w) => {
    const frames = batchOf(w, hues(6), 128);
    const deep = w.addNode(CreateVideo, { images: frames, fps: 10.0, bit_depth: 10, color_space: "sRGB" }).output.VIDEO;
    const parts = w.addNode(GetVideoComponents, { video: deep }).output;
    const rebuilt = w.addNode(CreateVideo, {
      images: parts.images,
      fps: parts.fps,
      bit_depth: parts.bit_depth,
      color_space: parts.color_space,
    }).output.VIDEO;
    const shallow = w.addNode(CreateVideo, { images: frames, fps: 10.0, bit_depth: 8 }).output.VIDEO;
    for (const [label, video] of [
      ["deep", deep],
      ["rebuilt", rebuilt],
      ["shallow", shallow],
    ] as const) {
      w.addNode(SaveVideo, { video, filename_prefix: label, format: "mp4", format_codec: "h264" });
    }

    return (out) => {
      for (const label of ["deep", "rebuilt", "shallow"]) {
        assertMp4(out.one(label), "avc1");
      }
      assert.equal(h264Profile(out.one("deep")), 110);
      assert.equal(h264Profile(out.one("rebuilt")), 110);
      assert.notEqual(h264Profile(out.one("shallow")), 110);
    };
  },
);
