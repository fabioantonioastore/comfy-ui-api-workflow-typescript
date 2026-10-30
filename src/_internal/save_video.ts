// The checks SaveVideo makes across its inputs. The dynamic combo "format"
// opens "format.codec", then "format.codec.encoding", then
// "format.codec.encoding.crf"; the flat "codec" group follows the same tree.
// The options are kept here too, so nodes/video.ts imports this module without
// a cycle.
import { InvalidNodeParameterValue } from "../errors/node.js";
import { validateChoice, validateRange } from "../nodes/validation.js";
import { repr } from "./format.js";
import { deepFreeze, own } from "./records.js";

const FORMATS = ["auto", "mp4", "mkv", "webm"] as const;
const CODECS = ["auto", "h264", "av1"] as const;
const ENCODINGS = ["auto", "re-encode"] as const;

/** The codecs each format offers, as in object_info. */
export const SAVEVIDEO_FORMAT_CODECS = deepFreeze({
  auto: ["auto", "h264", "av1"],
  mp4: ["auto", "h264", "av1"],
  mkv: ["auto", "h264", "av1"],
  webm: ["auto", "av1"],
} as const);

/** The crf range of each codec that takes an encoding, as in object_info. */
export const SAVEVIDEO_CRF_RANGES = deepFreeze({ h264: [0, 51], av1: [0, 63] } as const);

// Shared by the format.codec branch and the flat codec group: encoding only
// exists for h264 and av1, and crf only for "re-encode".
function validateCodecBranch(prefix: string, codec: unknown, encoding: unknown, crf: unknown): void {
  const encodingName = `${prefix}_encoding`;
  const crfName = `${encodingName}_crf`;
  if (encoding === undefined) {
    if (crf !== undefined) {
      throw new InvalidNodeParameterValue(`${crfName} only applies when ${encodingName} is 're-encode'`);
    }
    return;
  }
  const range = typeof codec === "string" ? own<readonly [number, number]>(SAVEVIDEO_CRF_RANGES, codec) : undefined;
  if (range === undefined) {
    throw new InvalidNodeParameterValue(
      `${encodingName} only applies when ${prefix} is h264 or av1, not ${repr(codec)}`,
    );
  }
  validateChoice(encodingName, encoding, ENCODINGS);
  if (encoding !== "re-encode") {
    if (crf !== undefined) {
      throw new InvalidNodeParameterValue(`${crfName} only applies when ${encodingName} is 're-encode'`);
    }
    return;
  }
  if (crf === undefined) {
    throw new InvalidNodeParameterValue(`${crfName} is required when ${encodingName} is 're-encode'`);
  }
  validateRange(crfName, crf, range[0], range[1]);
}

/** Checks the inputs of a SaveVideo node against each other. */
export function validateSaveVideo(inputs: Readonly<Record<string, unknown>>): void {
  const format = own(inputs, "format");
  validateChoice("format", format, FORMATS);
  const codecs: readonly string[] = SAVEVIDEO_FORMAT_CODECS[format as (typeof FORMATS)[number]];
  const formatCodec = own(inputs, "format_codec");
  if (typeof formatCodec !== "string" || !codecs.includes(formatCodec)) {
    throw new InvalidNodeParameterValue(
      `format_codec ${repr(formatCodec)} is not available for format ${repr(format)}; ` +
        `use one of ${repr([...codecs].sort())}`,
    );
  }
  validateCodecBranch(
    "format_codec",
    formatCodec,
    own(inputs, "format_codec_encoding"),
    own(inputs, "format_codec_encoding_crf"),
  );
  const codec = own(inputs, "codec");
  if (codec !== undefined) {
    validateChoice("codec", codec, CODECS);
  }
  validateCodecBranch("codec", codec, own(inputs, "codec_encoding"), own(inputs, "codec_encoding_crf"));
}
