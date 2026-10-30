// The SDXL cases of the Python test_runpod_workflows.py, which run on the GPU.
import assert from "node:assert/strict";

import { CFGNorm } from "../../src/nodes/advanced/guidance.js";
import type { Link } from "../../src/nodes/base.js";
import { LatentBlend, SelfAttentionGuidance } from "../../src/nodes/experimental.js";
import { SaveAnimatedWEBP } from "../../src/nodes/image.js";
import { ImageFromBatch } from "../../src/nodes/image/batch.js";
import { Canny, ImageSharpen } from "../../src/nodes/image/filters.js";
import { InvertMask } from "../../src/nodes/image/mask.js";
import { ImagePadForOutpaint } from "../../src/nodes/image/transform.js";
import { ImageScaleToMaxDimension } from "../../src/nodes/image/upscaling.js";
import { CLIPSetLastLayer, InpaintModelConditioning } from "../../src/nodes/model/conditioning.js";
import { CLIPTextEncodeSDXL } from "../../src/nodes/model/conditioning/stable_diffusion.js";
import {
  ConditioningAverage,
  ConditioningCombine,
  ConditioningConcat,
  ConditioningMultiply,
  ConditioningSetArea,
  ConditioningSetMask,
  ConditioningSetTimestepRange,
  ConditioningZeroOut,
} from "../../src/nodes/model/conditioning/transform.js";
import {
  LatentComposite,
  LatentUpscale,
  LatentUpscaleBy,
  SetLatentNoiseMask,
  VAEDecodeTiled,
  VAEEncode,
  VAEEncodeForInpaint,
  VAEEncodeTiled,
} from "../../src/nodes/model/latent.js";
import {
  LatentAdd,
  LatentBatchSeedBehavior,
  LatentInterpolate,
  LatentMultiply,
  LatentSubtract,
} from "../../src/nodes/model/latent/advanced.js";
import {
  LatentApplyOperation,
  LatentOperationSharpen,
  LatentOperationTonemapReinhard,
} from "../../src/nodes/model/latent/advanced/operations.js";
import { LatentFromBatch, RepeatLatentBatch } from "../../src/nodes/model/latent/batch.js";
import { LatentCrop, LatentFlip, LatentRotate } from "../../src/nodes/model/latent/transform.js";
import { CLIPMergeSimple, ModelMergeSimple } from "../../src/nodes/model/merging.js";
import { ModelSamplingDiscrete, RescaleCFG } from "../../src/nodes/model/patch.js";
import {
  EpsilonScaling,
  FreeU,
  FreeU_V2,
  HyperTile,
  PatchModelAddDownscale,
  PerturbedAttentionGuidance,
  TomePatchModel,
} from "../../src/nodes/model/patch/unet.js";
import { KSamplerAdvanced } from "../../src/nodes/model/sampling.js";
import { APG, SamplerCustom, SamplerCustomAdvanced } from "../../src/nodes/model/sampling/custom.js";
import { BasicGuider, CFGGuider, CFGOverride, DualCFGGuider } from "../../src/nodes/model/sampling/guiders.js";
import { RandomNoise } from "../../src/nodes/model/sampling/noise.js";
import { KSamplerSelect, SamplerDPMPP_2M_SDE, SamplerEulerAncestral } from "../../src/nodes/model/sampling/samplers.js";
import { BasicScheduler, ExponentialScheduler, KarrasScheduler } from "../../src/nodes/model/sampling/schedulers.js";
import { SplitSigmas } from "../../src/nodes/model/sampling/sigmas.js";
import { TextOverlay } from "../../src/nodes/text.js";
import { CreateVideo, GetVideoComponents, SaveVideo } from "../../src/nodes/video.js";
import type { Workflow } from "../../src/workflow.js";
import {
  combineMasks,
  crop,
  decode,
  emptyLatent,
  ksample,
  NEGATIVE,
  paste,
  prompt,
  quadrants,
  save,
  scale,
  sdxl,
  solid,
  solidMask,
  stitch,
  TextToImage,
  thumbnail,
} from "./blocks.js";
import {
  assertGenerated,
  assertMp4,
  BLUE,
  colorsOf,
  GRAY,
  GREEN,
  meanDifference,
  pngSize,
  RED,
  tuple,
  webpInfo,
  YELLOW,
} from "./media.js";
import { addCase, type Check } from "./registry.js";

/** How many of the files differ from each other. */
function distinct(files: readonly Buffer[]): number {
  return new Set(files.map((data) => data.toString("base64"))).size;
}

addCase(
  "sdxl_sampler_and_scheduler_sheet",
  "SDXL com quatro combinações de sampler/scheduler (euler, dpmpp_2m, dpmpp_2m_sde, " +
    "uni_pc) e uma folha de contato 2x2 montada com ImageScale + ImageStitch.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const positive = prompt(w, clip, "a lighthouse on a rocky coast at sunset, oil painting");
    const negative = prompt(w, clip, NEGATIVE);
    const latent = emptyLatent(w, 768, 768);
    const combos = [
      ["euler", "normal"],
      ["dpmpp_2m", "karras"],
      ["dpmpp_2m_sde", "exponential"],
      ["uni_pc", "sgm_uniform"],
    ] as const;
    const small: Link[] = [];
    for (const [sampler, scheduler] of combos) {
      const image = decode(w, ksample(w, model, positive, negative, latent, { sampler, scheduler }), vae);
      save(w, image, `${sampler}_${scheduler}`);
      small.push(scale(w, image, "area", 384, 384));
    }
    const [top, bottom] = [stitch(w, small[0]!, small[1]!), stitch(w, small[2]!, small[3]!)];
    save(w, stitch(w, top, bottom, "down"), "sheet");

    return (out) => {
      const images = combos.map(([sampler, scheduler]) => out.one(`${sampler}_${scheduler}`));
      for (const image of images) {
        assertGenerated(image, [768, 768]);
      }
      assert.equal(distinct(images), 4);
      assert.deepEqual(pngSize(out.one("sheet")), [768, 768]);
    };
  },
);

addCase(
  "sdxl_seed_reproducibility",
  "Semente reprodutível no SDXL: dois KSampler com a mesma semente dão o mesmo " +
    "resultado e uma semente diferente muda a imagem.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const positive = prompt(w, clip, "a bowl of ramen on a wooden table, food photography");
    const negative = prompt(w, clip, NEGATIVE);
    const latent = emptyLatent(w, 640, 640);
    for (const [label, seed] of [
      ["seed_a", 101],
      ["seed_a_again", 101],
      ["seed_b", 202],
    ] as const) {
      const image = decode(w, ksample(w, model, positive, negative, latent, { seed, steps: 12 }), vae);
      thumbnail(w, image, label, 32);
      if (label === "seed_a") {
        save(w, image, "full");
      }
    }

    return (out) => {
      const first = out.picture("seed_a");
      assert(meanDifference(first, out.picture("seed_a_again")) < 1.5);
      assert(meanDifference(first, out.picture("seed_b")) > 8);
      assertGenerated(out.one("full"), [640, 640]);
    };
  },
);

addCase(
  "sdxl_prompt_reaches_the_model",
  "O prompt chega ao modelo: 'parede vermelha' sai avermelhada e 'parede azul' sai " +
    "azulada, medido na cor média de miniaturas.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const negative = prompt(w, clip, NEGATIVE);
    const latent = emptyLatent(w, 768, 768);
    for (const [label, color] of [
      ["red_wall", "red"],
      ["blue_wall", "blue"],
    ] as const) {
      const text = `a plain wall painted bright ${color}, solid ${color} color, ${color}, minimalism`;
      const image = decode(w, ksample(w, model, prompt(w, clip, text), negative, latent), vae);
      save(w, image, label);
      thumbnail(w, image, `${label}_thumb`, 8);
    }

    return (out) => {
      let [r, g, b] = out.picture("red_wall_thumb").mean();
      assert(r > g + 40 && r > b + 40, tuple([r, g, b]));
      [r, g, b] = out.picture("blue_wall_thumb").mean();
      assert(b > r + 40, tuple([r, g, b]));
    };
  },
);

addCase(
  "sdxl_batch_and_latent_picks",
  "Lote de 4 latentes no SDXL: LatentFromBatch pega o 2º e o 3º, RepeatLatentBatch " +
    "duplica o 4º; as miniaturas batem com as do lote original.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const positive = prompt(w, clip, "a hot air balloon over green hills, watercolor");
    const negative = prompt(w, clip, NEGATIVE);
    const batch = ksample(w, model, positive, negative, emptyLatent(w, 576, 576, 4), { steps: 12 });
    const images = decode(w, batch, vae);
    save(w, images, "batch");
    thumbnail(w, images, "batch_thumb", 24);
    const picked = w.addNode(LatentFromBatch, { samples: batch, batch_index: 1, length: 2 }).output.LATENT;
    thumbnail(w, decode(w, picked, vae), "picked_thumb", 24);
    const fourth = w.addNode(LatentFromBatch, { samples: batch, batch_index: 3, length: 1 }).output.LATENT;
    const repeated = w.addNode(RepeatLatentBatch, { samples: fourth, amount: 2 }).output.LATENT;
    thumbnail(w, decode(w, repeated, vae), "repeated_thumb", 24);

    return (out) => {
      const files = out.of("batch");
      assert(files.length === 4 && distinct(files) === 4);
      for (const image of files) {
        assertGenerated(image, [576, 576]);
      }
      const thumbs = out.pictures("batch_thumb");
      const picks = out.pictures("picked_thumb");
      assert(meanDifference(picks[0]!, thumbs[1]!) < 2);
      assert(meanDifference(picks[1]!, thumbs[2]!) < 2);
      assert(out.pictures("repeated_thumb").every((copy) => meanDifference(copy, thumbs[3]!) < 2));
    };
  },
);

addCase(
  "sdxl_two_stage_advanced_matches_single_pass",
  "KSamplerAdvanced em duas etapas (0-12 com ruído restante, 12-20 sem novo ruído) dá " +
    "o mesmo resultado que uma passada única de 20 passos com euler.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const positive = prompt(w, clip, "a red fox in the snow, wildlife photography");
    const negative = prompt(w, clip, NEGATIVE);
    const latent = emptyLatent(w, 640, 640);
    const advanced = (
      source: Link,
      addNoise: "enable" | "disable",
      start: number,
      end: number,
      leftover: "enable" | "disable",
    ) =>
      w.addNode(KSamplerAdvanced, {
        model,
        add_noise: addNoise,
        noise_seed: 33,
        steps: 20,
        cfg: 6.5,
        sampler_name: "euler",
        scheduler: "normal",
        positive,
        negative,
        latent_image: source,
        start_at_step: start,
        end_at_step: end,
        return_with_leftover_noise: leftover,
      }).output.LATENT;

    const first = advanced(latent, "enable", 0, 12, "enable");
    const twoStage = decode(w, advanced(first, "disable", 12, 10000, "disable"), vae);
    const single = decode(w, advanced(latent, "enable", 0, 10000, "disable"), vae);
    save(w, twoStage, "two_stage");
    save(w, single, "single");
    thumbnail(w, twoStage, "two_stage_thumb", 32);
    thumbnail(w, single, "single_thumb", 32);

    return (out) => {
      assertGenerated(out.one("two_stage"), [640, 640]);
      assertGenerated(out.one("single"), [640, 640]);
      const difference = meanDifference(out.picture("two_stage_thumb"), out.picture("single_thumb"));
      assert(difference < 3, String(difference));
    };
  },
);

addCase(
  "sdxl_sampler_custom_split_sigmas",
  "SamplerCustom com KarrasScheduler dividido por SplitSigmas em duas metades (a 2ª sem " +
    "ruído) equivale à passada completa; o scheduler exponencial difere.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const positive = prompt(w, clip, "a glass terrarium with moss and tiny mushrooms, macro photo");
    const negative = prompt(w, clip, NEGATIVE);
    const latent = emptyLatent(w, 640, 640);
    const sampler = w.addNode(KSamplerSelect, { sampler_name: "euler" }).output.SAMPLER;
    const sigmas = w.addNode(KarrasScheduler, { steps: 16 }).output.SIGMAS;
    const halves = w.addNode(SplitSigmas, { sigmas, step: 8 }).output;
    const custom = (source: Link, schedule: Link, addNoise: boolean) =>
      w.addNode(SamplerCustom, {
        model,
        positive,
        negative,
        sampler,
        sigmas: schedule,
        latent_image: source,
        add_noise: addNoise,
        noise_seed: 11,
        cfg: 6.5,
      }).output;

    const first = custom(latent, halves.high_sigmas, true);
    const split = decode(w, custom(first.output, halves.low_sigmas, false).output, vae);
    const full = decode(w, custom(latent, sigmas, true).output, vae);
    const exponential = w.addNode(ExponentialScheduler, { steps: 16 }).output.SIGMAS;
    const other = decode(w, custom(latent, exponential, true).output, vae);
    save(w, split, "split");
    save(w, decode(w, first.denoised_output, vae), "halfway_preview");
    for (const [label, image] of [
      ["split", split],
      ["full", full],
      ["exponential", other],
    ] as const) {
      thumbnail(w, image, `${label}_thumb`, 32);
    }

    return (out) => {
      assertGenerated(out.one("split"), [640, 640]);
      assert.deepEqual(pngSize(out.one("halfway_preview")), [640, 640]);
      const [halved, whole, different] = ["split", "full", "exponential"].map((label) => out.picture(`${label}_thumb`));
      assert(meanDifference(halved!, whole!) < 3);
      assert(meanDifference(whole!, different!) > 1);
    };
  },
);

addCase(
  "sdxl_custom_advanced_guiders",
  "SamplerCustomAdvanced com três guiders (CFGGuider, DualCFGGuider, BasicGuider), " +
    "samplers e schedulers diferentes: três imagens distintas.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const positive = prompt(w, clip, "an astronaut riding a horse on the moon, digital art");
    const style = prompt(w, clip, "vivid neon colors, synthwave");
    const negative = prompt(w, clip, NEGATIVE);
    const latent = emptyLatent(w, 640, 640);
    const noise = w.addNode(RandomNoise, { noise_seed: 21 }).output.NOISE;
    const guiders = {
      cfg: w.addNode(CFGGuider, { model, positive, negative, cfg: 6.0 }).output.GUIDER,
      dual: w.addNode(DualCFGGuider, {
        model,
        cond1: positive,
        cond2: style,
        negative,
        style: "regular",
        cfg_conds: 6.0,
        cfg_cond2_negative: 3.0,
      }).output.GUIDER,
      basic: w.addNode(BasicGuider, { model, conditioning: positive }).output.GUIDER,
    };
    const samplers = {
      cfg: w.addNode(SamplerEulerAncestral, { eta: 1.0, s_noise: 1.0 }).output.SAMPLER,
      dual: w.addNode(SamplerDPMPP_2M_SDE, { solver_type: "midpoint", noise_device: "gpu" }).output.SAMPLER,
      basic: w.addNode(KSamplerSelect, { sampler_name: "dpmpp_2m" }).output.SAMPLER,
    };
    const schedulers = { cfg: "karras", dual: "exponential", basic: "sgm_uniform" } as const;
    for (const label of ["cfg", "dual", "basic"] as const) {
      const sigmas = w.addNode(BasicScheduler, { model, scheduler: schedulers[label], steps: 14 });
      const sampled = w.addNode(SamplerCustomAdvanced, {
        noise,
        guider: guiders[label],
        sampler: samplers[label],
        sigmas: sigmas.output.SIGMAS,
        latent_image: latent,
      });
      save(w, decode(w, sampled.output.output, vae), label);
    }

    return (out) => {
      const images = ["cfg", "dual", "basic"].map((label) => out.one(label));
      for (const image of images) {
        assertGenerated(image, [640, 640]);
      }
      assert.equal(distinct(images), 3);
    };
  },
);

addCase(
  "sdxl_img2img_keeps_color_layout",
  "img2img no SDXL a partir de quadrantes coloridos desenhados no próprio ComfyUI " +
    "(denoise 0,5): o layout de cores sobrevive à geração.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const source = quadrants(w, 768, [RED, GREEN, BLUE, YELLOW]);
    const latent = w.addNode(VAEEncode, { pixels: source, vae }).output.LATENT;
    const positive = prompt(w, clip, "an abstract painting made of four large color blocks, thick brush strokes");
    const image = decode(
      w,
      ksample(w, model, positive, prompt(w, clip, NEGATIVE), latent, { denoise: 0.5, steps: 16 }),
      vae,
    );
    save(w, source, "source");
    save(w, image, "painted");
    thumbnail(w, image, "painted_thumb", 8);

    return (out) => {
      assertGenerated(out.one("painted"), [768, 768]);
      const thumb = out.picture("painted_thumb");
      let [r, g, b] = thumb.mean([0, 0, 4, 4]);
      assert(r > g + 30 && r > b + 30, `top left ${tuple([r, g, b])}`);
      [r, g, b] = thumb.mean([4, 0, 8, 4]);
      assert(g > r + 30 && g > b + 30, `top right ${tuple([r, g, b])}`);
      [r, g, b] = thumb.mean([0, 4, 4, 8]);
      assert(b > r + 30 && b > g + 30, `bottom left ${tuple([r, g, b])}`);
      [r, g, b] = thumb.mean([4, 4, 8, 8]);
      assert(r > b + 30 && g > b + 30, `bottom right ${tuple([r, g, b])}`);
    };
  },
);

addCase(
  "sdxl_hires_fix_upscales_latent",
  "Hires fix no SDXL: base 640, LatentUpscaleBy (bislerp 1,5x) e LatentUpscale com " +
    "crop central, cada um refinado com denoise parcial.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const positive = prompt(w, clip, "a cozy cabin in a snowy pine forest at night, warm windows");
    const negative = prompt(w, clip, NEGATIVE);
    const base = ksample(w, model, positive, negative, emptyLatent(w, 640, 640), { steps: 12 });
    save(w, decode(w, base, vae), "base");
    const bigger = w.addNode(LatentUpscaleBy, { samples: base, upscale_method: "bislerp", scale_by: 1.5 });
    const hires = ksample(w, model, positive, negative, bigger.output.LATENT, { steps: 10, denoise: 0.45 });
    save(w, decode(w, hires, vae), "hires");
    const resized = w.addNode(LatentUpscale, {
      samples: base,
      upscale_method: "nearest-exact",
      width: 832,
      height: 832,
      crop: "center",
    });
    const refined = ksample(w, model, positive, negative, resized.output.LATENT, { steps: 10, denoise: 0.4 });
    save(w, decode(w, refined, vae), "resized");

    return (out) => {
      assertGenerated(out.one("base"), [640, 640]);
      assertGenerated(out.one("hires"), [960, 960]);
      assertGenerated(out.one("resized"), [832, 832]);
    };
  },
);

function inpaintCanvas(w: Workflow): [canvas: Link, mask: Link] {
  const canvas = solid(w, 768, 768, GRAY);
  const mask = combineMasks(w, solidMask(w, 0.0, 768, 768), solidMask(w, 1.0, 256, 256), "add", 256, 256);
  return [canvas, mask];
}

addCase(
  "sdxl_inpaint_keeps_outside_pixels",
  "Inpainting com VAEEncodeForInpaint numa tela cinza com máscara central; o resultado " +
    "é colado de volta com a máscara e o lado de fora fica exato.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const [canvas, mask] = inpaintCanvas(w);
    const latent = w.addNode(VAEEncodeForInpaint, { pixels: canvas, vae, mask, grow_mask_by: 8 });
    const positive = prompt(w, clip, "a shiny red apple on a gray table, studio photo");
    const painted = decode(w, ksample(w, model, positive, prompt(w, clip, NEGATIVE), latent.output.LATENT), vae);
    const final = paste(w, canvas, painted, 0, 0, mask);
    save(w, final, "final");
    save(w, crop(w, final, 0, 0, 64, 64), "outside");
    save(w, crop(w, final, 352, 352, 64, 64), "inside");

    return (out) => {
      assertGenerated(out.one("final"), [768, 768]);
      assert.deepEqual([...colorsOf(out.picture("outside"))], [tuple([128, 128, 128])]);
      assert(colorsOf(out.picture("inside")).size > 10, "nothing was painted inside the mask");
    };
  },
);

addCase(
  "sdxl_inpaint_conditioning_and_noise_mask",
  "Inpainting com InpaintModelConditioning e com SetLatentNoiseMask: fora da máscara a " +
    "imagem continua cinza, dentro aparece conteúdo novo.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const [canvas, mask] = inpaintCanvas(w);
    const positive = prompt(w, clip, "a blue ceramic vase with flowers, studio photo");
    const negative = prompt(w, clip, NEGATIVE);
    const conditioned = w.addNode(InpaintModelConditioning, {
      positive,
      negative,
      vae,
      pixels: canvas,
      mask,
      noise_mask: true,
    }).output;
    const first = decode(w, ksample(w, model, conditioned.positive, conditioned.negative, conditioned.latent), vae);
    const encoded = w.addNode(VAEEncode, { pixels: canvas, vae }).output.LATENT;
    const masked = w.addNode(SetLatentNoiseMask, { samples: encoded, mask }).output.LATENT;
    const second = decode(w, ksample(w, model, positive, negative, masked, { denoise: 0.9 }), vae);
    for (const [label, image] of [
      ["conditioned", first],
      ["noise_masked", second],
    ] as const) {
      save(w, image, label);
      save(w, crop(w, image, 0, 0, 64, 64), `${label}_outside`);
      save(w, crop(w, image, 352, 352, 64, 64), `${label}_inside`);
    }

    return (out) => {
      for (const label of ["conditioned", "noise_masked"]) {
        assertGenerated(out.one(label), [768, 768]);
        const [r, g, b] = out.picture(`${label}_outside`).mean();
        assert(
          [r, g, b].every((value) => Math.abs(value - 128) < 25),
          `${label} ${tuple([r, g, b])}`,
        );
        assert(colorsOf(out.picture(`${label}_inside`)).size > 10, label);
      }
    };
  },
);

addCase(
  "sdxl_outpaint_extends_canvas",
  "Outpainting no SDXL: imagem gerada é estendida 256 px para a direita com " +
    "ImagePadForOutpaint e a área nova é preenchida pelo modelo.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const positive = prompt(w, clip, "a wide beach with palm trees and turquoise water, travel photo");
    const negative = prompt(w, clip, NEGATIVE);
    const image = decode(w, ksample(w, model, positive, negative, emptyLatent(w, 640, 640), { steps: 12 }), vae);
    const padded = w.addNode(ImagePadForOutpaint, { image, left: 0, top: 0, right: 256, bottom: 0, feathering: 40 });
    const latent = w.addNode(VAEEncodeForInpaint, {
      pixels: padded.output.IMAGE,
      vae,
      mask: padded.output.MASK,
      grow_mask_by: 16,
    });
    const extended = decode(w, ksample(w, model, positive, negative, latent.output.LATENT, { steps: 14 }), vae);
    save(w, padded.output.IMAGE, "padded");
    save(w, extended, "extended");

    return (out) => {
      assert.deepEqual(pngSize(out.one("padded")), [896, 640]);
      assertGenerated(out.one("extended"), [896, 640]);
    };
  },
);

addCase(
  "sdxl_regional_prompts_left_and_right",
  "Prompts regionais: tulipas vermelhas à esquerda (ConditioningSetArea / SetMask) e " +
    "lago azul à direita; a metade esquerda sai mais vermelha.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const negative = prompt(w, clip, NEGATIVE);
    const scene = prompt(w, clip, "a landscape painting, wide view");
    const red = prompt(w, clip, "a field of bright red tulips, red flowers");
    const blue = prompt(w, clip, "a deep blue lake under a clear blue sky");
    const latent = emptyLatent(w, 1024, 512);
    const combined = (...conditionings: Link[]) => {
      let result = conditionings[0]!;
      for (const other of conditionings.slice(1)) {
        result = w.addNode(ConditioningCombine, { conditioning_1: result, conditioning_2: other }).output.CONDITIONING;
      }
      return result;
    };

    const left = w.addNode(ConditioningSetArea, { conditioning: red, width: 512, height: 512, x: 0, y: 0, strength: 1.0 });
    const right = w.addNode(ConditioningSetArea, {
      conditioning: blue,
      width: 512,
      height: 512,
      x: 512,
      y: 0,
      strength: 1.0,
    });
    const areas = combined(scene, left.output.CONDITIONING, right.output.CONDITIONING);
    const leftMask = combineMasks(w, solidMask(w, 0.0, 1024, 512), solidMask(w, 1.0, 512, 512), "add");
    const rightMask = w.addNode(InvertMask, { mask: leftMask }).output.MASK;
    const maskedLeft = w.addNode(ConditioningSetMask, {
      conditioning: red,
      mask: leftMask,
      set_cond_area: "mask bounds",
      strength: 1.0,
    });
    const maskedRight = w.addNode(ConditioningSetMask, {
      conditioning: blue,
      mask: rightMask,
      set_cond_area: "mask bounds",
      strength: 1.0,
    });
    const masks = combined(scene, maskedLeft.output.CONDITIONING, maskedRight.output.CONDITIONING);
    for (const [label, positive] of [
      ["areas", areas],
      ["masks", masks],
    ] as const) {
      const image = decode(w, ksample(w, model, positive, negative, latent, { steps: 16 }), vae);
      save(w, image, label);
      save(w, scale(w, image, "area", 16, 8), `${label}_thumb`);
    }

    return (out) => {
      for (const label of ["areas", "masks"]) {
        assertGenerated(out.one(label), [1024, 512]);
        const thumb = out.picture(`${label}_thumb`);
        const [leftR, , leftB] = thumb.mean([0, 0, 8, 8]);
        const [rightR, , rightB] = thumb.mean([8, 0, 16, 8]);
        assert(leftR - leftB > rightR - rightB + 25, `${label} ${tuple([leftR, leftB, rightR, rightB])}`);
      }
    };
  },
);

addCase(
  "sdxl_conditioning_arithmetic",
  "Aritmética de condicionamento: ConditioningAverage, Concat, SetTimestepRange " +
    "combinados, Multiply e ZeroOut no negativo; quatro imagens distintas.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const mountain = prompt(w, clip, "a snowy mountain peak, dramatic clouds");
    const beach = prompt(w, clip, "a tropical beach with palm trees");
    const negative = w.addNode(ConditioningZeroOut, { conditioning: prompt(w, clip, NEGATIVE) }).output.CONDITIONING;
    const early = w.addNode(ConditioningSetTimestepRange, { conditioning: mountain, start: 0.0, end: 0.5 });
    const late = w.addNode(ConditioningSetTimestepRange, { conditioning: beach, start: 0.5, end: 1.0 });
    const positives = {
      average: w.addNode(ConditioningAverage, {
        conditioning_to: mountain,
        conditioning_from: beach,
        conditioning_to_strength: 0.5,
      }).output.CONDITIONING,
      concat: w.addNode(ConditioningConcat, { conditioning_to: mountain, conditioning_from: beach }).output.CONDITIONING,
      timesteps: w.addNode(ConditioningCombine, {
        conditioning_1: early.output.CONDITIONING,
        conditioning_2: late.output.CONDITIONING,
      }).output.CONDITIONING,
      multiplied: w.addNode(ConditioningMultiply, { conditioning: mountain, multiplier: 1.2 }).output.CONDITIONING,
    };
    const latent = emptyLatent(w, 640, 640);
    for (const [label, positive] of Object.entries(positives)) {
      save(w, decode(w, ksample(w, model, positive, negative, latent, { steps: 12 }), vae), label);
    }

    return (out) => {
      const images = Object.keys(positives).map((label) => out.one(label));
      for (const image of images) {
        assertGenerated(image, [640, 640]);
      }
      assert.equal(distinct(images), 4);
    };
  },
);

/** Patches a model, as a node or a few added to the workflow. */
type Patch = (w: Workflow, model: Link) => Link;

const UNET_PATCHES_A: Readonly<Record<string, Patch>> = {
  freeu: (w, model) => w.addNode(FreeU, { model }).output.MODEL,
  freeu_v2: (w, model) => w.addNode(FreeU_V2, { model }).output.MODEL,
  pag: (w, model) => w.addNode(PerturbedAttentionGuidance, { model, scale: 2.0 }).output.MODEL,
  // At 640 px the depth-1 attention is 40x40 tokens; 128 px tiles (16 latent) split
  // it 2x2, and swap_size 1 keeps that choice deterministic.
  hypertile_depth_1: (w, model) =>
    w.addNode(HyperTile, { model, tile_size: 128, swap_size: 1, max_depth: 1 }).output.MODEL,
};
// ToMe and HyperTile's default max_depth=0 only touch full-resolution attention,
// which SDXL does not have, so on SDXL they leave the image as it was.
const UNET_NO_OPS_A: Readonly<Record<string, Patch>> = {
  tome: (w, model) => w.addNode(TomePatchModel, { model, ratio: 0.3 }).output.MODEL,
  hypertile_depth_0: (w, model) => w.addNode(HyperTile, { model, tile_size: 256 }).output.MODEL,
};
const UNET_PATCHES_B: Readonly<Record<string, Patch>> = {
  deep_shrink: (w, model) =>
    w.addNode(PatchModelAddDownscale, { model, downscale_method: "bicubic", upscale_method: "bicubic" }).output.MODEL,
  epsilon: (w, model) => w.addNode(EpsilonScaling, { model, scaling_factor: 1.005 }).output.MODEL,
  rescale_cfg: (w, model) => w.addNode(RescaleCFG, { model, multiplier: 0.7 }).output.MODEL,
  sag: (w, model) => w.addNode(SelfAttentionGuidance, { model, scale: 0.5, blur_sigma: 2.0 }).output.MODEL,
  apg: (w, model) => w.addNode(APG, { model, eta: 1.0, norm_threshold: 5.0 }).output.MODEL,
  cfg_norm: (w, model) => w.addNode(CFGNorm, { model, strength: 1.0 }).output.patched_model,
  cfg_override: (w, model) => w.addNode(CFGOverride, { model, cfg: 3.0 }).output.MODEL,
};

function patchCase(
  w: Workflow,
  text: string,
  changed: Readonly<Record<string, Patch>>,
  unchanged: Readonly<Record<string, Patch>>,
): Check {
  const [model, clip, vae] = sdxl(w);
  const positive = prompt(w, clip, text);
  const negative = prompt(w, clip, NEGATIVE);
  const latent = emptyLatent(w, 640, 640);
  const models = new Map<string, Link>([["baseline", model]]);
  for (const [label, build] of [...Object.entries(changed), ...Object.entries(unchanged)]) {
    models.set(label, build(w, model));
  }
  for (const [label, patched] of models) {
    const image = decode(w, ksample(w, patched, positive, negative, latent, { steps: 12 }), vae);
    save(w, image, label);
    thumbnail(w, image, `${label}_thumb`, 32);
  }

  return (out) => {
    const baseline = out.one("baseline");
    assertGenerated(baseline, [640, 640]);
    for (const label of Object.keys(changed)) {
      const image = out.one(label);
      assertGenerated(image, [640, 640]);
      assert(!image.equals(baseline), `${label} did not change the image`);
    }
    for (const label of Object.keys(unchanged)) {
      const difference = meanDifference(out.picture(`${label}_thumb`), out.picture("baseline_thumb"));
      assert(difference < 1.5, `${label} changed the image by ${difference}`);
    }
  };
}

addCase(
  "sdxl_unet_patches_part_one",
  "Patches de UNet no SDXL contra a linha de base: FreeU, FreeU_V2, PAG e HyperTile " +
    "(tiles de 128, max_depth 1) mudam a imagem; ToMe e HyperTile com max_depth 0 não, " +
    "porque o SDXL não tem atenção na resolução máxima.",
  { gpu: true },
  (w) => patchCase(w, "a vintage red car parked on a cobblestone street", UNET_PATCHES_A, UNET_NO_OPS_A),
);

addCase(
  "sdxl_unet_patches_part_two",
  "Mais sete patches (Deep Shrink, Epsilon Scaling, RescaleCFG, SAG, APG, CFGNorm, " +
    "CFGOverride) e ModelSamplingDiscrete eps, que não pode mudar nada.",
  { gpu: true },
  (w) => {
    const same: Readonly<Record<string, Patch>> = {
      eps: (flow, model) => flow.addNode(ModelSamplingDiscrete, { model, sampling: "eps", zsnr: false }).output.MODEL,
    };
    return patchCase(w, "a steaming cup of coffee on a rainy window sill", UNET_PATCHES_B, same);
  },
);

addCase(
  "sdxl_merge_identity_and_clip_skip",
  "ModelMergeSimple e CLIPMergeSimple de um modelo consigo mesmo não mudam a imagem; " +
    "CLIPSetLastLayer -2 também não (já é o padrão do SDXL), mas -3 muda.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const text = "a portrait of an old fisherman, dramatic lighting, oil painting";
    const negative = prompt(w, clip, NEGATIVE);
    const latent = emptyLatent(w, 640, 640);
    const mergedModel = w.addNode(ModelMergeSimple, { model1: model, model2: model, ratio: 0.5 }).output.MODEL;
    const mergedClip = w.addNode(CLIPMergeSimple, { clip1: clip, clip2: clip, ratio: 0.5 }).output.CLIP;
    const lastLayer = (layer: number) => w.addNode(CLIPSetLastLayer, { clip, stop_at_clip_layer: layer }).output.CLIP;

    const variants = {
      baseline: [model, clip],
      merged_model: [mergedModel, clip],
      merged_clip: [model, mergedClip],
      clip_skip_2: [model, lastLayer(-2)],
      clip_skip_3: [model, lastLayer(-3)],
    } as const;
    for (const [label, [unet, encoder]] of Object.entries(variants)) {
      const image = decode(w, ksample(w, unet, prompt(w, encoder, text), negative, latent, { steps: 12 }), vae);
      save(w, image, label);
      thumbnail(w, image, `${label}_thumb`, 32);
    }

    return (out) => {
      const baseline = out.picture("baseline_thumb");
      for (const label of Object.keys(variants)) {
        assertGenerated(out.one(label), [640, 640]);
      }
      assert(meanDifference(out.picture("merged_model_thumb"), baseline) < 2);
      assert(meanDifference(out.picture("merged_clip_thumb"), baseline) < 2);
      // SDXL already encodes with the penultimate layer of both text encoders.
      assert(meanDifference(out.picture("clip_skip_2_thumb"), baseline) < 2);
      assert(meanDifference(out.picture("clip_skip_3_thumb"), baseline) > 3);
    };
  },
);

addCase(
  "sdxl_size_conditioning",
  "CLIPTextEncodeSDXL com condicionamento de tamanho e de recorte diferentes: três " +
    "imagens distintas a partir do mesmo texto.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const negative = prompt(w, clip, NEGATIVE);
    const latent = emptyLatent(w, 768, 768);
    const variants = {
      full_size: { width: 1024, height: 1024, crop_w: 0, crop_h: 0 },
      cropped: { width: 1024, height: 1024, crop_w: 512, crop_h: 512 },
      low_res: { width: 512, height: 512, crop_w: 0, crop_h: 0 },
    };
    for (const [label, sizes] of Object.entries(variants)) {
      const positive = w.addNode(CLIPTextEncodeSDXL, {
        clip,
        text_g: "a majestic owl perched on a branch, detailed feathers",
        text_l: "owl, branch, forest",
        target_width: 1024,
        target_height: 1024,
        ...sizes,
      }).output.CONDITIONING;
      save(w, decode(w, ksample(w, model, positive, negative, latent, { steps: 12 }), vae), label);
    }

    return (out) => {
      const images = Object.keys(variants).map((label) => out.one(label));
      for (const image of images) {
        assertGenerated(image, [768, 768]);
      }
      assert.equal(distinct(images), 3);
    };
  },
);

addCase(
  "sdxl_latent_operations",
  "Operações de latente sobre uma geração 640x512: flip, rotação, crop, composite, " +
    "blend, interpolate, tonemap, sharpen e (a+a)*0,5 = a.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const negative = prompt(w, clip, NEGATIVE);
    const latent = emptyLatent(w, 640, 512);
    const base = ksample(w, model, prompt(w, clip, "a sailboat on a calm sea at dawn"), negative, latent, { steps: 12 });
    const other = ksample(w, model, prompt(w, clip, "a field of sunflowers under a blue sky"), negative, latent, {
      steps: 12,
    });

    const doubled = w.addNode(LatentAdd, { samples1: base, samples2: base }).output.LATENT;
    const variants = {
      base,
      flipped: w.addNode(LatentFlip, { samples: base, flip_method: "y-axis: horizontally" }).output.LATENT,
      rotated: w.addNode(LatentRotate, { samples: base, rotation: "90 degrees" }).output.LATENT,
      cropped: w.addNode(LatentCrop, { samples: base, width: 384, height: 384, x: 64, y: 64 }).output.LATENT,
      doubled_half: w.addNode(LatentMultiply, { samples: doubled, multiplier: 0.5 }).output.LATENT,
      blended: w.addNode(LatentBlend, { samples1: base, samples2: other, blend_factor: 0.5 }).output.LATENT,
      interpolated: w.addNode(LatentInterpolate, { samples1: base, samples2: other, ratio: 0.5 }).output.LATENT,
      tonemapped: w.addNode(LatentApplyOperation, {
        samples: base,
        operation: w.addNode(LatentOperationTonemapReinhard, { multiplier: 1.0 }).output.LATENT_OPERATION,
      }).output.LATENT,
      sharpened: w.addNode(LatentApplyOperation, {
        samples: base,
        operation: w.addNode(LatentOperationSharpen, { sharpen_radius: 9, sigma: 1.0, alpha: 0.1 }).output
          .LATENT_OPERATION,
      }).output.LATENT,
      composited: w.addNode(LatentComposite, {
        samples_to: base,
        samples_from: w.addNode(LatentCrop, { samples: other, width: 256, height: 256, x: 0, y: 0 }).output.LATENT,
        x: 64,
        y: 64,
        feather: 16,
      }).output.LATENT,
      subtracted: w.addNode(LatentSubtract, { samples1: base, samples2: base }).output.LATENT,
    };
    for (const [label, samples] of Object.entries(variants)) {
      const image = decode(w, samples, vae);
      save(w, image, label);
      if (label === "base" || label === "flipped" || label === "doubled_half") {
        thumbnail(w, image, `${label}_thumb`, 16);
      }
    }

    return (out) => {
      const sizes: Readonly<Record<string, readonly [number, number]>> = { rotated: [512, 640], cropped: [384, 384] };
      for (const label of Object.keys(variants)) {
        const size = sizes[label] ?? [640, 512];
        if (label === "subtracted") {
          assert.deepEqual(pngSize(out.one(label)), size);
        } else {
          assertGenerated(out.one(label), size, 30_000);
        }
      }
      const thumb = out.picture("base_thumb");
      assert(meanDifference(out.picture("doubled_half_thumb"), thumb) < 1);
      assert(meanDifference(out.picture("flipped_thumb"), thumb.mirrored()) < 25);
    };
  },
);

addCase(
  "sdxl_tiled_vae_round_trip",
  "VAEDecodeTiled e VAEEncodeTiled numa geração 1024x1024: o decode em tiles bate com o " +
    "normal e o encode em tiles alimenta um refinamento.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const positive = prompt(w, clip, "an ornate gothic cathedral interior, stained glass, light rays");
    const negative = prompt(w, clip, NEGATIVE);
    const base = ksample(w, model, positive, negative, emptyLatent(w, 1024, 1024), { steps: 12 });
    const full = decode(w, base, vae);
    const tiles = { tile_size: 512, overlap: 64, temporal_size: 64, temporal_overlap: 8 };
    const tiled = w.addNode(VAEDecodeTiled, { samples: base, vae, ...tiles }).output.IMAGE;
    const encoded = w.addNode(VAEEncodeTiled, { pixels: tiled, vae, ...tiles }).output.LATENT;
    const refined = decode(w, ksample(w, model, positive, negative, encoded, { steps: 8, denoise: 0.3 }), vae);
    save(w, tiled, "tiled");
    save(w, refined, "refined");
    thumbnail(w, full, "full_thumb", 32);
    thumbnail(w, tiled, "tiled_thumb", 32);

    return (out) => {
      assertGenerated(out.one("tiled"), [1024, 1024]);
      assertGenerated(out.one("refined"), [1024, 1024]);
      assert(meanDifference(out.picture("tiled_thumb"), out.picture("full_thumb")) < 4);
    };
  },
);

addCase(
  "sdxl_frames_to_video",
  "Seis gerações SDXL em lote (LatentBatchSeedBehavior random) viram um MP4 H.264, um " +
    "WEBP animado e voltam como quadros via GetVideoComponents.",
  { gpu: true },
  (w) => {
    const [model, clip, vae] = sdxl(w);
    const positive = prompt(w, clip, "a jellyfish glowing in the deep ocean, bioluminescence");
    const negative = prompt(w, clip, NEGATIVE);
    const batch = w.addNode(LatentBatchSeedBehavior, { samples: emptyLatent(w, 512, 512, 6), seed_behavior: "random" });
    const frames = decode(w, ksample(w, model, positive, negative, batch.output.LATENT, { steps: 10 }), vae);
    const video = w.addNode(CreateVideo, { images: frames, fps: 6.0 }).output.VIDEO;
    w.addNode(SaveVideo, { video, filename_prefix: "clip", format: "mp4", format_codec: "h264" });
    w.addNode(SaveAnimatedWEBP, {
      images: frames,
      method: "default",
      filename_prefix: "loop",
      fps: 6.0,
      lossless: false,
      quality: 80,
    });
    const parts = w.addNode(GetVideoComponents, { video }).output.images;
    const third = w.addNode(ImageFromBatch, { image: parts, batch_index: 3, length: 1 });
    save(w, third.output.IMAGE, "frame_3");

    return (out) => {
      assertMp4(out.one("clip"), "avc1");
      assert.deepEqual(webpInfo(out.one("loop")), [[512, 512], 6]);
      assertGenerated(out.one("frame_3"), [512, 512]);
    };
  },
);

addCase(
  "sdxl_postprocess_pipeline",
  "Pipeline TextToImage + ImageSharpen + TextOverlay + ImageScaleToMaxDimension + Canny " +
    "costurados num pôster: tamanhos conferidos em cada etapa.",
  { gpu: true },
  (w) => {
    const image = w.addPipeline(TextToImage, {
      text: "a futuristic city skyline at dusk, flying cars",
      width: 896,
      height: 640,
    }).output.IMAGE;
    const sharp = w.addNode(ImageSharpen, { image, sharpen_radius: 1, sigma: 1.0, alpha: 0.3 }).output.IMAGE;
    const titled = w.addNode(TextOverlay, {
      images: sharp,
      text: "ComfyUI via RunPod",
      font_size: 6.0,
      color: "#ffcc00",
      position: "bottom",
      align: "right",
      outline: true,
    }).output.images;
    const small = w.addNode(ImageScaleToMaxDimension, { image: titled, upscale_method: "lanczos", largest_size: 512 });
    const edges = w.addNode(Canny, { image: small.output.IMAGE, low_threshold: 0.2, high_threshold: 0.5 });
    save(w, titled, "titled");
    save(w, small.output.IMAGE, "small");
    save(w, stitch(w, small.output.IMAGE, edges.output.IMAGE, "down"), "poster");

    return (out) => {
      assertGenerated(out.one("titled"), [896, 640]);
      assert.deepEqual(pngSize(out.one("small")), [512, 366]);
      assert.deepEqual(pngSize(out.one("poster")), [512, 732]);
    };
  },
);
