// The special workflow of the Python test_runpod_workflows.py: two SDXL scenes,
// sampled and then refined at a larger size.
import { LatentUpscaleBy } from "../../src/nodes/model/latent.js";
import { Workflow } from "../../src/workflow.js";
import { decode, emptyLatent, ksample, prompt, save, sdxl } from "./blocks.js";

export const SPECIAL_NEGATIVE =
  "nsfw, nude, naked, swimsuit, bikini, underwear, blurry, low quality, deformed, " +
  "bad anatomy, extra limbs, extra fingers, watermark, text";

export const SPECIAL_SCENES = [
  [
    "menino_correndo_na_rua",
    "a cheerful young boy running down the middle of a quiet city street, motion, " +
      "sneakers, t-shirt and shorts, sunny afternoon, cinematic photo, shallow depth of " +
      "field, highly detailed",
    1001,
  ],
  [
    "menina_na_lagoa",
    "a young girl wearing a yellow summer dress standing on the grassy shore of a calm " +
      "lagoon, skipping a stone across the water, lush green trees, golden hour, " +
      "cinematic photo, highly detailed, family friendly",
    2002,
  ],
] as const;

export function specialWorkflow(): Workflow {
  const w = new Workflow();
  const [model, clip, vae] = sdxl(w);
  const negative = prompt(w, clip, SPECIAL_NEGATIVE);
  for (const [label, text, seed] of SPECIAL_SCENES) {
    const positive = prompt(w, clip, text);
    const base = ksample(w, model, positive, negative, emptyLatent(w, 1216, 832), { seed, steps: 30, cfg: 7.0 });
    const larger = w.addNode(LatentUpscaleBy, { samples: base, upscale_method: "bislerp", scale_by: 1.25 });
    const refined = ksample(w, model, positive, negative, larger.output.LATENT, { seed, steps: 18, denoise: 0.4 });
    save(w, decode(w, refined, vae), label);
  }
  return w;
}
