// The pipelines the other examples add: groups of nodes with parameters and
// named outputs, added all or nothing by Workflow.addPipeline.
import { Pipeline, type Link } from "comfy-ui-api-workflow";
import { SaveImage } from "comfy-ui-api-workflow/nodes/image";
import { ImageScaleBy } from "comfy-ui-api-workflow/nodes/image/upscaling";
import { CLIPTextEncode } from "comfy-ui-api-workflow/nodes/model/conditioning";
import { EmptyLatentImage, VAEDecode } from "comfy-ui-api-workflow/nodes/model/latent";
import { CheckpointLoaderSimple } from "comfy-ui-api-workflow/nodes/model/loaders";
import { KSampler } from "comfy-ui-api-workflow/nodes/model/sampling";
import { float, int, link, string } from "comfy-ui-api-workflow/nodes/schema";
import { CreateVideo, SaveVideo } from "comfy-ui-api-workflow/nodes/video";

export class TextToImage extends Pipeline {
  // Checked before build runs: an unknown, missing or mistyped parameter
  // rejects the pipeline. Those left out take the defaults of build.
  static override readonly PARAMETERS = {
    prompt: string(),
    negative: string({ optional: true }),
    checkpoint: string({ optional: true }),
    size: int({ optional: true }),
    steps: int({ optional: true }),
    seed: int({ optional: true }),
  };

  override build({
    prompt,
    negative = "blurry, low quality, distorted",
    checkpoint = "sd_xl_base_1.0.safetensors",
    size = 1024,
    steps = 25,
    seed = 42,
  }: {
    prompt: string;
    negative?: string;
    checkpoint?: string;
    size?: number;
    steps?: number;
    seed?: number;
  }) {
    const model = this.addNode(CheckpointLoaderSimple, { ckpt_name: checkpoint });
    const positiveText = this.addNode(CLIPTextEncode, { text: prompt, clip: model.output.CLIP });
    const negativeText = this.addNode(CLIPTextEncode, { text: negative, clip: model.output.CLIP });
    const latent = this.addNode(EmptyLatentImage, { width: size, height: size, batch_size: 1 });
    const sampler = this.addNode(KSampler, {
      model: model.output.MODEL,
      sampler_name: "dpmpp_2m",
      scheduler: "karras",
      positive: positiveText.output.CONDITIONING,
      negative: negativeText.output.CONDITIONING,
      latent_image: latent.output.LATENT,
      seed,
      steps,
      cfg: 7.0,
    });
    const image = this.addNode(VAEDecode, { samples: sampler.output.LATENT, vae: model.output.VAE });
    return { IMAGE: image.output.IMAGE };
  }
}

export class Upscale extends Pipeline {
  static override readonly PARAMETERS = { image: link(), scale: float({ optional: true }) };

  override build({ image, scale = 2.0 }: { image: Link; scale?: number }) {
    const scaled = this.addNode(ImageScaleBy, { image, upscale_method: "lanczos", scale_by: scale });
    return { IMAGE: scaled.output.IMAGE };
  }
}

export class SaveImages extends Pipeline {
  static override readonly PARAMETERS = { images: link(), prefix: string({ optional: true }) };

  override build({ images, prefix = "ComfyUI" }: { images: Link; prefix?: string }) {
    this.addNode(SaveImage, { images, filename_prefix: prefix });
  }
}

export class ImagesToVideo extends Pipeline {
  static override readonly PARAMETERS = {
    frames: link(),
    fps: float({ optional: true }),
    prefix: string({ optional: true }),
  };

  override build({ frames, fps = 8.0, prefix = "video/ComfyUI" }: { frames: Link; fps?: number; prefix?: string }) {
    const video = this.addNode(CreateVideo, { images: frames, fps });
    this.addNode(SaveVideo, {
      video: video.output.VIDEO,
      filename_prefix: prefix,
      format: "mp4",
      format_codec: "h264",
    });
    return { VIDEO: video.output.VIDEO };
  }
}
