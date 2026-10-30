// The failure cases of the Python test_runpod_workflows.py: workflows ComfyUI
// refuses or fails, refused by the library too when it can tell.
import {
  CheckpointLoaderSimple,
  DualCLIPLoader,
  LoraLoader,
  UNETLoader,
  UpscaleModelLoader,
  VAELoader,
} from "../../src/nodes/model/loaders.js";
import { ComfyNumberConvert, PreviewAny } from "../../src/nodes/utilities.js";
import { PrimitiveString } from "../../src/nodes/utilities/primitive.js";
import { calc, CHECKPOINT, decode, emptyLatent, ksample, NEGATIVE, prompt, save, solid } from "./blocks.js";
import { RED } from "./media.js";
import { addCase, addRejected } from "./registry.js";

addRejected(
  "comfy_rejects_value_below_minimum",
  "Largura 0 num EmptyImage: a biblioteca recusa e o ComfyUI também ('smaller than min').",
  { librarySays: "width", comfySays: "smaller than min of 1" },
  () => ({
    "1": { class_type: "EmptyImage", inputs: { width: 0, height: 8, batch_size: 1, color: 0 } },
    "2": { class_type: "SaveImage", inputs: { images: ["1", 0], filename_prefix: "never" } },
  }),
);

addRejected(
  "comfy_rejects_unknown_combo_value",
  "blend_mode inexistente no ImageBlend: recusado pela biblioteca e pelo ComfyUI ('Value not in list').",
  { librarySays: "blend_mode", comfySays: "Value not in list" },
  () => ({
    "1": { class_type: "EmptyImage", inputs: { width: 8, height: 8, batch_size: 1, color: 0 } },
    "2": {
      class_type: "ImageBlend",
      inputs: { image1: ["1", 0], image2: ["1", 0], blend_factor: 0.5, blend_mode: "glow" },
    },
    "3": { class_type: "SaveImage", inputs: { images: ["2", 0], filename_prefix: "never" } },
  }),
);

addRejected(
  "comfy_rejects_missing_required_input",
  "ImageInvert sem a entrada obrigatória image: recusado pelos dois lados ('Required input is missing').",
  { librarySays: "image", comfySays: "Required input is missing" },
  () => ({
    "1": { class_type: "ImageInvert", inputs: {} },
    "2": { class_type: "SaveImage", inputs: { images: ["1", 0], filename_prefix: "never" } },
  }),
);

addRejected(
  "comfy_rejects_unknown_node_type",
  "class_type que não existe: a biblioteca e o ComfyUI apontam o nó desconhecido.",
  { librarySays: "NotARealNode", comfySays: "NotARealNode" },
  () => ({
    "1": { class_type: "NotARealNode", inputs: {} },
    "2": { class_type: "SaveImage", inputs: { images: ["1", 0], filename_prefix: "never" } },
  }),
);

addRejected(
  "comfy_rejects_link_to_missing_node",
  "Link para um nó que não está no workflow: recusado pela biblioteca e o ComfyUI falha ao validar.",
  { librarySays: "ghost", comfySays: "Exception when validating node" },
  () => ({
    "1": { class_type: "SaveImage", inputs: { images: ["ghost", 0], filename_prefix: "never" } },
  }),
);

addRejected(
  "comfy_rejects_image_into_mask",
  "Imagem ligada numa entrada MASK: a biblioteca e o ComfyUI recusam o tipo errado.",
  {
    librarySays: "which is IMAGE, but the input takes MASK",
    comfySays: "received_type(IMAGE) mismatch input_type(MASK)",
  },
  () => ({
    "1": { class_type: "EmptyImage", inputs: { width: 8, height: 8, batch_size: 1, color: 0 } },
    "2": { class_type: "MaskToImage", inputs: { mask: ["1", 0] } },
    "3": { class_type: "SaveImage", inputs: { images: ["2", 0], filename_prefix: "never" } },
  }),
);

addCase(
  "division_by_zero_fails_at_runtime",
  "Divisão por zero no ComfyMathExpression: a biblioteca aceita o grafo, mas o ComfyUI " +
    "falha na execução com a mensagem do Python.",
  { failsWith: "division by zero" },
  (w) => {
    save(w, solid(w, calc(w, "a / b", { a: 64, b: 0 }).INT, 8, RED), "never");
  },
);

addCase(
  "number_convert_rejects_text",
  "ComfyNumberConvert recebe o texto 'doze': falha em tempo de execução com a mensagem do ComfyUI.",
  { failsWith: "Cannot convert string to number" },
  (w) => {
    const text = w.addNode(PrimitiveString, { value: "doze" }).output.STRING;
    const number = w.addNode(ComfyNumberConvert, { value: text }).output.INT;
    save(w, solid(w, number, 8, RED), "never");
  },
);

addCase(
  "worker_lists_models_on_placeholder",
  "Placeholder '__list__' nos loaders de checkpoint, LoRA, VAE, upscale, modelo de " +
    "difusão e text encoders (DualCLIPLoader): o worker responde com o inventário de " +
    "cada pasta (registrado no result.json).",
  { failsWith: [CHECKPOINT, "diffusion_models", "text_encoders"] },
  (w) => {
    const model = w.addNode(CheckpointLoaderSimple, { ckpt_name: "__list__" });
    const lora = w.addNode(LoraLoader, {
      model: model.output.MODEL,
      clip: model.output.CLIP,
      lora_name: "__list__",
      strength_model: 1.0,
      strength_clip: 1.0,
    });
    const vae = w.addNode(VAELoader, { vae_name: "__list__" });
    const upscaler = w.addNode(UpscaleModelLoader, { model_name: "__list__" });
    const unet = w.addNode(UNETLoader, { unet_name: "__list__", weight_dtype: "default" });
    const textEncoders = w.addNode(DualCLIPLoader, { clip_name1: "__list__", clip_name2: "__list__", type: "sdxl" });
    for (const output of [
      lora.output.MODEL,
      vae.output.VAE,
      upscaler.output.UPSCALE_MODEL,
      unet.output.MODEL,
      textEncoders.output.CLIP,
    ]) {
      w.addNode(PreviewAny, { source: output });
    }
  },
);

addCase(
  "missing_checkpoint_is_reported",
  "Checkpoint inexistente: o worker recusa antes de rodar, nomeia o arquivo que faltou " +
    "e lista os checkpoints disponíveis.",
  { failsWith: ["nao_existe.safetensors", CHECKPOINT] },
  (w) => {
    const loader = w.addNode(CheckpointLoaderSimple, { ckpt_name: "nao_existe.safetensors" });
    const positive = prompt(w, loader.output.CLIP, "a cat");
    const negative = prompt(w, loader.output.CLIP, NEGATIVE);
    const latent = ksample(w, loader.output.MODEL, positive, negative, emptyLatent(w, 512, 512));
    save(w, decode(w, latent, loader.output.VAE), "never");
  },
);
