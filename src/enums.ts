/** ComfyUI type names of common outputs. */
export const OutputTypes = {
  MODEL: "MODEL",
  CONDITIONING: "CONDITIONING",
  LATENT: "LATENT",
  CLIP: "CLIP",
  VAE: "VAE",
} as const;

export type OutputTypes = (typeof OutputTypes)[keyof typeof OutputTypes];
