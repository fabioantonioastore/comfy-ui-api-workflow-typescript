# comfy-ui-api-workflow for TypeScript

Build [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows in
TypeScript and export them in the API format that ComfyUI's `POST /prompt`
endpoint runs. A port of the Python library
[comfy-ui-api-workflow](https://github.com/fabioantonioastore/comfy-ui-api-workflow)
0.0.9, tested against what the Python library sends for every node class.

- **665 ComfyUI nodes as typed classes**, from `KSampler` to Wan, LTXV, Flux, 3D
  and partner API nodes.
- **Checked as you build**: input types, `min`/`max` ranges, combo choices,
  output names, socket types (no `LATENT` into a `CLIP` input) and cycles. A bad
  workflow fails in your code with a clear message, not in the ComfyUI queue.
  With TypeScript, many mistakes do not even compile.
- **Pipelines**: reusable groups of nodes with checked parameters and named
  outputs, added all or nothing.
- **Round trip**: load a workflow saved with ComfyUI's *Export (API)*, change
  it, export it again.
- No runtime dependencies, and no Node.js APIs. ESM with types, Node.js 20+.

## Installation

```bash
npm install comfy-ui-api-workflow
```

To install a commit that is not on npm yet, install from GitHub, where npm
builds the package on install. The code lives on the `typescript-port` branch
for now:

```bash
npm install github:fabioantonioastore/comfy-ui-api-workflow-typescript#typescript-port
```

npm 12 fetches git dependencies only when the project allows them, so with it,
run `npm config set allow-git=root --location=project` first. npm then warns
that the package's `prepare` script was blocked: that is its build, which
already ran when npm fetched the package.

## Quick start

```ts
import { Workflow, nodes } from "comfy-ui-api-workflow";

const workflow = new Workflow();
const checkpoint = workflow.addNode(nodes.model.loaders.CheckpointLoaderSimple, {
  ckpt_name: "sd_xl_base_1.0.safetensors",
});
const prompt = workflow.addNode(nodes.model.conditioning.CLIPTextEncode, {
  text: "a lighthouse at dusk",
  clip: checkpoint.output.CLIP,
});
const negative = workflow.addNode(nodes.model.conditioning.CLIPTextEncode, {
  text: "blurry",
  clip: checkpoint.output.CLIP,
});
const latent = workflow.addNode(nodes.model.latent.EmptyLatentImage, { width: 1024, height: 1024, batch_size: 1 });
const sampler = workflow.addNode(nodes.model.sampling.KSampler, {
  model: checkpoint.output.MODEL,
  positive: prompt.output.CONDITIONING,
  negative: negative.output.CONDITIONING,
  latent_image: latent.output.LATENT,
  sampler_name: "dpmpp_2m",
  scheduler: "karras",
  steps: 25,
  cfg: 7.0,
});
const image = workflow.addNode(nodes.model.latent.VAEDecode, {
  samples: sampler.output.LATENT,
  vae: checkpoint.output.VAE,
});
workflow.addNode(nodes.image.SaveImage, { images: image.output.IMAGE, filename_prefix: "lighthouse" });

console.log(JSON.stringify(workflow.getComfyUIWorkflowFormat(), null, 2));
```

The printed JSON is what ComfyUI's `/prompt` endpoint takes as `prompt`.

Mistakes are caught right away:

```ts
import { Workflow, errors, nodes } from "comfy-ui-api-workflow";

const workflow = new Workflow();
const latent = workflow.addNode(nodes.model.latent.EmptyLatentImage, { width: 512, height: 512 });

try {
  latent.width = 8;
} catch (error) {
  if (!(error instanceof errors.node.InvalidNodeParameterValue)) throw error;
  console.log(error.message); // width must be in closed range: [16, 16384]
}
```

With TypeScript, many mistakes do not even compile: a combo input only takes its
options (`sampler_name: "eular"` is a type error), outputs are named
(`checkpoint.output.LATENT` is one too) and required inputs must be given. The
checks still run, for what types cannot tell, such as ranges, socket types and
cycles, and for JavaScript callers.

## Pipelines

A pipeline adds a group of nodes from its parameters and returns some of their
outputs. `Workflow.addPipeline` adds all of its nodes or, if any is rejected,
none.

```ts
import { Pipeline, Workflow, nodes, type Link } from "comfy-ui-api-workflow";
import { float, link } from "comfy-ui-api-workflow/nodes/schema";

class Upscale extends Pipeline {
  static override readonly PARAMETERS = { image: link(), scale: float({ optional: true }) };

  override build({ image, scale = 2.0 }: { image: Link; scale?: number }) {
    const scaled = this.addNode(nodes.image.upscaling.ImageScaleBy, {
      image,
      upscale_method: "lanczos",
      scale_by: scale,
    });
    return { IMAGE: scaled.output.IMAGE };
  }
}

const workflow = new Workflow();
const photo = workflow.addNode(nodes.image.LoadImage, { image: "photo.png" });
const upscaled = workflow.addPipeline(Upscale, { image: photo.output.IMAGE, scale: 1.5 });
workflow.addNode(nodes.image.SaveImage, { images: upscaled.output.IMAGE, filename_prefix: "upscaled" });

console.log([...workflow.nodes.keys()].join(", ")); // LoadImage_0, ImageScaleBy_1, SaveImage_2
```

`PARAMETERS` declares what `build` takes, checked before it runs: an unknown,
missing or mistyped parameter rejects the pipeline. Without it, `build` takes
any parameters.

## Loading a workflow

```ts
import { Workflow } from "comfy-ui-api-workflow";

// A workflow saved with ComfyUI's Export (API), read with JSON.parse.
const exported = {
  "1": { class_type: "LoadImage", inputs: { image: "photo.png" } },
  "2": { class_type: "ImageScaleBy", inputs: { image: ["1", 0], upscale_method: "lanczos", scale_by: 2 } },
  "3": { class_type: "SaveImage", inputs: { images: ["2", 0], filename_prefix: "ComfyUI" } },
};

const workflow = new Workflow();
workflow.loadWorkflow(exported);
workflow.updateNode("2", { scale_by: 1.5 });

console.log(workflow.getComfyUIWorkflowFormat()["2"]?.inputs["scale_by"]); // 1.5
```

A loaded workflow is checked like one built in code, and keeps its node ids.

## Custom nodes

Nodes from custom node packs are classes too. `loadWorkflow` only knows the
library's nodes, but `addNode` takes any node class:

```ts
import { Workflow, nodes, type Link } from "comfy-ui-api-workflow";
import { BaseNode } from "comfy-ui-api-workflow/nodes/base";
import { float, link } from "comfy-ui-api-workflow/nodes/schema";

interface SharpenInputs {
  image: Link;
  strength?: number | Link;
}

class Sharpen extends BaseNode<SharpenInputs, "IMAGE"> {
  // The node type ComfyUI knows; the class name by default.
  static override readonly CLASS_TYPE = "Sharpen (Custom)";
  static override readonly INPUTS = {
    image: link(),
    strength: float({ link: true, default: 0.5, range: [0, 1] }),
  };
  static override readonly OUTPUTS = { IMAGE: 0 };
  // Optional: with them, links are checked against ComfyUI's socket types.
  static override readonly RETURN_TYPES = ["IMAGE"];
  static override readonly INPUT_SOCKET_TYPES = { image: "IMAGE", strength: "FLOAT" };
  declare strength: number | Link;
}

const workflow = new Workflow();
const photo = workflow.addNode(nodes.image.LoadImage, { image: "photo.png" });
const sharpened = workflow.addNode(Sharpen, { image: photo.output.IMAGE });
sharpened.strength = 0.8;

console.log(JSON.stringify(sharpened.inputs())); // {"image":["LoadImage_0",0],"strength":0.8}
```

## Examples

[`examples/`](examples) has the scripts of the Python library, with nodes and
pipelines. Run one with [tsx](https://tsx.is):

```bash
npx tsx examples/workflow_with_nodes.ts
```

## Differences from the Python library

The node classes, their inputs and outputs, the checks and the messages are
the Python library's. What changes is what TypeScript calls for:

| Python | TypeScript |
| --- | --- |
| `workflow.add_node(KSampler, seed=42)` | `workflow.addNode(KSampler, { seed: 42 })` |
| `workflow.update_node(node_id, steps=30)` | `workflow.updateNode(nodeId, { steps: 30 })` |
| `node.output["CLIP"]` | `node.output.CLIP` |
| `("CheckpointLoaderSimple_0", 1)` | `["CheckpointLoaderSimple_0", 1]` |
| `node.mask = None` | `node.mask = undefined`, or `null` |
| `def build(self, prompt, steps=25)` | `static PARAMETERS` and `build({ prompt, steps = 25 })` |

Methods are in camelCase (`getComfyUIWorkflowFormat`, `loadWorkflow`), while
inputs, outputs, class types and module paths keep their Python names
(`ckpt_name`, `output.CLIP`, `nodes.model.sampling`). Besides:

- **Integers** are JavaScript numbers, so integer inputs take safe integers: a
  seed goes up to `Number.MAX_SAFE_INTEGER` (2^53 - 1), where Python takes up
  to 2^64 - 1.
- **Messages** name JavaScript types (`an integer`, `a number`, `null`,
  `undefined`, `array`) and numbers (`[0.01, 8]`, where Python prints
  `[0.01, 8.0]`).
- **Loading order**: `loadWorkflow` reads the nodes in the order JavaScript
  gives the keys of an object, where integer ids such as `"3"` and `"10"` come
  first, in increasing order. When a workflow has several problems, the one
  reported can differ from Python's.
- **Ids and keys** are strings: node ids, input names and workflow keys.
- **`DA3Render.output_`**: the input that ComfyUI calls `output` would hide the
  outputs every node has, so here it is `output_`. It is still sent as `output`.
- **Pickling and copying** nodes has no counterpart.

## Bundling

`Workflow` imports all 665 node classes, since `loadWorkflow` can meet any of
them. Node ids (`KSampler_0`), pipeline names in messages and the class type of
a custom node without `CLASS_TYPE` come from class names, so keep them when
minifying: `keepNames: true` in esbuild, `keep_classnames: true` in Terser.

## Development

```bash
npm install        # also builds dist/
npm test           # unit tests (RunPod tests are skipped)
npm run coverage   # with coverage
npm run typecheck
RUNPOD=1 npm test  # also send real jobs to the RunPod endpoint in .env (paid GPU)
```

The RunPod tests run workflows on a [worker-comfyui](https://github.com/runpod-workers/worker-comfyui)
endpoint, set with `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` in `.env` or the
environment. The files the jobs return go to `runpod_outputs/`. Three suites port
the Python library's RunPod suites: a hundred workflows
(`tests/runpod_workflows.test.ts`), ten 30-second videos with sound
(`tests/runpod_videos.test.ts`) and a one-minute WAN 2.2 video of a lion in the
savanna (`tests/runpod_lion.test.ts`). Each job leaves a folder in
`runpod_outputs/workflows`, which `index.html` shows. The videos are measured with
`ffmpeg` and `ffprobe`; without them, those tests are skipped.

The node classes are generated from the Python library, which is expected next
to this project in `../comfy-ui-api-workflow`:

```bash
python3 -B scripts/extract_spec.py         # the node classes into spec/nodes.json
npm run generate                           # spec/nodes.json into src/nodes
python3 -B scripts/export_golden.py        # what Python sends, into tests/fixtures/golden.json
python3 -B scripts/export_runpod_golden.py # what its RunPod suites send, into tests/fixtures/runpod_golden.json
```

## License

MIT
