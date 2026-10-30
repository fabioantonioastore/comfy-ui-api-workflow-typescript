// The cases of the Python test_runpod_workflows.py that use the library's
// features: loading, updating and removing nodes, and pipelines.
import assert from "node:assert/strict";

import { NodeHasDependentsError } from "../../src/errors/node.js";
import { EmptyImage, SaveImage } from "../../src/nodes/image.js";
import { ImageInvert } from "../../src/nodes/image/color.js";
import { ImageFlip } from "../../src/nodes/image/transform.js";
import { Workflow } from "../../src/workflow.js";
import { flip, Framed, maskImage, quadrants, rotate, save, solid, Spotlight } from "./blocks.js";
import {
  assertPixel,
  assertQuadrants,
  assertSameRows,
  BLACK,
  BLUE,
  CYAN,
  FLIPS,
  gray,
  GRAY,
  GREEN,
  hues,
  MAGENTA,
  ORANGE,
  RED,
  WHITE,
  YELLOW,
} from "./media.js";
import { addCase, sent } from "./registry.js";

const API_JSON = {
  canvas: { class_type: "EmptyImage", inputs: { width: 48, height: 32, batch_size: 1, color: GREEN } },
  turn: { class_type: "ImageRotate", inputs: { image: ["canvas", 0], rotation: "90 degrees" } },
  frame: {
    class_type: "ImagePadForOutpaint",
    inputs: { image: ["turn", 0], left: 8, top: 8, right: 8, bottom: 8, feathering: 0 },
  },
  save: { class_type: "SaveImage", inputs: { images: ["frame", 0], filename_prefix: "loaded" } },
};

addCase(
  "loaded_api_json_runs",
  "JSON no formato da API com ids em texto carregado com load_workflow, alterado com " +
    "update_node e estendido com add_node usando get_node_output.",
  {},
  (w) => {
    w.loadWorkflow(JSON.parse(JSON.stringify(API_JSON)));
    w.updateNode("canvas", { color: ORANGE });
    const inverted = w.addNode(ImageInvert, { image: w.getNodeOutput("frame")["IMAGE"]! });
    save(w, inverted.output.IMAGE, "inverted");

    return (out) => {
      const loaded = out.picture("loaded");
      assert.deepEqual(loaded.size, [48, 64]);
      assertPixel(loaded, 24, 32, ORANGE);
      assertPixel(loaded, 2, 2, gray(0.5));
      const negative = out.picture("inverted");
      assertPixel(negative, 24, 32, [0, 127, 255]);
      assertPixel(negative, 2, 2, gray(0.5));
    };
  },
);

addCase(
  "update_node_rewires_inputs",
  "update_node troca o link de entrada e a cor depois de montado, e uma atribuição " +
    "direta (node.width = 56) passa pelas mesmas validações.",
  {},
  (w) => {
    const red = w.addNode(EmptyImage, { width: 16, height: 16, color: RED });
    const blue = w.addNode(EmptyImage, { width: 40, height: 24, color: BLUE });
    const inverted = w.addNode(ImageInvert, { image: red.output.IMAGE });
    save(w, inverted.output.IMAGE, "inverted");
    w.updateNode(inverted.nodeId, { image: blue.output.IMAGE });
    w.updateNode(blue.nodeId, { color: GREEN });
    blue.width = 56;

    return (out) => {
      const negative = out.picture("inverted");
      assert.deepEqual(negative.size, [56, 24]);
      assertPixel(negative, 10, 10, MAGENTA);
    };
  },
);

addCase(
  "remove_node_prunes_branch",
  "remove_node apaga um ramo inteiro (folha primeiro); remover um nó com dependentes " +
    "é recusado; só o ramo que ficou volta do ComfyUI.",
  {},
  (w) => {
    const base = solid(w, 24, 24, CYAN);
    save(w, base, "kept");
    const flipped = w.addNode(ImageFlip, { image: base, flip_method: FLIPS[0] });
    const dropped = w.addNode(SaveImage, { images: flipped.output.IMAGE, filename_prefix: "dropped" });
    assert.throws(() => w.removeNode(flipped.nodeId), NodeHasDependentsError);
    w.removeNode(dropped.nodeId);
    w.removeNode(flipped.nodeId);

    return (out) => {
      assert.deepEqual(out.of("dropped"), []);
      assertPixel(out.picture("kept"), 12, 12, CYAN);
    };
  },
);

addCase(
  "nested_framed_pipelines",
  "Pipeline Framed (GetImageSize + matemática + EmptyImage + composite) aplicada três " +
    "vezes em cadeia: as molduras ficam aninhadas na ordem certa.",
  {},
  (w) => {
    let image = solid(w, 32, 32, ORANGE);
    for (const [border, color] of [
      [4, WHITE],
      [8, BLACK],
      [2, RED],
    ] as const) {
      image = w.addPipeline(Framed, { image, border, color }).output.IMAGE;
    }
    save(w, image, "framed");

    return (out) => {
      const framed = out.picture("framed");
      assert.deepEqual(framed.size, [60, 60]);
      for (const [x, y, color] of [
        [0, 0, RED],
        [1, 30, RED],
        [5, 5, BLACK],
        [12, 12, WHITE],
        [30, 30, ORANGE],
      ] as const) {
        assertPixel(framed, x, y, color);
      }
    };
  },
);

addCase(
  "pipeline_with_two_outputs",
  "Pipeline Spotlight com duas saídas (IMAGE e MASK): GrowMask, MaskComposite com " +
    "posição calculada e composite sobre fundo preto.",
  {},
  (w) => {
    const spotlight = w.addPipeline(Spotlight, { image: solid(w, 64, 64, YELLOW), size: 16, grow: 4 });
    save(w, spotlight.output.IMAGE, "lit");
    save(w, maskImage(w, spotlight.output.MASK), "mask");

    return (out) => {
      const lit = out.picture("lit");
      for (const [x, y, color] of [
        [32, 32, YELLOW],
        [21, 21, YELLOW],
        [18, 18, BLACK],
        [45, 45, BLACK],
      ] as const) {
        assertPixel(lit, x, y, color);
      }
      const mask = out.picture("mask");
      assertPixel(mask, 20, 20, WHITE);
      assertPixel(mask, 19, 19, BLACK);
    };
  },
);

addCase(
  "reload_then_mutate",
  "Workflow montado, exportado em JSON, recarregado em outro Workflow, alterado com " +
    "update_node e estendido antes de rodar.",
  {},
  (w) => {
    const source = new Workflow();
    const image = quadrants(source, 32);
    save(source, image, "first");
    const [canvasId] = [...source.nodes].find(([, node]) => node instanceof EmptyImage && node.width === 32)!;
    w.loadWorkflow(sent(source));
    w.updateNode(canvasId, { color: MAGENTA });
    save(w, rotate(w, w.getNodeOutput(image[0])["IMAGE"]!, "180 degrees"), "turned");

    return (out) => {
      assertQuadrants(out.picture("first"), [MAGENTA, GREEN, BLUE, WHITE]);
      assertQuadrants(out.picture("turned"), [WHITE, BLUE, GREEN, MAGENTA]);
    };
  },
);

addCase(
  "two_dozen_outputs_in_one_job",
  "Um único job com 24 SaveImage de cores diferentes (roda de matizes): os 24 arquivos " +
    "voltam, cada um com a sua cor exata.",
  {},
  (w) => {
    const colors = hues(24);
    colors.forEach((color, index) => save(w, solid(w, 16, 16, color), `hue_${String(index).padStart(2, "0")}`));

    return (out) => {
      colors.forEach((color, index) => {
        assertPixel(out.picture(`hue_${String(index).padStart(2, "0")}`), 8, 8, color, 0);
      });
    };
  },
);

addCase(
  "forty_node_chain_returns_to_start",
  "Cadeia de 40 nós alternando ImageInvert e ImageFlip: com número par de cada, a saída " +
    "tem de ser idêntica à entrada.",
  {},
  (w) => {
    let image = quadrants(w, 32, [ORANGE, CYAN, MAGENTA, GRAY]);
    save(w, image, "start");
    for (let step = 0; step < 40; step++) {
      image = step % 2 ? flip(w, image, FLIPS[1]) : w.addNode(ImageInvert, { image }).output.IMAGE;
    }
    save(w, image, "end");

    return (out) => {
      assertSameRows(out.picture("end"), out.picture("start"));
    };
  },
);
