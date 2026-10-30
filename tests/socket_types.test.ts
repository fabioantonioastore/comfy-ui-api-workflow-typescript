// Port of the Python test_socket_types.py. The check against ComfyUI needs
// its object_info, saved as new_object_info.json at the project root.
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { NODE_TYPES } from "../src/_internal/registry.js";
import { BaseNode, type NodeType } from "../src/nodes/base.js";
import { ImageBlend } from "../src/nodes/image/filters.js";
import { ImageScale } from "../src/nodes/image/upscaling.js";
import { CheckpointLoaderSimple } from "../src/nodes/model/loaders.js";
import { KSampler } from "../src/nodes/model/sampling.js";
import { link } from "../src/nodes/schema.js";
import { CreateVideo } from "../src/nodes/video.js";

const OBJECT_INFO = new URL("../new_object_info.json", import.meta.url);
const TYPES = [...NODE_TYPES.values()].sort((a, b) => (a.CLASS_TYPE < b.CLASS_TYPE ? -1 : 1));

type Entry = readonly [unknown, Record<string, any>?];

interface ObjectInfoNode {
  readonly input: Readonly<Record<string, Readonly<Record<string, Entry>>>>;
  readonly output: readonly string[];
}

function socketType(entry: Entry): string {
  return typeof entry[0] === "string" ? entry[0] : "V1 COMBO";
}

/** The socket type of each input ComfyUI shows, under its ComfyUI name. */
function flattenedInputs(spec: ObjectInfoNode): Map<string, string> {
  const inputs = new Map<string, string>();
  const add = (name: string, entry: Entry): void => {
    const options = entry[1] ?? {};
    if (entry[0] === "COMFY_AUTOGROW_V3") {
      const template = options["template"];
      const section = Object.values(template.input as Record<string, Record<string, Entry>>).find(
        (fields) => Object.keys(fields).length > 0,
      )!;
      const value = Object.values(section)[0]!;
      const names: string[] =
        template.names ?? Array.from({ length: template.max }, (_, index) => `${template.prefix}${index}`);
      for (const letter of names) {
        inputs.set(`${name}.${letter}`, socketType(value));
      }
      return;
    }
    if (entry[0] === "COMFY_DYNAMICGROUP_V3") {
      for (let row = 0; row < options["max"]; row += 1) {
        for (const section of ["required", "optional"]) {
          for (const [field, fieldEntry] of Object.entries<Entry>(options["template"][section] ?? {})) {
            inputs.set(`${name}.${row}.${field}`, socketType(fieldEntry));
          }
        }
      }
      return;
    }
    inputs.set(name, socketType(entry));
    if (entry[0] === "COMFY_DYNAMICCOMBO_V3") {
      for (const option of options["options"]) {
        for (const section of ["required", "optional"]) {
          for (const [sub, subEntry] of Object.entries<Entry>(option.inputs[section] ?? {})) {
            add(`${name}.${sub}`, subEntry);
          }
        }
      }
    }
  };
  for (const section of ["required", "optional"]) {
    for (const [name, entry] of Object.entries(spec.input[section] ?? {})) {
      add(name, entry);
    }
  }
  return inputs;
}

class UntypedNode extends BaseNode<{ model: unknown }> {
  static override readonly CATEGORY = "test";
  static override readonly INPUTS = { model: link() };
}

describe("socket types", () => {
  it("are none for a node that declares none", () => {
    expect([UntypedNode.RETURN_TYPES, UntypedNode.INPUT_SOCKET_TYPES]).toEqual([[], {}]);
  });

  it("are given for every output of the node classes", () => {
    const missing = TYPES.filter(
      (type) =>
        type.RETURN_TYPES.length !== Object.keys(type.OUTPUTS).length ||
        !type.RETURN_TYPES.every((kind) => typeof kind === "string" && kind !== ""),
    ).map((type) => type.CLASS_TYPE);

    expect(missing).toEqual([]);
  });

  it("are given for every input of the node classes", () => {
    const missing = TYPES.filter((type) => {
      const sockets = Object.keys(type.INPUT_SOCKET_TYPES).sort();
      return (
        sockets.join("\n") !== Object.keys(type.INPUTS).sort().join("\n") ||
        !Object.values(type.INPUT_SOCKET_TYPES).every((kind) => typeof kind === "string" && kind !== "")
      );
    }).map((type) => type.CLASS_TYPE);

    expect(missing).toEqual([]);
  });

  it.each([
    [KSampler, "sampler_name", "V1 COMBO"],
    [KSampler, "scheduler", "V1 COMBO"],
    [CheckpointLoaderSimple, "ckpt_name", "V1 COMBO"],
    [ImageScale, "upscale_method", "V1 COMBO"],
    [CreateVideo, "bit_depth", "COMBO"],
    [ImageBlend, "blend_mode", "COMBO"],
  ] as [NodeType, string, string][])("give V1 combos a type of their own (%o.%s)", (type, name, kind) => {
    expect(type.INPUT_SOCKET_TYPES[name]).toBe(kind);
  });

  it.skipIf(!existsSync(OBJECT_INFO))("match the object_info of ComfyUI", () => {
    const specs = JSON.parse(readFileSync(OBJECT_INFO, "utf8")) as Record<string, ObjectInfoNode>;
    const different: [string, string][] = [];
    for (const type of TYPES) {
      const spec = specs[type.CLASS_TYPE]!;
      const names = new Map(Object.entries(type.INPUT_NAMES).map(([name, comfy]) => [comfy, name]));
      const inputs = Object.fromEntries(
        [...flattenedInputs(spec)].map(([name, kind]) => [names.get(name) ?? name, kind]),
      );
      if (JSON.stringify(type.RETURN_TYPES) !== JSON.stringify(spec.output)) {
        different.push([type.CLASS_TYPE, "outputs"]);
      }
      try {
        expect(type.INPUT_SOCKET_TYPES).toEqual(inputs);
      } catch {
        different.push([type.CLASS_TYPE, "inputs"]);
      }
    }

    expect(different).toEqual([]);
  });
});
