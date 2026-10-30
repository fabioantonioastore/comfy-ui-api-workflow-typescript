// Port of the Python test_runpod_workflows.py. A hundred workflows built with the
// library must send what the Python suite sends and survive a reload; with
// `RUNPOD=1 npm test` each one also runs on the RunPod endpoint in .env, and its
// files land in runpod_outputs/workflows, with an index.html.
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { deflateSync } from "node:zlib";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { InvalidWorkflowError } from "../src/errors/workflow.js";
import { Workflow } from "../src/workflow.js";
import { canonical, expectError } from "./helpers.js";
import { RUNPOD, runpodEndpoint, type RunPodJob } from "./runpod.js";
import { expectUploads, readRunPodGolden } from "./runpod_golden.js";
import { runAndRecord, type Endpoint } from "./runpod_workflows/gallery.js";
import { CASES, GPU_TIMEOUT, SPECIAL_SCENES, sent, specialWorkflow } from "./runpod_workflows/index.js";
import {
  assertGenerated,
  assertPixel,
  BLUE,
  crc32,
  decodePng,
  encodePng,
  pattern,
  RED,
  type Outputs,
  type PixelSource,
} from "./runpod_workflows/media.js";

const golden = readRunPodGolden();
const BUILT = CASES.filter((workflowCase) => workflowCase.librarySays === null);
const RAW = CASES.filter((workflowCase) => workflowCase.librarySays !== null);

describe("the RunPod workflows", () => {
  it("are the cases of the Python suite, in its order", () => {
    expect(CASES.map((workflowCase) => workflowCase.name)).toEqual(golden.workflows.map((expected) => expected.name));
  });

  it.each(golden.workflows)("$name sends what the Python case sends", (expected) => {
    const workflowCase = CASES.find((candidate) => candidate.name === expected.name);
    expect(workflowCase, `no case named ${expected.name}`).toBeDefined();
    const [payload] = workflowCase!.payload();

    expect(payload).toEqual(expected.payload);
    expectUploads(workflowCase!.uploads, expected.uploads);
    expect({
      description: workflowCase!.description,
      gpu: workflowCase!.gpu,
      failsWith: workflowCase!.failsWith,
      librarySays: workflowCase!.librarySays,
    }).toEqual({
      description: expected.description,
      gpu: expected.gpu,
      failsWith: expected.fails_with,
      librarySays: expected.library_says,
    });
  });

  it("send the special one the Python suite sends", () => {
    expect(sent(specialWorkflow())).toEqual(golden.special);
  });

  it.each(BUILT)("$name survives a reload", (workflowCase) => {
    const [payload] = workflowCase.payload();
    const workflow = new Workflow();

    workflow.loadWorkflow(JSON.parse(JSON.stringify(payload)));

    expect(sent(workflow)).toEqual(payload);
  });

  it.each(RAW)("$name is refused by the library, as ComfyUI refuses it", (workflowCase) => {
    const error = expectError(() => new Workflow().loadWorkflow(workflowCase.payload()[0]), InvalidWorkflowError);

    expect(error.message).toContain(workflowCase.librarySays);
  });

  it("are a hundred different workflows", () => {
    const payloads = new Set(CASES.map((workflowCase) => canonical(workflowCase.payload()[0])));

    expect(CASES).toHaveLength(100);
    expect(new Set(CASES.map((workflowCase) => workflowCase.name)).size).toBe(100);
    expect(payloads.size).toBe(100);
  });

  it("cover many node types", () => {
    const used = new Set(
      CASES.flatMap((workflowCase) =>
        Object.values(workflowCase.payload()[0]).map((node) => (node as { class_type: string }).class_type),
      ),
    );

    expect(used.size).toBeGreaterThanOrEqual(165);
  });

  it("keep the special one through a reload", () => {
    const payload = sent(specialWorkflow());
    const workflow = new Workflow();

    workflow.loadWorkflow(payload);

    expect(sent(workflow)).toEqual(payload);
  });
});

function filteredPng(width: number, height: number, pixel: PixelSource): Buffer {
  // Each row uses another PNG filter (none, sub, up, average, paeth), as encoders do.
  function predictor(kind: number, left: number, up: number, upLeft: number): number {
    if (kind === 1) {
      return left;
    }
    if (kind === 2) {
      return up;
    }
    if (kind === 3) {
      return Math.floor((left + up) / 2);
    }
    if (kind === 4) {
      const p = left + up - upLeft;
      const [pa, pb, pc] = [Math.abs(p - left), Math.abs(p - up), Math.abs(p - upLeft)];
      return pa <= pb && pa <= pc ? left : pb <= pc ? up : upLeft;
    }
    return 0;
  }

  const lines: Buffer[] = [];
  let previous = Buffer.alloc(width * 3);
  for (let y = 0; y < height; y++) {
    const row = Buffer.from(Array.from({ length: width }, (_, x) => [...pixel(x, y)]).flat());
    const kind = y % 5;
    const line = Buffer.from(
      row.map((value, i) => value - predictor(kind, i >= 3 ? row[i - 3]! : 0, previous[i]!, i >= 3 ? previous[i - 3]! : 0)),
    );
    lines.push(Buffer.from([kind]), line);
    previous = row;
  }
  const plain = encodePng(width, height, pixel);
  const start = plain.indexOf("IDAT") - 4;
  const length = plain.readUInt32BE(start);
  const idat = deflateSync(Buffer.concat(lines));
  const size = Buffer.alloc(4);
  size.writeUInt32BE(idat.length);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([Buffer.from("IDAT"), idat])));
  return Buffer.concat([plain.subarray(0, start), size, Buffer.from("IDAT"), idat, crc, plain.subarray(start + 12 + length)]);
}

describe("decodePng", () => {
  it("undoes every row filter", () => {
    const picture = decodePng(filteredPng(24, 16, pattern));
    const wrong: string[] = [];
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 24; x++) {
        if (picture.pixel(x, y).join() !== pattern(x, y).join()) {
          wrong.push(`(${x}, ${y})`);
        }
      }
    }

    expect(picture.size).toEqual([24, 16]);
    expect(wrong).toEqual([]);
  });
});

class FakeEndpoint implements Endpoint {
  constructor(readonly job: RunPodJob) {}

  async run(): Promise<RunPodJob> {
    return this.job;
  }
}

function redJob(): RunPodJob {
  const data = encodePng(4, 4, () => [255, 0, 0]).toString("base64");
  return {
    id: "job-1",
    status: "COMPLETED",
    output: { images: [{ filename: "red_00001_.png", type: "base64", data }] },
  };
}

describe("runAndRecord", () => {
  const anyWorkflow = { "1": { class_type: "EmptyImage", inputs: {} } };
  let gallery: string;

  beforeEach(() => {
    gallery = mkdtempSync(join(tmpdir(), "gallery-"));
  });

  afterEach(() => {
    rmSync(gallery, { recursive: true, force: true });
  });

  function result(folder: string): { status: string; message?: string } {
    return JSON.parse(readFileSync(join(gallery, folder, "result.json"), "utf8")) as { status: string; message?: string };
  }

  it("records a passing job in the gallery", async () => {
    const check = (out: Outputs) => assertPixel(out.picture("red"), 1, 1, RED, 0);

    await runAndRecord(new FakeEndpoint(redJob()), "001-red", "red", "um quadrado vermelho", anyWorkflow, {
      timeout: 1,
      check,
      gallery,
    });

    const page = readFileSync(join(gallery, "index.html"), "utf8");
    expect(result("001-red").status).toBe("passed");
    expect(existsSync(join(gallery, "001-red", "red_00001_.png"))).toBe(true);
    expect(page).toContain("001-red/red_00001_.png");
    expect(page).toContain("passou");
  });

  it("records a failing check, which still fails", async () => {
    const check = (out: Outputs) => assertPixel(out.picture("red"), 1, 1, BLUE, 0);

    await expect(
      runAndRecord(new FakeEndpoint(redJob()), "002-blue", "blue", "devia ser azul", anyWorkflow, {
        timeout: 1,
        check,
        gallery,
      }),
    ).rejects.toThrow("expected");

    expect(result("002-blue").status).toBe("failed");
    expect(result("002-blue").message).toContain("expected (0, 0, 255)");
    expect(readFileSync(join(gallery, "index.html"), "utf8")).toContain("falhou");
  });
});

// The timeouts of the cases count from when the job is sent, so give vitest a bit more.
const TEST_TIMEOUT = (GPU_TIMEOUT + 120) * 1000;

describe.skipIf(!RUNPOD)("the RunPod workflows on ComfyUI", () => {
  const numbered = CASES.map((workflowCase, index) => ({ name: workflowCase.name, workflowCase, number: index + 1 }));

  it.each(numbered)(
    "$name runs on ComfyUI",
    async ({ workflowCase, number }) => {
      const [payload, check] = workflowCase.payload();

      await runAndRecord(
        runpodEndpoint(),
        `${String(number).padStart(3, "0")}-${workflowCase.name}`,
        workflowCase.name,
        workflowCase.description,
        payload,
        {
          timeout: workflowCase.timeout,
          uploads: workflowCase.uploads,
          check,
          failsWith: workflowCase.failsWith,
        },
      );
    },
    TEST_TIMEOUT,
  );

  it(
    "special: a boy running down the street and a girl at the lagoon",
    async () => {
      const check = (out: Outputs) => {
        for (const [label] of SPECIAL_SCENES) {
          assertGenerated(out.one(label), [1520, 1040], 200_000);
        }
      };

      await runAndRecord(
        runpodEndpoint(),
        "000-especial-menino-e-menina",
        "especial",
        "Especial: um menino correndo no meio da rua e uma menina de vestido na beira de uma " +
          "lagoa jogando pedrinhas, SDXL 1216x832 com 30 passos e refinamento em 1520x1040.",
        sent(specialWorkflow()),
        { timeout: GPU_TIMEOUT, check },
      );
    },
    TEST_TIMEOUT,
  );
});
