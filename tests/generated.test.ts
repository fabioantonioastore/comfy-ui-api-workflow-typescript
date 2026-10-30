import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { generate, generatedFiles, readSpec, ROOT } from "../scripts/generate-nodes.js";

describe("the generated node modules", () => {
  const files = generate(readSpec());

  it("are what spec/nodes.json makes (run npm run generate)", () => {
    for (const [path, content] of files) {
      expect(readFileSync(join(ROOT, path), "utf8"), path).toBe(content);
    }
  });

  it("are all made by the spec", () => {
    expect(generatedFiles()).toEqual([...files.keys()].sort());
  });
});
