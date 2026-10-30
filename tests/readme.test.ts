// Port of the Python test_docs.py for the README, as it has no other docs:
// each TypeScript block type checks against the package, runs, and prints
// what the `console.log(...) // text` comments on its lines say.
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const README = readFileSync(join(ROOT, "README.md"), "utf8");
const BLOCKS = [...README.matchAll(/```ts\n(.*?)```/gs)].map((match) => match[1]!);
// Each block is a module of an ESM project, like the projects using the package.
const directory = mkdtempSync(join(tmpdir(), "readme-"));
writeFileSync(join(directory, "package.json"), JSON.stringify({ type: "module" }));
const files = BLOCKS.map((code, index) => {
  const file = join(directory, `block-${index + 1}.ts`);
  writeFileSync(file, code);
  return file;
});

afterAll(() => {
  rmSync(directory, { recursive: true, force: true });
});

// What a block says it prints: the comments after its console.log calls.
function printedComments(code: string): string[] {
  return [...code.matchAll(/^[ \t]*console\.log\(.*\);?[ \t]+\/\/ (.+)$/gm)].map((match) => match[1]!);
}

describe("the README", { timeout: 60_000 }, () => {
  it("has TypeScript examples, which say what they print", () => {
    expect(BLOCKS.length).toBeGreaterThan(3);
    expect(BLOCKS.flatMap(printedComments).length).toBeGreaterThan(3);
  });

  it("has examples that type check", () => {
    const config = join(directory, "tsconfig.json");
    writeFileSync(
      config,
      JSON.stringify({
        extends: join(ROOT, "tsconfig.json"),
        compilerOptions: { typeRoots: [join(ROOT, "node_modules", "@types")] },
        include: ["*.ts"],
      }),
    );

    const tsc = join(ROOT, "node_modules", "typescript", "bin", "tsc");
    const result = spawnSync(process.execPath, [tsc, "-p", config], { cwd: ROOT, encoding: "utf8" });

    expect(result.stdout + result.stderr).toBe("");
    expect(result.status).toBe(0);
  });

  it.each(BLOCKS.map((code, index) => [index + 1, code, files[index]!] as const))(
    "has example %i run and print what its comments say",
    (_, code, file) => {
      const printed = execFileSync(process.execPath, ["--import", "tsx", file], { cwd: ROOT, encoding: "utf8" });

      expect(printed.split("\n")).toEqual(expect.arrayContaining(printedComments(code)));
    },
  );
});
