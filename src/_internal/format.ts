// Formats values for error messages, so they read like the Python library's.
import { isRecord } from "./records.js";

/** Python's repr of the values messages show: 'text' in quotes, [lists] and {dicts}. */
export function repr(value: unknown): string {
  if (typeof value === "string") {
    return reprString(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(repr).join(", ")}]`;
  }
  if (isRecord(value)) {
    const entries = Object.entries(value).map(([key, item]) => `${reprString(key)}: ${repr(item)}`);
    return `{${entries.join(", ")}}`;
  }
  return String(value);
}

// What Python's str.isprintable() rejects, besides the space: the Other and
// Separator categories, so invisible characters show in messages.
const UNPRINTABLE = /[\p{C}\p{Z}]/u;

function reprString(text: string): string {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let body = "";
  for (const char of text) {
    if (char === "\\" || char === quote) {
      body += `\\${char}`;
    } else if (char === "\n") {
      body += "\\n";
    } else if (char === "\r") {
      body += "\\r";
    } else if (char === "\t") {
      body += "\\t";
    } else if (char !== " " && UNPRINTABLE.test(char)) {
      body += escapeCodePoint(char.codePointAt(0) ?? 0);
    } else {
      body += char;
    }
  }
  return `${quote}${body}${quote}`;
}

function escapeCodePoint(code: number): string {
  const hex = code.toString(16);
  if (code <= 0xff) {
    return `\\x${hex.padStart(2, "0")}`;
  }
  if (code <= 0xffff) {
    return `\\u${hex.padStart(4, "0")}`;
  }
  return `\\U${hex.padStart(8, "0")}`;
}

/** The kind of a value as messages name it: string, integer, float, array, null... */
export function typeName(value: unknown): string {
  if (value === null) {
    return "null";
  }
  if (Array.isArray(value)) {
    return "array";
  }
  if (typeof value === "number") {
    if (Number.isSafeInteger(value)) {
      return "integer";
    }
    if (Number.isInteger(value)) {
      return "unsafe integer";
    }
    return Number.isFinite(value) ? "float" : String(value);
  }
  if (typeof value === "object") {
    const name: unknown = Object.getPrototypeOf(value)?.constructor?.name;
    return typeof name === "string" && name !== "" && name !== "Object" ? name : "object";
  }
  return typeof value;
}

/** The name of what was thrown, like Python's type(error).__name__. */
export function errorTypeName(error: unknown): string {
  if (error instanceof Error) {
    return error.constructor.name || error.name;
  }
  return typeName(error);
}

/** The message of what was thrown, like Python's str(error). */
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** A message as part of a sentence: without its final period. */
export function sentence(error: unknown): string {
  return errorMessage(error).replace(/\.+$/, "");
}
