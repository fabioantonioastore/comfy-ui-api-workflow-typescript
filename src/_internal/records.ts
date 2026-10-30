// Plain objects used as dicts.

/** A plain object, like the ones JSON.parse and object literals make. */
export function isRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/** The value of an own key, so names like "constructor" never reach Object.prototype. */
export function own<T>(record: Readonly<Record<string, T>>, key: string): T | undefined {
  return Object.hasOwn(record, key) ? record[key] : undefined;
}

/** A copy of a value with every array and plain object it holds, none of them frozen. */
export function deepCopy<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map(deepCopy) as T;
  }
  if (isRecord(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, deepCopy(item)])) as T;
  }
  return value;
}

/** Freezes a value with everything it holds, so a shared default cannot be changed. */
export function deepFreeze<T>(value: T): T {
  if (typeof value === "object" && value !== null && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const item of Object.values(value)) {
      deepFreeze(item);
    }
  }
  return value;
}
