import type { Link } from "../nodes/validation.js";

// Names that tools read from any object to learn what it is, like await
// reading "then": they are not outputs, so reading them must not throw.
const PROBES = new Set([
  "then",
  "toJSON",
  "asymmetricMatch",
  "$$typeof",
  "nodeType",
  "tagName",
  "@@__IMMUTABLE_ITERABLE__@@",
  "@@__IMMUTABLE_RECORD__@@",
]);

/**
 * Lets each output be read as a property, like output.LATENT or
 * output["Audio VAE"]. An output wins over a member of the same name, and a
 * name that is neither throws the error made by missing.
 */
export function outputProxy<T extends object>(
  target: T,
  find: (name: string) => Link | undefined,
  missing: (name: string) => Error,
): T {
  return new Proxy(target, {
    get(object, property, receiver) {
      if (typeof property === "string") {
        const link = find(property);
        if (link !== undefined) {
          return link;
        }
        if (!(property in object) && !PROBES.has(property)) {
          throw missing(property);
        }
      }
      return Reflect.get(object, property, receiver);
    },
    has(object, property) {
      return (
        (typeof property === "string" && find(property) !== undefined) ||
        Reflect.has(object, property)
      );
    },
  });
}
