import { repr } from "../_internal/format.js";
import { WorkflowException } from "../exceptions.js";

export class InvalidWorkflowError extends WorkflowException {
  /** The node that could not be loaded, or null when the workflow as a whole is wrong. */
  readonly nodeId: string | null;
  readonly reason: string;

  constructor(nodeId: string | null, reason: string, options?: ErrorOptions) {
    super(
      nodeId === null
        ? `Cannot load the workflow: ${reason}.`
        : `Cannot load node ${repr(nodeId)}: ${reason}.`,
      options,
    );
    this.nodeId = nodeId;
    this.reason = reason;
  }
}
