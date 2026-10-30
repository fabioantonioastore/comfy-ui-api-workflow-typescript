import { repr } from "../_internal/format.js";
import { PipelineException } from "../exceptions.js";

export class PipelineError extends PipelineException {
  /** The name of the pipeline class. */
  readonly pipeline: string;
  readonly reason: string;

  constructor(pipeline: string, reason: string, options?: ErrorOptions) {
    super(
      `Cannot add pipeline ${repr(pipeline)}: ${reason}. No node of the pipeline was added.`,
      options,
    );
    this.pipeline = pipeline;
    this.reason = reason;
  }
}
