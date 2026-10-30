/** Every error this library throws on purpose extends this class. */
export class ComfyUIAPIWorkflowException extends Error {
  constructor(message?: string, options?: ErrorOptions) {
    super(message, options);
    // Not enumerable, like the name Error.prototype gives.
    Object.defineProperty(this, "name", {
      value: new.target.name,
      configurable: true,
      writable: true,
    });
  }
}

export class DataStructureException extends ComfyUIAPIWorkflowException {}

export class WorkflowException extends ComfyUIAPIWorkflowException {}

export class NodeException extends ComfyUIAPIWorkflowException {}

export class PipelineException extends ComfyUIAPIWorkflowException {}

export class OutputException extends ComfyUIAPIWorkflowException {}
