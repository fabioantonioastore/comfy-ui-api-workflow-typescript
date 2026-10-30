import { repr } from "./_internal/format.js";
import { outputProxy } from "./_internal/output.js";
import { pipelineState, setPipelineState } from "./_internal/state.js";
import { quoteNames } from "./errors/node.js";
import { OutputNotFoundError } from "./errors/output.js";
import { PipelineError } from "./errors/pipeline.js";
import type { BaseNode, NodeArgs, NodeType } from "./nodes/base.js";
import type { InputSpec } from "./nodes/schema.js";
import type { Link } from "./nodes/validation.js";

/** What a build returns: each output of the pipeline as a link. */
export type PipelineOutputs = Readonly<Record<string, Link>>;

/** The outputs of a pipeline, each read as the link it stands for. */
export type PipelineOutput<O extends string = string> = { readonly [K in O]: Link };

export const PipelineOutput = class PipelineOutput {
  readonly pipeline: string;
  readonly outputs: ReadonlyMap<string, Link>;

  constructor(pipeline: string, outputs: ReadonlyMap<string, Link>) {
    this.pipeline = pipeline;
    this.outputs = outputs;
    return outputProxy(
      this,
      (name) => outputs.get(name),
      (name) => {
        let message = `Pipeline ${repr(pipeline)} has no output ${repr(name)}.`;
        if (outputs.size > 0) {
          message += ` Its outputs are: ${quoteNames([...outputs.keys()])}.`;
        }
        return new OutputNotFoundError(message);
      },
    );
  }
} as unknown as new <O extends string = string>(
  pipeline: string,
  outputs: ReadonlyMap<string, Link>,
) => PipelineOutput<O>;

/** The names of the outputs a pipeline's build returns. */
export type OutputNames<P> = P extends { build(...args: any): infer R }
  ? Extract<keyof Exclude<R, void | null | undefined>, string>
  : string;

/** A pipeline class. */
export interface PipelineType<P extends Pipeline = Pipeline> {
  new (): P;
  readonly prototype: P;
  readonly name: string;
  readonly PARAMETERS?: Readonly<Record<string, InputSpec>>;
}

/** What a pipeline class takes when added: the parameters of its build. */
export type PipelineArgs<P extends PipelineType> = Parameters<InstanceType<P>["build"]>;

/**
 * A group of nodes added together. A subclass defines build, which adds the
 * nodes with addNode and returns the outputs; Workflow.addPipeline adds all of
 * them or none.
 */
export class Pipeline {
  /**
   * The parameters build takes, checked before it runs: unknown and missing
   * ones are rejected, and defaults fill the rest. Without it build takes any.
   */
  static readonly PARAMETERS?: Readonly<Record<string, InputSpec>>;

  constructor() {
    setPipelineState(this, { workflow: null, nodes: new Map(), outputs: new Map(), rejection: null });
  }

  /** The nodes the pipeline added, by id, in the order it added them. */
  get nodes(): ReadonlyMap<string, BaseNode> {
    return pipelineState(this).nodes;
  }

  get output(): PipelineOutput<OutputNames<this>> {
    return new PipelineOutput<OutputNames<this>>(this.constructor.name, pipelineState(this).outputs);
  }

  build(_parameters?: any): PipelineOutputs | null | undefined | void {
    throw new Error(`${this.constructor.name} does not define build`);
  }

  addNode<T extends NodeType>(type: T, ...args: NodeArgs<T>): InstanceType<T> {
    const state = pipelineState(this);
    if (state.workflow === null) {
      const reason = "addNode only works inside build, while Workflow.addPipeline adds the pipeline";
      throw new PipelineError(this.constructor.name, reason);
    }
    let node: InstanceType<T>;
    try {
      node = state.workflow.addNode(type, ...args);
    } catch (error) {
      state.rejection ??= { position: state.nodes.size + 1, type, error };
      throw error;
    }
    state.nodes.set(node.nodeId, node);
    return node;
  }
}
