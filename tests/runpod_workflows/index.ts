// The hundred workflows of the Python test_runpod_workflows.py. Each module adds
// the cases of one of its sections, so they are imported in the Python order.
import "./geometry.js";
import "./color.js";
import "./masks.js";
import "./uploads.js";
import "./logic.js";
import "./video.js";
import "./library.js";
import "./failures.js";
import "./sdxl.js";

export { CASES, GPU_TIMEOUT, sent, TIMEOUT, type Case, type Check } from "./registry.js";
export { SPECIAL_SCENES, specialWorkflow } from "./special.js";
