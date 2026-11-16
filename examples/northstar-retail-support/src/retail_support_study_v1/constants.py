from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION = 10
STUDY_ID = "agent-lens-retail-support-study-v10"
DESTINATION_PROJECT = "wandb/agent-lens-retail-support-study-v2"
INFERENCE_PROJECT = "wandb/agent-lens-retail-support-inference-v2"

# The V2 destination above is retained for the frozen-bundle experiment.  The
# article evidence comes from a fresh run whose spans are created around the
# actual application operations, not reconstructed after the cohort finishes.
LIVE_AGENT_LENS_PROJECTS = {
    "canary": "wandb/agent-lens-retail-support-live-canary-v10-r4",
    "full": "wandb/agent-lens-retail-support-live-v10",
}
LIVE_AGENT_NAME = "retail-support-generalist"
LIVE_INSTRUMENTATION_VERSION = "2"

TAU3_VERSION = "1.0.1"
# v1.0.1 is an annotated tag. Pin both the tag object requested in the study
# plan and the source commit it cryptographically peels to.
TAU3_TAG_OBJECT = "b711c1ead46f55111bf765cf44d5da8bacc2d28c"
TAU3_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TAU3_REPOSITORY = "https://github.com/sierra-research/tau2-bench"
TAU3_LICENSE = "MIT"
TASK_COUNT = 114

SUPPORT_MODEL = "zai-org/GLM-5.3-Flash"
USER_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
SUPPORT_TEMPERATURE = 0.0
USER_TEMPERATURE = 0.4
SUPPORT_MAX_OUTPUT_TOKENS = 4_096
USER_MAX_OUTPUT_TOKENS = 512
# DeepSeek V4 Flash 0731 enables reasoning by default. The simulated customer
# only needs to respond naturally to the latest support turn, so disable hidden
# reasoning explicitly rather than spending its small response allowance on it.
USER_ENABLE_THINKING = False
USER_CONFORMANCE_RETRIES = 2
USER_REPAIR_SEED_OFFSET = 1_000_000
WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"

PROFILES = ("ordinary", "concise", "skeptical")
TRIAL_INDICES = (0, 1, 2, 3)
FULL_COORDINATE_COUNT = TASK_COUNT * len(PROFILES) * len(TRIAL_INDICES)
CANARY_TASK_GROUPS = {
    "order_management": (
        "17",
        "40",
        "41",
        "87",
        "24",
        "76",
        "113",
    ),
    "returns_resolution": (
        "18",
        "21",
        "45",
        "51",
        "53",
        "73",
        "84",
    ),
    "generalist": (
        "12",
        "16",
        "59",
        "39",
        "55",
        "68",
    ),
}
CANARY_TASK_COUNT = 20
CANARY_COORDINATE_COUNT = CANARY_TASK_COUNT * len(PROFILES)
# Generation-two through generation-five recovery locks belong to the retained
# v9 12-task canary and must continue to verify independently of the fresh v10
# canary matrix.
LEGACY_CANARY_COORDINATE_COUNT = 36
CANARY_SELECTION_POLICY = "fixed-agent-stratified-v2"
CANARY_TRIAL_ASSIGNMENT_POLICY = "round-robin-task-profile-v1"
CANARY_TASK_IDS = tuple(
    task_id
    for group in (
        "order_management",
        "returns_resolution",
        "generalist",
    )
    for task_id in CANARY_TASK_GROUPS[group]
)

# The tuple order is part of the preview contract. The two count tuples must
# reconcile exactly with the 1,368-coordinate full cohort and 60-coordinate
# canary, respectively.
MODEL_ROUTE_COUNTS_FULL = (
    ("wandb-glm-5-3-flash", 821),
    ("wandb-deepseek-v4-1-flash", 205),
    ("wandb-kimi-k2-6", 137),
    ("anthropic-claude-haiku-4-5", 137),
    ("anthropic-claude-sonnet-5", 68),
)
MODEL_ROUTE_COUNTS_CANARY = tuple(
    (route_id, 12) for route_id, _ in MODEL_ROUTE_COUNTS_FULL
)

TRANSPORT_RETRIES = 2
INTERNAL_SERVER_RETRY_DELAYS_SECONDS = (30.0, 60.0)
OTHER_RETRYABLE_TRANSPORT_DELAYS_SECONDS = (1.0, 2.0)
MAX_CONVERSATION_STEPS = 40
MAX_AGENT_TOOL_CALLS = 24
CANARY_COST_CAP_USD = 25.0
FULL_COST_CAP_USD = 200.0
FULL_COST_PROJECTION_MARGIN = 0.25

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CANARY_EXECUTION_LOCK_PATH = PACKAGE_ROOT / "canary-execution.lock.json"
PREPARED_DIR = PACKAGE_ROOT / "prepared"
PRIVATE_DIR = PACKAGE_ROOT / "private"
RUNS_DIR = PACKAGE_ROOT / "runs"
RECEIPTS_DIR = PACKAGE_ROOT / "receipts"
RUNTIME_SNAPSHOTS_DIR = PACKAGE_ROOT / "runtime-snapshots"
SOURCE_LOCK_PATH = PACKAGE_ROOT / "source.lock.json"
# Earlier previews and all digest-addressed runtime/run artifacts remain
# frozen. V10 writes a distinct authoring pointer before materializing its own
# immutable runtime snapshot.
PREVIEW_PATH = PACKAGE_ROOT / "preview-live-v10.json"

WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    }
)

FORBIDDEN_PUBLIC_KEY_FRAGMENTS = (
    "expected",
    "evaluation_criteria",
    "gold",
    "ground_truth",
    "private_label",
    "review_label",
    "scenario",
    "simulator_prompt",
    "source_url",
    "task_id",
    "task_truth",
)
