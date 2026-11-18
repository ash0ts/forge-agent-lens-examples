# Release gate report

Decision: **pass**. CI exit code: **0**. Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B`.

Synthetic hand-authored tickets. This gate decides whether a prompt change may proceed; it deploys nothing.

| Variant | Split | Correct / cases | Accuracy | Complete |
| --- | --- | --- | --- | --- |
| baseline | all | 64 / 64 | 1.000 | 64 |
| baseline | development | 16 / 16 | 1.000 | 16 |
| baseline | heldout | 48 / 48 | 1.000 | 48 |
| candidate | all | 64 / 64 | 1.000 | 64 |
| candidate | development | 16 / 16 | 1.000 | 16 |
| candidate | heldout | 48 / 48 | 1.000 | 48 |

Negative control (deliberately bad prompt): 2 / 8 correct, rejected by the gate.

Policy checks:

- valid_membership: pass
- required_heldout_count: pass
- complete_responses: pass
- strict_label_contract: pass
- candidate_absolute_floor: pass
- every_category_floor: pass
- candidate_delta_floor: pass
- paired_regression_budget: pass
- negative_control_rejected: pass
