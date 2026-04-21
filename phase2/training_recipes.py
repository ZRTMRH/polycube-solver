"""
Grid-specific recommended supervised-training recipes.
"""


def recommended_training_recipe(grid_size):
    """Return a pragmatic default training recipe for a target grid size."""
    if grid_size >= 5:
        return {
            "num_instances": 160,
            "num_negatives": 0,
            "epochs": 45,
            "batch_size": 48,
            "lr": 1e-3,
            "hidden_dim": 160,
            "num_residual_blocks": 8,
            "value_head_type": "gap",
            "policy_head_type": "conv",
            "use_context_features": True,
            "lambda_value": 1.0,
            "lambda_policy": 0.5,
            "instance_source": "constructive",
            "constructive_variant": "robust",
        }

    if grid_size == 4:
        return {
            "num_instances": 80,
            "num_negatives": 1,
            "epochs": 35,
            "batch_size": 64,
            "lr": 1e-3,
            "hidden_dim": 128,
            "num_residual_blocks": 6,
            "value_head_type": "gap",
            "policy_head_type": "conv",
            "use_context_features": True,
            "lambda_value": 1.0,
            "lambda_policy": 0.3,
            "instance_source": "constructive",
            "constructive_variant": "mixed",
        }

    return {
        "num_instances": 100,
        "num_negatives": 2,
        "epochs": 50,
        "batch_size": 64,
        "lr": 1e-3,
        "hidden_dim": 128,
        "num_residual_blocks": 6,
        "value_head_type": "gap",
        "policy_head_type": "conv",
        "use_context_features": False,
        "lambda_value": 1.0,
        "lambda_policy": 0.3,
        "instance_source": "auto",
        "constructive_variant": "mixed",
    }
