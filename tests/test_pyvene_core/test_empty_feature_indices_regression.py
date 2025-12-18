"""
Regression test for empty feature_indices bug.

This test ensures that including interventions with feature_indices=[] (empty list)
produces the same outputs as excluding those interventions entirely.

Bug context: Previously, when using the featurizer wrapper pattern, including
attention heads with feature_indices=[] produced different logit distributions
than excluding those heads entirely, even though empty feature_indices should
mean "intervene on no features" (i.e., no intervention).

The bug was fixed by ensuring pyvene properly handles empty subspaces.
"""

import pytest
import torch
from causalab.neural.pipeline import LMPipeline
from causalab.neural.LM_units import AttentionHead
from causalab.neural.token_position_builder import TokenPosition
from causalab.neural.model_units import Featurizer
from causalab.neural.pyvene_core.interchange import batched_interchange_intervention
from causalab.neural.pyvene_core.intervenable_model import prepare_intervenable_model
from causalab.neural.model_units import InterchangeTarget


@pytest.mark.slow
class TestEmptyFeatureIndicesRegression:
    """Regression tests for the empty feature_indices bug."""

    @pytest.fixture
    def model_name(self):
        """Model to use for testing."""
        return "meta-llama/Llama-3.2-1B-Instruct"

    @pytest.fixture
    def pipeline(self, model_name):
        """Create a pipeline for testing."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = LMPipeline(model_name, max_new_tokens=1, device=device)
        pipeline.tokenizer.padding_side = "left"
        return pipeline

    @pytest.fixture
    def test_batch(self):
        """Create test batch with base and counterfactual inputs."""
        return {
            "input": [{"raw_input": "The banana is yellow. What color is the banana?"}],
            "counterfactual_inputs": [
                [{"raw_input": "The banana is green. What color is the banana?"}]
            ],
        }

    @pytest.fixture
    def token_position(self, pipeline, test_batch):
        """Create token position indexer for last token."""
        return TokenPosition(
            lambda x: [len(pipeline.load(x)["input_ids"][0]) - 1],
            pipeline,
            id="last_token",
        )

    def test_empty_feature_indices_matches_excluded_heads(
        self, pipeline, test_batch, token_position
    ):
        """
        Test that including heads with feature_indices=[] produces the same output
        as excluding those heads entirely.

        This is a regression test for a bug where empty feature_indices caused
        different outputs during interventions.
        """
        # Get model config
        num_heads = pipeline.get_num_attention_heads()
        p_config = pipeline.model.config
        if hasattr(p_config, "head_dim"):
            head_size = p_config.head_dim
        else:
            if hasattr(p_config, "n_head"):
                num_heads = p_config.n_head
            elif hasattr(p_config, "num_attention_heads"):
                num_heads = p_config.num_attention_heads
            elif hasattr(p_config, "num_heads"):
                num_heads = p_config.num_heads
            head_size = pipeline.model.config.hidden_size // num_heads

        # Setup 1: All heads included, some with empty feature_indices
        units_1 = []
        for layer, head, feature_idx in [
            (0, 3, None),
            (0, 7, []),
            (0, 9, []),
            (0, 12, None),
        ]:
            featurizer = Featurizer(n_features=head_size)
            unit = AttentionHead(
                layer=layer,
                head=head,
                token_indices=token_position,
                featurizer=featurizer,
                feature_indices=feature_idx,
                target_output=True,
                shape=(head_size,),
            )
            units_1.append(unit)
        interchange_target_1 = InterchangeTarget([units_1])

        # Setup 2: Only heads with feature_indices=None (exclude empty ones)
        units_2 = []
        for layer, head in [(0, 3), (0, 12)]:
            featurizer = Featurizer(n_features=head_size)
            unit = AttentionHead(
                layer=layer,
                head=head,
                token_indices=token_position,
                featurizer=featurizer,
                feature_indices=None,
                target_output=True,
                shape=(head_size,),
            )
            units_2.append(unit)
        interchange_target_2 = InterchangeTarget([units_2])

        # Run interventions for both setups
        intervenable_model_1 = prepare_intervenable_model(
            pipeline, interchange_target_1, intervention_type="interchange"
        )

        with torch.no_grad():
            output_1 = batched_interchange_intervention(
                pipeline,
                intervenable_model_1,
                test_batch,
                interchange_target_1,
                output_scores=True,
            )

        del intervenable_model_1
        torch.cuda.empty_cache()

        intervenable_model_2 = prepare_intervenable_model(
            pipeline, interchange_target_2, intervention_type="interchange"
        )

        with torch.no_grad():
            output_2 = batched_interchange_intervention(
                pipeline,
                intervenable_model_2,
                test_batch,
                interchange_target_2,
                output_scores=True,
            )

        del intervenable_model_2
        torch.cuda.empty_cache()

        # Compare outputs
        sequences_match = torch.equal(output_1["sequences"], output_2["sequences"])
        assert sequences_match, "Generated sequences should match"

        # Compare logits (if available)
        if (
            "scores" in output_1
            and "scores" in output_2
            and output_1["scores"]
            and output_2["scores"]
        ):
            logits_1 = output_1["scores"][0][0]
            logits_2 = output_2["scores"][0][0]

            max_diff = torch.max(torch.abs(logits_1 - logits_2)).item()
            mean_diff = torch.mean(torch.abs(logits_1 - logits_2)).item()

            # Assert that logits are very close (within numerical precision)
            # Using a reasonable tolerance for floating point comparisons
            assert max_diff < 1e-4, (
                f"Max logit difference {max_diff} exceeds tolerance. This indicates the empty feature_indices bug has regressed."
            )
            assert mean_diff < 1e-5, (
                f"Mean logit difference {mean_diff} exceeds tolerance. This indicates the empty feature_indices bug has regressed."
            )

            # Also check with torch.allclose for more robust comparison
            logits_match = torch.allclose(logits_1, logits_2, rtol=1e-5, atol=1e-8)
            assert logits_match, (
                "Logits should match within tolerance. The empty feature_indices bug may have regressed."
            )

    def test_all_empty_feature_indices_equivalent_to_no_intervention(
        self, pipeline, test_batch, token_position
    ):
        """
        Test that all heads having feature_indices=[] is equivalent to not
        doing any intervention at all.
        """
        # Get model config
        p_config = pipeline.model.config
        if hasattr(p_config, "head_dim"):
            head_size = p_config.head_dim
        else:
            num_heads = (
                p_config.num_attention_heads
                if hasattr(p_config, "num_attention_heads")
                else p_config.n_head
            )
            head_size = pipeline.model.config.hidden_size // num_heads

        # Setup: All heads with empty feature_indices
        units = []
        for layer, head in [(0, 3), (0, 7), (0, 9), (0, 12)]:
            featurizer = Featurizer(n_features=head_size)
            unit = AttentionHead(
                layer=layer,
                head=head,
                token_indices=token_position,
                featurizer=featurizer,
                feature_indices=[],  # Empty - no intervention
                target_output=True,
                shape=(head_size,),
            )
            units.append(unit)
        interchange_target = InterchangeTarget([units])

        # Run intervention with all empty feature_indices
        intervenable_model = prepare_intervenable_model(
            pipeline, interchange_target, intervention_type="interchange"
        )

        with torch.no_grad():
            intervened_output = batched_interchange_intervention(
                pipeline,
                intervenable_model,
                test_batch,
                interchange_target,
                output_scores=True,
            )

        del intervenable_model
        torch.cuda.empty_cache()

        # Run without any intervention
        with torch.no_grad():
            base_inputs = test_batch["input"]
            no_intervention_output = pipeline.generate(base_inputs, output_scores=True)

        # Compare outputs
        sequences_match = torch.equal(
            intervened_output["sequences"], no_intervention_output["sequences"]
        )
        assert sequences_match, (
            "With all empty feature_indices, output should match no-intervention case"
        )

        # Compare logits if available
        if (
            "scores" in intervened_output
            and intervened_output["scores"]
            and "scores" in no_intervention_output
            and no_intervention_output["scores"]
        ):
            logits_intervened = intervened_output["scores"][0][0]
            logits_no_intervention = no_intervention_output["scores"][0][0]

            max_diff = torch.max(
                torch.abs(logits_intervened - logits_no_intervention)
            ).item()

            # Allow for slightly more tolerance here since we're comparing across different code paths
            assert max_diff < 1e-3, (
                f"Max logit difference {max_diff} with all empty feature_indices should be minimal"
            )
