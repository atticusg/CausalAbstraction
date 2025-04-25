import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Callable, Tuple

from experiments.experiments import *
from model_units.LM_units import *
from model_units.model_units import *
from pipeline import LMPipeline
from task import Task


class PatchResidualStream(InterventionExperiment):
    def __init__(self,
                 pipeline: LMPipeline,
                 task: Task,
                 layers: List[int],
                 token_positions: List[TokenPosition],
                 checker: Callable,
                 featurizers: Dict[Tuple[int, str], Featurizer] = None,
                 **kwargs):
        """
        Initialize ResidualStreamExperiment for analyzing residual stream interventions.
        
        Args:
            pipeline: LMPipeline object for model execution
            task: Task object containing counterfactual datasets
            layers: List of layer indices to analyze
            token_positions: List of ComponentIndexers for token positions
            checker: Function to evaluate output accuracy
            features: Dict mapping (layer, position.id) to NoFeatures instances
            target_output: Whether to use block output (True) or block input (False)
            metadata: Function to extract metadata from model units
            id: Identifier for the experiment
        """
        self.featurizers = featurizers if featurizers is not None else {}

        # Generate all combinations of model units without feature_indices
        model_units_lists = []
        for layer in layers:
            for pos in token_positions:
                featurizer = self.featurizers.get((layer, pos.id), Featurizer(n_features=pipeline.model.config.hidden_size))
                model_units_lists.append([
                    ResidualStream(
                        layer=layer,
                        token_indices=pos,
                        featurizer=featurizer,
                        shape=(pipeline.model.config.hidden_size,),
                        feature_indices=None, 
                        target_output=True
                    )
                ])

        metadata = lambda x: {"layer": x[0].component.get_layer(), "position": x[0].component.get_index_id()}

        super().__init__(
            pipeline=pipeline,
            task=task,
            model_units_list=model_units_lists,
            checker=checker,
            metadata=metadata,
            **kwargs
        )
        
        self.layers = layers
        self.token_positions = token_positions

    def build_SAE_feature_intervention(self, sae_loader):
        for model_units in self.model_units_list:
            for unit in model_units:
                layer = unit.component.get_layer()
                sae = sae_loader(layer)
                unit.set_featurizer(SAEFeaturizer(sae))

    def plot_heatmaps(self, results: Dict, save_path: str = None, average_counterfactuals: bool = False):
            """
            Generate heatmaps visualizing intervention scores across layers and positions.
            
            Args:
                results: Dictionary containing experiment results from interpret_results()
                save_path: Optional path to save the generated plots. If None, displays plots interactively.
                average_counterfactuals: If True, averages scores across counterfactual datasets
            """
            
            # Extract unique layers and positions from metadata
            # Extract all model units from the first dataset for metadata
            first_dataset = next(iter(results["dataset"]))
            
            # Create a mapping from model_unit string to metadata
            metadata_map = {}
            for unit_str, unit_data in results["dataset"][first_dataset]["model_unit"].items():
                if "metadata" in unit_data:
                    metadata_map[unit_str] = unit_data["metadata"]
            
            # Extract unique layers and positions
            layers = sorted(list(set(metadata["layer"] for metadata in metadata_map.values() if "layer" in metadata)), reverse=True)
            positions = list(set(metadata["position"] for metadata in metadata_map.values() if "position" in metadata))
            
            if average_counterfactuals:
                # Create a score matrix to average across datasets
                score_matrix = np.zeros((len(layers), len(positions)))
                dataset_count = 0
                
                # Sum scores across all datasets
                for dataset_name in results["dataset"]:
                    temp_matrix = np.zeros((len(layers), len(positions)))
                    valid_entries = False
                    
                    # Fill temporary matrix
                    for i, layer in enumerate(layers):
                        for j, pos in enumerate(positions):
                            for unit_str, unit_data in results["dataset"][dataset_name]["model_unit"].items():
                                if "metadata" in unit_data and "accuracy" in unit_data:
                                    metadata = unit_data["metadata"]
                                    if metadata.get("layer") == layer and metadata.get("position") == pos:
                                        temp_matrix[i, j] = unit_data["accuracy"]
                                        valid_entries = True
                    
                    if valid_entries:
                        score_matrix += temp_matrix
                        dataset_count += 1
                
                # Calculate average (avoid division by zero)
                if dataset_count > 0:
                    score_matrix /= dataset_count
                
                # Create averaged heatmap
                plt.figure(figsize=(10, 6))
                display_matrix = np.round(score_matrix * 100, 2).astype(int)
                
                sns.heatmap(
                    score_matrix,
                    xticklabels=positions,
                    yticklabels=layers,
                    cmap='viridis',
                    annot=display_matrix,
                    fmt='.2f',
                    cbar_kws={'label': 'Accuracy (%)'},
                    vmin=0,
                    vmax=1,
                )
                
                plt.yticks(rotation=0)
                plt.xlabel('Position')
                plt.ylabel('Layer')
                plt.title(f'Intervention Accuracy - Average across {dataset_count} datasets\nTask: {results["task_name"]}')
                
                if save_path:
                    plt.savefig(os.path.join(save_path, f'heatmap_average_{results["task_name"]}.png'), 
                            bbox_inches='tight', 
                            dpi=300)
                    plt.close()
                else:
                    plt.show()
            
            else:
                # Get dataset names
                dataset_names = list(results["dataset"].keys())
                
                # Track if we have valid data for any dataset
                any_valid_entries = False
                
                # Create individual heatmaps for each dataset
                for dataset_name in dataset_names:
                    score_matrix = np.zeros((len(layers), len(positions)))
                    valid_entries = False
                    
                    # Fill score matrix
                    for i, layer in enumerate(layers):
                        for j, pos in enumerate(positions):
                            for unit_str, unit_data in results["dataset"][dataset_name]["model_unit"].items():
                                if "metadata" in unit_data and "accuracy" in unit_data:
                                    metadata = unit_data["metadata"]
                                    if metadata.get("layer") == layer and metadata.get("position") == pos:
                                        score_matrix[i, j] = unit_data["accuracy"]
                                        valid_entries = True
                    
                    if valid_entries:
                        any_valid_entries = True
                        
                        # Create a new figure for each dataset
                        plt.figure(figsize=(8, 6))
                        display_matrix = np.round(score_matrix * 100, 2)
                        
                        sns.heatmap(
                            score_matrix,
                            xticklabels=positions,
                            yticklabels=layers,
                            cmap='viridis',
                            annot=display_matrix,
                            fmt='.2f',
                            cbar_kws={'label': 'Accuracy (%)'},
                            vmin=0,
                            vmax=1,
                        )
                        
                        plt.yticks(rotation=0)
                        plt.xlabel('Position')
                        plt.ylabel('Layer')
                        plt.title(f'Intervention Accuracy - Dataset: {dataset_name}\nTask: {results["task_name"]}')
                        plt.tight_layout()
                        
                        if save_path:
                            # Create a file name with the dataset name
                            # Remove/replace problematic characters in the dataset name for the filename
                            safe_dataset_name = dataset_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                            plt.savefig(os.path.join(save_path, f'heatmap_{safe_dataset_name}_{results["task_name"]}.png'), 
                                    bbox_inches='tight', 
                                    dpi=300)
                            plt.close()
                        else:
                            plt.show()
                
                if not any_valid_entries and save_path is None:
                    print("No valid data found for visualization.")

class PatchIOIHeads(InterventionExperiment):
       def __init__(self,
                 pipeline: LMPipeline,
                 task: Task,
                 layers: List[int],
                 layer_head_list: List[Tuple[int, int]],
                 token_positions: List[TokenPosition],
                 checker: Callable,
                 featurizers: Dict[Tuple[int, str], Featurizer] = None,
                 **kwargs):
        """
        Initialize ResidualStreamExperiment for analyzing residual stream interventions.
        
        Args:
            pipeline: LMPipeline object for model execution
            task: Task object containing counterfactual datasets
            layers: List of layer indices to analyze
            token_positions: List of ComponentIndexers for token positions
            checker: Function to evaluate output accuracy
            features: Dict mapping (layer, position.id) to NoFeatures instances
            target_output: Whether to use block output (True) or block input (False)
            metadata: Function to extract metadata from model units
            id: Identifier for the experiment
        """
        if layer_head_list is None:
            layer_head_list = [(7, 3), (7, 9), (8, 6), (8, 10)]
        self.featurizers = featurizers if featurizers is not None else {}
        # Generate all combinations of model units without feature_indices
        model_units_lists = []
        head_size = pipeline.model.config.hidden_size//pipeline.model.config.n_head
        for layer, head in layer_head_list:
            featurizer = Featurizer(n_features=head_size)
            model_units_lists.append(
                AttentionHead(
                    layer=layer,
                    head=head,
                    token_indices=token_positions,
                    featurizer=featurizer,
                    feature_indices=None, 
                    target_output=True,
                    shape = (head_size,)
                )
            )
            
        metadata = lambda x: {"layer": x[0].component.get_layer(), "position": x[0].component.get_index_id()}
        super().__init__(
            pipeline=pipeline,
            task=task,
            model_units_list=[model_units_lists],
            checker=checker,
            metadata=metadata,
            **kwargs
        )
        
        self.layers = layers
        self.token_positions = token_positions
 