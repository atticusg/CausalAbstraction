from typing import List, Union 
import pyvene as pv
import torch

# COMPONENTS, i.e., hidden vectors inside of neural network models
class ComponentIndexer:
    def __init__(self, indexer, id="null"):
        self.indexer = indexer
        self.id = id

    def index(self, input):
        return self.indexer(input)

    def repr(self):
        return f"ComponentIndexer(id={self.id})"


class Component:
    def __init__(self,
                layer: int,
                component_type: str,
                indices_func: Union[ComponentIndexer,List[int]],
                unit: str = "pos"):
        """
        Initialize base component with dynamic or static indices.
        
        Args:
            layer (int): The layer of the IntervenableModel where the component is located.
            component_type (str): The type component within the layer (e.g., "block_output", "mlp", etc.).
            indices_func: Either a ComponentIndexer for dynamic indices or a List for static indices.
            unit (str): The model variable being indexed (for the residual stream "pos" and for attention heads, pos.h).
        """
        self.component_type = component_type
        self.unit = unit
        self.layer = layer
        
        # If indices_func is a list, convert it to a constant function ComponentIndexer
        if isinstance(indices_func, list):
            constant_indices = indices_func
            self._indices_func = ComponentIndexer(
                lambda _: constant_indices,
                id=f"constant_{constant_indices}"
            )
        else:
            self._indices_func = indices_func

    def get_layer(self) -> int:
        """Get the layer number."""
        return self.layer
    
    def set_layer(self, layer: int):
        """Set the layer number."""
        self.layer = layer
    
    def get_index_id(self) -> str:
        """Get the index id."""
        return self._indices_func.id

    def index(self, input) -> List:
        """Get the component indices."""
        return self._indices_func.index(input)

    def __repr__(self):
        return (f"{self.__class__.__name__}(layer={self.get_layer()}, "
                f"component_type='{self.component_type}', "
                f"component_indices={str(self._indices_func)}, "
                f"unit={self.unit}")

class StaticComponent(Component):
    def __init__(self,
                layer: int,
                component_type: str, 
                component_indices: List,
                unit: str = "pos"):
        """
        Initialize a static component where layer and indices are fixed.
        
        Args:
            layer (int): The layer of the IntervenableModel where the component is located.
            component_type (str): The type component within the layer.
            component_indices (list): An index to the target hidden vector.
            unit (str): The unit being indexed (default="pos").
        """
        super().__init__(layer, component_type, component_indices, unit)

# FEATURIZERS AND FEATURES, e.g., the dimensions of hidden vectors or orthgonal directions of hidden vectors or sparse linear features from an SAE 
def build_feature_interchange_intervention(featurizer, inverse_featurizer):
    class FeatureInterchangeIntervention(pv.TrainableIntervention, pv.DistributedRepresentationIntervention):
        """Intervention in a custom featurized space."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.featurizer = featurizer
            self.inverse_featurizer = inverse_featurizer
        
        def forward(self, base, source, subspaces=None):
            # Apply featurizer to base and source
            featurized_base, base_error = self.featurizer(base)
            featurized_source, _ = self.featurizer(source)
            # Perform intervention in the featurized space
            intervened_features = None
            if subspaces is None or all(inner_list is None or all(element is None for element in inner_list) for inner_list in subspaces):
                intervened_features = featurized_source
            else:
                intervened_features = pv.models.intervention_utils._do_intervention_by_swap(
                            featurized_base,
                            featurized_source,
                            "interchange",
                            self.interchange_dim,
                            subspaces,
                            subspace_partition=self.subspace_partition,
                            use_fast=self.use_fast,
                        )
            output = self.inverse_featurizer(intervened_features, base_error)
            # Apply inverse featurizer to get back to the original space
            return output.to(base.dtype)

        def __str__(self):
            return f"CustomFeaturizedSpaceIntervention({self.featurizer.__name__})"
    return FeatureInterchangeIntervention

def build_feature_collect_intervention(featurizer):
    class FeatureCollectIntervention(pv.CollectIntervention):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.featurizer = featurizer
        
        def forward(self, base, source=None, subspaces=None):
            featurized_base, _ = self.featurizer(base)
            features = pv.models.intervention_utils._do_intervention_by_swap(
                        featurized_base,
                        source,
                        "collect",
                        self.interchange_dim,
                        subspaces,
                        subspace_partition=self.subspace_partition,
                        use_fast=self.use_fast,
                    )
            return features
    return FeatureCollectIntervention

def build_feature_mask_intervention(featurizer, inverse_featurizer, n_features):
    class FeatureMaskIntervention(pv.TrainableIntervention):
        """Intervention that applies differential binary masking in a custom featurized space."""
        def __init__(self,  **kwargs):
            super().__init__(**kwargs)
            self.featurizer = featurizer
            self.inverse_featurizer = inverse_featurizer
            self.temperature = torch.tensor(1e-2)
            self.mask = torch.nn.Parameter(torch.zeros(n_features), requires_grad=True)
        
        def get_temperature(self):
            return self.temperature
        
        def set_temperature(self, temp):
            self.temperature = temp
        
        def forward(self, base, source, subspaces=None):
            # Apply featurizer to base and source
            featurized_base, base_error = self.featurizer(base)
            featurized_source, _ = self.featurizer(source)
            
            input_dtype, model_dtype = featurized_base.dtype, self.mask.dtype
            featurized_base = featurized_base.to(model_dtype)
            featurized_source = featurized_source.to(model_dtype)
            
            # Apply masking intervention in the featurized space
            if self.training:
                mask_sigmoid = torch.sigmoid(self.mask / self.temperature)
                # Apply weighted combination based on mask
                featurized_output = (1.0 - mask_sigmoid) * featurized_base + mask_sigmoid * featurized_source
            else:
                # In inference mode, apply a binary mask based on learned threshold
                mask_binary = (torch.sigmoid(self.mask) > 0.5).float()
                # Only replace the features where mask_binary is 1
                featurized_output = (1.0 - mask_binary) * featurized_base + mask_binary * featurized_source
            
            # Apply inverse featurizer to get back to the original space
            output = self.inverse_featurizer(featurized_output.to(input_dtype), base_error)
            
            return output.to(base.dtype)
            
        def get_sparsity_loss(self):
            if self.mask is None:
                return torch.tensor(0.0)
                
            mask_sigmoid = torch.sigmoid(self.mask / torch.tensor(self.temperature))
            return torch.norm(mask_sigmoid, p=1)
            
        def __str__(self):
            return f"FeatureMaskIntervention({self.featurizer.__name__})"
            
    return FeatureMaskIntervention

class IdentityFeaturizerModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x, None 

class IdentityInverseFeaturizerModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x, error):
        return x

class Featurizer:
    def __init__(self, featurizer=IdentityFeaturizerModule(),
                 inverse_featurizer=IdentityInverseFeaturizerModule(),
                 n_features=None,
                 id="null"):
        self.featurizer = featurizer
        self.inverse_featurizer = inverse_featurizer
        self.n_features = n_features
        self.id = id
    
    def get_interchange_intervention(self):
        if not hasattr(self, "interchange_intervention"):
            self.interchange_intervention = build_feature_interchange_intervention(self.featurizer, self.inverse_featurizer) 
        return self.interchange_intervention
    
    def get_collect_intervention(self):
        if not hasattr(self, "collect_intervention"):
            self.collect_intervention = build_feature_collect_intervention(self.featurizer)
        return self.collect_intervention
    
    def get_mask_intervention(self):
        if not hasattr(self, "mask_intervention"):
            self.mask_intervention = build_feature_mask_intervention(self.featurizer, self.inverse_featurizer, self.n_features)
        return self.mask_intervention

    
    def featurize(self, x):
        return self.featurizer(x)
    
    def inverse_featurize(self, x, error):
        return self.inverse_featurizer(x, error)
    
    def save_modules(self, path):
        """
        Save featurizer and inverse featurizer modules to disk.
        
        Args:
            path (str): Base path for saving the modules
            
        Returns:
            tuple: Paths to the saved featurizer and inverse featurizer files
        """
        # Extract class names without full module path
        featurizer_class = self.featurizer.__class__.__name__
        inverse_featurizer_class = self.inverse_featurizer.__class__.__name__
        
        if featurizer_class == 'SAEFeaturizerModule':
            return "", ""
        # Determine what additional config needs to be saved based on class type
        if featurizer_class == 'SubspaceFeaturizerModule':
            # For SubspaceFeaturizer, we need to save the rotation matrix
            additional_config = {
                'rotation_matrix': self.featurizer.rotate.weight.clone().detach(),
                'requires_grad': self.featurizer.rotate.weight.requires_grad
            }
        else:
            additional_config = {}
        
        # Save model metadata
        model_info = {
            'featurizer_class': featurizer_class,
            'inverse_featurizer_class': inverse_featurizer_class,
            'n_features': self.n_features,
            'additional_config': additional_config
        }
        
        # Save state dictionaries
        torch.save({
            'model_info': model_info,
            'state_dict': self.featurizer.state_dict()
        }, path + "_featurizer") 
        
        torch.save({
            'model_info': model_info,
            'state_dict': self.inverse_featurizer.state_dict()
        }, path + "_inverse_featurizer")
        
        return path + "_featurizer", path + "_inverse_featurizer"

    @classmethod
    def load_modules(cls, path):
        """
        Load featurizer and inverse featurizer modules from disk.
        
        Args:
            path (str): Base path for the saved modules
            
        Returns:
            Featurizer: A new instance with loaded modules
        """
        # Load saved data
        featurizer_data = torch.load(path + "_featurizer")
        inverse_featurizer_data = torch.load(path + "_inverse_featurizer")
        
        model_info = featurizer_data['model_info']
        featurizer_class = model_info['featurizer_class']
        additional_config = model_info['additional_config']
        
        # Initialize appropriate modules based on the saved class names
        if featurizer_class == 'SubspaceFeaturizerModule':
            # Recreate the rotate layer
            rotation_matrix = additional_config['rotation_matrix']
            requires_grad = additional_config['requires_grad']
            
            # Create a rotation layer with the saved dimensions
            rotate_layer = pv.models.layers.LowRankRotateLayer(
                rotation_matrix.shape[0], 
                rotation_matrix.shape[1], 
                init_orth=False
            )
            rotate_layer.weight.data.copy_(rotation_matrix)
            rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
            rotate_layer.requires_grad_(requires_grad)
            
            # Create the modules
            featurizer = SubspaceFeaturizerModule(rotate_layer)
            inverse_featurizer = SubspaceInverseFeaturizerModule(rotate_layer)
            
        elif featurizer_class == 'SAEFeaturizerModule':
            # Recreate the SAE
            from transformer_lens.utils import to_device  # Import here if needed
            
            # Create a SAE with the saved configuration
            sae_config = additional_config['sae_config']
            sae = to_device(pv.models.sparse_autoencoder.SparseAutoencoder.from_dict(sae_config))
            
            # Load weights
            sae_weights = additional_config['sae_weights']
            sae.encoder.weight.data.copy_(sae_weights['encoder'])
            sae.decoder.weight.data.copy_(sae_weights['decoder'])
            if sae_weights['bias'] is not None:
                sae.encoder.bias.data.copy_(sae_weights['bias'])
            
            sae.requires_grad_(additional_config['requires_grad'])
            
            # Create the modules
            featurizer = SAEFeaturizerModule(sae)
            inverse_featurizer = SAEInverseFeaturizerModule(sae)
        elif featurizer_class == 'IdentityFeaturizerModule':
            # Create identity modules
            featurizer = IdentityFeaturizerModule()
            inverse_featurizer = IdentityInverseFeaturizerModule()
        else:
            raise ValueError(f"Unknown featurizer class: {featurizer_class}")
        
        # Load state dicts
        featurizer.load_state_dict(featurizer_data['state_dict'])
        inverse_featurizer.load_state_dict(inverse_featurizer_data['state_dict'])
        
        # Create and return new featurizer instance
        instance = cls(featurizer, inverse_featurizer, n_features=model_info['n_features'])
        return instance


class SubspaceFeaturizerModule(torch.nn.Module):
    def __init__(self, rotate_layer):
        super().__init__()
        self.rotate = rotate_layer
        
    def forward(self, x):
        r = self.rotate.weight.T
        f = x.to(r.dtype) @ r.T
        error = x - (f @ r).to(x.dtype)
        return f, error


class SubspaceInverseFeaturizerModule(torch.nn.Module):
    def __init__(self, rotate_layer):
        super().__init__()
        self.rotate = rotate_layer
        
    def forward(self, f, error):
        r = self.rotate.weight.T
        return (f.to(r.dtype) @ r).to(f.dtype) + error.to(f.dtype)


class SubspaceFeaturizer(Featurizer):
    def __init__(self, shape=None, rotation_subspace=None, trainable=True, id="subspace"):
        assert shape is not None or rotation_subspace is not None, "Either shape or rotation_subspace must be provided."
        if shape is not None:
            self.rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=True)
        elif rotation_subspace is not None:
            shape = rotation_subspace.shape
            self.rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=False)
            self.rotate.weight.data.copy_(rotation_subspace)
        self.rotate = torch.nn.utils.parametrizations.orthogonal(self.rotate)

        if not trainable:
            self.rotate.requires_grad_(False)

        # Create module-based featurizer and inverse_featurizer
        featurizer = SubspaceFeaturizerModule(self.rotate)
        inverse_featurizer = SubspaceInverseFeaturizerModule(self.rotate)
            
        super().__init__(featurizer, inverse_featurizer, n_features=self.rotate.weight.shape[1], id=id)

class SAEFeaturizerModule(torch.nn.Module):
    def __init__(self, sae):
        super().__init__()
        self.sae = sae
    
    def forward(self, x):
        features = self.sae.encode(x.to(self.sae.dtype))
        error = x - self.sae.decode(features).to(x.dtype)
        return features.to(x.dtype), error

class SAEInverseFeaturizerModule(torch.nn.Module):
    def __init__(self, sae):
        super().__init__()
        self.sae = sae
    
    def forward(self, features, error):
        x_recon = self.sae.decode(features.to(self.sae.dtype)).to(features.dtype)
        return x_recon + error.to(features.dtype)

class SAEFeaturizer(Featurizer):
    def __init__(self, sae, trainable=False):
        self.sae = sae
        self.sae.requires_grad_(trainable)
        
        # Create module-based featurizer and inverse_featurizer
        featurizer = SAEFeaturizerModule(self.sae)
        inverse_featurizer = SAEInverseFeaturizerModule(self.sae)
        
        super().__init__(featurizer, inverse_featurizer, n_features=self.sae.cfg.to_dict()["d_sae"])

# MODEL UNITS, i.e., components and features together, which specify an exact location to perform an intervention

class AtomicModelUnit:
    def __init__(self,
                component: Component,
                featurizer= Featurizer(),
                feature_indices: List[int] = None, 
                shape=None,
                id="null"):
        """
        Initialize an AtomicModelUnit object.
        """
        self.id = id
        self.component = component
        self.feature_indices = feature_indices
        self.featurizer = featurizer 
        self.shape = shape
    
    def get_shape(self):
        return self.shape

    def index_component(self, input):
        return self.component.index(input)

    def get_feature_indices(self):
        return self.feature_indices
    
    def set_feature_indices(self, feature_indices: List[int]):
        self.feature_indices = feature_indices
        return
    
    def set_featurizer(self, featurizer):
        self.featurizer = featurizer
        return

    def is_static(self):
        return isinstance(self.component, StaticComponent)

    def set_layer(self, layer: int):
        self.component.layer = layer
        return
    
    def get_layer(self):
        return self.component.layer
    
    def set_subspace_intervention(self, shape=None, rotation_subspace=None, trainable=True, id="subspace"):
        self.featurizer = SubspaceFeaturizer(shape=shape, rotation_subspace=rotation_subspace, trainable=trainable, id=id)

    def create_intervention_config(self, group_key, intervention_type):
        config = {
            "component": self.component.component_type,
            "unit": self.component.unit,
            "layer": self.component.layer,
            "group_key": group_key, 
            }
        if intervention_type == "interchange":
            config["intervention_type"] = self.featurizer.get_interchange_intervention()
        elif intervention_type == "collect":
            config["intervention_type"] = self.featurizer.get_collect_intervention()
        elif intervention_type == "mask":
            config["intervention_type"] = self.featurizer.get_mask_intervention()
        
        return config
    
    def __repr__(self):
        return f"id='{self.id}'" 
