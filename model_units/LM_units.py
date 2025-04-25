from model_units.model_units import *
from pipeline import LMPipeline
from typing import List, Union

class TokenPosition(ComponentIndexer):
    def __init__(self,
                indexer,
                pipeline: LMPipeline,
                **kwargs
                ):
        """
        Initialize a TokenPosition object.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            max_length (int): The maximum length of the input.
        """
        super().__init__(indexer, **kwargs)
        self.pipeline = pipeline
        self.tokenizer = pipeline.tokenizer

    def highlight_selected_token(self, prompt: str) -> str:
        """
        Highlight selected tokens in the prompt based on the indexer.
        Returns a string with highlighted tokens marked in **bold**.
        
        Args:
            prompt (str): The input prompt to process.
            
        Returns:
            str: A string with highlighted tokens marked
        """
        # Tokenize the input prompt
        ids = self.pipeline.load(prompt)["input_ids"][0]
        
        # Get token positions to highlight using the indexer
        highlighted_positions = self.index(prompt) 
        
        if not isinstance(highlighted_positions, list):
            highlighted_positions = [highlighted_positions]
        
        # Validate indices
        highlighted_positions = [
            pos for pos in highlighted_positions 
            if 0 <= pos < len(ids)
        ]
        
        # Format tokens with highlighting
        formatted_tokens = []
        for i, id in enumerate(ids):
            id = self.tokenizer.decode(id)
            if i in highlighted_positions:
                formatted_tokens.append(f"**{id}**")
            else:
                formatted_tokens.append(id)
        
        return "".join(formatted_tokens)

def get_last_token_index(prompt, pipeline):
    input_ids = list(pipeline.load(prompt)["input_ids"][0])
    return [len(input_ids) - 1]

class ResidualStream(AtomicModelUnit):
    def __init__(self,
                layer: int,
                token_indices: Union[List[int], ComponentIndexer], 
                featurizer=Featurizer(),
                shape=None,
                feature_indices=None,
                target_output=False):
        """
        Initialize a ResidualStreamVector object.
        """
        component_type = "block_output" if target_output else "block_input"
        if isinstance(token_indices, ComponentIndexer):
            tok_id = token_indices.id
        else:
            tok_id = token_indices
        id = f"ResidualStream(Layer:{layer},Token:{tok_id})"
        unit = "pos"
        self.unit = unit
        component = None
        if isinstance(token_indices, list):
            component = StaticComponent(layer, component_type, token_indices, unit)
        elif isinstance(token_indices, ComponentIndexer):
            component = Component(layer, component_type, token_indices, unit)
        super().__init__(component, featurizer, feature_indices, shape, id=id)

class AttentionHead(AtomicModelUnit):
    def __init__(self,
                layer: int,
                head: int,
                token_indices: Union[List[int], ComponentIndexer], 
                featurizer=Featurizer(),
                shape=None,
                feature_indices=None,
                target_output=False):
        """
        Initialize an AttentionHead object.
        """
        self.head = head
        component_type = "head_attention_value_output" if target_output else "head_attention_value_input"
        if isinstance(token_indices, ComponentIndexer):
            tok_id = token_indices.id
        else:
            tok_id = token_indices
        id = f"AttentionHead(Layer:{layer},Token:{tok_id})"
        self.unit = "h.pos"
        component = None
        if isinstance(token_indices, list):
            component = StaticComponent(layer, component_type, token_indices, self.unit)
        elif isinstance(token_indices, ComponentIndexer):
            component = Component(layer, component_type, token_indices, self.unit)
        super().__init__(component, featurizer, feature_indices, shape, id=id)
    
    def index_component(self, input):
        return [
            [self.head],
            self.component.index(input)
        ]
    