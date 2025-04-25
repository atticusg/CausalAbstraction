from abc import ABC, abstractmethod
from typing import Union, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc


class Pipeline(ABC):
    """Base abstract Pipeline class"""
    def __init__(self,
                 model_or_name,
                 **kwargs):
        self.model_or_name = model_or_name
        self._setup_model(**kwargs)

    @abstractmethod
    def _setup_model(self):
        """Initialize model and any other required components"""
        pass

    @abstractmethod
    def load(self, raw_input):
        """Convert raw input into model-ready format"""
        pass

    @abstractmethod
    def dump(self, model_output):
        """Convert model output into desired format"""
        pass

    @abstractmethod
    def generate(self, prompt, max_new_tokens):
        """Generate output from input prompt"""
        pass

    def intervenable_generate(intervenable_model, base, sources, map, feature_indices, max_new_tokens):
        """Generate output for IntervenableModel in pyvene"""
        pass


class LMPipeline(Pipeline):
    """Pipeline specifically for language models"""
    def __init__(self,
                 model_or_name: str,
                 max_new_tokens: int = 3,
                 max_length = None,
                 logit_labels: bool =False,
                 position_ids: bool = False,
                 **kwargs):
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.logit_labels = logit_labels
        self.position_ids = position_ids
        super().__init__(model_or_name, **kwargs)

    def _setup_model(self, **kwargs):
        if isinstance(self.model_or_name, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_or_name, dtype=kwargs.get("dtype", torch.float16))
            self.model = AutoModelForCausalLM.from_pretrained(self.model_or_name, config=kwargs.get("config", None))
            self.model = self.model.to(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            self.model = self.model.to(kwargs.get("dtype", torch.float16))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_or_name.config.name_or_path, dtype=kwargs.get("dtype", torch.float16))
            self.model = self.model_or_name

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[-1]

    def load(self, raw_input: Union[str, List[str]], max_length=None, padding_side=None, add_special_tokens=True):
        """
        Tokenize and prepare text input
        Handles both single strings and lists of strings
        """
        if max_length is None:
            max_length = self.max_length
        # Convert single string to list for consistent handling
        if isinstance(raw_input, str):
            raw_input = [raw_input]

        inputs = self.tokenizer(
            raw_input,
            padding='max_length' if max_length else True,
            max_length=max_length,
            truncation=max_length is not None,
            return_tensors='pt',
            padding_side=padding_side,
            add_special_tokens=add_special_tokens
        )
        if self.position_ids:
            inputs["position_ids"] = self.model.prepare_inputs_for_generation(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )["position_ids"]
        # Move all tensors to the model's device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)
        return inputs 

    def dump(self, model_output, is_logits=True):
        """
        Convert a (batched) model output to a (list of) decoded string(s)
        """
        generated_ids = None
        # Turn the model output into a tensor 
        # of shape batch_size x num_tokens (x logits)
        if isinstance(model_output, tuple) or isinstance(model_output, list):
            if len(model_output) == 1:
                # If output is a single tensor, extract it
                model_output = model_output[0].unsqueeze(1)
            else:
                # If output is a list of tensors, stack them
                model_output = torch.stack(model_output, dim=1)
        # Turn the tensor into the shape of (batch_size, num_tokens)
        if model_output.dim() == 3 or is_logits: 
            # If output is 2D and logits, take the argmax to get token IDs
            # and add a batch dimension
            generated_ids = model_output.argmax(dim=-1)
        elif model_output.dim() == 1:
            generated_ids = generated_ids.unsqueeze(0)
        else:
            generated_ids = model_output

        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        # Return single string if input was unbatched
        return decoded[0] if len(decoded) == 1 else decoded
    
    def tokenize_label(self, text_label, length):
        """Convert label text to token sequence padded ON THE RIGHT up to length"""
        if isinstance(text_label, str):
            text_label = [text_label]
        current_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        result = self.tokenizer(
            text_label, 
            padding="max_length",
            truncation=True,
            max_length=length, 
            return_tensors="pt", 
            add_special_tokens=False)
        self.tokenizer.padding_side = current_padding_side
        #Return sequence of tokens
        return result["input_ids"]

    def generate(self, prompt: Union[str, List[str]]):
        """
        Generate text from prompt(s)
        Handles both single strings and lists of strings
        """
        # Convert single string to list for consistent handling
        if isinstance(prompt, str):
            prompt = [prompt]
        inputs = self.load(prompt)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=False,
            )
        scores = [score.clone().detach().to("cpu") for score in output.scores]

        del output.scores
        del output.sequences
        del output.past_key_values
        del output
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
        return scores


    def intervenable_generate(self, intervenable_model, base, sources,
                              inv_locations, feature_indices,
                              output_scores=False):
        with torch.no_grad():
            output = intervenable_model.generate(
                base,
                sources=sources,
                unit_locations=inv_locations,
                subspaces=feature_indices,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=output_scores,
                intervene_on_prompt=True,
                do_sample=False,
                use_cache=False,
            )
        # Output probabilities in batch_size * num_token * vocab_size or
        #        token_ids in batch_size * num_token
        return (output[-1].scores if output_scores else
                output[-1].sequences[:, -self.max_new_tokens:])

    def get_num_layers(self):
        """Get the number of layers in the model"""
        return self.model.config.num_hidden_layers