import collections
import random
from typing import Dict
import torch
import gc

from causal_model import CausalModel
from datasets import Dataset
from model_units.model_units import ComponentIndexer
from tqdm.auto import tqdm


class Task:
    def __init__(self, 
                causal_model:CausalModel, 
                counterfactual_datasets: Dict[str, Dataset],
                input_dumper= None,
                output_dumper= None,
                input_loader= None,
                id="null"
                ):
        self.causal_model = causal_model
        self.input_dumper = input_dumper if input_dumper is not None else lambda x: x
        self.output_dumper = output_dumper if output_dumper is not None else lambda x: x
        self.input_loader = input_loader if input_loader is not None else lambda x: x 
        self.counterfactual_datasets = counterfactual_datasets
        self.id = id

        inputs = []
        labels = []
        for _, dataset in self.counterfactual_datasets.items():
            for input in tqdm(dataset["input"]):
                inputs.append(input)
                setting = self.causal_model.run_forward(input)
                labels.append(self.output_dumper(setting))
        self.all_data = dataset.from_dict({"input": inputs, "label": labels})

        self.raw_all_data = self.create_raw_data(self.all_data)
        self.raw_counterfactual_datasets = {name: self.create_raw_data(self.counterfactual_datasets[name]) for name in self.counterfactual_datasets}

    def create_raw_data(self, dataset):
        raw_inputs, raw_counterfactual_inputs, labels = [], [], []
        for example in dataset:
            raw_inputs.append(self.input_dumper(example["input"]))
            if "counterfactual_inputs" in example:
                raw_counterfactual_inputs.append([self.input_dumper(x) for x in example["counterfactual_inputs"]])
            if "label" in example:
                labels.append(example["label"])
        data = {"input": raw_inputs}
        if len(raw_counterfactual_inputs) > 0:
            data["counterfactual_inputs"] = raw_counterfactual_inputs
        if len(labels) > 0:
            data["label"] = labels
        return Dataset.from_dict(data)
    
    def display_counterfactual_data(self):
        for dataset_name, dataset in self.counterfactual_datasets.items():
            print(f"Dataset '{dataset_name}':")
            example = dataset[0]
            print(f"Input: {self.input_dumper(example['input'])}")
            print(f"Counterfactual Inputs:")
            for counterfactual_input in example["counterfactual_inputs"]:
                print(f"{self.input_dumper(counterfactual_input)}")
            print()
    
    def sample_raw_input(self):
        return random.choice(self.raw_all_data["input"])
    
    def label_probe_data(self, dataset, target_variables):
        inputs = []
        labels = []
        label_to_setting = {}

        new_id = 0
        for example in dataset:
            inputs.append(example["input"])
            setting = self.causal_model.run_forward(example["input"])
            target_labels = [str(setting[var]) for var in target_variables]
            if "".join(target_labels) in label_to_setting:
                id = label_to_setting["".join(target_labels)]
            else:
                id = new_id
                label_to_setting["".join(target_labels)] = new_id
                new_id += 1
            labels.append(id)
        return Dataset.from_dict({"input": inputs, "label": labels}), label_to_setting

    
    def label_counterfactual_data(self, name_or_dataset, target_variables):
        if isinstance(name_or_dataset, str):
            dataset = self.counterfactual_datasets[name_or_dataset]
        else:
            dataset = name_or_dataset

        inputs = []
        counterfactual_inputs = []
        labels = []
        for example in dataset:
            if len(example["counterfactual_inputs"]) != len(target_variables) and len(example["counterfactual_inputs"]) == 1:
                example["counterfactual_inputs"] = [example["counterfactual_inputs"][0]] * len(target_variables)
            inputs.append(self.input_dumper(example["input"]))
            counterfactual_inputs.append([self.input_dumper(counterfactual_input) for counterfactual_input in example["counterfactual_inputs"]])
            setting = self.causal_model.run_interchange(example["input"], dict(zip(target_variables, example["counterfactual_inputs"])))
            labels.append(self.output_dumper(setting))
        result = {"input": inputs, "counterfactual_inputs": counterfactual_inputs}
        if target_variables is not None:
            result["label"] = labels
        return Dataset.from_dict(result)

    def filter(self, pipeline, checker, batch_size=32, verbose=False):
        """
        Filter dataset based on agreement between pipeline and causal model outputs,
        processing and filtering data in batches for efficiency.
        
        Args:
            pipeline: Model pipeline that processes inputs
            checker: Function that compares model output with expected output
            batch_size: Size of batches for processing (default: 32)
            verbose: Whether to print filtering statistics (default: False)
        """
        filtered_datasets = {}
        total_original = 0
        total_kept = 0
        
        # Process each counterfactual dataset
        for dataset_name, dataset in self.counterfactual_datasets.items():
            filtered_data = collections.defaultdict(list)
            dataset_original = len(dataset["input"])
            total_original += dataset_original
            
            # Process dataset in batches
            for b_i in tqdm(range(0, len(dataset["input"]), batch_size)):
                # Get batch of original inputs and their counterfactuals
                orig_inputs = dataset["input"][b_i:b_i + batch_size]
                all_cf_inputs = dataset["counterfactual_inputs"][
                    b_i:b_i + batch_size]
                # Process original inputs in batch
                orig_dumped = [self.input_dumper(x) for x in orig_inputs]
                orig_preds = pipeline.dump(pipeline.generate(
                    orig_dumped))
                orig_expected = [
                    self.output_dumper(self.causal_model.run_forward(x))
                    for x in orig_inputs]
                orig_valid = [
                    checker(pred, exp)
                    for pred, exp in zip(orig_preds, orig_expected)]
                # Process counterfactuals in batch.
                cf_dumped = [[self.input_dumper(x) for x in cf_inputs]
                              if orig_valid[i] else None
                              for i, cf_inputs in enumerate(all_cf_inputs)]
                cf_dumped_flatten = [x for xs in cf_dumped if xs for x in xs]
                if not cf_dumped_flatten:
                  continue
                cf_pred_flatten = pipeline.dump(
                        pipeline.generate(cf_dumped_flatten))
                cf_preds, offset = [], 0
                for i, xs in enumerate(cf_dumped):
                  cf_preds.append(cf_pred_flatten[offset:offset + len(xs)]
                                  if xs else None)
                  offset += len(xs) if xs else 0
                cf_expected = [
                    [self.output_dumper(self.causal_model.run_forward(x))
                     for x in cf_inputs]
                    for cf_inputs in all_cf_inputs]
                cf_valid = [all(checker(pred, exp) for pred, exp in zip(
                                cf_preds[i], cf_expected[i]))
                            if cf_preds[i] else None
                            for i in range(len(cf_preds))]
                # Filter valid original and counterfactual input pairs.
                for idx, is_orig_valid in enumerate(orig_valid):
                    if not is_orig_valid or not cf_valid[idx]:
                        continue
                    # If both pass, add to filtered data
                    filtered_data["input"].append(orig_inputs[idx])
                    filtered_data["counterfactual_inputs"].append(
                            all_cf_inputs[idx])

                del orig_inputs, all_cf_inputs, orig_dumped, orig_preds, orig_expected
                del orig_valid, cf_dumped, cf_dumped_flatten, cf_pred_flatten, cf_preds
                del cf_expected, cf_valid
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

            # Create filtered dataset
            filtered_datasets[dataset_name] = Dataset.from_dict(filtered_data)
            dataset_kept = len(filtered_data["input"])
            total_kept += dataset_kept
            
            if verbose:
                print(f"Dataset '{dataset_name}': kept {dataset_kept}/{dataset_original} examples " 
                    f"({(dataset_kept/dataset_original)*100:.1f}%)")
        
        # Update counterfactual datasets
        self.counterfactual_datasets = filtered_datasets
        
        inputs = []
        labels = []
        for _, dataset in self.counterfactual_datasets.items():
            if "input" not in dataset.features:
                continue
            for input in dataset["input"]:
                inputs.append(input)
                setting = self.causal_model.run_forward(input)
                labels.append(self.output_dumper(setting))
        self.all_data = dataset.from_dict({"input": inputs, "label": labels})

        # Recreate all_data with filtered results
        self.raw_all_data = self.create_raw_data(self.all_data)
        self.raw_counterfactual_datasets = {name: self.create_raw_data(self.counterfactual_datasets[name]) for name in self.counterfactual_datasets}
        
        if verbose:
            print(f"\nTotal filtering results:")
            print(f"Original examples: {total_original}")
            print(f"Kept examples: {total_kept}")
            print(f"Overall keep rate: {(total_kept/total_original)*100:.1f}%")
