from typing import List, Dict, Callable, Tuple, Union
import gc, json, os, collections, random

import pyvene as pv
import torch
import numpy as np
import transformers
from sklearn.decomposition import PCA
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_units.model_units import *
from pipeline import Pipeline
from task import Task

def _delete(intervenable_model):
    intervenable_model.set_device("cpu", set_model=False)
    del intervenable_model
    gc.collect()
    torch.cuda.empty_cache()
    return

def _prepare_intervenable_model(pipeline: Pipeline, model_unit_lists: List[AtomicModelUnit], intervention_type:str="interchange"):
    static = True
    for model_units in model_unit_lists:
        for model_unit in model_units:
            if not model_unit.is_static():
                static = False

    configs = []
    i = 0
    for model_units in model_unit_lists:
        for model_unit in model_units:
            config = model_unit.create_intervention_config(0, intervention_type)
            i += 1
            configs.append(config)
    intervention_config = pv.IntervenableConfig(configs)
    intervenable_model = pv.IntervenableModel(intervention_config, model = pipeline.model, use_fast=static)
    intervenable_model.set_device(pipeline.model.device)
    return intervenable_model

def _prepare_interventable_inputs(pipeline, intervenable_model, batch, model_units):
    batched_base = batch["input"]
    batched_sources = list(zip(batch["counterfactual_inputs"]))[0]

    if model_units[0].unit == 'h.pos': # Attention heads
        base_indices = [
            [[model_unit.index_component(base)[0] for base in batched_base],
            [model_unit.index_component(base)[1] for base in batched_base]]
            for model_unit in model_units
        ]

        source_indices = [
            [[model_unit.index_component(source)[0] for source in batched_sources[0]], 
            [model_unit.index_component(source)[1] for source in batched_sources[0]]]
            for model_unit in model_units
        ]

    else:
        base_indices = [
            [model_unit.index_component(base) for base in batched_base]
            for model_unit in model_units
        ]

        source_indices = [
            [model_unit.index_component(source) for source in batched_source]
            for model_unit, batched_source in zip(model_units, batched_sources)
        ]

    feature_indices= [
        [model_unit.get_feature_indices() for _ in range(len(batched_base))]
        for model_unit in model_units
    ]

    batched_base = pipeline.load(batched_base)
    batched_sources = [pipeline.load(batched_source) for batched_source in batched_sources]
    if pipeline.tokenizer.padding_side == "left":
        if model_units[0].unit != 'h.pos': # Attention heads
            pad_token_id = pipeline.tokenizer.pad_token_id
            base_indices = [
                [[j + (base==pad_token_id).sum().item() for j in index] for base, index in zip(batched_base["input_ids"], indices)]
                for indices in base_indices
            ]

            source_indices = [
                [[j + (source==pad_token_id).sum().item() for j in index] for source, index in zip(batched_source["input_ids"], indices)]
                for indices, batched_source in zip(source_indices, batched_sources)
            ]

    inv_locations = {"sources->base": (source_indices, base_indices)}
    return batched_base, batched_sources, inv_locations, feature_indices


def _train_with_batched_interchange_intervention(pipeline, intervenable_model, batch, model_units):
    batched_base, batched_sources, inv_locations, feature_indices = _prepare_interventable_inputs(
        pipeline, intervenable_model, batch, model_units)

    if not pipeline.logit_labels:
        batched_inv_label = batch['label']
        batched_inv_label = pipeline.load(
            batched_inv_label, max_length=pipeline.max_new_tokens, padding_side='right', add_special_tokens=False)
        for k in batched_base:
            batched_base[k] = torch.cat([batched_base[k], batched_inv_label[k]], dim=-1)
        _, counterfactual_logits = intervenable_model(
            batched_base, batched_sources, unit_locations=inv_locations, subspaces=feature_indices)
        labels = batched_inv_label['input_ids']
        return {
            # Keep the logits that correspond to the label tokens.
            'logits': counterfactual_logits.logits[:, -labels.shape[-1] - 1 : -1],
            'labels': labels}
    else:
        _, counterfactual_logits = intervenable_model(
            batched_base, batched_sources, unit_locations=inv_locations, subspaces=feature_indices)
        return {
            # Keep the logits that correspond to the label tokens.
            'logits': counterfactual_logits.logits[:,-1]}


def _batched_interchange_intervention(pipeline, intervenable_model, batch, model_units, output_scores=False):
    batched_base, batched_sources, inv_locations, feature_indices = _prepare_interventable_inputs(
        pipeline, intervenable_model, batch, model_units)

    output = pipeline.intervenable_generate(
        intervenable_model, batched_base, batched_sources, inv_locations, feature_indices,
        output_scores=output_scores)

    for batched in [batched_base] + batched_sources:
        for k, v in batched.items():
            if v is not None and isinstance(v, torch.Tensor):
                batched[k] = v.cpu()
    return output

def _run_interchange_interventions(
    pipeline: Pipeline,
    counterfactual_dataset: Dataset,
    model_units: List[AtomicModelUnit],
    verbose: bool = False,
    batch_size=32,
    output_scores=False):
    #Load intervenable model
    intervenable_model = _prepare_intervenable_model(
        pipeline,
        [model_units],
        intervention_type="interchange")
    #Load Data
    dataloader = DataLoader(
        counterfactual_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    all_outputs = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches", disable=not verbose)):

        with torch.no_grad():
            scores_or_sequencs = _batched_interchange_intervention(
                    pipeline, intervenable_model, batch, model_units,
                    output_scores=output_scores)
            if output_scores:
                scores_or_sequencs = torch.stack([score.clone().detach().to("cpu") for score in scores_or_sequencs], 1)
            else:
                scores_or_sequencs = scores_or_sequencs.clone().detach().to("cpu")
            
            all_outputs.append(scores_or_sequencs)
        # Clear CUDA cache after each batch
        gc.collect()
        torch.cuda.empty_cache()

    _delete(intervenable_model)
    return all_outputs

def compute_metrics(eval_preds, eval_labels, pad_token_id):
    """Computes sequence-level and token-level accuracy."""

    # Get predicted token ids by taking argmax over vocab dimension
    predicted_token_ids = torch.argmax(eval_preds, dim=-1)

    # Create mask to ignore pad tokens in labels
    mask = (eval_labels != pad_token_id)

    # Calculate token-level accuracy (only for non-pad tokens)
    correct_tokens = (predicted_token_ids == eval_labels) & mask
    token_accuracy = correct_tokens.sum().float() / mask.sum() if mask.sum() > 0 else torch.tensor(1.0)

    # Calculate sequence-level accuracy (sequence correct if all non-pad tokens correct)
    sequence_correct = torch.stack([torch.all(correct_tokens[i, mask[i]]) for i in range(eval_labels.shape[0])])
    sequence_accuracy = sequence_correct.float().mean() if len(sequence_correct) > 0 else torch.tensor(1.0)

    return {
        "accuracy": float(sequence_accuracy.item()),
        "token_accuracy": float(token_accuracy.item())
    }

def compute_cross_entropy_loss(eval_preds, eval_labels, pad_token_id):
    """Computes cross-entropy loss over the last n tokens."""

    # Reshape predictions to (batch_size * sequence_length, vocab_size)
    batch_size, seq_length, vocab_size = eval_preds.shape
    preds_flat = eval_preds.reshape(-1, vocab_size)

    # Reshape labels to (batch_size * sequence_length)
    labels_flat = eval_labels.reshape(-1)

    # Create mask for non-pad tokens
    mask = labels_flat != pad_token_id

    # Only compute loss on non-pad tokens by filtering predictions and labels
    active_preds = preds_flat[mask]
    active_labels = labels_flat[mask]

    # Compute cross entropy loss
    loss = torch.nn.functional.cross_entropy(active_preds, active_labels)

    return loss

def _train_intervention(pipeline: Pipeline,
                        model_units: List[AtomicModelUnit],
                        counterfactual_dataset: Dataset,
                        intervention_type: str,
                        config: Dict, 
                        verbose=False,
                        custom_loss=None
                        ):
    intervenable_model = _prepare_intervenable_model(pipeline, [model_units], intervention_type=intervention_type)
    intervenable_model.disable_model_gradients()
    intervenable_model.eval()

    dataloader = DataLoader(
        counterfactual_dataset,
        batch_size=config["batch_size"],
        shuffle=config.get("shuffle", True),
    )

    #set up optimizer
    num_epoch = config['training_epoch']
    regularization_coefficient = config['regularization_coefficient']
    optimizer_params = []
    for k, v in intervenable_model.interventions.items():
        if verbose:
            print(f"Intervention: {k}")
        if isinstance(v, tuple):
            v = v[0]
        for i, param in enumerate(v.parameters()):
            if verbose:
                print(f"  Parameter {i}: requires_grad = {param.requires_grad}, shape = {param.shape}")
        # TODO: Check if this is a reference or a copy.
        optimizer_params += list(v.parameters())
    optimizer = torch.optim.AdamW(optimizer_params,
                                    lr=config['init_lr'],
                                    weight_decay=0)
    scheduler = transformers.get_scheduler('constant',
                                optimizer=optimizer,
                                num_training_steps=num_epoch *
                                len(dataloader))
    if verbose:
        print("Model trainable parameters: ", pv.count_parameters(intervenable_model.model))
        print("Intervention trainable parameters: ", intervenable_model.count_parameters())
    temperature_schedule = None
    if (intervention_type == "mask"):
        temperature_start, temperature_end = config['temperature_schedule']
        temperature_schedule = torch.linspace(temperature_start, temperature_end,
                                            num_epoch * len(dataloader) +
                                            1).to(pipeline.model.dtype).to(pipeline.model.device)
        for k, v in intervenable_model.interventions.items():
            if isinstance(v, tuple):
                intervenable_model.interventions[k][0].set_temperature(
                    temperature_schedule[scheduler._step_count])
            else:
                intervenable_model.interventions[k].set_temperature(
                    temperature_schedule[scheduler._step_count])

    train_iterator = trange(0, int(num_epoch), desc="Epoch")
    tb_writer = SummaryWriter(config['log_dir'])
    for epoch in train_iterator:
        epoch_iterator = tqdm(dataloader,
                            desc=f"Epoch: {epoch}",
                            position=0,
                            leave=True,
                            disable=not verbose
                            )
        aggreated_stats = collections.defaultdict(list)
        for step, batch in enumerate(epoch_iterator):
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to(pipeline.model.device)
            # Run training step.
            if custom_loss is None:
                logits_and_labels = _train_with_batched_interchange_intervention(
                    pipeline,
                    intervenable_model,
                    batch,
                    model_units
                )
                # Only compute the accuracy of the last N tokens, i.e., the label tokens.
                labels = logits_and_labels['labels']
                counterfactual_logits = logits_and_labels['logits']
                eval_metrics = compute_metrics(counterfactual_logits,
                                            labels,
                                            pad_token_id=pipeline.tokenizer.pad_token_id)
                loss = compute_cross_entropy_loss(counterfactual_logits,
                                                labels,
                                                pad_token_id=pipeline.tokenizer.pad_token_id)
            else:
                pred = _train_with_batched_interchange_intervention(
                    pipeline,
                    intervenable_model,
                    batch,
                    model_units,
                )
                counterfactual_logits = pred["logits"]
                loss = custom_loss(counterfactual_logits, batch["label"])

            # Add sparsity loss for Differential Binary Masking.
            for k, v in intervenable_model.interventions.items():
                if intervention_type == "mask":
                    if isinstance(v, tuple):
                        loss = loss + regularization_coefficient * intervenable_model.interventions[k][0].get_sparsity_loss()
                        intervenable_model.interventions[k][0].set_temperature(
                            temperature_schedule[scheduler._step_count])
                    else:
                        loss = loss + regularization_coefficient * intervenable_model.interventions[k].get_sparsity_loss()

            aggreated_stats['loss'].append(loss.item())
            if custom_loss is None:
                aggreated_stats['acc'].append(eval_metrics["accuracy"])
            epoch_iterator.set_postfix(
                {k: round(np.mean(aggreated_stats[k]), 2) for k in aggreated_stats})

            # Backprop.
            loss.backward()
            optimizer.step()
            scheduler.step()
            intervenable_model.set_zero_grad()

            # Logging.
            if step % 10 == 0:
                tb_writer.add_scalar("lr",
                                    scheduler.get_last_lr()[0], scheduler._step_count)
                tb_writer.add_scalar("loss", loss, scheduler._step_count)
            if step < 2 and epoch == 0 and verbose:
                print('Base:',
                    pipeline.tokenizer.batch_decode(pipeline.load(batch['input'])["input_ids"]))
                print('Source:',
                    [pipeline.tokenizer.batch_decode(pipeline.load(source)["input_ids"]) for source in batch['counterfactual_inputs']])
                print('Output:',
                    pipeline.tokenizer.batch_decode(torch.argmax(counterfactual_logits, dim=-1))
                    )
                if custom_loss is None:
                    print('Label:',
                        pipeline.tokenizer.batch_decode(labels))
    tb_writer.flush()
    tb_writer.close()

    if intervention_type == "mask":
        for kv, model_unit in zip(intervenable_model.interventions.items(), model_units):
            k, v = kv
            if isinstance(v, tuple):
                v = v[0]
            mask_binary = (torch.sigmoid(v.mask) > 0.5).float().cpu()
            indices = list(torch.nonzero(mask_binary).squeeze().numpy())
            model_unit.set_feature_indices(indices)
            if verbose:
                print(f"Number Selected features: {len(indices)}")
                print(f"Selected features: {indices}")
    _delete(intervenable_model)

class InterventionExperiment:
    def __init__(self,
            pipeline: Pipeline,
            task: Task,
            model_units_list: List[AtomicModelUnit],
            checker: Callable,
            metadata=lambda x: None,
            config=None):
        self.pipeline = pipeline
        self.task = task
        self.model_units_list = model_units_list
        self.checker = checker
        self.metadata = metadata
        self.config = {"batch_size": 32} if config is None else config 
        if "evaluation_batch_size" not in self.config:
            self.config["evaluation_batch_size"] = self.config["batch_size"]
        if "method_name" not in self.config:
            self.config["method_name"] = "InterventionExperiment"
        if "output_scores" not in self.config:
            self.config["output_scores"] = False 

    def save_featurizers(self, model_units, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        f_dirs, invf_dirs, indices_dir = [], [], []
        for model_unit in model_units:
            filename = os.path.join(model_dir, model_unit.id)
            f_dir, invf_dir = model_unit.featurizer.save_modules(filename)
            with open(filename + "_indices", "w") as f:
                indices = model_unit.get_feature_indices()
                if indices is not None:
                    json.dump([int(i) for i in indices], f)
                else:
                    json.dump(None, f)
            f_dirs.append(f_dir)
            invf_dirs.append(invf_dir)
            indices_dir.append(filename + "_indices")
        return f_dirs, invf_dirs, indices_dir

    def load_featurizers(self, model_dir):
        for model_units in self.model_units_list:
            for model_unit in model_units:
                filename = os.path.join(model_dir, model_unit.id)
                if os.path.exists(filename + "_featurizer") and os.path.exists(filename + "_inverse_featurizer"):
                    model_unit.set_featurizer(Featurizer.load_modules(filename))
                if os.path.exists(filename + "_indices"):
                    indices = None
                    with open(filename + "_indices", "r") as f:
                        indices = json.load(f)
                    model_unit.set_feature_indices(indices)
        return

    def perform_interventions(self, dataset_names= None, verbose: bool = False) -> Dict:
        """
        Compute intervention scores across multiple counterfactual datasets and model units.

        Args:
            pipeline: Pipeline object for model execution
            counterfactual_datasets: Dict mapping dataset names to Dataset objects
            model_units_lists: List of model units to intervene on
            checker: Function to evaluate output accuracy
            verbose: Whether to show progress bars

        Returns:
            Dict mapping dataset names to scores for each model unit configuration
        """
        if dataset_names is None:
            dataset_names = self.task.counterfactual_datasets.keys()
        results = {"method_name": self.config["method_name"],
                    "model_name": self.pipeline.model.__class__.__name__,
                    "task_name": self.task.id,
                    "dataset": {dataset_name: {
                        "model_unit": {str(unit): None for unit in self.model_units_list}}
                    for dataset_name in dataset_names}}
        for dataset_name in dataset_names:
            for model_units in self.model_units_list:
                # Run interventions
                if verbose:
                    print(f"Running interventions for {dataset_name} with model units {model_units}")
                raw_outputs = _run_interchange_interventions(
                    pipeline=self.pipeline,
                    counterfactual_dataset=self.task.raw_counterfactual_datasets[dataset_name],
                    model_units=model_units,
                    verbose=verbose,
                    output_scores=self.config["output_scores"],
                    batch_size=self.config["evaluation_batch_size"]
                )
                results["dataset"][dataset_name]["model_unit"][str(model_units)] = {
                    "raw_outputs": raw_outputs,
                    "metadata": self.metadata(model_units)}
        return results

    def interpret_results(self, raw_results, target_variables, save_dir=None, use_raw_output=False):
        """
        Compute accuracy of raw results based on target variables and checker function.

        Args:
            raw_results: Raw results to evaluate
            target_variables: Target variables to evaluate
            checker: Function to compare model output with expected output
            save_dir: Directory to save processed results to (optional)
        """
        # Create a new results dictionary without raw outputs
        processed_results = {
            "method_name": raw_results["method_name"],
            "model_name": raw_results["model_name"],
            "task_name": raw_results["task_name"],
            "target_variables": "-".join(target_variables),
            "dataset": {}
        }

        # Process each dataset and model unit
        for dataset_name in raw_results["dataset"]:
            processed_results["dataset"][dataset_name] = {"model_unit": {}}

            for model_units in raw_results["dataset"][dataset_name]["model_unit"]:
                # Get raw outputs and compute text representations
                raw_outputs = raw_results["dataset"][dataset_name]["model_unit"][model_units]["raw_outputs"]

                texts = []
                for raw_output in raw_outputs:
                    texts += self.pipeline.dump(raw_output, is_logits=self.config["output_scores"])

                # Get metadata from original results
                metadata = raw_results["dataset"][dataset_name]["model_unit"][model_units].get("metadata", None)

                # Compute accuracy scores
                labeled_data = self.task.label_counterfactual_data(dataset_name, target_variables)
                scores = []
                for example, text, raw_output in zip(labeled_data, texts, raw_outputs[0]):
                    if use_raw_output:
                        score = self.checker(raw_output, example["label"])
                    else:
                        score = self.checker(text, example["label"])
                    if isinstance(score, torch.Tensor):
                        score = score.item()
                    scores.append(score)
                
                # Store processed results (without raw outputs)
                processed_results["dataset"][dataset_name]["model_unit"][model_units] = {
                    "accuracy": sum(scores) / len(scores),
                    "metadata": metadata
                }

        # Save processed results to directory if provided
        if save_dir is not None:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Save the processed results to a JSON file
            file_name = "results.json"
            for k in ["method_name", "model_name", "task_name", "target_variables"]:
                file_name = processed_results[k] + file_name
            with open(os.path.join(save_dir, file_name), "w") as f:
                json.dump(processed_results, f, indent=2)

        # Return the processed results
        return processed_results

    def collect_activations(self, dataset, verbose=False):
        intervenable_model = _prepare_intervenable_model(self.pipeline, self.model_units_list, intervention_type="collect")
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=False
        )

        list_lens = [len(model_units) for model_units in self.model_units_list]
        data = [[] for _ in range(len(self.model_units_list))]
        # Process dataset in batches
        for batch in tqdm(dataloader, desc="Processing batches", disable=not verbose):
            # Prepare batch data
            batched_base = batch["input"]
            b = len(batched_base)

            if self.model_units_list[0][0].unit == 'h.pos': # Attention heads
                base_indices = [
                    [[model_unit.index_component(base)[0] for base in batched_base],
                    [model_unit.index_component(base)[1] for base in batched_base]]
                    for model_units in self.model_units_list for model_unit in model_units
                ]
            else:
                base_indices = [
                    [model_unit.index_component(base) for base in batched_base]
                    for model_units in self.model_units_list for model_unit in model_units 
                ]
            map = {"sources->base": (base_indices,base_indices)}
            batched_base = self.pipeline.load(batched_base)

            if self.pipeline.tokenizer.padding_side == "left":
                pad_token_id = self.pipeline.tokenizer.pad_token_id
                if self.model_units_list[0][0].unit == 'h.pos': # Attention heads
                    base_indices = [
                        [
                            indices[0], 
                            [ [j + (base==pad_token_id).sum().item() for j in tokens] 
                            for base, tokens in zip(batched_base["input_ids"], indices[1])
                            ]
                        ]
                        for indices in base_indices
                    ]
                else:
                    base_indices = [
                        [[j + (base==pad_token_id).sum().item() for j in index] for base, index in zip(batched_base["input_ids"], indices)]
                        for indices in base_indices
                    ]
            activations = intervenable_model(batched_base,unit_locations=map)[0][1]

            #clear memory from gpu
            activations_cpu = [activation.cpu() for activation in activations]
            batched_base = [batched_base[k].cpu() for k in batched_base]
            del batched_base
            del activations


            c = 0
            for i, length in enumerate(list_lens):
                batched_activations = []
                for j in range(length):
                    batched_activations.append(torch.cat(activations_cpu[(c+ j)*b:(c+ j+1)*b], dim=0))
                data[i].append(torch.cat(batched_activations, dim=0).view(-1,batched_activations[0].shape[-1]))
                c += length

        return [torch.cat(x, dim =0) for x in data]

    def build_PCA_feature_interventions(self, dataset, n_components=None, verbose=False):
        features = self.collect_activations(dataset)

        PCA_featurizers = []
        for X in features:
            n = min(X.shape[0], X.shape[1])
            if n_components is not None:
                n = min(n, n_components)
            device = X.device
            dtype = X.dtype
            X = np.array(X.to("cpu"))
            # Normalize input features to zero mean unit variance.
            pca_mean = np.mean(X, axis=0, keepdims=True)
            pca_std = X.var(axis=0)**0.5
            epsilon = 1e-6
            pca_std = np.where(pca_std < epsilon, epsilon, pca_std)
            X = (X - pca_mean) / pca_std
            pca = PCA(n_components=n)
            pca.fit(X)
            PCA_featurizers.append(torch.tensor(pca.components_).to(device).to(dtype))
            if verbose:
                print(f'PCA normalized inputs: min={X.min():.2f} max={X.max():.2f}'
                        f' mean={X.mean():.2f}')
                print(f'PCA explained variance: {[round(float(x),2) for x in pca.explained_variance_ratio_]}')

        for model_units, rotation in zip(self.model_units_list, PCA_featurizers):
            for model_unit in model_units:
                model_unit.set_subspace_intervention(rotation_subspace=rotation.T, trainable=False)
        return PCA_featurizers

    def select_features_linear_probe(self, dataset, target_variables, threshold=0.01, top_k=None,
                        model_type="classification", verbose=False):
        """
        Train a linear model to predict target variables and select feature indices based on weights.

        Args:
            dataset: Dataset to collect features from
            target_variables: Target variables to predict
            threshold: Threshold for feature selection based on weights (default=0.01)
            top_k: Number or percentage of top features to select. If None, use threshold method (default=None)
            model_type: Type of model to train ("classification" or "regression") (default="classification")
            batch_size: Batch size for feature collection (default=8)
            verbose: Whether to show progress bars (default=False)

        Returns:
            List of selected feature indices for each model unit
        """
        features = self.collect_activations(self.task.create_raw_data(dataset),verbose=verbose)
        labeled_data, label_to_setting = self.task.label_probe_data(dataset, target_variables)
        labels = torch.tensor([example["label"] for example in labeled_data])

        selected_indices = []
        for feature_group in features:
            X = feature_group.cpu().numpy()
            y = labels.cpu().numpy()

            # Train linear model based on specified model_type
            from sklearn.linear_model import LogisticRegression, Lasso

            if model_type == "classification":
                model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
                model.fit(X, y)
                weights = model.coef_
                accuracy = model.score(X, y)
                if verbose:
                    print(f"Classification accuracy = {accuracy:.4f}")
                    print(weights.shape)
                if len(weights.shape) > 1:
                    weights = np.abs(weights).sum(axis=0)
                else:
                    weights = np.abs(weights)
            else:  # regression
                model = Lasso(alpha=0.01)
                model.fit(X, y)
                if verbose:
                    print(f"R^2 = {model.score(X, y):.4f}")
                weights = np.abs(model.coef_)

            # Select features
            if top_k is None:
                # Use threshold method
                selected = np.where(weights > threshold)[0].tolist()
            else:
                # Use top_k method
                if isinstance(top_k, float) and 0 < top_k < 1:
                    # Interpret as percentage if between 0 and 1
                    k = max(1, int(top_k * len(weights)))
                else:
                    # Use as absolute number
                    k = min(int(top_k), len(weights))
                selected = np.argsort(weights)[-k:].tolist()

            selected_indices.append(selected)

        # Set the selected feature indices for each model unit
        for model_units, indices in zip(self.model_units_list, selected_indices):
            for model_unit in model_units:
                model_unit.set_feature_indices(indices)
                if verbose:
                    print(f"Number Selected features: {len(indices)}")
                    print(f"Selected features: {indices}")

        return selected_indices


    def train_interventions(self, dataset_names, target_variables, method="DAS", model_dir=None, verbose=False, custom_loss=None):

        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        defaults = {
            "training_epoch": 3,
            "init_lr": 1e-2,
            "regularization_coefficient": 1e-4,
            "max_output_tokens": 1,
            "log_dir": "logs",
            "n_features": 32,
            "temperature_schedule": (1.0, 0.01),
            "batch_size": 32
        }

        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value 

        assert method in ["DAS", "DBM"]
        counterfactual_dataset = []
        for dataset_name in dataset_names:
            counterfactual_dataset += self.task.label_counterfactual_data(dataset_name, target_variables)
        if method == "DAS":
            intervention_type = "interchange"
        elif method == "DBM":
            intervention_type = "mask"
        for model_units in self.model_units_list:
            for model_unit in model_units:
                if method == "DAS":
                    model_unit.set_subspace_intervention(shape=(model_unit.shape[0], self.config["n_features"]), id="DAS")
                    model_unit.set_feature_indices(None)
            _train_intervention(self.pipeline, model_units, counterfactual_dataset, intervention_type, self.config, verbose=verbose, custom_loss=custom_loss)
            if model_dir is not None:
                self.save_featurizers(model_units, model_dir)
        return self