from experiments.LM_experiments import PatchResidualStream, PatchIOIHeads
import os
import gc
import torch

def residual_stream_baselines(
    pipeline=None, 
    task=None, 
    token_positions=None, 
    train_data=None, 
    test_data=None, 
    config=None, 
    target_variables=None, 
    checker=None, 
    start=None, 
    end=None, 
    verbose=False,
    model_dir=None,
    results_dir=None,
    methods=["full_vector", "DAS", "DBM+PCA", "DBM", "DBM+SAE"]
    ):

    def heatmaps(results, config):
        heatmap_path = os.path.join(results_dir, "heatmaps", config["method_name"], 
                         pipeline.model.__class__.__name__, "-".join(target_variables))

        # Create directory if it doesn't exist
        if not os.path.exists(heatmap_path):
            os.makedirs(heatmap_path)
        experiment.plot_heatmaps(results, save_path=heatmap_path)
        experiment.plot_heatmaps(results, average_counterfactuals=True, save_path=heatmap_path)
    
    def clear_memory():
        # Clear Python garbage collector
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force a synchronization point to ensure memory is freed
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if "full_vector" in methods:
        # Full vector method
        config["method_name"] = "full_vector"
        experiment = PatchResidualStream(pipeline, task, list(range(start,end)), token_positions, checker, config=config)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose)
        processed_results = experiment.interpret_results(raw_results, target_variables, save_dir=results_dir)
        heatmaps(processed_results, config)
        
        # Release memory before next experiment
        del experiment, raw_results, processed_results
        clear_memory()

    if "DAS" in methods:
        # DAS method
        config["method_name"] = "DAS"
        experiment = PatchResidualStream(pipeline, task, list(range(start,end)), token_positions, checker, config=config)
        experiment.train_interventions(train_data, target_variables, method="DAS", verbose=verbose, model_dir=model_dir)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose)
        processed_results = experiment.interpret_results(raw_results, target_variables, save_dir=results_dir)
        heatmaps(processed_results, config)
        
        # Release memory before next experiment
        del experiment, raw_results, processed_results
        clear_memory()

    if "DBM+PCA" in methods:
        # DBM+PCA method
        config["method_name"] = "DBM+PCA"
        experiment = PatchResidualStream(pipeline, task, list(range(start,end)), token_positions, checker, config=config)
        experiment.build_PCA_feature_interventions(task.raw_all_data, verbose=verbose)
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose)
        processed_results = experiment.interpret_results(raw_results, target_variables, save_dir=results_dir)
        heatmaps(processed_results, config)
        
        # Release memory before next experiment
        del experiment, raw_results, processed_results
        clear_memory()

    if "DBM" in methods:
        # DBM method
        config["method_name"] = "DBM"
        experiment = PatchResidualStream(pipeline, task, list(range(start,end)), token_positions, checker, config=config)
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose)
        processed_results = experiment.interpret_results(raw_results, target_variables, save_dir=results_dir)
        heatmaps(processed_results, config)
        
        # Release memory before next experiment
        del experiment, raw_results, processed_results
        clear_memory()

    # Gemma SAE method (conditional)
    if hasattr(pipeline.model, 'config') and hasattr(pipeline.model.config, '_name_or_path') and pipeline.model.config._name_or_path == "google/gemma-2-2b" and "DBM+SAE" in methods:
        config["method_name"] = "DBM+SAE"
        from sae_lens import SAE

        def sae_loader(layer):
            sae, _, _ = SAE.from_pretrained(
                release = "gemma-scope-2b-pt-res-canonical",
                sae_id = f"layer_{layer}/width_16k/canonical",
                device = "cpu",
            )
            return sae

        experiment = PatchResidualStream(pipeline, task, list(range(start,end)), token_positions, checker, config=config)
        experiment.build_SAE_feature_intervention(sae_loader)
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose)
        processed_results = experiment.interpret_results(raw_results, target_variables, save_dir=results_dir)
        heatmaps(processed_results, config)
        
        # Final memory cleanup
        del experiment, raw_results, processed_results, sae_loader
        clear_memory()

    if hasattr(pipeline.model, 'config') and hasattr(pipeline.model.config, '_name_or_path') and pipeline.model.config._name_or_path == "meta-llama/Meta-Llama-3.1-8B-Instruct" and "DBM+SAE" in methods:
        config["method_name"] = "DBM+SAE"
        from sae_lens import SAE

        def sae_loader(layer):
            sae, _, _ = SAE.from_pretrained(
                release = "llama_scope_lxr_8x",
                sae_id = f"l{layer}r_8x",
                device = "cpu",
            )
            return sae

        experiment = PatchResidualStream(pipeline, task, list(range(start,end)), token_positions, checker, config=config)
        experiment.build_SAE_feature_intervention(sae_loader)
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose)
        processed_results = experiment.interpret_results(raw_results, target_variables, save_dir=results_dir)
        heatmaps(processed_results, config)
        
        # Final memory cleanup
        del experiment, raw_results, processed_results, sae_loader
        clear_memory()

def ioi_baselines(
    pipeline=None, 
    task=None, 
    token_positions=None, 
    train_data=None, 
    test_data=None, 
    config=None, 
    target_variables=None, 
    checker=None, 
    start=None, 
    end=None, 
    verbose=False,
    model_dir=None,
    results_dir=None,
    custom_loss=None,
    heads_list=None,
    skip=[]
    ):

    def clear_memory():
        # Clear Python garbage collector
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force a synchronization point to ensure memory is freed
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if "full_vector" not in skip:
        # Full vector method
        config["method_name"] = "full_vector"
        experiment = PatchIOIHeads(pipeline, task, list(range(start,end)), heads_list, token_positions, checker, config=config)
        raw_results = experiment.perform_interventions(test_data, verbose=False)
        processed_results = experiment.interpret_results(raw_results, target_variables, save_dir=results_dir, use_raw_output=True)
        
        # Release memory before next experiment
        del experiment, raw_results, processed_results
        clear_memory()

    if "DAS" not in skip:
        # DAS method
        config["method_name"] = "DAS"
        experiment = PatchIOIHeads(pipeline, task, list(range(start,end)), heads_list, token_positions, checker, config=config)
        experiment.train_interventions(train_data, target_variables, method="DAS", verbose=verbose, model_dir=model_dir, custom_loss=custom_loss)
        clear_memory()
        raw_results = experiment.perform_interventions(test_data, verbose=verbose)
        processed_results = experiment.interpret_results(raw_results, target_variables, save_dir=results_dir, use_raw_output=True)
        
        # Release memory before next experiment
        del experiment, raw_results, processed_results
        clear_memory()

    if "DBM" not in skip:
        # DBM method
        config["method_name"] = "DBM"
        experiment = PatchIOIHeads(pipeline, task, list(range(start,end)), heads_list, token_positions, checker, config=config)
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose, custom_loss=custom_loss)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose)
        processed_results = experiment.interpret_results(raw_results, target_variables, save_dir=results_dir, use_raw_output=True)
        # Release memory before next experiment
        del experiment, raw_results, processed_results
        clear_memory()

    if "DBM+PCA" not in skip:
        # DBM+PCA method
        config["method_name"] = "DBM+PCA"
        experiment = PatchIOIHeads(pipeline, task, list(range(start,end)), heads_list, token_positions, checker, config=config)
        experiment.build_PCA_feature_interventions(task.raw_all_data, verbose=verbose)
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose,custom_loss=custom_loss)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose)
        processed_results = experiment.interpret_results(raw_results, target_variables, save_dir=results_dir, use_raw_output=True)
        # Release memory before next experiment
        del experiment, raw_results, processed_results
        clear_memory()