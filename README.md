# Learning from rationales

This is 
 * Generating/downloading text classification datasets, especially those with rationales
 * Running model training experiments (i.e. sweeps over hyperparameter selections) over them
 

## Quickstart

### Installation

Clone the branch:
`git clone -b slim_rationale https://github.com/shcarton/learn_from_explanations_v2.git [destination]`
 
Install the requirements: enter the project root directory, and run `pip install -e .` to install various modules in the requirements.txt .

### Validate environment

Run 

`python -u [project_root]/deployment_test.py` 

to make sure Python is finding the GPUs okay

### Set global config options

Edit `[project_root]/config/global_config.py` to choose where stuff gets downloaded, outputted, etc. By default, these things happen in dedicated subdirectories within the project directory. 

### Download and process the multirc dataset

Run 

`python -u [project_root]/processing_scripts/download_and_process_eraser_datasets.py`

To download and process the MultiRC dataset from the ERASER benchmark collection website. 

* There are other ERASER datasets you can download, but you have to uncomment them in the script. 

* BERT-base accuracy is about 68-69 on this dataset. 

### Train and test various models on MultiRC dataset

The config file `[project_root]/config/run_training_experiment/multirc_example.py` has been provided to test the modeling code on this dataset

Run the training code:

`python -u [project_root]/experiment_scripts/run_training_experiment.py --config config.run_training_experiment.multirc_example`


* You can encourage the rationale model to make sparser or less sparse rationales by upping or lowering the sparsity_loss_weight parameter from 0.1 in the config file. 


### Inspect the output

Each model produces a bunch of output, which can be seen in `[global_output_directory]/run_experiment/multirc_example/multirc/[model_name]/default_trainer/[config_name]/`

If we refer to the above as `[model_output_dir]`, then there's a few specific files of interest:
 * `[model_output_dir]/test_output/test_epoch_eval.json`: test performance`
 * `[model_output_dir]/test_output/epoch_-1_predictions.json`: model output on test set
 * `[model_output_dir]/test_prediction_sample.html`: sample of test predictions
 
 
 ### FAQ
 #### How does the file naming work in the output directory?
  
  The directory name for a trained model is based on the hyperparameters that were set in the config for that model. For example, `agb=10_bs=gumbel_softmax_hrlw=0.0_ms=multiply_zero_slw=0.1_p=True` means:
   
   * accumulate_grad_batchs(agb) = 10; run 10 batches before doing a step, for a higher effective batch size
   * binarization_strategy (bs) = gumbel_softmax; use Gumbel Softmax to binarize the generated rationale
   * human_rationale_loss_weight (hrlw) = 0.0; don't train the extractor layer to mimic human rationales
   * masking_strategy (ms) = multiply_zero; mask input to the predictor by just multiplying input embeddings by 0
   * sparsity_loss_weight(slw) = 0.1; weight the sparsity of the rationale mask by this amount
   * pretrained (p) = True; the model has been pretrained in some capacity (in this case, the predictor has been pretrained on full input)
   
 The hyperparameters that make it into this name are ones that are included as lists in the config. So:
 
 `sparsity_loss_weight=[0.l]` --> "...slw=0.1..."
 
 while:
 
 `sparsity_loss_weight=0.1` --> [not present]
 
 If no named hyperparameters are specified, then the output directory is named "default". 
 
 #### How do I try several different values for a hyperparameter?
 
 Set the value to a multi-item list, e.g: 
  `sparsity_loss_weight=[0.l, 0.15, 0.2]` 
  
  If you do this for multiple hyperparameters, every combination will be trained. So:
  
   `sparsity_loss_weight=[0.l, 0.15, 0.2]` 
   
`human_rationale_loss_weight=[0.0, 1.0]` 

will result in 6 models. 

#### What should I change for debugging versus doing long runs

Mainly `train_model_in_new_process` in the script parameters. Set it to `False` for debugging, `True` for training multiple models. 

Also, 
`			
'limit_train_batches': 10,
'limit_val_batches': 10,
'limit_test_batches': 10
`
in the trainer_params or trainer config can be helpful. 

#### What if I have other questions?

Email me at samuel.carton@gmail.com


