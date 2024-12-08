# *A Study on the Efficiency of Combined Reconstruction and Poisoning Attacks in Federated Learning.*

This is the repository with the code for the paper: *A Study on the Efficiency of Combined Reconstruction and Poisoning Attacks in Federated Learning.* The code is structured in the following way:

<pre>
/
├── conda_envs
│   └── environment_mlflow.py (Environment file to run experiments)
│   └── environment_mlflow.py (Environment file to run experiments with DP.)
│   └── environment_gpu.py (Environment file to run experiments with GPU support. Incompatible with DP.)
├── experiments
│   └── federated_gan.py (Where all the code with Federated Learning and attacks is present.)
│   └── experiment_*.py (The different experiment configurations.)
│   └── experiment_paper_attack_combination_with_global_SGD.py (It reproduces the 10x10 matrix of class reconstructed-class poisoned attack)
│   └── experiment_paper_distribution_influence.py (Experiment to check how data distributions affects attacks.)
│   └── experiment_paper_dp_comparison.py (It reproduces the attack with DP applied)
│   └── dataloader_dirichlet.py (Utility to generate dirichlet distributions)
│   └── dataloader.py (Generation of data distributions)
├── conda_envs
│   └── data_processing_*.py Generate visualization for corresponding experiment
│   └── *.png visualizations for the corresponding experiment



</pre>

## Requirements

- Conda installed: Either anaconda or miniconda.

## Installation

We recommend to create a new environment, in order to reproduce our experiments. To do so, use the following commands.

```sh
conda env create --name crpa --file=environment_mlflow.yml
conda activate crpa
```

If you wish to run any experiment with DP, you need to also install the packages from the DP environment. You may do so using:

```sh
conda env update --file environment_privacy.yml
```

Now any experiments using DP can be run.

In case you possess gpu capabilities and want to use them, please install the packages in another environment. We have not managed to make run DP with the GPU.


## Running the experiments.

In order to run experiments, write a experiment file following the structure from the provided experiments, and then just execute it.

## Visualizing the experiments.

Generate the visualizations by writing the MLRun names of the chosen experiments to the file.