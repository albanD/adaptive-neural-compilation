# Adaptive compilation

This folder contains the code for the model and the adaptation by learning.

## Install

#### Torch
Instructions to install Torch can be found [here](http://torch.ch/docs/getting-started.html#_)

#### Custom package
This package can be installed by running `luarocks make` in this folder.

### Test your installation

`th -e "nc=require 'nc';nc.test()"`

## Usage

### CLI tools

The `cli` folder contains a set of tools to use our module.
To see the command line options associated with each tool, you can run `th tool.lua --help`.

* [`train.lua`](cli/train.lua) is the cli tool to either adapt an initial program or train from scratch a program based on a task configuration file.
* [`eval.lua`](cli/eval.lua) is the cli tool to evaluate the performances of a model.
* [`plot.lua`](cli/plot.lua) is the cli to plot from different metrics from the csv generated during training.
* [`gnc.lua`](cli/gnc.lua) is the cli to perform "decompilation" on a trained model.
* [`run_xp.lua`](cli/run_xp.lua) is the cli tool to perform cross validation or run the same experiment multiple times using multiple threads. This tool does not take command line arguments but what is executed is defined inside the file [here](cli/run_xp.lua#L10-L18).

### Sample compiled programs

The [`examples/`](examples/) folder contains a set a different sample tasks (described in the paper) and sample compiled programs to solve them.
An example with specific definition for each field can be found in the [`default`](examples/default.lua) configuration file.

### Paper experiments

The [`paper-experiment/run-experiments.sh`](paper-experiment/run-experiments.sh) contains the selected hyperparameters for each result reported in the paper. Since many factors can affect the RNG state, the selected random seeds may not lead to the exact same results. You cna use the `run_xp.lua` script to find a correct random seed for your current system.

## The `nc` package

This packages aims at providing all the necessary tools for running and leaning the model described in the paper.

* Our differentiable RAM model is defined in [`layers/dRAM.lua`](layers/dRAM.lua). It uses the following elements:
    * The execution Machine in [`layers/machine.lua`](layers/machine.lua) that will execute the command of the Controller.
    * All the instructions are defined in the [`ops/`](ops/) folder.
    * The initialisation of the registers is handled by the [`layers/initialModule.lua`](layers/initialModule.lua).
* The training for the adaptation is mainly done with the [`utils/trainer.lua`](utils/trainer.lua) tool. It is based on the following elements:
    * The loss to learn better algorithm is defined in [`layers/algCrit.lua`](layers/algCrit.lua)
    * The optimisation algorithms used (plain `sgd` and `adam`) are in the [`optim/`](optim/) folder.
