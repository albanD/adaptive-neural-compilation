#!/bin/bash
# This script will reproduce the experiments presented in the "Adaptive Neural Computation" paper.
# The seeds are set to make sure that you can reproduce our exact results.
# To explore the quality of convergence, remove the --seed arguments
# If you want to train on your own problem / data, take a look at the examples in the examples/ folder

# Access
pushd ..
th cli/train.lua examples/access.lua --optim sgd --batch 1 --biased_sample \
   --alpha 10 --beta 0 --delta 1 --gamma 0 --lr 1 --sharp 2 \
   --it 2000 --val 100 --max_rec 10 --seed 1463572200 \
   --save_name paper-experiment/biased-access/ --save_it 100 --print_val 100 \
   --decompile
th cli/plot.lua --single-run --path-to-csv paper-experiment/biased-access/plot.csv \
   --output-path paper-experiment/biased-access/figure/ --file-only
popd

#increment
pushd ..
th cli/train.lua examples/increment.lua --optim adam --batch 1 --biased_sample \
   --alpha 1 --beta 0 --delta 5 --gamma 0.1 --lr 0.1 --sharp 5 \
   --it 7200 --val 100 --max_rec 50 --seed 1463505709 \
   --save_name paper-experiment/biased-increment/ --save_it 100 --print_val 100 \
   --decompile
th cli/plot.lua --single-run --path-to-csv paper-experiment/biased-increment/plot.csv \
   --output-path paper-experiment/biased-increment/figure/ --file-only
popd

#swap
pushd ..
th cli/train.lua examples/swap.lua --optim adam --batch 1 --biased_sample \
   --alpha 1 --beta 0 --delta 10 --gamma 0 --lr 0.1 --sharp 3 \
   --it 2000 --val 100 --max_rec 15 --seed 1463512912 \
   --save_name paper-experiment/biased-swap/ --save_it 100 --print_val 100 \
   --decompile
th cli/plot.lua --single-run --path-to-csv paper-experiment/biased-swap/plot.csv \
   --output-path paper-experiment/biased-swap/figure/ --file-only
popd

# List-k
pushd ..
th cli/train.lua examples/listk.lua --optim adam --batch 5 --biased_sample \
   --alpha 10 --beta 10 --gamma 0.1 --delta 20 --sharp 5 \
   --lr 1 --it 2000 --val 100 --max_rec 75 --seed 1463436986 \
   --save_name paper-experiment/biased-listk/ --save_it 100 --print_val 100 \
   --decompile
th cli/plot.lua --single-run --path-to-csv  paper-experiment/biased-listk/plot.csv \
   --output-path paper-experiment/biased-listk/figure/ --file-only
popd


# Bubble sort
pushd ..
th cli/train.lua examples/bubble_sort.lua --optim adam --batch 5 --biased_sample \
   --alpha 10 --beta 5 --gamma 0.1 --delta 5 --sharp 6 \
   --lr 1 --it 700 --seed  1463505705 --max_rec 75 --decompile \
   --save_name paper-experiment/biased-bubble-sort/ --save_it 100 --print_val 100 \
   --decompile
th cli/plot.lua --single-run --path-to-csv paper-experiment/biased-bubble-sort/plot.csv \
   --output-path paper-experiment/biased-bubble-sort/figure/ --file-only
popd


# Loop addition
pushd ..
th cli/train.lua examples/loopy_add.lua --optim adam --batch 1 --sample \
   --alpha 10 --beta 0 --gamma 0 --delta 10 --sharp 4 --val 100\
   --lr 0.1 --it 3000 --seed 1464172448 --max_rec 65 --decompile \
   --save_name paper-experiment/biased-addition/ --save_it 100 --print_val 100
th cli/plot.lua --single-run --path-to-csv paper-experiment/biased-addition/plot.csv \
   --output-path paper-experiment/biased-addition/figure/ --file-only
popd
