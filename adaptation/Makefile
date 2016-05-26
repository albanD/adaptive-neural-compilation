build: FORCE
	luarocks make

run_xp: build
	th cli/run_xp.lua

plot:
	th cli/plot.lua --single-run

run-tests: build
	th -lnc -e "require('nc').test('${TEST}')"

eval-bst: build
	th eval.lua examples/walkBST.lua --sample

eval-swap: build
	th eval.lua examples/swap.lua --sample

eval-reverse: build
	th eval.lua examples/reverse.lua --sample

eval-permutation: build
	th eval.lua examples/permutation.lua --sample --max_rec 100

eval-bubble: build
	th eval.lua examples/bubble_sort.lua --sample --max_rec 150

eval-merge: build
	th eval.lua examples/merge.lua --sample --max_rec 100

eval-listk: build
	th eval.lua examples/listk.lua --sample

eval-listSearch: build
	th eval.lua examples/listSearch.lua --sample

eval-increment: build
	th eval.lua examples/increment.lua --sample

eval-copy: build
	th eval.lua examples/copy.lua --sample

eval-access: build
	th eval.lua examples/access.lua --sample

eval-inc : build
	th eval.lua examples/increment.lua --biased_sample --max_rec 100

train-inc : build clean-output
	th train.lua examples/increment.lua --alpha 0 --biased_sample --decompile --it 100 --max_rec 50

train-access : build clean-output
	th train.lua examples/access.lua --alpha 0 --biased_sample --decompile --it 100

train-swap : build clean-output
	th train.lua examples/swap.lua --alpha 0 --val 1 --train_size 1 --it 10000 --lr 0.05 --max_rec 50

train-listk : build clean-output
	th cli/train.lua examples/listk.lua --alpha 10 --beta 10 --gamma 1.5 --delta 5\
									--train_size 100 --val_on_train --biased_sample \
									--it 10000  --lr 0.5 --max_rec 50 \
									--decompile --save_it 10

train-bubble: build clean-output
	th cli/train.lua examples/bubble_sort.lua --alpha 10 --beta 5 --gamma 0.5 --delta 0\
									--train_size 100 --val_on_train --biased_sample \
									--it 1000  --lr 0.5 --max_rec 75 \
									--decompile --save_it 10

train-bubble-from-scratch: build clean-output
	th cli/train.lua examples/bubble_sort.lua --alpha 10 --beta 5 --gamma 0.5 --delta 0\
									--train_size 100 --val_on_train --biased_sample \
									--it 1000  --lr 0.5 --max_rec 75 \
									--decompile --save_it 10 --rand_dr

train-listkbiased: build clean-output
	th cli/train.lua examples/listk-biased.lua --biased_sample

clean-output:
	-rm -rf output/*

FORCE:
