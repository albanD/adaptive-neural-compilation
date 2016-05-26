local nn = require 'nn'
local nc = {}

-- Put the submodules we want in the table of the main module
nc.distUtils = require 'nc.distUtils'
nc.layers = require 'nc.layers'
nc.decompiler = require 'nc.decompiler'
nc.optim = require 'nc.optim'
nc.utils = require 'nc.utils'

-- Include ops in the ram
torch.include('nc', 'add_op.lua')
torch.include('nc', 'dec_op.lua')
torch.include('nc', 'inc_op.lua')
torch.include('nc', 'jez_op.lua')
torch.include('nc', 'max_op.lua')
torch.include('nc', 'min_op.lua')
torch.include('nc', 'read_op.lua')
torch.include('nc', 'stop_op.lua')
torch.include('nc', 'sub_op.lua')
torch.include('nc', 'write_op.lua')
torch.include('nc', 'zero_op.lua')

-- Include the ram machine and controller to layers
torch.include('nc', 'initialModule.lua')
torch.include('nc', 'machine.lua')
torch.include('nc', 'dRAM.lua')
torch.include('nc', 'algCrit.lua')

-- Include the various optimisers
torch.include('nc', 'sgd.lua')
torch.include('nc', 'adam.lua')
torch.include('nc', 'priors.lua')
torch.include('nc', 'infinity_prior.lua')
torch.include('nc', 'softmaxed_prior.lua')


-- Include the various trainers
torch.include('nc', 'trainer.lua')
torch.include('nc', 'plotter.lua')

-- We add the test last since everything else should be ready
-- before initialising it
nc.test = require 'nc.test'

return nc
