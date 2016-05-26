local torch = require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
local nc = require 'nc'
local trainer = require('nc.utils')["trainer"]

-- Parse the command line arguments
local opt = trainer.parse_opt(arg)

-- Load the example problem
local config = dofile(opt.config)

-- Run the training
trainer.train(opt, config)
