-- This is a config for the increment task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Increment"
-- Number of available registers (excluding the RI)
config.nb_registers = 6
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 7
-- Initial state of the registers
config.registers_init = torch.Tensor{0, f, 6, 0, 0, f}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)

-- Program
config.nb_states = 7
config.program = {
   torch.Tensor{0, 1, 1, 0, 0, 4, f},  -- first arguments
   torch.Tensor{f, 2, f, 1, f, 3, f},  -- second arguments
   torch.Tensor{1, 5, 1, 5, 0, 5, 5},  -- target register
   torch.Tensor{8,10, 2, 9, 2,10, 0}   -- instruction to operate
}


-- Sample input memory
-- We increment all elements of the list
-- Our input list is {4, 5, 6, 7}
-- The output should be, inplace {5, 6, 7, 8}
config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 4
config.example_input[2] = 5
config.example_input[3] = 6
config.example_input[4] = 7

config.example_output = config.example_input:clone()
config.example_output[1] = 5
config.example_output[2] = 6
config.example_output[3] = 7
config.example_output[4] = 8

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)

config.gen_sample = function()
   local list_length = 1 + math.floor(torch.uniform()*(config.memory_size-2))


   local input = torch.floor(torch.rand(config.memory_size) * (config.memory_size-1))+1
   input:narrow(1, list_length + 1, config.memory_size - list_length):fill(0)
   local output = input:clone()
   output:copy(output+1)
   output:narrow(1, list_length + 1, config.memory_size - list_length):fill(0)

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask:narrow(1,1, list_length+1):fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

config.gen_biased_sample = function()
   -- This is biased because we the lists are always composed of the same value so we don't need to always read
    local val = 1 + math.floor(torch.uniform()*(config.memory_size-2))

    local input = torch.ones(config.memory_size) * val
    input[config.memory_size] = 0
    local output = torch.ones(config.memory_size) * (val + 1)
    output[config.memory_size] = 0

    input = distUtils.toDistTensor(input, config.memory_size)
    output = distUtils.toDistTensor(output, config.memory_size)
    local loss_mask = torch.ones(config.memory_size, config.memory_size)
    return input, output, loss_mask
end


return config
