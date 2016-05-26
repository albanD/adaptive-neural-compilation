-- This a config for the permutataion task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Permutation"
-- Number of available registers (excluding the RI)
config.nb_registers = 8
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 15
-- Initial state of the registers
config.registers_init = torch.Tensor{-1,f,4,0,0,0,11,f}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)
-- Program
config.nb_states = 12
config.program = {
    torch.Tensor{0, 0, 1, 3, 5, 1, 0, 1, 5, 5, 4, f},
    torch.Tensor{f, f, 2, 4, f, 6, 1, f, 1, f, 2, f},
    torch.Tensor{0, 1, 7, 7, 1, 7, 1, 1, 7, 5, 7, 7},
    torch.Tensor{2, 8,10,10, 8,10, 3, 8, 9, 2,10, 0},
}
-- Sample input memory
-- We ask for the permutation {1, 3, 2} of the list {6, 8, 7}
-- So this should output {6, 7, 8} at position {1, 2, 3}
config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 1
config.example_input[2] = 3
config.example_input[3] = 2
config.example_input[4] = 0
config.example_input[5] = 6
config.example_input[6] = 8
config.example_input[7] = 7

config.example_output = config.example_input:clone()
config.example_output[1] = 6
config.example_output[2] = 7
config.example_output[3] = 8

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)

config.gen_sample = function()
   -- Note: the task here described is a bit different than the one
   -- used in the original Neural Ram paper
   local possible_list_length = math.floor((config.memory_size -1)/2)
   local list_length = 2 + math.floor(torch.uniform() * (possible_list_length - 2))
   local array_of_value = torch.floor(torch.rand(list_length) * (config.memory_size-1))+1
   local permutation = torch.randperm(list_length)

   local input = torch.zeros(config.memory_size)
   input:narrow(1, 1, list_length):copy(permutation)
   input:narrow(1, 2 + list_length, list_length):copy(array_of_value)

   local output = input:clone()
   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   for i=1,list_length do
      output[i] = array_of_value[permutation[i]]
      loss_mask[i]:fill(1)
   end

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

return config
