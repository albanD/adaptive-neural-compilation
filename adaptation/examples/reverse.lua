-- This a config for the reverse task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Reverse"
-- Number of available registers (excluding the RI)
config.nb_registers = 8
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 15
-- Initial state of the registers
config.registers_init = torch.Tensor{1,f,5,0,1,0,11,f}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)
-- Program
config.nb_states = 12
config.program = {
    torch.Tensor{5, 0, 1, 0, 3, 0, 0, 0, 5, 5, 3, f},
    torch.Tensor{f, f, 2, f, 4, f, 6, f, 1, f, 2, f},
    torch.Tensor{5, 1, 7, 0, 7, 0, 7, 1, 7, 5, 7, 7},
    torch.Tensor{8, 8,10, 2,10, 5,10, 8, 9, 2,10, 0},
}
-- Sample input memory
-- Should output {2, 4, 3} at positions {6, 7, 8}
config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 5
config.example_input[2] = 3
config.example_input[3] = 4
config.example_input[4] = 2
config.example_input[5] = 0

config.example_output = config.example_input:clone()
config.example_output[6] = 2
config.example_output[7] = 4
config.example_output[8] = 3

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)

config.gen_sample = function()
   local possible_list_length = math.floor((config.memory_size -1)/2)
   local list_length = 2 + math.floor(torch.uniform() * (possible_list_length-2))
   local array_of_value = torch.floor(torch.rand(list_length) * (config.memory_size-1))+1


   local input = torch.zeros(config.memory_size)
   input[1] = list_length+1
   input:narrow(1, 2, list_length):copy(array_of_value)

   local output = input:clone()
   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   for i=1,list_length do
      output[list_length+1+i] = array_of_value[list_length+1-i]
      loss_mask[list_length+1+i]:fill(1)
   end

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

return config
