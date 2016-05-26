-- This a config for the copy task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Copy"
-- Number of available registers (excluding the RI)
config.nb_registers = 7
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 15
-- Initial state of the registers
config.registers_init = torch.Tensor{1,0,f,7,0,1,f}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)
-- Program
config.nb_states = 8
config.program = {
    torch.Tensor{1, 0, 2, 1, 0, 1, 4, f},
    torch.Tensor{f, f, 3, 2, f, f, 5, f},
    torch.Tensor{1, 2, 6, 6, 0, 1, 6, 6},
    torch.Tensor{8, 8,10, 9, 2, 2,10, 0},
}
-- Sample input memory
-- Should output {3, 4, 2} at addresses {6, 7, 8}
config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 5
config.example_input[2] = 3
config.example_input[3] = 4
config.example_input[4] = 2
config.example_input[5] = 0

config.example_output = config.example_input:clone()
config.example_output[6] = 3
config.example_output[7] = 4
config.example_output[8] = 2
config.example_output[9] = 0

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)

config.gen_sample = function()
   local list_length = 1 + math.floor(torch.uniform()*(math.floor((config.memory_size-2)/2)-1))
   local where_to_write = list_length + 3 + math.floor(torch.uniform()*(config.memory_size-2*list_length-3))


   local input = torch.floor(torch.rand(config.memory_size) * (config.memory_size-1))+1
   input[1] = where_to_write
   input:narrow(1, list_length + 2, config.memory_size - (list_length+1)):fill(0)
   local output = input:clone()
   output:narrow(1, where_to_write+1, list_length)
      :copy(input:narrow(1, 2, list_length))

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask:narrow(1, where_to_write+1, list_length):fill(1)


   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

config.gen_biased_sample = function()
   -- This is biased because the length of the list are always the same
   -- and the target is always the same.
   -- memory-size is odd
   local input = torch.floor(torch.rand(config.memory_size) * (config.memory_size-1))+1
   input[1] = (config.memory_size+1)/2
   input:narrow(1, (config.memory_size+1)/2 +1 , (config.memory_size-1)/2):fill(0)
   local output = input:clone()
   output:narrow(1, (config.memory_size+1)/2 +1 , (config.memory_size-1)/2)
      :copy(input:narrow(1, 2, (config.memory_size-1)/2))

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   local loss_mask = torch.ones(config.memory_size, config.memory_size)
   return input, output, loss_mask
end

return config
