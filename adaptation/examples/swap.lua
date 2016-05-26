-- This is a config for the Swap task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Swap"
-- Number of available registers (excluding the RI)
config.nb_registers = 5
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 10
-- Initial state of the registers
config.registers_init = torch.Tensor{0, 1, f, f, 2}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)

-- Program
config.nb_states = 9
config.program = {
   torch.Tensor{0, 0, 1, 1, 0, 1, 0, 1, f},  -- first arguments
   torch.Tensor{f, 4, f, 4, f, f, 3, 2, f},  -- second arguments
   torch.Tensor{0, 0, 1, 1, 2, 3, 4, 4, 4},  -- target register
   torch.Tensor{8, 3, 8, 3, 8, 8, 9, 9, 0}   -- instruction to operate
}


-- Sample input memory
-- the first two values are addresses in the list
-- In this sample, we will switch the third and fourth value (index 2 and 3)
-- Our input list is {4,5,6,7}
-- The output list should be {4,5,7,6}
config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 2
config.example_input[2] = 3
config.example_input[3] = 4
config.example_input[4] = 5
config.example_input[5] = 6
config.example_input[6] = 7

config.example_output = config.example_input:clone()
config.example_output[5] = 7
config.example_output[6] = 6

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)

local function randint(a, b)
   return a + math.floor(torch.uniform() * (b-a))
end

config.gen_sample = function()
   local possible_list_length = math.floor(config.memory_size-2)
   local list_length = randint(2, possible_list_length)
   local array_of_value = torch.floor(torch.rand(list_length) * (config.memory_size-1))+1

   local p = randint(0, list_length-1)
   local q = randint(0, list_length-1)

   local input = torch.zeros(config.memory_size)
   input[1] = p
   input[2] = q
   input:narrow(1, 3, list_length):copy(array_of_value)

   local output = input:clone()
   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   output[3+p] = array_of_value[1+q]
   loss_mask[3+p]:fill(1)
   output[3+q] = array_of_value[1+p]
   loss_mask[3+q]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

config.gen_biased_sample = function()
   local possible_list_length = math.floor(config.memory_size-2)
   local list_length = 3 + math.floor(torch.uniform() * (possible_list_length-3))
   local array_of_value = torch.floor(torch.rand(list_length) * (config.memory_size-1))+1

   local p = 0
   local q = 2

   local input = torch.zeros(config.memory_size)
   input[1] = p
   input[2] = q
   input:narrow(1, 3, list_length):copy(array_of_value)

   local output = input:clone()
   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   output[3+p] = array_of_value[1+q]
   loss_mask[3+p]:fill(1)
   output[3+q] = array_of_value[1+p]
   loss_mask[3+q]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end



return config
