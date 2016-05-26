-- This a config for the Sort task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Sort"
-- Number of available registers (excluding the RI)
config.nb_registers = 10
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 21
-- Initial state of the registers
config.registers_init = torch.Tensor{18,16,8,0,0,0,0,1,0,0}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)
-- Program
config.nb_states = 21
config.program = {
    torch.Tensor{8,8,8,4,5,4,3,6,5,3,8,8,8,8,9,6,7,9,9,7,6},
    torch.Tensor{9,9,9,1,4,3,2,6,4,6,5,9,4,9,9,6,0,9,9,9,6},
    torch.Tensor{5,8,4,9,3,3,9,9,3,9,9,8,9,8,7,9,9,9,8,7,9},
    torch.Tensor{8,2,8,10,6,4,10,10,4,10,9,5,9,2,1,10,10,0,1,2,10},
}

-- Sample input memory
config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 5
config.example_input[2] = 3
config.example_input[3] = 4
config.example_input[4] = 2
config.example_input[5] = 0

config.example_output = torch.zeros(config.memory_size)
config.example_output[1] = 2
config.example_output[2] = 3
config.example_output[3] = 4
config.example_output[4] = 5
config.example_output[5] = 0

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)

local function randint(a,b)
   return a + math.floor(torch.uniform() * (b-a))
end

config.gen_sample = function()
   local max_list_length = 4
   local list_length = randint(1, max_list_length)

   local input_vals = torch.floor(torch.rand(list_length) * (config.memory_size-1)) + 1
   local input = torch.zeros(config.memory_size)
   input:narrow(1, 1, list_length):copy(input_vals)

   local output = torch.zeros(config.memory_size)
   local output_vals = input_vals:sort(1, false)
   output:narrow(1,1, list_length):copy(output_vals)

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask:narrow(1, 1, list_length):fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end


config.gen_biased_sample = function()
   -- This is biased because only the first two elements should
   -- potentially be changed, the rest wou
   local max_list_length = 4
   local max_value_for_first_elements = config.memory_size / 2
   local list_length = randint(3, max_list_length)
   local first_elt = randint(1, max_value_for_first_elements)
   local second_elt = randint(1, max_value_for_first_elements)

   local biggest, smallest
   if first_elt > second_elt then
      biggest = first_elt
      smallest = second_elt
   else
      biggest = second_elt
      smallest = first_elt
   end

   local input_vals = torch.floor(torch.rand(list_length-2) * (config.memory_size-biggest)) + biggest
   input_vals = input_vals:sort(1, false)
   local input = torch.zeros(config.memory_size)
   input[1] = first_elt
   input[2] = second_elt
   input:narrow(1, 3, list_length-2):copy(input_vals)

   local output = torch.zeros(config.memory_size)
   local output_vals = input_vals:sort(1, false)
   output[1] = smallest
   output[2] = biggest
   output:narrow(1,3, list_length-2):copy(output_vals)

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask:narrow(1, 1, list_length):fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end


return config
