-- This is a config for the increment task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Merge"
-- Number of available registers (excluding the RI)
config.nb_registers = 13
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 30
-- Initial state of the registers
config.registers_init = torch.Tensor{0, 1, 2, f, f, f, 14, 0, 3, 18, 20, 26, f}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)

-- Program
config.nb_states = 27
config.program = {
   torch.Tensor{0, 1, 2, 0, 1, 3, 4, 3, 3, 5, 2, 2, 0, 7, 2, 2, 1, 7, 1, 4, 2 ,0, 2, 0, 3, 7, f},  -- first arguments
   torch.Tensor{f, f, f, f, f, 9,10, 4, 5, 6, 3, f, f, 8, 4, f, f, 8, 7, 7, 3, f, f, f,11,10, f},  -- second arguments
   torch.Tensor{0, 1, 2, 3, 4,12,12, 5, 5,12,12, 2, 0,12,12, 2, 1,12, 0, 3,12, 0, 2, 3,12,12,12},  -- target register
   torch.Tensor{8, 8, 8, 8, 8,10,10, 6, 4,10, 9, 2, 2,10, 9, 2, 2,10, 3, 3 ,9, 2, 2, 8,10,10, 0}   -- instruction to operate
}


-- Sample input memory
-- The goal is to merge two sorted lists together
-- Our lists are {1, 3, 5, 6, 7, 8} {2, 4, 9}
-- The lists are prefixed with three values: start_of_first_array, start_of_second_array, start_of_output
-- The lists are delimited by the zero value.
-- Therefore, the initial memory is, for our two lists
-- {3,10,14, 8, 7, 6, 5, 3, 1, 0, 9, 4, 2, 0, f, f, f, f, f, f, f, f, f}
-- And the output memory should be:
-- {3,10,14, 8, 7, 6, 5, 3, 1, 0, 9, 4 ,2, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1}
config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 3
config.example_input[2] = 10
config.example_input[3] = 14
config.example_input[4] = 8
config.example_input[5] = 7
config.example_input[6] = 6
config.example_input[7] = 5
config.example_input[8] = 3
config.example_input[9] = 1
config.example_input[10] = 0
config.example_input[11] = 9
config.example_input[12] = 4
config.example_input[13] = 2
config.example_input[14] = 0
config.example_input[15] = f
config.example_input[16] = f
config.example_input[17] = f
config.example_input[18] = f
config.example_input[19] = f
config.example_input[20]= f
config.example_input[21] = f
config.example_input[22] = f
config.example_input[23] = f

config.example_output = config.example_input:clone()
config.example_output[15] = 9
config.example_output[16] = 8
config.example_output[17] = 7
config.example_output[18] = 6
config.example_output[19] = 5
config.example_output[20]= 4
config.example_output[21] = 3
config.example_output[22] = 2
config.example_output[23] = 1


config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)

config.gen_sample = function()
   local possible_list_length_1 = math.floor(config.memory_size/2) - 6 -- 3 for the first arguments, 2 for the final 0 delimiters, 1 for because the second list needs to have at least 1 elements
   local list_length_1 = 2 + math.floor(torch.uniform() * (possible_list_length_1 - 2) )

   local possible_list_length_2 = math.floor(config.memory_size/2) - 5 - list_length_1
   local list_length_2 = 1 + math.floor(torch.uniform() * (possible_list_length_2 -1))


   local list_1 = torch.floor(torch.rand(list_length_1) * (config.memory_size -1 ))+1
   local list_2 = torch.floor(torch.rand(list_length_2) * (config.memory_size -1 ))+1

   local merged_list = torch.cat(list_1, list_2)
   list_1 = list_1:sort(1, true)
   list_2 = list_2:sort(1, true)
   merged_list = merged_list:sort(1, true)

   local input = torch.zeros(config.memory_size)

   input[1] = 3 -- where does the first one start?
   input[2] = 4 + list_length_1 -- where does the second one start?
   input[3] = 4 + list_length_1 + list_length_2 + 1 -- where do we start to write?

   input:narrow(1, 4, list_length_1):copy(list_1)
   input:narrow(1, 4 + list_length_1 + 1, list_length_2):copy(list_2)

   local output = input:clone()
   output:narrow(1, 4 + list_length_1 + list_length_2 + 2, list_length_1+list_length_2):copy(merged_list)

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask:narrow(1, 4 + list_length_1 + list_length_2 + 2, list_length_1+list_length_2+1):fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end


return config
