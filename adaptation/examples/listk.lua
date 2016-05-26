-- This is a config for the listk task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "listK"
-- Number of available registers (excluding the RI)
config.nb_registers = 7
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 20
-- Initial state of the registers
config.registers_init = torch.Tensor{0, 1, 2, 6, 0, 2, f}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)

-- Program
config.nb_states = 11
config.program = {
   torch.Tensor{1, 2, 0, 1, 1, 4, 0, 0, 2, f},  -- first arguments
   torch.Tensor{f, f, f, f, 3, 5, f, f, 0, f},  -- second arguments
   torch.Tensor{1, 2, 0, 1, 6, 6, 0, 0, 6, 6},  -- target register
   torch.Tensor{8, 8, 8, 5,10,10, 2, 8, 9, 0}   -- instruction to operate
}


-- Sample input memory
-- The goal is to go along a linked list
-- Our list is {4, 5, 6, 7}
-- As a linked list, where each node is [address of next node, value] this is {3+2*1, 5, 3+2*3, 6, 3+2*0,4,f,7}
-- Before the list, there is pointer_to_first_element, number_of_hops, where to write the value
-- Therefore, the initial memory is, doing two hops and writing at the third pos
-- {7, 2, 2, 5, 5, 9, 6, 3, 4, f, 7}
-- And the output memory should just have the 3-rd address changed, with the second element of the list
-- {7, 2, 5, 5, 5, 9, 6, 3, 4, f, 7}
config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 7
config.example_input[2] = 2
config.example_input[3] = 2
config.example_input[4] = 5
config.example_input[5] = 5
config.example_input[6] = 9
config.example_input[7] = 6
config.example_input[8] = 3
config.example_input[9] = 4
config.example_input[10] = f
config.example_input[11] = 7

config.example_output = config.example_input:clone()
config.example_output[3] = 5

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)

local function randint(a, b)
   return a + math.floor(torch.uniform()*(b-a))
end

config.gen_sample = function()
   local possible_list_length = math.floor((config.memory_size - 3)/2)
   local list_length = randint(2, possible_list_length) -- what is the length of a list


   local list_to_store = torch.floor(torch.rand(list_length) * (config.memory_size-1))+1


   local head_of_the_list
   local which_element_of_the_list = randint(1, list_length-1)

   local input = torch.rand(config.memory_size)
   local already_used = {}
   -- fill the list in reverse order
   local pointer_to_next_element = 0
   for i=list_length,1,-1 do
      local pos_in_the_linked_list
      while true do
         pos_in_the_linked_list = randint(0, possible_list_length-1)
         if not already_used[pos_in_the_linked_list] then
            break
         end
      end
      already_used[pos_in_the_linked_list] = true
      local input_index_for_pointer = 4 + 2 * pos_in_the_linked_list
      local input_index_for_value = 4 + 2 * pos_in_the_linked_list + 1

      input[input_index_for_pointer] = pointer_to_next_element
      input[input_index_for_value] = list_to_store[i]

      pointer_to_next_element = input_index_for_pointer-1
   end

   -- Where does the linked list start?
   head_of_the_list = pointer_to_next_element
   input[1] = head_of_the_list

   -- What is the value that we search for
   input[2] = which_element_of_the_list

   -- Where do we write the output?
   local out = 2
   input[3] = out

   -- Figure out which one is the first (in case there is several in the list)
   local output = input:clone()
   output[3] = list_to_store[which_element_of_the_list]

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask[3]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

config.gen_biased_sample = function ()
   -- These samples are biased because the linked-list elements are
   -- not randomly aligned in memory, in fact they are in contiguous
   -- space, and in the correct order.
   local possible_list_length = math.floor((config.memory_size - 3)/2)
   local list_length = randint(2, possible_list_length) -- what is the length of a list


   local list_to_store = torch.floor(torch.rand(list_length) * (config.memory_size-1))+1


   local which_element_of_the_list =  randint(1, list_length)

   local input = torch.rand(config.memory_size)

   -- fill the list in reverse order
   local pointer_to_next_element = 0
   for i=1, list_length do

      local input_index_for_pointer = 2 + 2 * i
      local input_index_for_value = 2 + 2 * i + 1

      input[input_index_for_pointer] = input_index_for_pointer + 2 -1
      input[input_index_for_value] = list_to_store[i]

   end

   -- Where does the linked list start?
   local head_of_the_list = 3
   input[1] = head_of_the_list

   -- What is the value that we search for
   input[2] = which_element_of_the_list

   -- Where do we write the output?
   local out = 2
   input[3] = out

   -- Figure out which one is the first (in case there is several in the list)
   local output = input:clone()
   output[3] = list_to_store[which_element_of_the_list]

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask[3]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask


end


return config
