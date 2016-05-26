-- This a config for the walk BST task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "WalkBST"
-- Number of available registers (excluding the RI)
config.nb_registers = 8
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 30
-- Initial state of the registers
config.registers_init = torch.Tensor{0,1,2,8,0,2,f,f}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)
-- Program
config.nb_states = 11
config.program = {
    torch.Tensor{0, 1, 2, 7, 0, 0, 2, 4, 0, 1, f},
    torch.Tensor{f, f, f, 3, 7, f, f, 5, f, 0, f},
    torch.Tensor{0, 1, 7, 6, 0, 0, 2, 6, 0, 6, 6},
    torch.Tensor{8, 8, 8,10, 3, 8, 2,10, 8, 9, 0},
}
-- Sample input memory
-- The goal is to go along a BST
-- Our BST is
--          8
--        /   \
--       6     10
--      / \   / \
--      5 7  9   11
-- As a BST, where each node is [value, address left, address right]
-- Before the BST, there is pointer_to_first_element, where to write the value and a
-- list of operation whether going left (value 1) or right (value 2)
-- Therefore, the initial memory is, doing two hops: left then right
-- {14, 1, 1, 2, 0, 5, f, f, 7, f, f, 10,17,20, 8,23,11, 9, f, f, 11, f, f, 6, 5, 8}
-- And the output memory should just have the 3-rd address changed, with the pointer to the second element
-- {14, 7, 1, 2, 0, 5, f, f, 7, f, f, 10,17,20, 8,23,11, 9, f, f, 11, f, f, 6, 5, 8}
config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 14
config.example_input[2] = 1
config.example_input[3] = 1
config.example_input[4] = 2
config.example_input[5] = 0
config.example_input[6] = 5
config.example_input[7] = f
config.example_input[8] = f
config.example_input[9] = 7
config.example_input[10] = f
config.example_input[11] = f
config.example_input[12] = 10
config.example_input[13] = 17
config.example_input[14] = 20
config.example_input[15] = 8
config.example_input[16] = 23
config.example_input[17] = 11
config.example_input[18] = 9
config.example_input[19] = f
config.example_input[20] = f
config.example_input[21] = 11
config.example_input[22] = f
config.example_input[23] = f
config.example_input[24] = 6
config.example_input[25] = 5
config.example_input[26] = 8

config.example_output = config.example_input:clone()
config.example_output[2] = 7

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)

local function randint(a, b)
   return a + math.floor(torch.uniform() * (b-a))
end

local make_bst

make_bst = function(ordered_tensor_to_fill, already_used, possible_list_length, input_tensor)
   local nb_elements_to_fill = ordered_tensor_to_fill:size(1)
   local median = math.ceil(nb_elements_to_fill/2)
   local value_to_write = ordered_tensor_to_fill[median]

   local pos_in_the_random_order
   while true  do
      pos_in_the_random_order = randint(0, possible_list_length)
      if not already_used[pos_in_the_random_order] then
         break
      end
   end
   already_used[pos_in_the_random_order] = true
   local head_of_the_list = 7 + 3*pos_in_the_random_order
   input_tensor[head_of_the_list] = value_to_write

   local nb_elt_to_the_left = median - 1
   local pointer_to_the_left
   if nb_elt_to_the_left>0 then
      pointer_to_the_left = make_bst(ordered_tensor_to_fill:narrow(1,1,nb_elt_to_the_left),
                                     already_used,
                                     possible_list_length,
                                     input_tensor)
   else
      pointer_to_the_left = f
   end
   input_tensor[head_of_the_list+1] = pointer_to_the_left

   local nb_elt_to_the_right = nb_elements_to_fill - median
   local pointer_to_the_right
   if nb_elt_to_the_right > 0 then
      pointer_to_the_right = make_bst(ordered_tensor_to_fill:narrow(1, median+1, nb_elt_to_the_right),
                                      already_used,
                                      possible_list_length,
                                      input_tensor)
   else
      pointer_to_the_right = f
   end
   input_tensor[head_of_the_list+2] = pointer_to_the_right

   return head_of_the_list-1
end



config.gen_sample = function()
   local possible_list_length = math.floor((config.memory_size - 7)/3) -- -2 for root and out, -4 for the max depth, -1 for the null terminator of the sequence of directions
   local list_length = randint(3, possible_list_length) -- what is the length of a list


   local list_to_store = torch.floor(torch.rand(list_length) * (config.memory_size-1))+1
   list_to_store = list_to_store:sort(1)
   local already_used = {}

   local input = torch.zeros(config.memory_size)

   local head_of_the_list = make_bst(list_to_store, already_used, possible_list_length, input)
   input[1] = head_of_the_list
   input[2] = 1 -- where to write the target
   -- Figure out which element we are looking for:
   local val = list_to_store[randint(1, list_length)]
   -- Figure out the path that leads to it
   local pointer_to_current = head_of_the_list
   local current_val = input[pointer_to_current+1]

   for i=1,4 do
      if val<current_val then
         input[2+i] = 1
         pointer_to_current = input[pointer_to_current+1+1]
         current_val = input[pointer_to_current+1]
      elseif val>current_val then
         input[2+i] = 2
         pointer_to_current = input[pointer_to_current+1+2]
         current_val = input[pointer_to_current+1]
      else
         break
      end
   end



   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   local output = input:clone()
   output[2] = val
   loss_mask[2]:fill(1)
   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)

   return input, output, loss_mask
end


return config
