-- This a config for the copy task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "LinearSearch"
-- Number of available registers (excluding the RI)
config.nb_registers = 7
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 15
-- Initial state of the registers
config.registers_init = torch.Tensor{6,0,1,0,0,1,0}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)
-- Program
config.nb_states = 8
config.program = {
    torch.Tensor{4,5,1,1,5,4,4,6},
    torch.Tensor{6,6,3,0,6,2,5,6},
    torch.Tensor{3,1,1,6,5,6,6,6},
    torch.Tensor{8,8,4,10,2,10,9,0},
}

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
config.example_input[10] = 2

config.example_output = config.example_input:clone()
config.example_output[1] = 7

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)

local function randint(a, b)
   return a + math.floor(torch.uniform()*(b-a))
end

config.gen_sample = function()
   -- If memory size is 15, the worst case takes 72 iterations
   -- Remove 5 iterations per decrease in the max size
   local max_size = config.memory_size - 1
   local list_length = randint(2, max_size)

   -- We don't want any duplicates in the list
   local full_list = torch.randperm(max_size)


   local input = torch.zeros(config.memory_size)
   input:narrow(1,2, list_length):copy(full_list:narrow(1,1,list_length))
   -- Which element are we looking for?
   local index = randint(1, list_length)
   input[1] = full_list[index]

   local output = input:clone()
   output[1] = index

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask[1]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

config.gen_biased_sample_not_before_k = function(k)
   -- This is biased because the correct value can not be in the first (k-1) ones.
   local max_size = config.memory_size - 1
   local list_length = randint(k, max_size)

   -- We don't want any duplicates in the list
   local full_list = torch.randperm(max_size)


   local input = torch.zeros(config.memory_size)
   input:narrow(1,2, list_length):copy(full_list:narrow(1,1,list_length))
   -- Which element are we looking for?
   local index = randint(k, list_length)
   input[1] = full_list[index]

   local output = input:clone()
   output[1] = index

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask[1]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

local not_before_k_ateur = function(k)
   return function()
      return config.gen_biased_sample_not_before_k(k)
   end
end


config.gen_biased_only_at_even_pos = function()
   -- This is biased because the correct value can only be even
   local max_size = config.memory_size - 1
   local list_length = randint(2, max_size)

   -- We don't want any duplicates in the list
   local full_list = torch.randperm(max_size)


   local input = torch.zeros(config.memory_size)
   input:narrow(1,2, list_length):copy(full_list:narrow(1,1,list_length))
   -- Which element are we looking for?
   local index = randint(1, math.floor(list_length/2))
   index = 2 * index
   input[1] = full_list[index]

   local output = input:clone()
   output[1] = index

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask[1]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

config.gen_biased_only_at_odd_pos = function()
   -- This is biased because the correct value can only be odd
   local max_size = config.memory_size - 1
   local list_length = randint(2, max_size)

   -- We don't want any duplicates in the list
   local full_list = torch.randperm(max_size)


   local input = torch.zeros(config.memory_size)
   input:narrow(1,2, list_length):copy(full_list:narrow(1,1,list_length))
   -- Which element are we looking for?
   local index = randint(0, math.floor((list_length-1)/2))
   index = 1+ 2 * index
   input[1] = full_list[index]

   local output = input:clone()
   output[1] = index

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask[1]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

config.gen_biased_sorted_down = function()
   -- This is biased because all the elements are sorted in decreasing order
   local max_size = config.memory_size - 1
   local list_length = randint(2, max_size)

   -- We don't want any duplicates in the list
   local full_list = torch.randperm(max_size)
   full_list = full_list:narrow(1,1,list_length):sort(1, true)

   local input = torch.zeros(config.memory_size)
   input:narrow(1,2, list_length):copy(full_list)
   -- Which element are we looking for?
   local index = randint(0, math.floor((list_length-1)/2))
   index = 1+ 2 * index
   input[1] = full_list[index]

   local output = input:clone()
   output[1] = index

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask[1]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

config.gen_biased_sorted_up = function()
   -- This is biased because all the elements are sorted in increasing order
   local max_size = config.memory_size - 1
   local list_length = randint(2, max_size)

   -- We don't want any duplicates in the list
   local full_list = torch.randperm(max_size)
   full_list = full_list:narrow(1,1,list_length):sort(1, false)

   local input = torch.zeros(config.memory_size)
   input:narrow(1,2, list_length):copy(full_list)
   -- Which element are we looking for?
   local index = randint(0, math.floor((list_length-1)/2))
   index = 1+ 2 * index
   input[1] = full_list[index]

   local output = input:clone()
   output[1] = index

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask[1]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

config.gen_biased_natural = function()
   -- This is biased because the list is always the sequence of natural orders
   -- (potentially wrapping around)
   local max_size = config.memory_size - 1
   local list_length = randint(2, max_size)


   local list_start = randint(1, config.memory_size-1)
   -- We don't want any duplicates in the list
   local full_list = torch.cat(torch.range(list_start, config.memory_size-1), torch.range(0, list_start),1)

   local input = torch.zeros(config.memory_size)
   input:narrow(1,2, list_length):copy(full_list:narrow(1,1,list_length))
   -- Which element are we looking for?
   local index = randint(0, math.floor((list_length-1)/2))
   index = 1+ 2 * index
   input[1] = full_list[index]

   local output = input:clone()
   output[1] = index

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask[1]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

config.gen_biased_natural_reverse = function()
   -- This is biased because the list is always the sequence of natural orders, in reverse orders
   -- (potentially wrapping around)
   local max_size = config.memory_size - 1
   local list_length = randint(2, max_size)


   local list_start = randint(1, config.memory_size-1)
   -- We don't want any duplicates in the list
   local full_list = torch.cat(torch.range(0, list_start):sort(1,true),
                               torch.range(list_start, config.memory_size-1):sort(1,true),1)


   local input = torch.zeros(config.memory_size)
   input:narrow(1,2, list_length):copy(full_list:narrow(1,1,list_length))
   -- Which element are we looking for?
   local index = randint(0, math.floor((list_length-1)/2))
   index = 1+ 2 * index
   input[1] = full_list[index]

   local output = input:clone()
   output[1] = index

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask[1]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask
end

config.gen_biased_sample = {
   config.gen_biased_only_at_even_pos,
   config.gen_biased_only_at_odd_pos,
   config.gen_biased_sorted_down,
   config.gen_biased_sorted_up,
   config.gen_biased_natural,
   config.gen_biased_natural_reverse,
   not_before_k_ateur(3),
   not_before_k_ateur(5),
   not_before_k_ateur(8)
}

return config
