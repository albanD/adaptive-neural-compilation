-- This a config for the copy task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Loopy-add"
-- Number of available registers (excluding the RI)
config.nb_registers = 6
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 15
-- Initial state of the registers
config.registers_init = torch.Tensor{6,2,0,1,0,0}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)
-- Program
config.nb_states = 8
config.program = {
    torch.Tensor{4,3,3,3,4,2,2,5},
    torch.Tensor{5,5,0,5,5,1,4,5},
    torch.Tensor{4,3,5,3,4,5,5,5},
    torch.Tensor{8,8,10,5,2,10,9,0},
}

config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 3
config.example_input[2] = 4

config.example_output = config.example_input:clone()
config.example_output[1] = 7

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)


local function randint(a, b)
   return a + math.floor(torch.uniform()*(b-a))
end

config.gen_sample = function()
   local arg1 = randint(0, config.memory_size-2)
   local arg2 = randint(0, config.memory_size-arg1-1)

   local input = torch.zeros(config.memory_size)
   input[1] = arg1
   input[2] = arg2

   local output = input:clone()
   output[1] = arg1 + arg2

   local loss_mask = torch.zeros(config.memory_size, config.memory_size)
   loss_mask[1]:fill(1)

   input = distUtils.toDistTensor(input, config.memory_size)
   output = distUtils.toDistTensor(output, config.memory_size)
   return input, output, loss_mask

end

return config
