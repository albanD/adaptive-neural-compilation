-- This a sample config file for the CLI tool
-- This sample reads two values in the memory adds them and writes the
-- result in the third position.

local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Example"
-- Number of available registers (excluding IR)
config.nb_registers = 4
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number representable
config.memory_size = 5
-- Initial state of the registers: zero, one, two and flat
config.registers_init = torch.Tensor{0, 1, 2, f}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)
-- Program
config.nb_states = 5
config.program = {
    torch.Tensor{0, 1, 0, 2, f},
    torch.Tensor{f, f, 1, 0, f},
    torch.Tensor{0, 1, 0, 3, 3},
    torch.Tensor{8, 8, 2, 9, 0},
}

-- Sample single input memory
-- Input
config.example_input = torch.zeros(config.memory_size) -- Set all input to 0
config.example_input[1] = 1 -- Set first value to 1
config.example_input[2] = 2 -- Set second value to 2
-- Output
config.example_output = config.example_input:clone() -- Output is a clone of the input
config.example_output[3] = 1+2 -- Where we change the third value with 1+2
-- Make them distribution
config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUt
-- Mask
-- Here we select the whole memory
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)ils.toDistTensor(config.example_output, config.memory_size)

-- Util function to draw an integer in [a,b)
local function randint(a,b)
   return a + math.floor(torch.uniform() * (b-a))
end

-- Function that will generate a random sample
config.gen_sample = function()
    -- An input contains random values in the first two memory cells
    local input = torch.zeros(config.memory_size)
    input[1] = randint(0,config.memory_size)
    input[2] = randint(0,config.memory_size)

    -- The output contains also the result of the sum (modulo the representable number)
    local output = input:clone()
    output[3] = (input[1] + input[2]) % config.memory_size

    -- Mask is only the third memory cell
    local loss_mask = torch.zeros(config.memory_size, config.memory_size)
    loss_mask[3]:fill(1)

    input = distUtils.toDistTensor(input, config.memory_size)
    output = distUtils.toDistTensor(output, config.memory_size)
    return input, output, loss_mask
end

-- Function that will generate a biased random sample
-- In this case, the first input value is constant
config.gen_sample = function()
    -- An input contains random values in the first two memory cells
    local input = torch.zeros(config.memory_size)
    input[1] = 2 -- Fix value
    input[2] = randint(0,config.memory_size)

    -- The output contains also the result of the sum (modulo the representable number)
    local output = input:clone()
    output[3] = (input[1] + input[2]) % config.memory_size

    -- Mask is only the third memory cell
    local loss_mask = torch.zeros(config.memory_size, config.memory_size)
    loss_mask[3]:fill(1)

    input = distUtils.toDistTensor(input, config.memory_size)
    output = distUtils.toDistTensor(output, config.memory_size)
    return input, output, loss_mask
end

return config
