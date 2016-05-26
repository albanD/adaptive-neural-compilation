-- This a config for the access task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Access"
-- Number of available registers (excluding the RI)
config.nb_registers = 3
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 10
-- Initial state of the registers
config.registers_init = torch.Tensor{0, 0, f}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)
-- Program
config.nb_states = 5
config.program = {
    torch.Tensor{0, 1, 1, 0, f},
    torch.Tensor{f, f, f, 1, f},
    torch.Tensor{1, 1, 1, 2, 2},
    torch.Tensor{8, 2, 8, 9, 0},
}

-- Sample input memory
-- We ask for the third argument of the list (the list is 0 indexed, so this is the 2)
-- Our list is {6, 7, 8, 9}
-- So this should output the number 8 on the first line
config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 2
config.example_input[2] = 6
config.example_input[3] = 7
config.example_input[4] = 8
config.example_input[5] = 9

config.example_output = config.example_input:clone()
config.example_output[1] = 8

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)

config.gen_sample = function()
    local input = torch.floor(torch.rand(config.memory_size)*config.memory_size)
    if input[1]+2 > config.memory_size then
        input[1] = input[1] - 3
    end
    local output = input:clone()
    output[1] = input[input[1]+2]

    local loss_mask = torch.zeros(config.memory_size, config.memory_size)
    loss_mask[1]:fill(1)

    input = distUtils.toDistTensor(input, config.memory_size)
    output = distUtils.toDistTensor(output, config.memory_size)
    return input, output, loss_mask
end


config.gen_biased_sample = function()
    -- This is biased because we always ask to get the 3rd element of the array
    local input = torch.floor(torch.rand(config.memory_size)*config.memory_size)
    input[1] = 3
    local output = input:clone()
    output[1] = input[5]

    local loss_mask = torch.zeros(config.memory_size, config.memory_size)
    loss_mask[1]:fill(1)

    input = distUtils.toDistTensor(input, config.memory_size)
    output = distUtils.toDistTensor(output, config.memory_size)
    return input, output, loss_mask
end

return config
