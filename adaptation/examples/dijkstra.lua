-- This a config for the shortest path problem
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Copy"
-- Number of available registers (excluding the RI)
config.nb_registers = 23
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 100
-- Initial state of the registers
config.registers_init = torch.Tensor{56,49,35,31,16,14,12,3,0,0,0,0,0,99,0,0,1,1,0,0,0,0,0}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)
-- Program
config.nb_states = 8
config.program = {
    torch.Tensor{19,19,14,16,9,18,18,18,18,16,15,14,19,14,17,16,16,12,12,16,16,16,19,18,18,21,10,10,9,18,14,19,14,13,14,18,12,12,9,18,18,18,9,21,9,9,11,12,14,21,9,19,18,18,18,14,22},
    torch.Tensor{22,14,22,22,6,13,22,10,22,22,22,7,14,3,20,22,22,3,22,22,22,22,12,12,22,11,11,9,4,11,4,22,22,14,22,22,22,15,1,22,22,22,2,11,11,2,14,14,2,13,0,20,20,22,14,5,22},
    torch.Tensor{19,18,10,9,22,22,18,22,18,16,15,22,22,22,16,16,12,22,12,16,11,16,18,18,10,11,9,9,22,22,22,18,12,21,20,18,12,9,22,11,18,9,22,9,9,22,21,20,22,9,22,18,18,18,22,22,22},
    torch.Tensor{8,3,2,8,10,9,2,9,2,2,2,10,9,10,3,8,8,10,5,2,8,2,3,3,8,3,6,4,10,9,10,5,5,3,5,2,2,4,10,8,2,8,10,7,4,10,3,3,10,4,10,3,3,2,9,10,0},
}

config.example_input = torch.zeros(config.memory_size)
config.example_input[1] = 49 -- output address
config.example_input[2] = 14 -- in[15]
config.example_input[3] = 19 -- in[20]
config.example_input[4] = 22 -- in[23]
config.example_input[5] = 25 -- in[26]
config.example_input[6] = 28 -- in[29]
config.example_input[7] = 0

                             -- from node 1
config.example_input[15] = 2 -- node 2
config.example_input[16] = 2 -- 1-2 = 2
config.example_input[17] = 4 -- node 4
config.example_input[18] = 5 -- 1-4 = 5
config.example_input[19] = 0

                             -- from node 2
config.example_input[20] = 3 -- node 3
config.example_input[21] = 1 -- 2-3 = 1
config.example_input[22] = 0

                             -- from node 3
config.example_input[23] = 5 -- node 5
config.example_input[24] = 1 -- 3-5 = 1
config.example_input[25] = 0

                             -- from node 4
config.example_input[26] = 5 -- node 5
config.example_input[27] = 6 -- 4-5 = 6
config.example_input[28] = 0

                             -- from node 5
config.example_input[29] = 0

config.example_output = config.example_input:clone()
config.example_output[50] = 0
config.example_output[51] = 0
config.example_output[52] = 2
config.example_output[53] = 0
config.example_output[54] = 3
config.example_output[55] = 0
config.example_output[56] = 5
config.example_output[57] = 0
config.example_output[58] = 4
config.example_output[59] = 0

config.example_input = distUtils.toDistTensor(config.example_input, config.memory_size)
config.example_output = distUtils.toDistTensor(config.example_output, config.memory_size)
config.example_loss_mask = torch.ones(config.memory_size, config.memory_size)


return config
