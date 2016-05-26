-- This a config for the copy task
local config = {}
-- For conversion to distributions
local distUtils = require 'nc.distUtils'
local f = distUtils.flatDist

-- Name of the algorithm
config.name = "Copy"
-- Number of available registers (excluding the RI)
config.nb_registers = 7
-- Number of instructions this program uses
config.nb_existing_ops = 11
-- Size of the memory tape and largest number addressable
config.memory_size = 15
-- Initial state of the registers
config.registers_init = torch.Tensor#{init_string}
config.registers_init = distUtils.toDistTensor(config.registers_init, config.memory_size)
-- Program
config.nb_states = 8
config.program = {
    torch.Tensor#{first_arg_string},
    torch.Tensor#{second_arg_string},
    torch.Tensor#{target_string},
    torch.Tensor#{instruction_string},
}

return config
