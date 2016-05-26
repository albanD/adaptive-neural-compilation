local ops = require 'nc.ops'
local Read_op, parent = torch.class('nc.Read_op', 'nc.Abstract_op', ops)
ops.nb_existing_ops = ops.nb_existing_ops + 1

function Read_op:__init(memory_size)
    parent.__init(self, memory_size)
end

function Read_op:updateOutput(input)
    local address = input[1][1]
    -- The second argument is junk and will be ignored.
    local memory_tape = input[2]

    self.output:resize(memory_tape:size(2))
    self.output:mv(memory_tape:t(), address)

    return self.output
end

function Read_op:updateGradInput(input, gradOutput)
    local address = input[1][1]
    -- The second argument is junk and its gradient will br 0
    local memory_tape = input[2]
    assert(address:size(1)==gradOutput:size(1), "GradOutput incorrect size in read op")

    if not (type(self.gradInput) == "table") then
        self.gradInput = {}
        self.gradInput[1] = {}
        self.gradInput[1][1] = torch.Tensor(address:size(1))
        self.gradInput[1][2] = torch.Tensor(address:size(1))
        self.gradInput[2] = torch.Tensor(memory_tape:size())
    end
    self.gradInput[1][1]:zero()
    self.gradInput[1][2]:zero()
    self.gradInput[2]:zero()

    self.gradInput[1][1]:addmv(0, 1, memory_tape, gradOutput)
    self.gradInput[2]:ger(address, gradOutput)

    return self.gradInput
end
