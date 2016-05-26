local ops = require 'nc.ops'
local distUtils = require 'nc.distUtils'
local Zero_op, parent = torch.class('nc.Zero_op', 'nc.Abstract_op', ops)
ops.nb_existing_ops = ops.nb_existing_ops + 1

function Zero_op:__init(memory_size)
    parent.__init(self, memory_size)
end

function Zero_op:updateOutput(input)
    self.output = distUtils.toDist(0, self.memory_size)
   	return self.output
end

function Zero_op:updateGradInput(input, gradOutput)
    -- The gradient on the input does not depend on the gradient on the output
    -- so just return 0
    local dist1 = input[1][1]
    assert(dist1:size(1)==gradOutput:size(1), "GradOutput incorrect size in zero op")

    if not (type(self.gradInput) == "table") then
        self.gradInput = {}
        self.gradInput[1] = {}
        self.gradInput[1][1] = torch.Tensor(dist1:size(1))
        self.gradInput[1][2] = torch.Tensor(dist1:size(1))
        self.gradInput[2] = torch.Tensor(input[2]:size())
    end
    self.gradInput[1][1]:zero()
    self.gradInput[1][2]:zero()
    self.gradInput[2]:zero()

    return self.gradInput
end
