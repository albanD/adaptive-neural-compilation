local ops = require 'nc.ops'
local distUtils = require 'nc.distUtils'
local Add_op, parent = torch.class('nc.Add_op', 'nc.Abstract_op', ops)
ops.nb_existing_ops = ops.nb_existing_ops + 1

function Add_op:__init(memory_size)
	parent.__init(self, memory_size)
end

function Add_op:updateOutput(input)
	local dist1 = input[1][1]
	local dist2 = input[1][2]
	self.output = distUtils.addDist(dist1, dist2, self.output)
   return self.output
end

function Add_op:updateGradInput(input, gradOutput)
    local dist1 = input[1][1]
    local dist2 = input[1][2]
    assert(dist1:size(1)==gradOutput:size(1), "Grad output incorrect size in add op")

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

    for i=1,dist1:size(1) do
        local acc_1 = 0
        local acc_2 = 0
        for c=1,gradOutput:size(1) do
            -- dist2[(c-i)%self.memory_size+1] = sum_b dist2[b] 1_{i+b==c}
            acc_1 = acc_1 + gradOutput[c]*dist2[(c-i)%self.memory_size+1]
            acc_2 = acc_2 + gradOutput[c]*dist1[(c-i)%self.memory_size+1]
        end
        self.gradInput[1][1][i] = acc_1
        self.gradInput[1][2][i] = acc_2
    end

    return self.gradInput
end
