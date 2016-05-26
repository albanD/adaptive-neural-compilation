local ops = require 'nc.ops'
local distUtils = require 'nc.distUtils'
local Min_op, parent = torch.class('nc.Min_op', 'nc.Abstract_op', ops)
ops.nb_existing_ops = ops.nb_existing_ops + 1

function Min_op:__init(memory_size)
	parent.__init(self, memory_size)
end

function Min_op:updateOutput(input)
	local dist1 = input[1][1]	
	local dist2 = input[1][2]
	self.output = distUtils.minDist(dist1, dist2, self.output)
   return self.output
end

function Min_op:updateGradInput(input, gradOutput)
    local dist1 = input[1][1]
    local dist2 = input[1][2]
    assert(dist1:size(1)==gradOutput:size(1), "GradOutput incorrect size in min op")

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

    -- Get reverse cumsum
    local rcum1 = torch.Tensor(dist1:storage())
    distUtils.reverseTensor(rcum1)
    rcum1 = torch.cumsum(rcum1)
    distUtils.reverseTensor(rcum1)
    local rcum2 = torch.Tensor(dist2:storage())
    distUtils.reverseTensor(rcum2)
    rcum2 = torch.cumsum(rcum2)
    distUtils.reverseTensor(rcum2)

    for i=1,dist1:size(1) do
        -- sum_c (gradOutput[c] * sum_b dist2[b] * 1_{i+1==c}
        local acc_1 = gradOutput[i]*rcum2[i]
        local acc_2 = gradOutput[i]*rcum1[i]
        for c=1,i-1 do
            acc_1 = acc_1 + gradOutput[c]*dist2[c]
            acc_2 = acc_2 + gradOutput[c]*dist1[c]
        end

        self.gradInput[1][1][i] = acc_1
        self.gradInput[1][2][i] = acc_2
    end

    return self.gradInput
end

