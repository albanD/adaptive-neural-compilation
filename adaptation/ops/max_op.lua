local ops = require 'nc.ops'
local distUtils = require 'nc.distUtils'
local Max_op, parent = torch.class('nc.Max_op', 'nc.Abstract_op', ops)
ops.nb_existing_ops = ops.nb_existing_ops + 1

function Max_op:__init(memory_size)
	parent.__init(self, memory_size)
end

function Max_op:updateOutput(input)
	local dist1 = input[1][1]	
	local dist2 = input[1][2]
	self.output = distUtils.maxDist(dist1, dist2, self.output)
   return self.output
end

function Max_op:updateGradInput(input, gradOutput)
    local dist1 = input[1][1]
    local dist2 = input[1][2]
    assert(dist1:size(1)==gradOutput:size(1), "GradOutput incorrect size in max op")

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

    local cum1 = dist1:cumsum()
    local cum2 = dist2:cumsum()

    for i=1,dist1:size(1) do
        -- sum_c (gradOutput[c] * sum_b dist2[b] * 1_{i+1==c}
        local acc_1 = gradOutput[i]*cum2[i]
        local acc_2 = gradOutput[i]*cum1[i]
        for c=i+1,dist1:size(1) do
            acc_1 = acc_1 + gradOutput[c]*dist2[c]
            acc_2 = acc_2 + gradOutput[c]*dist1[c]
        end

        self.gradInput[1][1][i] = acc_1
        self.gradInput[1][2][i] = acc_2
    end

    return self.gradInput
end
