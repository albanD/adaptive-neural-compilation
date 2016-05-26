local ops = require 'nc.ops'
local distUtils = require 'nc.distUtils'
local Dec_op, parent = torch.class('nc.Dec_op', 'nc.Abstract_op', ops)
ops.nb_existing_ops = ops.nb_existing_ops + 1

function Dec_op:__init(memory_size)
	parent.__init(self, memory_size)
end

function Dec_op:updateOutput(input)
	local dist1 = input[1][1]	
	local dist2 = distUtils.toDist(1, self.memory_size)
	self.output = distUtils.subDist(dist1, dist2, self.output)
   return self.output
end

function Dec_op:updateGradInput(input, gradOutput)
    local dist1 = input[1][1]
    local dist2 = input[1][2]
    assert(dist1:size(1)==gradOutput:size(1), "GradOutput incorrect size in dec op")

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
        -- sum_c (gradOutput[c] * sum_b dist2[b] * 1_{i-1==c}
        self.gradInput[1][1][i] = gradOutput[(i-2)%self.memory_size+1]
    end

    return self.gradInput
end
