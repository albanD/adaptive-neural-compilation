local layers = require 'nc.layers'
local distUtils = require 'nc.distUtils'
local initialModule, parent = torch.class('nc.InitialModule', 'nn.Module', layers)

function initialModule:__init(nb_registers, memory_size)
	parent.__init(self)
  self.memory_size = memory_size
  self.weight = torch.rand(nb_registers+1, memory_size)
	self.gradWeight = torch.Tensor(nb_registers+1, memory_size)
  self.softMax = nn.SoftMax()
end

function initialModule:updateOutput(input)
  local stop_tensor = torch.Tensor({0, 1})
  local dist_registers = self.softMax:forward(self.weight)
  self.output = {input, dist_registers, stop_tensor}
	return self.output
end

function initialModule:registersInit(R)
  assert(R:size(1)==self.weight:size(1)-1, "Dimension 1 should contain one entry per register")
  assert(R:size(2)==self.weight:size(2), "Dimension 2 should contain one entry per number allowed")
  self.weight[1]:copy(distUtils.toDist(0, self.memory_size)*R:max())
  self.weight:narrow(1,2,R:size(1)):copy(R)
end

function initialModule:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):copy(gradOutput[1])
  return self.gradInput
end

function initialModule:accGradParameters(input, gradOutput, scale)
  local reg_go = self.softMax:backward(self.weight, gradOutput[2])
  self.gradWeight:add(scale, reg_go)
end
