local ops = require 'nc.ops'
local distUtils = require 'nc.distUtils'
local Write_op, parent = torch.class('nc.Write_op', 'nc.Abstract_op', ops)
ops.nb_existing_ops = ops.nb_existing_ops + 1

function Write_op:__init(memory_size)
  parent.__init(self, memory_size)
  self.keep_memory = torch.zeros(self.memory_size, self.memory_size)
  self.new_memory = torch.zeros(self.memory_size, self.memory_size)
  self.ones = torch.ones(self.memory_size)
end

function Write_op:updateOutput(input)
  -- This module has no execution to do and its input is going to the junk register
  self.output = distUtils.toDist(0, self.memory_size)
  return self.output
end

function Write_op:update_memory(weight, address, value, memory_tape)
   -- Write down 'value' at 'address' in 'memory_tape' (modified by reference)
   -- This is a soft write, parametrized by 'weight'
   torch.cmul(self.keep_memory, torch.ger((self.ones - address), self.ones), memory_tape)
   torch.ger(self.new_memory, address, value)
   memory_tape:copy(memory_tape * (1 - weight) + (self.keep_memory + self.new_memory) * weight)
end

function Write_op:updateGradInput(input, gradOutput)
    -- The gradient on the input does not depend on the gradient on the output
    -- so just return 0
    local dist1 = input[1][1]
    assert(dist1:size(1)==gradOutput:size(1), "GradOutput incorrect size in write op")

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

function Write_op:grad_update_memory(weight, address, value, memory_tape,
  gradOutput,
  grad_weight, ind_grad_weight, grad_address, grad_value, grad_memory_tape)

  -- weight
  grad_weight[ind_grad_weight] = grad_weight[ind_grad_weight] + torch.cmul(
    gradOutput,
    self.keep_memory + self.new_memory - memory_tape
  ):sum()
  -- address
  grad_address:add(weight, torch.cmul(
    gradOutput,
    torch.ger(self.ones, value) - memory_tape
  ):sum(2))
  -- value
  grad_value:add(weight, torch.cmul(
    gradOutput,
    torch.ger(address, self.ones)
  ):sum(1))
  -- mem
  grad_memory_tape:add(torch.cmul(
    gradOutput,
    torch.ger((self.ones - address), self.ones)*weight + 1 - weight
  ))
end