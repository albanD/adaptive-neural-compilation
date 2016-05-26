local layers = require 'nc.layers'
local distUtils = require 'nc.distUtils'
local algCrit, parent = torch.class('nc.AlgCrit', 'nn.Criterion', layers)

local grad = require 'autograd'

-- All the loss functions
-- This gives the order of the outputs
local loss_ordering = {
  "time",
  "response",
}

-- The parameters for the loss functions:

-- alpha is the weight to add to the loss defined over the last output
-- of the network, be it that we stopped using a STOP instruction or
-- because we were out of iterations.

-- beta is the weight for the loss defined over every iteration,
-- asking them to be exact, weighted by their probability that we are
-- done / have already stopped.

-- gamma is the weight for a penalty on all iterations for not
-- stopping.

-- delta is the weight for not having stopped before the end.


-- Implementation of the functions
local loss_functions = {}
loss_functions["response"] = function(input, target, params)
  local T = #input

  local target_end_memory = target[1]
  local target_mask = target[2]

  -- In order to not compute the loss over all the memory,
  -- we set the memory to the correct value for the part that we ignore
  local modded_input = (torch.cmul(target_mask, input[T][1])) - (torch.cmul(target_mask-1, target_end_memory))

  -- Get the correct answer
  local diff = modded_input - target_end_memory
  local err_response = params.alpha * torch.sum(torch.pow(diff, 2))

  return err_response
end

loss_functions["time"] = function(input, target, params)
  local T = #input

  local target_end_memory = target[1]
  local target_mask = target[2]

  -- Reduce time
  local err_time = 0
  for t=2,T do
      local modded_input = (torch.cmul(target_mask, input[t][1])) - (torch.cmul(target_mask-1, target_end_memory))
      local err = torch.sum(torch.pow(modded_input - target_end_memory, 2))
      err_time = err_time + params.beta * (input[t][3][1] - input[t-1][3][1]) * err
  end
  for t=1,T do
    err_time = err_time + params.gamma * (1 - input[t][3][1])
  end

  if grad.util.lt(input[T][3][1], 0.9) then
    err_time = err_time + params.delta * (1 - input[T][3][1])
  end

  return err_time
end


-- Should not modify below this point !
local grad_functions = {}
for i, name in ipairs(loss_ordering) do
  grad_functions[name] = grad(loss_functions[name], {optimize=true})
end

function algCrit:__init(params)
    parent.__init(self)
    self.params = params
    self.gradInput = {}
end

function algCrit:updateOutput(input, target)
   -- target is a table, contaning
   -- [1] the ground truth target
   -- [2] A mask, as a tensor containing only 0 and 1
   --                  -> 0 means that the loss on it should be ignored
   --                  -> 1 means that the loss should be counted
   
  local losses = {}
  for i, name in ipairs(loss_ordering) do
    table.insert(losses, loss_functions[name](input, target, self.params))
  end

  return unpack(losses)
end

local function zeroTableCopy(t1, t2)
   for k, v in pairs(t2) do
      if (torch.type(v) == "table") then
         t1[k] = zeroTableCopy(t1[k] or {}, t2[k])
      else
         if not t1[k] then
            t1[k] = v:clone():zero()
         else
            t1[k]:zero()
         end
      end
   end
   for k, v in pairs(t1) do
      if not t2[k] then
         t1[k] = nil
      end
   end
   return t1
end

local function updateTable(t1, t2)
  for k, v in pairs(t2) do
    if (torch.type(v) == "table") then
      t1[k] = updateTable(t1[k] or {}, t2[k])
    else
      t1[k]:add(t2[k])
    end
  end
  return t1
end

function algCrit:updateGradInput(input, target)
    -- Compute the agregated gradInput
    zeroTableCopy(self.gradInput, input)
    for i, name in ipairs(loss_ordering) do
      local grad = grad_functions[name](input, target, self.params)
      updateTable(self.gradInput, grad)
    end
    return self.gradInput
end
