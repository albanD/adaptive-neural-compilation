require 'rnn'
local layers = require 'nc.layers'
local ops = require 'nc.ops'
local decompiler = require 'nc.decompiler'
local distUtils = require 'nc.distUtils'
local dRAM, parent = torch.class('nc.DRAM', 'nn.Recurrent', layers)

local create_DistributionCreator, initLinear

function dRAM:__init(nb_registers, memory_size)
  assert(nb_registers, "Number of registers must be specified")
  assert(memory_size, "Size of the memory must be specified")
  self.nb_registers = nb_registers
  self.memory_size = memory_size

  self.firstArgLinear = create_DistributionCreator(memory_size, nb_registers)
  self.secondArgLinear = create_DistributionCreator(memory_size, nb_registers)
  self.OutputLinear = create_DistributionCreator(memory_size, nb_registers)
  self.instructionLinear = create_DistributionCreator(memory_size, ops.nb_existing_ops)

  -- Create the controller
  local controller = nn.ConcatTable()
  controller:add(nn.Sequential()
                    :add(nn.SelectTable(2)) -- Take the registers
                    :add(nn.Narrow(1,1,1))       -- Take only the first register
                    :add(nn.ConcatTable()
                            :add(self.firstArgLinear) -- First argument
                            :add(self.secondArgLinear) -- Second argument
                            :add(self.OutputLinear) -- Output register
                            :add(self.instructionLinear) -- Instruction to execute
                        )
                )
  controller:add(nn.Identity())
  -- Input of the controller is:
  -- { [M x M] [R x M] t_stop }
  -- Output of the controller is:
  -- {  {[R] [R] [R] [I]}  , {[M x M] [R x M] t_stop} }


  -- Create the execution machine
  self.machine = layers.RamMachine(nb_registers+1, memory_size)
  -- Input of the machine is:
  -- {  {[R] [R] [R] [I]}  , {[M x M] [R x M] t_stop} }
  -- Output of the machine is:
  -- { [M x M] [R x M] t_stop }



  -- Create a single step differentiable RAM
  self.singleStepRAM = nn.Sequential()
  self.singleStepRAM:add(controller)
  self.singleStepRAM:add(self.machine)

  -- The module used only at start
  self.startModule = layers.InitialModule(nb_registers, memory_size)

  -- Complete RAM
  parent.__init(self,
    self.startModule,
    nn.Identity(),
    self.singleStepRAM,
    nn.Identity(),
    9999,
    nn.SelectTable(2))
end

function dRAM:executeProgram(initialMemory, max_it)
  local outputs = self:forwardProgram(initialMemory, max_it)
  return outputs[#outputs], #outputs
end

function dRAM:traceProgram(initialMemory, max_it)
  print(decompiler.dump_registers(self.startModule.weight, self.nb_registers, true))
  print(" ")
  print(decompiler.dump_initial_state(self.startModule.weight[1], true))
  if max_it == nil or max_it == -1 then
    max_it = 9999
  end
  self:forget()

  local it = 1
  self:forward(initialMemory)
  local output = {nil,nil,{0}}
  while output[3][1] <= 0.9 do
    it = it + 1
    output = self:forward()
    local data = self.sharedClones[it]:get(1):get(2):get(1).output
    io.write(decompiler.dump_line(data[1][1], data[1][2], data[1][3], data[1][4], it))
    print(distUtils.toNumberTensor(data[2][2]):view(1,-1))
    if it > max_it then
      break
    end
  end

  return output, it
end

function dRAM:forwardProgram(initialMemory, max_it)
  if max_it == nil or max_it == -1 then
    max_it = 9999
  end
  self:forget()

  local it = 1
  local outputs = {}
  outputs[1] = self:forward(initialMemory)
  while outputs[it][3][1] <= 0.9 do
    it = it + 1
    outputs[it] = self:forward()
    if it > max_it then
      break
    end
  end

  return outputs
end

function dRAM:backwardProgram(initialMemory, gradOutputs)
  for it=#gradOutputs,2,-1 do
    self:backward(nil, gradOutputs[it])
  end
  local gradInput = self:backward(initialMemory, gradOutputs[1])

  return gradInput
end

function dRAM:flashHardProgram(config, sharpening_factor)
  local max_sizes = {
    config.nb_registers,
    config.nb_registers,
    config.nb_registers,
    config.nb_existing_ops
  }
  local dist_program = distUtils.toDistTable(config.program, max_sizes)

  self:flashSoftProgram(dist_program, config.registers_init, sharpening_factor)

end

function dRAM:flashSoftProgram(transfer, register, sharpening_factor)
  assert(transfer[1]:dim()==2, "arg1 transfer tensor should have 2 dimensions")
  assert(transfer[1]:size(1)<=self.memory_size, "too many states in arg1 transfer")
  assert(transfer[1]:size(2)==self.nb_registers, "distributions in arg1 state invalid size")
  assert(transfer[2]:dim()==2, "arg2 transfer tensor should have 2 dimensions")
  assert(transfer[2]:size(2)==self.nb_registers, "distributions in arg2 state invalid size")
  assert(transfer[3]:dim()==2, "out transfer tensor should have 2 dimensions")
  assert(transfer[3]:size(2)==self.nb_registers, "distributions in out state invalid size")
  assert(transfer[4]:dim()==2, "instr transfer tensor should have 2 dimensions")
  assert(transfer[4]:size(2)==ops.nb_existing_ops, "distributions in instr state invalid size")
  assert(transfer[1]:size(1)==transfer[2]:size(1) and
         transfer[1]:size(1)==transfer[3]:size(1) and
         transfer[1]:size(1)==transfer[4]:size(1), "Non matching number of states")
  assert(register:dim()==2, "register should have 2 dimensions")
  assert(register:size(1)==self.nb_registers, "invalid number of registers")
  assert(register:size(2)==self.memory_size, "distributions in register invalid size")

  initLinear(self.firstArgLinear, transfer[1]*sharpening_factor)
  initLinear(self.secondArgLinear, transfer[2]*sharpening_factor)
  initLinear(self.OutputLinear, transfer[3]*sharpening_factor)
  initLinear(self.instructionLinear, transfer[4]*sharpening_factor)

  self.startModule:registersInit(register*sharpening_factor)
end

initLinear = function(layer, weights)
  layer:get(1).weight:narrow(2,1,weights:size(1)):copy(weights:t())
end

create_DistributionCreator = function(input_size, distribution_size)
  local dist_creator = nn.Sequential()
  dist_creator:add(nn.Linear(input_size, distribution_size, false))
  dist_creator:add(nn.SoftMax())
  dist_creator:add(nn.View(-1))
  return dist_creator
end

-- Put back the original getParameters from nn and not the dpnn one
function dRAM:getParameters()
   -- get parameters
   local parameters,gradParameters = self:parameters()
   local p, g = dRAM.flatten(parameters), dRAM.flatten(gradParameters)
   assert(p:nElement() == g:nElement(),
      'check that you are sharing parameters and gradParameters')
   for i=1,#parameters do
      assert(parameters[i]:storageOffset() == gradParameters[i]:storageOffset(),
         'misaligned parameter at ' .. tostring(i))
   end
   return p, g
end
