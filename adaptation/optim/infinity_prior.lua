local optim = require 'nc.optim'
local distUtils = require 'nc.distUtils'

local Infinity_prior, parent = torch.class('nc.Infinity_prior', 'nc.Abstract_prior', optim.priors)

function Infinity_prior:__init(config, initial_weight, annealing_param)
   local max_sizes = {
      config.nb_registers,
      config.nb_registers,
      config.nb_registers,
      config.nb_existing_ops
   }
   local dist_program = distUtils.toDistTable(config.program, max_sizes)
   self.to_add = {}
   -- Prior over the initial state
   local ini_state_to_add = torch.ones(config.memory_size) * -1
   ini_state_to_add[1] = 1
   self.to_add[1] = torch.cat(ini_state_to_add:view(1,-1), torch.mul(config.registers_init, 2):add(-1):ceil(), 1)
   for param_set = 1 , #max_sizes do
      self.to_add[param_set+1] = torch.zeros(max_sizes[param_set], config.memory_size)
      local weights = torch.mul(dist_program[param_set], 2):add(-1):ceil()
      self.to_add[param_set+1]:narrow(2,1,weights:size(1)):copy(weights:t())
   end

   self.initial_weight = initial_weight
   self.annealing_param = annealing_param
   self.t = 0
   parent.__init(self)
end


function Infinity_prior:update_gradients(parameters_table, grad_parameters_table)
   self.t = self.t + 1
   -- Need to have the minus to penalise being different and not encourage it
   local weight = - self.initial_weight / (1 + self.annealing_param * self.t)

   for param_set = 1, #grad_parameters_table do
      grad_parameters_table[param_set]:add(weight, self.to_add[param_set])
   end
end
