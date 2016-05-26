local optim = require 'nc.optim'
local distUtils = require 'nc.distUtils'
local nn = require 'nn'

local Softmaxed_prior, parent = torch.class('nc.Softmaxed_prior', 'nc.Abstract_prior', optim.priors)

function Softmaxed_prior:__init(config, initial_weight, annealing_param)
   local max_sizes = {
      config.nb_registers,
      config.nb_registers,
      config.nb_registers,
      config.nb_existing_ops
   }
   local dist_program = distUtils.toDistTable(config.program, max_sizes)

   self.softmaxer = nn.SoftMax()

   self.reference = {}
   self.mask = {}


   -- Prior over the initial state
   local ini_state_to_add = torch.zeros(config.memory_size)
   ini_state_to_add[1] = 1
   self.mask[1] = torch.ones(config.nb_registers + 1, config.memory_size)
   for reg = 1, config.registers_init:size(1) do
      if config.registers_init[reg]:max() ~= 1 then
         self.mask[1][reg+1]:fill(0)
      end
   end
   self.reference[1] = torch.cat(ini_state_to_add:view(1,-1), config.registers_init, 1)


   -- Prior over the instructions
   for param_set = 1 , #max_sizes do
      self.reference[param_set+1] = torch.zeros(max_sizes[param_set], config.memory_size)
      self.reference[param_set+1]:narrow(2,1,dist_program[param_set]:size(1)):copy(dist_program[param_set]:t())
      self.mask[param_set+1] = torch.cat(torch.ones(max_sizes[param_set], config.nb_states),
                                         torch.zeros(max_sizes[param_set], config.memory_size - config.nb_states), 2)
      for reg = 1, config.nb_states do
         if self.reference[param_set+1]:select(2, reg):max() ~= 1 then
            self.mask[param_set+1]:select(2, reg):fill(0)
         end
      end
   end
   -- We probably need to have a mask to not put the prior over the things that we don't care about.

   self.initial_weight = initial_weight
   self.annealing_param = annealing_param
   self.t = 0
   parent.__init(self)
end


function Softmaxed_prior:update_gradients(parameters_table, grad_parameters_table)
   self.t = self.t + 1
   -- Need to have the minus to penalise being different and not encourage it
   local weight = - self.initial_weight / (1 + self.annealing_param * self.t)
   for param_set = 1, #grad_parameters_table do
      local softmaxed_param
      if param_set == 1 then -- the alignment is not proper for the softmax on the others
         softmaxed_param = self.softmaxer:forward(parameters_table[param_set])
      else
         softmaxed_param = self.softmaxer:forward(parameters_table[param_set]:t()):t()
      end
      local loss_grad = (self.reference[param_set] - softmaxed_param):mul(2)
      local param_grad
      if param_set == 1 then
         param_grad = self.softmaxer:backward(parameters_table[param_set], loss_grad)
      else
         param_grad = self.softmaxer:backward(parameters_table[param_set]:t(), loss_grad:t()):t()
      end
      -- put the mask down here
      grad_parameters_table[param_set]:add(weight, param_grad:cmul(self.mask[param_set]))
   end
end
