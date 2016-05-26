local optim = require 'nc.optim'

local adam = {}

local param_shaped_table = function(parameters)
   local tab = {}

   for param_set = 1, #parameters do
      tab[param_set] = parameters[param_set]:clone():zero()
   end

   return tab
end


function adam.update_parameters(parameters, gradParameters, state)
   local param_set_nb = #parameters

   local state = state or {}

   -- Default parameters
   local learning_rate = state.learning_rate or 0.001
   local beta1 = state.beta1 or 0.9
   local beta2 = state.beta2 or 0.999
   local epsilon = state.epsilon or 1e-8

   -- State initialisation, in case it wasn't done
   state.t = state.t or 0
   state.m = state.m or param_shaped_table(parameters)
   state.v = state.v or param_shaped_table(parameters)
   state.denom = state.denom or param_shaped_table(parameters) -- not necessary but this way we don't reallocate at each iteration

   -- Updates
   state.t = state.t + 1

   local step_size = learning_rate * math.sqrt(1-beta2^state.t)/ (1- beta1^state.t)
   for param_set = 1, #parameters do
      -- Update first order moment
      state.m[param_set]:mul(beta1):add(1-beta1,gradParameters[param_set])

      -- Update second order moment
      state.v[param_set]:mul(beta2):addcmul(1-beta2, gradParameters[param_set], gradParameters[param_set])

      torch.sqrt(state.denom[param_set], state.v[param_set])
      state.denom[param_set]:add(epsilon)
      parameters[param_set]:addcdiv(-step_size, state.m[param_set], state.denom[param_set])
   end

end

optim.optimisers["adam"] = adam
