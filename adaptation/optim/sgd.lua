local optim = require 'nc.optim'

local sgd = {}


function sgd.update_parameters(parameters, gradParameters, state)
   for param_set=1, #parameters do
      parameters[param_set]:add(-state.learning_rate, gradParameters[param_set])
   end
end

optim.optimisers["sgd"] = sgd
