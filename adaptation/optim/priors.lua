local optim = require 'nc.optim'

local Abstract_prior = torch.class('nc.Abstract_prior', optim.priors)

function Abstract_prior:__init()
end
