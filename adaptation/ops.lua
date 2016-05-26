local ops = {}
ops.nb_existing_ops = 0
local Abstract_op, parent = torch.class('nc.Abstract_op', 'nn.Module', ops)

function Abstract_op:__init(memory_size)
   self.memory_size = memory_size
   parent.__init(self)
end

function Abstract_op:updateOutput(input)
   -- input is of the form
   -- { { arg1, arg2}  memory_tape}
   -- { { [M] , [M] }   [M x M]   }

   -- This abstract operation should never be called.
   error('Call to unimplemented operation')
   return self.output
end


return ops
