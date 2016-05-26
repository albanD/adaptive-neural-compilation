require 'nn'
local ops = require 'nc.ops'
local layers = require 'nc.layers'
local distUtils = require 'nc.distUtils'

local RamMachine, parent = torch.class('nc.RamMachine', 'nn.Module', layers)


function RamMachine:__init(nb_registers, memory_size)
   parent.__init(self)
   -- Initialise parameters
   self.memory_size = memory_size
   self.nb_registers = nb_registers
   -- Initialise instruction set
   self.instruction_set = {}
   local current_pos = 1
   -- Load the instruction set
   self.stop_op_index = current_pos
   table.insert(self.instruction_set, ops.Stop_op(memory_size));     current_pos = current_pos + 1
   table.insert(self.instruction_set, ops.Zero_op(memory_size));     current_pos = current_pos + 1
   table.insert(self.instruction_set, ops.Inc_op(memory_size));      current_pos = current_pos + 1
   table.insert(self.instruction_set, ops.Add_op(memory_size));      current_pos = current_pos + 1
   table.insert(self.instruction_set, ops.Sub_op(memory_size));      current_pos = current_pos + 1
   table.insert(self.instruction_set, ops.Dec_op(memory_size));      current_pos = current_pos + 1
   table.insert(self.instruction_set, ops.Min_op(memory_size));      current_pos = current_pos + 1
   table.insert(self.instruction_set, ops.Max_op(memory_size));      current_pos = current_pos + 1
   table.insert(self.instruction_set, ops.Read_op(memory_size));     current_pos = current_pos + 1
   self.write_op_index = current_pos
   table.insert(self.instruction_set, ops.Write_op(memory_size));    current_pos = current_pos + 1
   self.jump_op_index = current_pos
   table.insert(self.instruction_set, ops.Jez_op(memory_size));      current_pos = current_pos + 1

   self.arg1 = torch.zeros(self.memory_size)
   self.arg2 = torch.zeros(self.memory_size)
   self.out_vector = torch.zeros(self.memory_size)
   self.out_instr = torch.Tensor(ops.nb_existing_ops, self.memory_size)
end


function RamMachine:updateOutput(input)
   -- Input is
   -- {  {[R] [R] [R] [I]}  , {[M x M] [R x M] [2]} }
   -- Arg1-distribution, Arg2-distribution, out-distribution, Instruction distribution
   -- Memory-distribution, Register distribution, StopTensor
   -- Output is
   -- { [M x M] [R x M] [2] }
   -- Memory-distribution, Register distribution, StopTensor
   local operation_parameters = input[1]
   local memory_tape = input[2][1]:clone()
   local registers = input[2][2]:clone()
   local stop_tensor = input[2][3]:clone()
   local instruction_register = registers[1]
   local standard_registers = registers:narrow(1, 2, registers:size(1)-1)

   torch.mv(self.arg1, standard_registers:t(), operation_parameters[1])
   torch.mv(self.arg2, standard_registers:t(), operation_parameters[2])

   -- Compute the output of the operations
   for index, op in ipairs(self.instruction_set) do
      self.out_instr[index]:copy(op:updateOutput({{self.arg1, self.arg2}, memory_tape}))
   end

   -- Modify the memory tape according to the write instruction
   -- At the moment, this is the only one that has any effect on the memory tape.
   local write_coefficient = operation_parameters[4][self.write_op_index]
   self.instruction_set[self.write_op_index]:update_memory(write_coefficient, self.arg1, self.arg2, memory_tape)

   -- Weight the output of each instruction according to the probability it is used.
   self.out_vector = torch.mv(self.out_instr:t(), operation_parameters[4])
   -- Reshape the vectors so that they can be multiplied correctly with each other.
   self.out_vector = self.out_vector:view(1,-1)
   local out_mat = torch.expand(self.out_vector, self.nb_registers-1, self.memory_size)
   local operation_parameters_3 = operation_parameters[3]:view(-1,1)
   local register_write_coefficient = torch.expandAs(operation_parameters_3, out_mat)
   -- Multiply them together so that we have the version of the registers
   local written_registers_unrolled = torch.cmul(out_mat, register_write_coefficient)

   -- For each of the output, we had a probability of writing to it. We also have the
   -- inverse probability of not writing to it (discretely, this corresponds to writing to another register
   -- and therefore leaving this one intact)
   local keep_write_coefficient = register_write_coefficient*(-1) + 1
   local kept_registers = torch.cmul(standard_registers, keep_write_coefficient)


   -- Update the standard registers
   standard_registers:copy(kept_registers + written_registers_unrolled)

   -- Update the instruction registers
   local jump_coefficient = operation_parameters[4][self.jump_op_index]*self.arg1[1]
   -- Standard increment
   local nojumped_instruction = distUtils.addDist(instruction_register, distUtils.toDist(1, self.memory_size))
   -- Increment if the conditional jump was selected. We pick up the second argument as new RI.
   local jumped_instruction = self.arg2
   -- Do the mixing of the instructions positions.
   instruction_register:copy(jumped_instruction*jump_coefficient + nojumped_instruction*(1-jump_coefficient))

   -- Update the stopping criterion
   local stop_coefficient = operation_parameters[4][self.stop_op_index]
   stop_tensor[1] = stop_tensor[1] + stop_coefficient * stop_tensor[2]
   stop_tensor[2] = (1-stop_coefficient) * stop_tensor[2]


   self.output = {memory_tape, registers, stop_tensor}

   return self.output
end

function RamMachine:updateGradInput(input, gradOutput)
   -- Input is
   -- {  {[R] [R] [R] [I]}  , {[M x M] [R x M] t_stop} } and { [M x M] [R x M] t_stop }
   -- Arg1-distribution, Arg2-distribution, out-distribution, Instruction distribution
   -- Output is
   -- {  {[R] [R] [R] [I]}  , {[M x M] [R x M] t_stop} }
   -- Memory-distribution, Register distribution

   if not (type(self.gradInput) == "table") then
      self.gradInput = {
         {
            torch.zeros(self.nb_registers-1),
            torch.zeros(self.nb_registers-1),
            torch.zeros(self.nb_registers-1),
            torch.zeros(ops.nb_existing_ops)
         },
         {
            torch.zeros(self.memory_size, self.memory_size),
            torch.zeros(self.nb_registers, self.memory_size),
            torch.zeros(2)
         }
      }
      self.arg1GradInput = torch.zeros(self.memory_size)
      self.arg2GradInput = torch.zeros(self.memory_size)
      self.outGradInput = torch.zeros(self.memory_size)
   else
      self.gradInput[1][1]:zero()
      self.gradInput[1][2]:zero()
      self.gradInput[1][3]:zero()
      self.gradInput[1][4]:zero()
      self.gradInput[2][1]:zero()
      self.gradInput[2][2]:zero()
      self.gradInput[2][3]:zero()
      self.arg1GradInput:zero()
      self.arg2GradInput:zero()
      self.outGradInput:zero()
   end


   local operation_parameters = input[1]
   local memory_tape = input[2][1]
   local registers = input[2][2]
   local stop_tensor = input[2][3]:clone()
   local instruction_register = registers[1]
   local standard_registers = registers:narrow(1, 2, registers:size(1)-1)

   --------
   -- Update the stopping criterion
   -- Not what we want to do but remove dependancy
   self.gradInput[1][4][self.stop_op_index] = self.gradInput[1][4][self.stop_op_index] + gradOutput[3][1] * stop_tensor[2]
   self.gradInput[1][4][self.stop_op_index] = self.gradInput[1][4][self.stop_op_index] - gradOutput[3][2] * stop_tensor[2]
   self.gradInput[2][3][1] = self.gradInput[2][3][1] + gradOutput[3][1] + operation_parameters[4][self.stop_op_index] * gradOutput[3][2]
   self.gradInput[2][3][2] = self.gradInput[2][3][2] - gradOutput[3][2] * operation_parameters[4][self.stop_op_index]

   --------
   -- Update the instruction registers
   local instruction_register_grad_output = gradOutput[2][1]
   local IR_plus_one = distUtils.addDist(instruction_register, distUtils.toDist(1, self.memory_size))
   -- arg1
   self.arg1GradInput[1] = self.arg1GradInput[1] + torch.cmul(
      instruction_register_grad_output,
      self.arg2 * operation_parameters[4][self.jump_op_index] - IR_plus_one * operation_parameters[4][self.jump_op_index]
   ):sum()
   -- arg2
   self.arg2GradInput:add(instruction_register_grad_output * self.arg1[1] * operation_parameters[4][self.jump_op_index])
   -- jmp instr
   self.gradInput[1][4][self.jump_op_index] = self.gradInput[1][4][self.jump_op_index] + torch.cmul(
      instruction_register_grad_output,
      self.arg2 * self.arg1[1] - IR_plus_one * self.arg1[1]
   ):sum()
   -- IR
   local IR_coeff = (1-self.arg1[1]*operation_parameters[4][self.jump_op_index])
   local shifted_irgo = torch.zeros(self.memory_size)
   shifted_irgo[self.memory_size] = instruction_register_grad_output[1]
   shifted_irgo:narrow(1,1,self.memory_size-1):copy(instruction_register_grad_output:narrow(1,2,self.memory_size-1))
   self.gradInput[2][2]:narrow(1,1,1):add(shifted_irgo * IR_coeff)

   --------
   -- Update the standard registers
   local std_register_grad_output = gradOutput[2]:narrow(1,2,self.nb_registers-1)
   local operation_parameters_3 = operation_parameters[3]:view(-1,1)
   local register_write_coefficient = torch.expand(operation_parameters_3, self.nb_registers-1, self.memory_size)
   local out_mat = torch.expand(self.out_vector, self.nb_registers-1, self.memory_size)

   -- std reg
   self.gradInput[2][2]:narrow(1,2,self.nb_registers-1):add(torch.cmul(
      std_register_grad_output,
      register_write_coefficient*(-1) + 1
   ))
   -- param3
   self.gradInput[1][3]:add(torch.cmul(
      std_register_grad_output,
      out_mat-standard_registers
   ):sum(2))
   -- outMat
   self.outGradInput:add(torch.cmul(
      std_register_grad_output,
      register_write_coefficient
   ):sum(1))

   --------
   -- Grad of the memory
   local write_coefficient = operation_parameters[4][self.write_op_index]
   self.instruction_set[self.write_op_index]:grad_update_memory(
      write_coefficient,
      self.arg1,
      self.arg2,
      memory_tape,
      gradOutput[1],
      self.gradInput[1][4],
      self.write_op_index,
      self.arg1GradInput,
      self.arg2GradInput,
      self.gradInput[2][1])

   --------
   -- Grad of the operations
   local gi
   local go
   for index, op in ipairs(self.instruction_set) do
      -- grad on instr weight
      self.gradInput[1][4][index] = self.gradInput[1][4][index] + torch.cmul(
         self.outGradInput,
         self.out_instr[index]
      ):sum()
      go = self.outGradInput * operation_parameters[4][index]
      gi = op:updateGradInput({{self.arg1, self.arg2}, memory_tape}, go)
      -- Grad on arg1
      self.arg1GradInput:add(gi[1][1])
      -- Grad on arg2
      self.arg2GradInput:add(gi[1][2])
      -- Grad on mem
      self.gradInput[2][1]:add(gi[2])
   end

   --------
   -- Grad on arg1
   -- param1
   self.gradInput[1][1]:addmv(0, 1, standard_registers, self.arg1GradInput)
   -- reg
   self.gradInput[2][2]:narrow(1,2,self.nb_registers-1):addr(operation_parameters[1], self.arg1GradInput)

   --------
   -- Grad on arg2
   -- param2
   self.gradInput[1][2]:addmv(0, 1, standard_registers, self.arg2GradInput)
   -- reg
   self.gradInput[2][2]:narrow(1,2,self.nb_registers-1):addr(operation_parameters[2], self.arg2GradInput)

   return self.gradInput
end

function RamMachine:__tostring__()
   return "nc.RamMachine :)"
end
