local torch = require 'torch'
local torchtest = torch.TestSuite()
local mytester
local seed = os.time()
local precision = 1e-6
local perturbation = 1e-6

local distUtils = require 'nc.distUtils'
local ops = require 'nc.ops'
local layers = require 'nc.layers'

local function testDistOp(f, op, name)
	local M = 100
	local dist1 = torch.rand(M)
	dist1 = dist1 / dist1:sum()
	local dist2 = torch.rand(M)
	dist2 = dist2 / dist2:sum()

	local res = f(dist1, dist2)

	local gt = res.new():resizeAs(res):zero()
	for c=0,gt:size(1)-1 do
		for a=0,dist1:size(1)-1 do
			for b=0,dist2:size(1)-1 do
				if op(a,b,gt:size(1))==c then
					gt[c+1] = gt[c+1] + dist1[a+1] * dist2[b+1]
				end
			end
		end
	end

	mytester:assertlt((res-gt):abs():max(), precision, name..' dist invalid')
end

function torchtest.testAddDist()
	testDistOp(
		distUtils.addDist,
		function(a,b,c)
			return (a+b)%c
		end,
		"Addition"
	)
end

function torchtest.testSubDist()
	testDistOp(
		distUtils.subDist,
		function(a,b,c)
			return (a-b)%c
		end,
		"Substraction"
	)
end

function torchtest.testMaxDist()
	testDistOp(
		distUtils.maxDist,
		function(a,b,c)
			return math.max(a,b)
		end,
		"Max"
	)
end

function torchtest.testMinDist()
	testDistOp(
		distUtils.minDist,
		function(a,b,c)
			return math.min(a,b)
		end,
		"Min"
	)
end

function torchtest.testAddDistQuantized()
	local M = 100
	local dist1 = torch.zeros(M)
	local ind1 = math.ceil(math.random(M))
	dist1[ind1] = 1
	local dist2 = torch.zeros(M)
	local ind2 = math.ceil(math.random(M))
	dist2[ind2] = 1

	local res = distUtils.addDist(dist1, dist2)

	mytester:asserteq(res:sum(), 1, 'Dist add invalid on quantized values')
	mytester:asserteq(res[(ind1-1+ind2-1)%M+1], 1, 'Dist add invalid')
end

function torchtest.testMaxDistQuantized()
	local M = 100
	local dist1 = torch.zeros(M)
	local ind1 = math.ceil(math.random(M))
	dist1[ind1] = 1
	local dist2 = torch.zeros(M)
	local ind2 = math.ceil(math.random(M))
	dist2[ind2] = 1

	local res = distUtils.maxDist(dist1, dist2)

	mytester:asserteq(res:sum(), 1, 'Dist max invalid on quantized values')
	mytester:asserteq(res[math.max(ind1,ind2)], 1, 'Dist max invalid')
end

function torchtest.testMemoryWrite()
	local M = 5
	local bin_memory_tape = torch.ones(M)
	local dist_memory_tape = distUtils.toDistTensor(bin_memory_tape, M)
	local ref_dist_memory_tape = dist_memory_tape:clone() -- reference to compare
	local value_to_write = distUtils.toDist(4, M)
	local address_to_write = distUtils.toDist(0, M)
	local op = ops.Write_op(M)

	-- This shouldn't affect anything
	op:updateOutput({{address_to_write, value_to_write}, dist_memory_tape})
	-- so this should still be equals to the ref
	mytester:eq(ref_dist_memory_tape, dist_memory_tape)

	-- This should update the memory because this is the side-effect function
	-- This test does it with a full write, on a binary address
	op:update_memory(1, address_to_write, value_to_write, dist_memory_tape)
	-- The reference needs to be changed
	ref_dist_memory_tape[1]:copy(value_to_write)
	mytester:eq(ref_dist_memory_tape, dist_memory_tape)

	-- Let's now try a soft write, on a binary address, at the second line
	local target_value = torch.zeros(M)
	target_value[2] = 0.5; target_value[5] = 0.5;
	ref_dist_memory_tape[2]:copy(target_value)
	op:update_memory(0.5, distUtils.toDist(1,M), value_to_write, dist_memory_tape)
	mytester:eq(ref_dist_memory_tape, dist_memory_tape)


	-- Let's now do a full write, on a soft address
	local soft_address = torch.zeros(M)
	soft_address[3] = 0.5; soft_address[4] = 0.5;
	-- At the address, 3 and 4, at the moment, we have the value 1
	-- We will write the value 4.
	ref_dist_memory_tape[3][2] = 0.5; ref_dist_memory_tape[3][5]=0.5;
	ref_dist_memory_tape[4][2] = 0.5; ref_dist_memory_tape[4][5]=0.5;
	op:update_memory(1, soft_address, value_to_write, dist_memory_tape)
	mytester:eq(ref_dist_memory_tape, dist_memory_tape)

	-- Soft write with soft address
	-- We're going to go back to a clean state: all addresses contain only ones
	dist_memory_tape = distUtils.toDistTensor(bin_memory_tape, M)
	ref_dist_memory_tape = distUtils.toDistTensor(bin_memory_tape, M)
	-- We are going to write at address 3 and 4 equally, the value 4, with half-proba to do it
	-- This means that to have the value 4, it needs -> the write to be executed (0.5)
	--                                               -> the write to be applied to it (0.5)
	-- This is a total probability of 0.25
	-- Start to manually craft the output that we are expecting
	ref_dist_memory_tape[3][2] = 0.75; ref_dist_memory_tape[3][5]=0.25;
	ref_dist_memory_tape[4][2] = 0.75; ref_dist_memory_tape[4][5]=0.25;
	op:update_memory(0.5, soft_address, value_to_write, dist_memory_tape)
	mytester:eq(ref_dist_memory_tape, dist_memory_tape)
end

function torchtest.testMemoryRead()
	local M = 5
	local bin_memory_tape = torch.ones(M)
	local dist_memory_tape = distUtils.toDistTensor(bin_memory_tape, M)
	local ref_dist_memory_tape = dist_memory_tape:clone()
	local junk_argument = distUtils.toDist(4, M)
	local address_to_read = distUtils.toDist(0, M)
	local op = ops.Read_op(M)

	local read_value = op:updateOutput({{address_to_read, junk_argument}, dist_memory_tape})
	-- This shouldn't have changed
	mytester:eq(dist_memory_tape, ref_dist_memory_tape)
	-- we should have read the proper value
	mytester:eq(read_value, distUtils.toDist(1, M))

	--Change the memory on one of the address and perform a soft read
	dist_memory_tape[1]:copy(junk_argument)
	ref_dist_memory_tape = dist_memory_tape:clone()
	-- Now the tape contains 1 on each address, except the first one where it contains 4
	local soft_address_to_read = torch.zeros(M)
	soft_address_to_read[1] = 0.5; soft_address_to_read[2] = 0.5;
	-- We will read from the first two address
	local expected_read_value = torch.zeros(M)
	expected_read_value[2] = 0.5; expected_read_value[5] = 0.5;
	-- Execute the read
	read_value = op:updateOutput({{soft_address_to_read, junk_argument}, dist_memory_tape})
	-- The memory shouldn't have changed
	mytester:eq(dist_memory_tape, ref_dist_memory_tape)
	-- we should have read the proper value
	--print(read_value)
	--print(expected_read_value)
	mytester:eq(read_value, expected_read_value)
end

local function finDiffOpTest(mod, name)
	local M = 10
	mod = mod(M)

	-- Get a random input
	local dist1 = torch.rand(M)
	dist1 = dist1 / dist1:sum()
	local dist2 = torch.rand(M)
	dist2 = dist2 / dist2:sum()
	local mem = torch.rand(M, M)
	for i=1,M do
		mem[i]:copy(mem[i] / mem[i]:sum())
	end
	local input = {{dist1, dist2}, mem}

	-- Get a null gradOutput
	local gradOutput = torch.zeros(M)

	local dist1_err_fd = torch.zeros(M, M)
	local dist1_err_backward = torch.zeros(M, M)
	local dist2_err_fd = torch.zeros(M, M)
	local dist2_err_backward = torch.zeros(M, M)
	local mem_err_fd = torch.zeros(M*M, M)
	local mem_err_backward = torch.zeros(M*M, M)

	-- Compute the *_backward versions
	mod:forward(input)
	for i=1,M do
		gradOutput[i] = 1
		local gradInput = mod:backward(input, gradOutput)
		dist1_err_backward:select(2,i):copy(gradInput[1][1])
		dist2_err_backward:select(2,i):copy(gradInput[1][2])
		mem_err_backward:select(2,i):copy(gradInput[2])
		gradOutput[i] = 0
	end

	-- Compute the *_fd verions (finite difference)
	local outa = torch.Tensor(M)
	local outb = torch.Tensor(M)
	-- For dist1
	for i=1,M do
		local original_in = input[1][1][i]
		input[1][1][i] = original_in - perturbation
		outa:copy(mod:forward(input))
		input[1][1][i] = original_in + perturbation
		outb:copy(mod:forward(input))
		input[1][1][i] = original_in

		outb:add(-1, outa):div(2*perturbation)
		dist1_err_fd[i]:copy(outb)
	end
	-- For dist2
	for i=1,M do
		local original_in = input[1][2][i]
		input[1][2][i] = original_in - perturbation
		outa:copy(mod:forward(input))
		input[1][2][i] = original_in + perturbation
		outb:copy(mod:forward(input))
		input[1][2][i] = original_in

		outb:add(-1, outa):div(2*perturbation)
		dist2_err_fd[i]:copy(outb)
	end
	-- For mem
	for i=1,M do
		for j=1,M do
			local original_in = input[2][i][j]
			input[2][i][j] = original_in - perturbation
			outa:copy(mod:forward(input))
			input[2][i][j] = original_in + perturbation
			outb:copy(mod:forward(input))
			input[2][i][j] = original_in

			outb:add(-1, outa):div(2*perturbation)
			mem_err_fd[j+M*(i-1)]:copy(outb)
		end
	end

	-- Compare the results
	mytester:eq(dist1_err_fd, dist1_err_backward, precision, "Invalid gradInput on dist1 for "..name)
	mytester:eq(dist2_err_fd, dist2_err_backward, precision, "Invalid gradInput on dist2 for "..name)
	mytester:eq(mem_err_fd, mem_err_backward, precision, "Invalid gradInput on mem for "..name)
end

local opsToTest = {
	{ops.Add_op, "AddOp"},
	{ops.Dec_op, "DecOp"},
	{ops.Inc_op, "IncOp"},
	{ops.Jez_op, "JezOp"},
	{ops.Max_op, "MaxOp"},
	{ops.Min_op, "MinOp"},
	{ops.Read_op, "ReadOp"},
	{ops.Stop_op, "StopOp"},
	{ops.Sub_op, "SubOp"},
	{ops.Write_op, "WriteOp"},
	{ops.Zero_op, "ZeroOp"},
}
for _, todo in pairs(opsToTest) do
	torchtest["testGrad"..todo[2]] = function()
		finDiffOpTest(todo[1], todo[2])
	end
end

torchtest.finDiffInitialModTest = function()
	local name = "InitialModule"
	local M = 10
	local R = 5
	local R_inside = R + 1
	local perturbation = 1e-6

	local startModule = layers.InitialModule(R, M)

	-- Get a random input
	local input
	do
		local mem = torch.rand(M, M)
		for i=1,M do
			mem[i]:copy(mem[i] / mem[i]:sum())
		end
		input = mem
	end

	-- Get a null gradOutput
	local gradOutput
	do
		local mem = torch.zeros(M, M)
		local reg = torch.zeros(R_inside, M)
		local stop_tensor = torch.zeros(2)
		gradOutput = {mem, reg, stop_tensor}
	end

	local mem_err_fd = {torch.zeros(M*M, M*M), torch.zeros(M*M, R_inside*M), torch.zeros(M*M, 2)}
	local mem_err_backward = {torch.zeros(M*M, M*M), torch.zeros(M*M, R_inside*M), torch.zeros(M*M, 2)}
	local weight_err_fd = {torch.zeros(R_inside*M, M*M), torch.zeros(R_inside*M, R_inside*M), torch.zeros(R_inside*M, 2)}
	local weight_err_backward = {torch.zeros(R_inside*M, M*M), torch.zeros(R_inside*M, R_inside*M), torch.zeros(R_inside*M, 2)}

	-- Compute the *_backward versions
	-- For out mem
	for i=1,M do
		for j=1,M do
			startModule:zeroGradParameters()
			startModule:forward(input)
			gradOutput[1][i][j] = 1
			local gradInput = startModule:backward(input, gradOutput)
			mem_err_backward[1]:select(2,j+(i-1)*M):copy(gradInput)
			weight_err_backward[1]:select(2,j+(i-1)*M):copy(startModule.gradWeight)
			gradOutput[1][i][j] = 0
		end
	end
	-- For out reg
	for i=1,R_inside do
		for j=1,M do
			startModule:zeroGradParameters()
			startModule:forward(input)
			gradOutput[2][i][j] = 1
			local gradInput = startModule:backward(input, gradOutput)
			mem_err_backward[2]:select(2,j+(i-1)*M):copy(gradInput)
			weight_err_backward[2]:select(2,j+(i-1)*M):copy(startModule.gradWeight)
			gradOutput[2][i][j] = 0
		end
	end
	-- For out stop
	for i=1,2 do
		startModule:zeroGradParameters()
		startModule:forward(input)
		gradOutput[3][i] = 1
		local gradInput = startModule:backward(input, gradOutput)
		mem_err_backward[3]:select(2,i):copy(gradInput)
		weight_err_backward[3]:select(2,i):copy(startModule.gradWeight)
		gradOutput[3][i] = 0
	end

	-- Compute the *_fd verions (finite difference)
	local outmema = torch.Tensor(M*M)
	local outmemb = torch.Tensor(M*M)
	local outrega = torch.Tensor(R_inside*M)
	local outregb = torch.Tensor(R_inside*M)
	local outstopa = torch.Tensor(2)
	local outstopb = torch.Tensor(2)
	-- For mem
	for i=1,M do
		for j=1,M do
			local original_in = input[i][j]
			input[i][j] = original_in - perturbation
			local output = startModule:forward(input)
			outmema:copy(output[1])
			outrega:copy(output[2])
			outstopa:copy(output[3])
			input[i][j] = original_in + perturbation
			output = startModule:forward(input)
			outmemb:copy(output[1])
			outregb:copy(output[2])
			outstopb:copy(output[3])
			input[i][j] = original_in

			outmemb:add(-1, outmema):div(2*perturbation)
			outregb:add(-1, outrega):div(2*perturbation)
			outstopb:add(-1, outstopa):div(2*perturbation)
			mem_err_fd[1][j+(i-1)*M]:copy(outmemb)
			mem_err_fd[2][j+(i-1)*M]:copy(outregb)
			mem_err_fd[3][j+(i-1)*M]:copy(outstopb)
		end
	end
	-- For weight
	for i=1,R_inside do
		for j=1,M do
			local original_in = startModule.weight[i][j]
			startModule.weight[i][j] = original_in - perturbation
			local output = startModule:forward(input)
			outmema:copy(output[1])
			outrega:copy(output[2])
			outstopa:copy(output[3])
			startModule.weight[i][j] = original_in + perturbation
			output = startModule:forward(input)
			outmemb:copy(output[1])
			outregb:copy(output[2])
			outstopb:copy(output[3])
			startModule.weight[i][j] = original_in

			outmemb:add(-1, outmema):div(2*perturbation)
			outregb:add(-1, outrega):div(2*perturbation)
			outstopb:add(-1, outstopa):div(2*perturbation)
			weight_err_fd[1][j+(i-1)*M]:copy(outmemb)
			weight_err_fd[2][j+(i-1)*M]:copy(outregb)
			weight_err_fd[3][j+(i-1)*M]:copy(outstopb)
		end
	end

	mytester:eq(mem_err_fd[1], mem_err_backward[1], precision, "Invalid mem gradInput on mem wrt mem for "..name)
	mytester:eq(mem_err_fd[2], mem_err_backward[2], precision, "Invalid mem gradInput on mem wrt reg for "..name)
	mytester:eq(mem_err_fd[3], mem_err_backward[3], precision, "Invalid mem gradInput on mem wrt stop for "..name)
	mytester:eq(weight_err_fd[1], weight_err_backward[1], precision, "Invalid gradWeight on mem wrt mem for "..name)
	mytester:eq(weight_err_fd[2], weight_err_backward[2], precision, "Invalid gradWeight on mem wrt reg for "..name)
	mytester:eq(weight_err_fd[3], weight_err_backward[3], precision, "Invalid gradWeight on mem wrt stop for "..name)

end

torchtest.finDiffMachineTest = function()
	-- Read finDiffOpTest before this one!
	-- This one is just more complex because it has mutiple outputs
	local name = "Machine"
	local M = 10
	local R = 6
	local R_inside = R + 1
	local nb_instructions = 11
	local mod = layers.RamMachine(R_inside, M)


	-- Get a random input
	local input
	do
		local arg1 = torch.rand(R)
		arg1 = arg1 / arg1:sum()
		local arg2 = torch.rand(R)
		arg2 = arg2 / arg2:sum()
		local out = torch.rand(R)
		out = out / out:sum()
		local instr = torch.rand(nb_instructions)
		instr = instr / instr:sum()
		local mem = torch.rand(M, M)
		for i=1,M do
			mem[i]:copy(mem[i] / mem[i]:sum())
		end
		local reg = torch.rand(R_inside, M)
		for i=1,R_inside do
			reg[i]:copy(reg[i] / reg[i]:sum())
		end
		local stop_tensor = torch.rand(2)
		input = {{arg1, arg2, out, instr}, {mem ,reg, stop_tensor}}
	end

	-- Get a null gradOutput
	local gradOutput
	do
		local mem = torch.zeros(M, M)
		local reg = torch.zeros(R_inside, M)
		local stop_tensor = torch.zeros(2)
		gradOutput = {mem, reg, stop_tensor}
	end

	local arg1_err_fd = {torch.zeros(R, M*M), torch.zeros(R, R_inside*M), torch.zeros(R, 2)}
	local arg1_err_backward = {torch.zeros(R, M*M), torch.zeros(R, R_inside*M), torch.zeros(R, 2)}
	local arg2_err_fd = {torch.zeros(R, M*M), torch.zeros(R, R_inside*M), torch.zeros(R, 2)}
	local arg2_err_backward = {torch.zeros(R, M*M), torch.zeros(R, R_inside*M), torch.zeros(R, 2)}
	local out_err_fd = {torch.zeros(R, M*M), torch.zeros(R, R_inside*M), torch.zeros(R, 2)}
	local out_err_backward = {torch.zeros(R, M*M), torch.zeros(R, R_inside*M), torch.zeros(R, 2)}
	local instr_err_fd = {torch.zeros(nb_instructions, M*M), torch.zeros(nb_instructions, R_inside*M), torch.zeros(nb_instructions, 2)}
	local instr_err_backward = {torch.zeros(nb_instructions, M*M), torch.zeros(nb_instructions, R_inside*M), torch.zeros(nb_instructions, 2)}
	local mem_err_fd = {torch.zeros(M*M, M*M), torch.zeros(M*M, R_inside*M), torch.zeros(M*M, 2)}
	local mem_err_backward = {torch.zeros(M*M, M*M), torch.zeros(M*M, R_inside*M), torch.zeros(M*M, 2)}
	local reg_err_fd = {torch.zeros(R_inside*M, M*M), torch.zeros(R_inside*M, R_inside*M), torch.zeros(R_inside*M, 2)}
	local reg_err_backward = {torch.zeros(R_inside*M, M*M), torch.zeros(R_inside*M, R_inside*M), torch.zeros(R_inside*M, 2)}
	local stop_err_fd = {torch.zeros(1, M*M), torch.zeros(1, R_inside*M), torch.zeros(1, 2)}
	local stop_err_backward = {torch.zeros(1, M*M), torch.zeros(1, R_inside*M), torch.zeros(1, 2)}


	-- Compute the *_backward versions
	mod:forward(input)
	-- For out mem
	for i=1,M do
		for j=1,M do
			gradOutput[1][i][j] = 1
			local gradInput = mod:backward(input, gradOutput)
			arg1_err_backward[1]:select(2,j+(i-1)*M):copy(gradInput[1][1])
			arg2_err_backward[1]:select(2,j+(i-1)*M):copy(gradInput[1][2])
			out_err_backward[1]:select(2,j+(i-1)*M):copy(gradInput[1][3])
			instr_err_backward[1]:select(2,j+(i-1)*M):copy(gradInput[1][4])
			mem_err_backward[1]:select(2,j+(i-1)*M):copy(gradInput[2][1])
			reg_err_backward[1]:select(2,j+(i-1)*M):copy(gradInput[2][2])
			gradOutput[1][i][j] = 0
		end
	end
	-- For out reg
	for i=1,R_inside do
		for j=1,M do
			gradOutput[2][i][j] = 1
			local gradInput = mod:backward(input, gradOutput)
			arg1_err_backward[2]:select(2,j+(i-1)*M):copy(gradInput[1][1])
			arg2_err_backward[2]:select(2,j+(i-1)*M):copy(gradInput[1][2])
			out_err_backward[2]:select(2,j+(i-1)*M):copy(gradInput[1][3])
			instr_err_backward[2]:select(2,j+(i-1)*M):copy(gradInput[1][4])
			mem_err_backward[2]:select(2,j+(i-1)*M):copy(gradInput[2][1])
			reg_err_backward[2]:select(2,j+(i-1)*M):copy(gradInput[2][2])
			gradOutput[2][i][j] = 0
		end
	end
	-- For out stop
	gradOutput[3][1] = 1
	local gradInput = mod:backward(input, gradOutput)
	arg1_err_backward[3]:select(2,1):copy(gradInput[1][1])
	arg2_err_backward[3]:select(2,1):copy(gradInput[1][2])
	out_err_backward[3]:select(2,1):copy(gradInput[1][3])
	instr_err_backward[3]:select(2,1):copy(gradInput[1][4])
	mem_err_backward[3]:select(2,1):copy(gradInput[2][1])
	reg_err_backward[3]:select(2,1):copy(gradInput[2][2])
	gradOutput[3][1] = 0
	gradOutput[3][2] = 1
	local gradInput = mod:backward(input, gradOutput)
	arg1_err_backward[3]:select(2,2):copy(gradInput[1][1])
	arg2_err_backward[3]:select(2,2):copy(gradInput[1][2])
	out_err_backward[3]:select(2,2):copy(gradInput[1][3])
	instr_err_backward[3]:select(2,2):copy(gradInput[1][4])
	mem_err_backward[3]:select(2,2):copy(gradInput[2][1])
	reg_err_backward[3]:select(2,2):copy(gradInput[2][2])
	gradOutput[3][2] = 0

	-- Compute the *_fd verions (finite difference)
	local outmema = torch.Tensor(M*M)
	local outmemb = torch.Tensor(M*M)
	local outrega = torch.Tensor(R_inside*M)
	local outregb = torch.Tensor(R_inside*M)
	local outstopa = torch.Tensor(2)
	local outstopb = torch.Tensor(2)
	-- For arg1
	for i=1,R do
		local original_in = input[1][1][i]
		input[1][1][i] = original_in - perturbation
		local output = mod:forward(input)
		outmema:copy(output[1])
		outrega:copy(output[2])
		outstopa:copy(output[3])
		input[1][1][i] = original_in + perturbation
		output = mod:forward(input)
		outmemb:copy(output[1])
		outregb:copy(output[2])
		outstopb:copy(output[3])
		input[1][1][i] = original_in

		outmemb:add(-1, outmema):div(2*perturbation)
		outregb:add(-1, outrega):div(2*perturbation)
		outstopb:add(-1, outstopa):div(2*perturbation)
		arg1_err_fd[1][i]:copy(outmemb)
		arg1_err_fd[2][i]:copy(outregb)
		arg1_err_fd[3][i]:copy(outstopb)
	end
	-- For arg2
	for i=1,R do
		local original_in = input[1][2][i]
		input[1][2][i] = original_in - perturbation
		local output = mod:forward(input)
		outmema:copy(output[1])
		outrega:copy(output[2])
		outstopa:copy(output[3])
		input[1][2][i] = original_in + perturbation
		output = mod:forward(input)
		outmemb:copy(output[1])
		outregb:copy(output[2])
		outstopb:copy(output[3])
		input[1][2][i] = original_in

		outmemb:add(-1, outmema):div(2*perturbation)
		outregb:add(-1, outrega):div(2*perturbation)
		outstopb:add(-1, outstopa):div(2*perturbation)
		arg2_err_fd[1][i]:copy(outmemb)
		arg2_err_fd[2][i]:copy(outregb)
		arg2_err_fd[3][i]:copy(outstopb)
	end
	-- For out
	for i=1,R do
		local original_in = input[1][3][i]
		input[1][3][i] = original_in - perturbation
		local output = mod:forward(input)
		outmema:copy(output[1])
		outrega:copy(output[2])
		outstopa:copy(output[3])
		input[1][3][i] = original_in + perturbation
		output = mod:forward(input)
		outmemb:copy(output[1])
		outregb:copy(output[2])
		outstopb:copy(output[3])
		input[1][3][i] = original_in

		outmemb:add(-1, outmema):div(2*perturbation)
		outregb:add(-1, outrega):div(2*perturbation)
		outstopb:add(-1, outstopa):div(2*perturbation)
		out_err_fd[1][i]:copy(outmemb)
		out_err_fd[2][i]:copy(outregb)
		out_err_fd[3][i]:copy(outstopb)
	end
	-- For instr
	for i=1,nb_instructions do
		local original_in = input[1][4][i]
		input[1][4][i] = original_in - perturbation
		local output = mod:forward(input)
		outmema:copy(output[1])
		outrega:copy(output[2])
		outstopa:copy(output[3])
		input[1][4][i] = original_in + perturbation
		output = mod:forward(input)
		outmemb:copy(output[1])
		outregb:copy(output[2])
		outstopb:copy(output[3])
		input[1][4][i] = original_in

		outmemb:add(-1, outmema):div(2*perturbation)
		outregb:add(-1, outrega):div(2*perturbation)
		outstopb:add(-1, outstopa):div(2*perturbation)
		instr_err_fd[1][i]:copy(outmemb)
		instr_err_fd[2][i]:copy(outregb)
		instr_err_fd[3][i]:copy(outstopb)
	end
	-- For mem
	for i=1,M do
		for j=1,M do
			local original_in = input[2][1][i][j]
			input[2][1][i][j] = original_in - perturbation
			local output = mod:forward(input)
			outmema:copy(output[1])
			outrega:copy(output[2])
			outstopa:copy(output[3])
			input[2][1][i][j] = original_in + perturbation
			output = mod:forward(input)
			outmemb:copy(output[1])
			outregb:copy(output[2])
			outstopb:copy(output[3])
			input[2][1][i][j] = original_in

			outmemb:add(-1, outmema):div(2*perturbation)
			outregb:add(-1, outrega):div(2*perturbation)
			outstopb:add(-1, outstopa):div(2*perturbation)
			mem_err_fd[1][j+(i-1)*M]:copy(outmemb)
			mem_err_fd[2][j+(i-1)*M]:copy(outregb)
			mem_err_fd[3][i]:copy(outstopb)
		end
	end
	-- For reg
	for i=1,R_inside do
		for j=1,M do
			local original_in = input[2][2][i][j]
			input[2][2][i][j] = original_in - perturbation
			local output = mod:forward(input)
			outmema:copy(output[1])
			outrega:copy(output[2])
			outstopa:copy(output[3])
			input[2][2][i][j] = original_in + perturbation
			output = mod:forward(input)
			outmemb:copy(output[1])
			outregb:copy(output[2])
			outstopb:copy(output[3])
			input[2][2][i][j] = original_in

			outmemb:add(-1, outmema):div(2*perturbation)
			outregb:add(-1, outrega):div(2*perturbation)
			outstopb:add(-1, outstopa):div(2*perturbation)
			reg_err_fd[1][j+(i-1)*M]:copy(outmemb)
			reg_err_fd[2][j+(i-1)*M]:copy(outregb)
			reg_err_fd[3][i]:copy(outstopb)
		end
	end

	mytester:eq(arg1_err_fd[1], arg1_err_backward[1], precision, "Invalid gradInput on arg1 wrt mem for "..name)
	mytester:eq(arg1_err_fd[2], arg1_err_backward[2], precision, "Invalid gradInput on arg1 wrt reg for "..name)
	mytester:eq(arg1_err_fd[3], arg1_err_backward[3], precision, "Invalid gradInput on arg1 wrt stop for "..name)
	mytester:eq(arg2_err_fd[1], arg2_err_backward[1], precision, "Invalid gradInput on arg2 wrt mem for "..name)
	mytester:eq(arg2_err_fd[2], arg2_err_backward[2], precision, "Invalid gradInput on arg2 wrt reg for "..name)
	mytester:eq(arg2_err_fd[3], arg2_err_backward[3], precision, "Invalid gradInput on arg2 wrt stop for "..name)
	mytester:eq(out_err_fd[1], out_err_backward[1], precision, "Invalid gradInput on out wrt mem for "..name)
	mytester:eq(out_err_fd[2], out_err_backward[2], precision, "Invalid gradInput on out wrt reg for "..name)
	mytester:eq(out_err_fd[3], out_err_backward[3], precision, "Invalid gradInput on out wrt stop for "..name)
	mytester:eq(instr_err_fd[1], instr_err_backward[1], precision, "Invalid gradInput on instr wrt mem for "..name)
	mytester:eq(instr_err_fd[2], instr_err_backward[2], precision, "Invalid gradInput on instr wrt reg for "..name)
	mytester:eq(instr_err_fd[3], instr_err_backward[3], precision, "Invalid gradInput on instr wrt stop for "..name)
	mytester:eq(mem_err_fd[1], mem_err_backward[1], precision, "Invalid gradInput on mem wrt mem for "..name)
	mytester:eq(mem_err_fd[2], mem_err_backward[2], precision, "Invalid gradInput on mem wrt reg for "..name)
	mytester:eq(mem_err_fd[3], mem_err_backward[3], precision, "Invalid gradInput on mem wrt stop for "..name)
	mytester:eq(reg_err_fd[1], reg_err_backward[1], precision, "Invalid gradInput on reg wrt mem for "..name)
	mytester:eq(reg_err_fd[2], reg_err_backward[2], precision, "Invalid gradInput on reg wrt reg for "..name)
	mytester:eq(reg_err_fd[3], reg_err_backward[3], precision, "Invalid gradInput on reg wrt stop for "..name)

end

torchtest.finDiffDRAMTest = function()
	local name = "dRAM"
	local M = 10
	local R = 3
	local R_inside = R + 1
	local nb_instructions = 11

	-- The weight here are randomly initialised
	local diff_ram = layers.DRAM(R, M)

	-- Get a random input
	local input = torch.rand(M, M)
	for i=1,M do
		input[i]:copy(input[i] / input[i]:sum())
	end
	
	-- Get two null gradOutput
	local gradOutput
	do
		local mem = torch.zeros(M, M)
		local reg = torch.zeros(R_inside, M)
		local stop_tensor = torch.zeros(2)
		gradOutput = {
			{mem:clone(), reg:clone(), stop_tensor:clone()},
			{mem:clone(), reg:clone(), stop_tensor:clone()},
		}
	end

	-- Rows consist of on which of the iteration you put on a gradient
	-- Cols consist on 1-> memory, 2-> registers, 3-> stop signals

	local mem_err_fd = {
		{torch.zeros(M*M, M*M), torch.zeros(M*M, R_inside*M), torch.zeros(M*M, 2)},
		{torch.zeros(M*M, M*M), torch.zeros(M*M, R_inside*M), torch.zeros(M*M, 2)}
	}
	local mem_err_backward = {
		{torch.zeros(M*M, M*M), torch.zeros(M*M, R_inside*M), torch.zeros(M*M, 2)},
		{torch.zeros(M*M, M*M), torch.zeros(M*M, R_inside*M), torch.zeros(M*M, 2)}
	}
	local reg_err_fd = {
		{torch.zeros(R_inside*M, M*M), torch.zeros(R_inside*M, R_inside*M), torch.zeros(R_inside*M, 2)},
		{torch.zeros(R_inside*M, M*M), torch.zeros(R_inside*M, R_inside*M), torch.zeros(R_inside*M, 2)}
	}
	local reg_err_backward = {
		{torch.zeros(R_inside*M, M*M), torch.zeros(R_inside*M, R_inside*M), torch.zeros(R_inside*M, 2)},
		{torch.zeros(R_inside*M, M*M), torch.zeros(R_inside*M, R_inside*M), torch.zeros(R_inside*M, 2)}
	}

	local first_arg_err_fd = {
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)},
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	}
	local first_arg_err_backward = {
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)},
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	}
	local second_arg_err_fd = {
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)},
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	}
	local second_arg_err_backward = {
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)},
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	}
	local out_arg_err_fd = {
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)},
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	}
	local out_arg_err_backward = {
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)},
		{torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	}
	local instruction_arg_err_fd = {
		{torch.zeros(nb_instructions*M, M*M), torch.zeros(nb_instructions*M, R_inside*M), torch.zeros(nb_instructions*M, 2)},
		{torch.zeros(nb_instructions*M, M*M), torch.zeros(nb_instructions*M, R_inside*M), torch.zeros(nb_instructions*M, 2)}
	}
	local instruction_arg_err_backward = {
		{torch.zeros(nb_instructions*M, M*M), torch.zeros(nb_instructions*M, R_inside*M), torch.zeros(nb_instructions*M, 2)},
		{torch.zeros(nb_instructions*M, M*M), torch.zeros(nb_instructions*M, R_inside*M), torch.zeros(nb_instructions*M, 2)}
	}

	local ram_params, ram_gradParams = diff_ram:parameters()

	-- Compute the *_backward versions
	-- For out mem
	for i=1,M do
		for j=1,M do
			for it=1,2 do
				diff_ram:forget()
				diff_ram:zeroGradParameters()
				diff_ram:forward(input)
				diff_ram:forward()
				gradOutput[it][1][i][j] = 1
				diff_ram:backward(nil, gradOutput[2])
				local gradInput = diff_ram:backward(input, gradOutput[1])
				mem_err_backward[it][1]:select(2,j+(i-1)*M):copy(gradInput)
				reg_err_backward[it][1]:select(2,j+(i-1)*M):copy(diff_ram.startModule.gradWeight)
				first_arg_err_backward[it][1]:select(2,j+(i-1)*M):copy(ram_gradParams[2])
				second_arg_err_backward[it][1]:select(2,j+(i-1)*M):copy(ram_gradParams[3])
				out_arg_err_backward[it][1]:select(2,j+(i-1)*M):copy(ram_gradParams[4])
				instruction_arg_err_backward[it][1]:select(2,j+(i-1)*M):copy(ram_gradParams[5])
				gradOutput[it][1][i][j] = 0
			end
		end
	end
	-- For out reg
	for i=1,R_inside do
		for j=1,M do
			for it=1,2 do
				diff_ram:forget()
				diff_ram:zeroGradParameters()
				diff_ram:forward(input)
				diff_ram:forward()
				gradOutput[it][2][i][j] = 1
				diff_ram:backward(nil, gradOutput[2])
				local gradInput = diff_ram:backward(input, gradOutput[1])
				mem_err_backward[it][2]:select(2,j+(i-1)*M):copy(gradInput)
				reg_err_backward[it][2]:select(2,j+(i-1)*M):copy(diff_ram.startModule.gradWeight)
				first_arg_err_backward[it][2]:select(2,j+(i-1)*M):copy(ram_gradParams[2])
				second_arg_err_backward[it][2]:select(2,j+(i-1)*M):copy(ram_gradParams[3])
				out_arg_err_backward[it][2]:select(2,j+(i-1)*M):copy(ram_gradParams[4])
				instruction_arg_err_backward[it][2]:select(2,j+(i-1)*M):copy(ram_gradParams[5])
				gradOutput[it][2][i][j] = 0
			end
		end
	end
	-- For out stop
	for i=1,2 do
		for it=1,2 do
			gradOutput[it][3][i] = 1
			diff_ram:forget()
			diff_ram:zeroGradParameters()
			diff_ram:forward(input)
			diff_ram:forward()
			diff_ram:backward(nil, gradOutput[2])
			local gradInput = diff_ram:backward(input, gradOutput[1])
			mem_err_backward[it][3]:select(2,i):copy(gradInput)
			reg_err_backward[it][3]:select(2,i):copy(diff_ram.startModule.gradWeight)
			first_arg_err_backward[it][3]:select(2,i):copy(ram_gradParams[2])
			second_arg_err_backward[it][3]:select(2,i):copy(ram_gradParams[3])
			out_arg_err_backward[it][3]:select(2,i):copy(ram_gradParams[4])
			instruction_arg_err_backward[it][3]:select(2,i):copy(ram_gradParams[5])
			gradOutput[it][3][i] = 0
		end
	end

	-- Compute the *_fd verions (finite difference)
	local outmem1a = torch.Tensor(M*M)
	local outmem1b = torch.Tensor(M*M)
	local outreg1a = torch.Tensor(R_inside*M)
	local outreg1b = torch.Tensor(R_inside*M)
	local outstop1a = torch.Tensor(2)
	local outstop1b = torch.Tensor(2)
	local outmem2a = torch.Tensor(M*M)
	local outmem2b = torch.Tensor(M*M)
	local outreg2a = torch.Tensor(R_inside*M)
	local outreg2b = torch.Tensor(R_inside*M)
	local outstop2a = torch.Tensor(2)
	local outstop2b = torch.Tensor(2)
	-- For mem
	for i=1,M do
		for j=1,M do
			local original_in = input[i][j]
			input[i][j] = original_in - perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1a:copy(out1[1])
			outreg1a:copy(out1[2])
			outstop1a:copy(out1[3])
			outmem2a:copy(out2[1])
			outreg2a:copy(out2[2])
			outstop2a:copy(out2[3])
			input[i][j] = original_in + perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1b:copy(out1[1])
			outreg1b:copy(out1[2])
			outstop1b:copy(out1[3])
			outmem2b:copy(out2[1])
			outreg2b:copy(out2[2])
			outstop2b:copy(out2[3])
			input[i][j] = original_in

			outmem1b:add(-1, outmem1a):div(2*perturbation)
			outreg1b:add(-1, outreg1a):div(2*perturbation)
			outstop1b:add(-1, outstop1a):div(2*perturbation)
			outmem2b:add(-1, outmem2a):div(2*perturbation)
			outreg2b:add(-1, outreg2a):div(2*perturbation)
			outstop2b:add(-1, outstop2a):div(2*perturbation)
			mem_err_fd[1][1][j+(i-1)*M]:copy(outmem1b)
			mem_err_fd[1][2][j+(i-1)*M]:copy(outreg1b)
			mem_err_fd[1][3][i]:copy(outstop1b)
			mem_err_fd[2][1][j+(i-1)*M]:copy(outmem2b)
			mem_err_fd[2][2][j+(i-1)*M]:copy(outreg2b)
			mem_err_fd[2][3][j+(i-1)*M]:copy(outstop2b)
		end
	end
	-- For reg
	for i=1,R_inside do
		for j=1,M do
			local original_in = diff_ram.startModule.weight[i][j]
			diff_ram.startModule.weight[i][j] = original_in - perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1a:copy(out1[1])
			outreg1a:copy(out1[2])
			outstop1a:copy(out1[3])
			outmem2a:copy(out2[1])
			outreg2a:copy(out2[2])
			outstop2a:copy(out2[3])
			diff_ram.startModule.weight[i][j] = original_in + perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1b:copy(out1[1])
			outreg1b:copy(out1[2])
			outstop1b:copy(out1[3])
			outmem2b:copy(out2[1])
			outreg2b:copy(out2[2])
			outstop2b:copy(out2[3])
			diff_ram.startModule.weight[i][j] = original_in

			outmem1b:add(-1, outmem1a):div(2*perturbation)
			outreg1b:add(-1, outreg1a):div(2*perturbation)
			outstop1b:add(-1, outstop1a):div(2*perturbation)
			outmem2b:add(-1, outmem2a):div(2*perturbation)
			outreg2b:add(-1, outreg2a):div(2*perturbation)
			outstop2b:add(-1, outstop2a):div(2*perturbation)
			reg_err_fd[1][1][j+(i-1)*M]:copy(outmem1b)
			reg_err_fd[1][2][j+(i-1)*M]:copy(outreg1b)
			reg_err_fd[1][3][i]:copy(outstop1b)
			reg_err_fd[2][1][j+(i-1)*M]:copy(outmem2b)
			reg_err_fd[2][2][j+(i-1)*M]:copy(outreg2b)
			reg_err_fd[2][3][j+(i-1)*M]:copy(outstop2b)
		end
	end
	-- For arg1 of the weights
	for i=1,R do
		for j=1,M do
			local original_in = ram_params[2][i][j]
			ram_params[2][i][j] = original_in - perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1a:copy(out1[1])
			outreg1a:copy(out1[2])
			outstop1a:copy(out1[3])
			outmem2a:copy(out2[1])
			outreg2a:copy(out2[2])
			outstop2a:copy(out2[3])
			ram_params[2][i][j] = original_in + perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1b:copy(out1[1])
			outreg1b:copy(out1[2])
			outstop1b:copy(out1[3])
			outmem2b:copy(out2[1])
			outreg2b:copy(out2[2])
			outstop2b:copy(out2[3])
			ram_params[2][i][j] = original_in

			outmem1b:add(-1, outmem1a):div(2*perturbation)
			outreg1b:add(-1, outreg1a):div(2*perturbation)
			outstop1b:add(-1, outstop1a):div(2*perturbation)
			outmem2b:add(-1, outmem2a):div(2*perturbation)
			outreg2b:add(-1, outreg2a):div(2*perturbation)
			outstop2b:add(-1, outstop2a):div(2*perturbation)
			first_arg_err_fd[1][1][j+(i-1)*M]:copy(outmem1b)
			first_arg_err_fd[1][2][j+(i-1)*M]:copy(outreg1b)
			first_arg_err_fd[1][3][i]:copy(outstop1b)
			first_arg_err_fd[2][1][j+(i-1)*M]:copy(outmem2b)
			first_arg_err_fd[2][2][j+(i-1)*M]:copy(outreg2b)
			first_arg_err_fd[2][3][j+(i-1)*M]:copy(outstop2b)
		end
	end
	-- For arg2 of the weights
	for i=1,R do
		for j=1,M do
			local original_in = ram_params[3][i][j]
			ram_params[3][i][j] = original_in - perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1a:copy(out1[1])
			outreg1a:copy(out1[2])
			outstop1a:copy(out1[3])
			outmem2a:copy(out2[1])
			outreg2a:copy(out2[2])
			outstop2a:copy(out2[3])
			ram_params[3][i][j] = original_in + perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1b:copy(out1[1])
			outreg1b:copy(out1[2])
			outstop1b:copy(out1[3])
			outmem2b:copy(out2[1])
			outreg2b:copy(out2[2])
			outstop2b:copy(out2[3])
			ram_params[3][i][j] = original_in

			outmem1b:add(-1, outmem1a):div(2*perturbation)
			outreg1b:add(-1, outreg1a):div(2*perturbation)
			outstop1b:add(-1, outstop1a):div(2*perturbation)
			outmem2b:add(-1, outmem2a):div(2*perturbation)
			outreg2b:add(-1, outreg2a):div(2*perturbation)
			outstop2b:add(-1, outstop2a):div(2*perturbation)
			second_arg_err_fd[1][1][j+(i-1)*M]:copy(outmem1b)
			second_arg_err_fd[1][2][j+(i-1)*M]:copy(outreg1b)
			second_arg_err_fd[1][3][i]:copy(outstop1b)
			second_arg_err_fd[2][1][j+(i-1)*M]:copy(outmem2b)
			second_arg_err_fd[2][2][j+(i-1)*M]:copy(outreg2b)
			second_arg_err_fd[2][3][j+(i-1)*M]:copy(outstop2b)
		end
	end
	-- For out arg of the weights
	for i=1,R do
		for j=1,M do
			local original_in = ram_params[4][i][j]
			ram_params[4][i][j] = original_in - perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1a:copy(out1[1])
			outreg1a:copy(out1[2])
			outstop1a:copy(out1[3])
			outmem2a:copy(out2[1])
			outreg2a:copy(out2[2])
			outstop2a:copy(out2[3])
			ram_params[4][i][j] = original_in + perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1b:copy(out1[1])
			outreg1b:copy(out1[2])
			outstop1b:copy(out1[3])
			outmem2b:copy(out2[1])
			outreg2b:copy(out2[2])
			outstop2b:copy(out2[3])
			ram_params[4][i][j] = original_in

			outmem1b:add(-1, outmem1a):div(2*perturbation)
			outreg1b:add(-1, outreg1a):div(2*perturbation)
			outstop1b:add(-1, outstop1a):div(2*perturbation)
			outmem2b:add(-1, outmem2a):div(2*perturbation)
			outreg2b:add(-1, outreg2a):div(2*perturbation)
			outstop2b:add(-1, outstop2a):div(2*perturbation)
			out_arg_err_fd[1][1][j+(i-1)*M]:copy(outmem1b)
			out_arg_err_fd[1][2][j+(i-1)*M]:copy(outreg1b)
			out_arg_err_fd[1][3][i]:copy(outstop1b)
			out_arg_err_fd[2][1][j+(i-1)*M]:copy(outmem2b)
			out_arg_err_fd[2][2][j+(i-1)*M]:copy(outreg2b)
			out_arg_err_fd[2][3][j+(i-1)*M]:copy(outstop2b)
		end
	end

	-- For instruction arg of the weights
	for i=1,nb_instructions do
		for j=1,M do
			local original_in = ram_params[5][i][j]
			ram_params[5][i][j] = original_in - perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1a:copy(out1[1])
			outreg1a:copy(out1[2])
			outstop1a:copy(out1[3])
			outmem2a:copy(out2[1])
			outreg2a:copy(out2[2])
			outstop2a:copy(out2[3])
			ram_params[5][i][j] = original_in + perturbation
			diff_ram:forget()
			local out1 = diff_ram:forward(input)
			local out2 = diff_ram:forward()
			outmem1b:copy(out1[1])
			outreg1b:copy(out1[2])
			outstop1b:copy(out1[3])
			outmem2b:copy(out2[1])
			outreg2b:copy(out2[2])
			outstop2b:copy(out2[3])
			ram_params[5][i][j] = original_in

			outmem1b:add(-1, outmem1a):div(2*perturbation)
			outreg1b:add(-1, outreg1a):div(2*perturbation)
			outstop1b:add(-1, outstop1a):div(2*perturbation)
			outmem2b:add(-1, outmem2a):div(2*perturbation)
			outreg2b:add(-1, outreg2a):div(2*perturbation)
			outstop2b:add(-1, outstop2a):div(2*perturbation)
			instruction_arg_err_fd[1][1][j+(i-1)*M]:copy(outmem1b)
			instruction_arg_err_fd[1][2][j+(i-1)*M]:copy(outreg1b)
			instruction_arg_err_fd[1][3][i]:copy(outstop1b)
			instruction_arg_err_fd[2][1][j+(i-1)*M]:copy(outmem2b)
			instruction_arg_err_fd[2][2][j+(i-1)*M]:copy(outreg2b)
			instruction_arg_err_fd[2][3][j+(i-1)*M]:copy(outstop2b)
		end
	end


	-- Assertion over the gradients towards memory
	mytester:eq(mem_err_fd[1][1], mem_err_backward[1][1], precision, "Invalid gradInput on mem wrt mem after 1 it for "..name)
	mytester:eq(mem_err_fd[1][2], mem_err_backward[1][2], precision, "Invalid gradInput on mem wrt reg after 1 it for "..name)
	mytester:eq(mem_err_fd[1][3], mem_err_backward[1][3], precision, "Invalid gradInput on mem wrt stop after 1 it for "..name)
	
	mytester:eq(mem_err_fd[2][1], mem_err_backward[2][1], precision, "Invalid gradInput on mem wrt mem after 2 it for "..name)
	mytester:eq(mem_err_fd[2][2], mem_err_backward[2][2], precision, "Invalid gradInput on mem wrt reg after 2 it for "..name)
	mytester:eq(mem_err_fd[2][3], mem_err_backward[2][3], precision, "Invalid gradInput on mem wrt stop after 2 it for "..name)


	-- Assertion over the gradients towards registers
	mytester:eq(reg_err_fd[1][1], reg_err_backward[1][1], precision, "Invalid gradWeight on reg wrt mem after 1 it for "..name)
	mytester:eq(reg_err_fd[1][2], reg_err_backward[1][2], precision, "Invalid gradWeight on reg wrt reg after 1 it for "..name)
	mytester:eq(reg_err_fd[1][3], reg_err_backward[1][3], precision, "Invalid gradWeight on reg wrt stop after 1 it for "..name)
	
	mytester:eq(reg_err_fd[2][1], reg_err_backward[2][1], precision, "Invalid gradWeight on reg wrt mem after 2 it for "..name)
	mytester:eq(reg_err_fd[2][2], reg_err_backward[2][2], precision, "Invalid gradWeight on reg wrt reg after 2 it for "..name)
	mytester:eq(reg_err_fd[2][3], reg_err_backward[2][3], precision, "Invalid gradWeight on reg wrt stop after 2 it for "..name)

	-- Assertion over the gradients towards arg1
	mytester:eq(first_arg_err_fd[1][1], first_arg_err_backward[1][1], precision, "Invalid gradWeight on first_arg wrt mem after 1 it for "..name)
	mytester:eq(first_arg_err_fd[1][2], first_arg_err_backward[1][2], precision, "Invalid gradWeight on first_arg wrt reg after 1 it for "..name)
	mytester:eq(first_arg_err_fd[1][3], first_arg_err_backward[1][3], precision, "Invalid gradWeight on first_arg wrt stop mem after 1 it for "..name)

	mytester:eq(first_arg_err_fd[2][1], first_arg_err_backward[2][1], precision, "Invalid gradWeight on first_arg wrt mem after 2 it for "..name)
	mytester:eq(first_arg_err_fd[2][2], first_arg_err_backward[2][2], precision, "Invalid gradWeight on first_arg wrt reg after 2 it for "..name)
	mytester:eq(first_arg_err_fd[2][3], first_arg_err_backward[2][3], precision, "Invalid gradWeight on first_arg wrt stop after 2 it for "..name)

	-- Assertion over the gradients towards arg2
	mytester:eq(second_arg_err_fd[1][1], second_arg_err_backward[1][1], precision, "Invalid gradWeight on second_arg wrt mem after 1 it for "..name)
	mytester:eq(second_arg_err_fd[1][2], second_arg_err_backward[1][2], precision, "Invalid gradWeight on second_arg wrt reg after 1 it for "..name)
	mytester:eq(second_arg_err_fd[1][3], second_arg_err_backward[1][3], precision, "Invalid gradWeight on second_arg wrt stop mem after 1 it for "..name)

	mytester:eq(second_arg_err_fd[2][1], second_arg_err_backward[2][1], precision, "Invalid gradWeight on second_arg wrt mem after 2 it for "..name)
	mytester:eq(second_arg_err_fd[2][2], second_arg_err_backward[2][2], precision, "Invalid gradWeight on second_arg wrt reg after 2 it for "..name)
	mytester:eq(second_arg_err_fd[2][3], second_arg_err_backward[2][3], precision, "Invalid gradWeight on second_arg wrt stop after 2 it for "..name)

	-- Assertion over the gradients towards out
	mytester:eq(out_arg_err_fd[1][1], out_arg_err_backward[1][1], precision, "Invalid gradWeight on out_arg wrt mem after 1 it for "..name)
	mytester:eq(out_arg_err_fd[1][2], out_arg_err_backward[1][2], precision, "Invalid gradWeight on out_arg wrt reg after 1 it for "..name)
	mytester:eq(out_arg_err_fd[1][3], out_arg_err_backward[1][3], precision, "Invalid gradWeight on out_arg wrt stop mem after 1 it for "..name)

	mytester:eq(out_arg_err_fd[2][1], out_arg_err_backward[2][1], precision, "Invalid gradWeight on out_arg wrt mem after 2 it for "..name)
	mytester:eq(out_arg_err_fd[2][2], out_arg_err_backward[2][2], precision, "Invalid gradWeight on out_arg wrt reg after 2 it for "..name)
	mytester:eq(out_arg_err_fd[2][3], out_arg_err_backward[2][3], precision, "Invalid gradWeight on out_arg wrt stop after 2 it for "..name)

	-- Assertion over the gradients towards instruction
	mytester:eq(instruction_arg_err_fd[1][1], instruction_arg_err_backward[1][1], precision, "Invalid gradWeight on instruction_arg wrt mem after 1 it for "..name)
	mytester:eq(instruction_arg_err_fd[1][2], instruction_arg_err_backward[1][2], precision, "Invalid gradWeight on instruction_arg wrt reg after 1 it for "..name)
	mytester:eq(instruction_arg_err_fd[1][3], instruction_arg_err_backward[1][3], precision, "Invalid gradWeight on instruction_arg wrt stop mem after 1 it for "..name)

	mytester:eq(instruction_arg_err_fd[2][1], instruction_arg_err_backward[2][1], precision, "Invalid gradWeight on instruction_arg wrt mem after 2 it for "..name)
	mytester:eq(instruction_arg_err_fd[2][2], instruction_arg_err_backward[2][2], precision, "Invalid gradWeight on instruction_arg wrt reg after 2 it for "..name)
	mytester:eq(instruction_arg_err_fd[2][3], instruction_arg_err_backward[2][3], precision, "Invalid gradWeight on instruction_arg wrt stop after 2 it for "..name)

end

torchtest.finDiffWeightCriterion = function()
	local name = "dRAM weight"
	local M = 10
	local R = 5
	local R_inside = R + 1
	local nb_instructions = 11
	local bad_case = false
	-- If you set the bad case to true,
	-- this test is going to run by looking into the working directory
	-- for stored differentiable ram, input, output and mask
	-- Otherwise, it is going to run with random input/output/parameters


	local diff_ram, input, mask, gt

	if bad_case then
		diff_ram = torch.load("bad.t7")
		diff_ram:double()
		input = torch.load("input.t7"):double()
		input = input[1]
		mask = torch.load("mask.t7"):double()
		gt = torch.load("gt.t7"):double()
		gt = gt[1]
	else
		diff_ram = layers.DRAM(R, M)
		-- Get a random input
		input = torch.rand(M, M)
		gt = torch.rand(M,M)
		for i=1,M do
			input[i]:copy(input[i] / input[i]:sum())
			gt[i]:copy(gt[i] / gt[i]:sum())
		end
		mask = torch.ones(M, M) -- full mask
	end
	local params = {
		alpha = 10,
		beta = 0,
		gamma = 0,
		delta = 0,
	}
	local criterion = layers.AlgCrit(params)


	-- Compute the finite-difference based gradient
	local parameters, _ = diff_ram:parameters()
	diff_ram:forget()
	local fd_grad={}
	for param_set = 1, #parameters do
		local params = parameters[param_set]:view(-1)
		fd_grad[param_set] = torch.Tensor(params:size(1))
		for i=1, params:size(1) do
			-- perturb plus
			diff_ram:forget()
			local original_value = params[i]
			params[i] = original_value + perturbation
			local output_plus = diff_ram:forwardProgram(input, 10)
			local _, err_response_plus = criterion:forward(output_plus, {gt, mask})

			-- perturb minus
			diff_ram:forget()
			params[i] = original_value - perturbation
			local output_minus = diff_ram:forwardProgram(input, 10)
			local _, err_response_minus = criterion:forward(output_minus, {gt, mask})

			params[i]= original_value
			fd_grad[param_set][i] = (err_response_plus - err_response_minus) / (2*perturbation)
		end
		fd_grad[param_set]:resizeAs(parameters[param_set])
	end

	-- Compute the gradient using the forward backward
	diff_ram:forget()
	diff_ram:zeroGradParameters()
	local fb_param, fb_grad = diff_ram:parameters()
	local output = diff_ram:forwardProgram(input, 10)
	local err_time, err_response = criterion:forward(output, {gt, mask})
	local gradOutput = criterion:backward(output, {gt, mask})
	diff_ram:backwardProgram(input, gradOutput)

	-- Check whether all the coordinates of the gradients are correct
	mytester:eq(fb_grad, fd_grad, precision, "Different gradients between backprop and finite diff")

	-- Compute the inner product of the two gradients and their norm
	-- This test is weak but would potentially provide additional information
	-- in case it is wrong.
	local fb_grad_norm = 0
	local fd_grad_norm = 0
	local inner = 0
	for param_set = 1, #parameters do
		fb_grad_norm = fb_grad_norm + math.pow(fb_grad[param_set]:norm(), 2)
		fd_grad_norm = fd_grad_norm + math.pow(fd_grad[param_set]:norm(), 2)
		-- print(fb_grad[param_set])
		-- print(fd_grad[param_set])
		-- print(torch.dot(fb_grad[param_set], fd_grad[param_set]))
		inner = inner + torch.dot(fb_grad[param_set], fd_grad[param_set])
	end
	inner = inner /(math.sqrt(fb_grad_norm)*math.sqrt(fd_grad_norm))
	mytester:eq(inner, 1, 0.00001, "The two gradients computed have a significantly different direction")
	-- TODO Rudy fix below
	--mytester:eq(math.sqrt(fb_grad_norm), math.sqrt(fd_grad_norm), precision, "The two gradients have a wildly different norm")
end

torchtest.finDiffWeightTest = function()
	local name = "dRAM weight"
	local M = 10
	local R = 3
	local R_inside = R + 1
	local nb_instructions = 11

	local diff_ram = layers.DRAM(R, M)

	-- Get a random input
	local input = torch.rand(M, M)
	for i=1,M do
		input[i]:copy(input[i] / input[i]:sum())
	end

	-- Get a null gradOutput
	local gradOutput
	do
		local mem = torch.zeros(M, M)
		local first_arg = torch.zeros(R_inside, M)
		local stop_tensor = torch.zeros(2)
		gradOutput = {mem, first_arg, stop_tensor}
	end

	-- Get pointers to the weights
	local weights
	do
		weights = {
			diff_ram.firstArgLinear:get(1).weight,
			diff_ram.secondArgLinear:get(1).weight,
			diff_ram.OutputLinear:get(1).weight,
			diff_ram.instructionLinear:get(1).weight,
			diff_ram.startModule.weight
		}
	end
	-- Get pointers to the gradients
	local gradWeights
	do
		gradWeights = {
			diff_ram.firstArgLinear:get(1).gradWeight,
			diff_ram.secondArgLinear:get(1).gradWeight,
			diff_ram.OutputLinear:get(1).gradWeight,
			diff_ram.instructionLinear:get(1).gradWeight,
			diff_ram.startModule.gradWeight
		}
	end

	local arg1_err_fd = {torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	local arg1_err_backward = {torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	local arg2_err_fd = {torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	local arg2_err_backward = {torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	local out_err_fd = {torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	local out_err_backward = {torch.zeros(R*M, M*M), torch.zeros(R*M, R_inside*M), torch.zeros(R*M, 2)}
	local instr_err_fd = {torch.zeros(nb_instructions*M, M*M), torch.zeros(nb_instructions*M, R_inside*M), torch.zeros(nb_instructions*M, 2)}
	local instr_err_backward = {torch.zeros(nb_instructions*M, M*M), torch.zeros(nb_instructions*M, R_inside*M), torch.zeros(nb_instructions*M, 2)}
	local reg_err_fd = {torch.zeros(R_inside*M, M*M), torch.zeros(R_inside*M, R_inside*M), torch.zeros(R_inside*M, 2)}
	local reg_err_backward = {torch.zeros(R_inside*M, M*M), torch.zeros(R_inside*M, R_inside*M), torch.zeros(R_inside*M, 2)}

	-- Compute the *_backward versions
	-- For out mem
	for i=1,M do
		for j=1,M do
			diff_ram:forget()
			diff_ram:zeroGradParameters()
			diff_ram:forward(input)
			gradOutput[1][i][j] = 1
			diff_ram:backward(input, gradOutput)
			arg1_err_backward[1]:select(2,j+(i-1)*M):copy(gradWeights[1])
			arg2_err_backward[1]:select(2,j+(i-1)*M):copy(gradWeights[2])
			out_err_backward[1]:select(2,j+(i-1)*M):copy(gradWeights[3])
			instr_err_backward[1]:select(2,j+(i-1)*M):copy(gradWeights[4])
			reg_err_backward[1]:select(2,j+(i-1)*M):copy(gradWeights[5])
			gradOutput[1][i][j] = 0
		end
	end
	-- For out reg
	for i=1,R_inside do
		for j=1,M do
			diff_ram:forget()
			diff_ram:zeroGradParameters()
			diff_ram:forward(input)
			gradOutput[2][i][j] = 1
			diff_ram:backward(input, gradOutput)
			arg1_err_backward[2]:select(2,j+(i-1)*M):copy(gradWeights[1])
			arg2_err_backward[2]:select(2,j+(i-1)*M):copy(gradWeights[2])
			out_err_backward[2]:select(2,j+(i-1)*M):copy(gradWeights[3])
			instr_err_backward[2]:select(2,j+(i-1)*M):copy(gradWeights[4])
			reg_err_backward[2]:select(2,j+(i-1)*M):copy(gradWeights[5])
			gradOutput[2][i][j] = 0
		end
	end
	-- For out stop
	for i=1,2 do
		diff_ram:forget()
		diff_ram:zeroGradParameters()
		diff_ram:forward(input)
		gradOutput[3][i] = 1
		diff_ram:backward(input, gradOutput)
		arg1_err_backward[3]:select(2,i):copy(gradWeights[1])
		arg2_err_backward[3]:select(2,i):copy(gradWeights[2])
		out_err_backward[3]:select(2,i):copy(gradWeights[3])
		instr_err_backward[3]:select(2,i):copy(gradWeights[4])
		reg_err_backward[3]:select(2,i):copy(gradWeights[5])
		gradOutput[3][i] = 0
	end

	-- Compute the *_fd verions (finite difference)
	local outmema = torch.Tensor(M*M)
	local outmemb = torch.Tensor(M*M)
	local outrega = torch.Tensor(R_inside*M)
	local outregb = torch.Tensor(R_inside*M)
	local outstopa = torch.Tensor(2)
	local outstopb = torch.Tensor(2)
	-- For arg1
	for i=1,R do
		for j=1,M do
			local original_in = weights[1][i][j]
			weights[1][i][j] = original_in - perturbation
			diff_ram:forget()
			local out = diff_ram:forward(input)
			outmema:copy(out[1])
			outrega:copy(out[2])
			outstopa:copy(out[3])
			weights[1][i][j] = original_in + perturbation
			diff_ram:forget()
			out = diff_ram:forward(input)
			outmemb:copy(out[1])
			outregb:copy(out[2])
			outstopb:copy(out[3])
			weights[1][i][j] = original_in

			outmemb:add(-1, outmema):div(2*perturbation)
			outregb:add(-1, outrega):div(2*perturbation)
			outstopb:add(-1, outstopa):div(2*perturbation)
			arg1_err_fd[1][j+(i-1)*M]:copy(outmemb)
			arg1_err_fd[2][j+(i-1)*M]:copy(outregb)
			arg1_err_fd[3][i]:copy(outstopb)
		end
	end
	-- For arg2
	for i=1,R do
		for j=1,M do
			local original_in = weights[2][i][j]
			weights[2][i][j] = original_in - perturbation
			diff_ram:forget()
			local out = diff_ram:forward(input)
			outmema:copy(out[1])
			outrega:copy(out[2])
			outstopa:copy(out[3])
			weights[2][i][j] = original_in + perturbation
			diff_ram:forget()
			out = diff_ram:forward(input)
			outmemb:copy(out[1])
			outregb:copy(out[2])
			outstopb:copy(out[3])
			weights[2][i][j] = original_in

			outmemb:add(-1, outmema):div(2*perturbation)
			outregb:add(-1, outrega):div(2*perturbation)
			outstopb:add(-1, outstopa):div(2*perturbation)
			arg2_err_fd[1][j+(i-1)*M]:copy(outmemb)
			arg2_err_fd[2][j+(i-1)*M]:copy(outregb)
			arg2_err_fd[3][i]:copy(outstopb)
		end
	end
	-- For out
	for i=1,R do
		for j=1,M do
			local original_in = weights[3][i][j]
			weights[3][i][j] = original_in - perturbation
			diff_ram:forget()
			local out = diff_ram:forward(input)
			outmema:copy(out[1])
			outrega:copy(out[2])
			outstopa:copy(out[3])
			weights[3][i][j] = original_in + perturbation
			diff_ram:forget()
			out = diff_ram:forward(input)
			outmemb:copy(out[1])
			outregb:copy(out[2])
			outstopb:copy(out[3])
			weights[3][i][j] = original_in

			outmemb:add(-1, outmema):div(2*perturbation)
			outregb:add(-1, outrega):div(2*perturbation)
			outstopb:add(-1, outstopa):div(2*perturbation)
			out_err_fd[1][j+(i-1)*M]:copy(outmemb)
			out_err_fd[2][j+(i-1)*M]:copy(outregb)
			out_err_fd[3][i]:copy(outstopb)
		end
	end
	-- For instr
	for i=1,nb_instructions do
		for j=1,M do
			local original_in = weights[4][i][j]
			weights[4][i][j] = original_in - perturbation
			diff_ram:forget()
			local out = diff_ram:forward(input)
			outmema:copy(out[1])
			outrega:copy(out[2])
			outstopa:copy(out[3])
			weights[4][i][j] = original_in + perturbation
			diff_ram:forget()
			out = diff_ram:forward(input)
			outmemb:copy(out[1])
			outregb:copy(out[2])
			outstopb:copy(out[3])
			weights[4][i][j] = original_in

			outmemb:add(-1, outmema):div(2*perturbation)
			outregb:add(-1, outrega):div(2*perturbation)
			outstopb:add(-1, outstopa):div(2*perturbation)
			instr_err_fd[1][j+(i-1)*M]:copy(outmemb)
			instr_err_fd[2][j+(i-1)*M]:copy(outregb)
			instr_err_fd[3][i]:copy(outstopb)
		end
	end
	-- For reg
	for i=1,R_inside do
		for j=1,M do
			local original_in = weights[5][i][j]
			weights[5][i][j] = original_in - perturbation
			diff_ram:forget()
			local out = diff_ram:forward(input)
			outmema:copy(out[1])
			outrega:copy(out[2])
			outstopa:copy(out[3])
			weights[5][i][j] = original_in + perturbation
			diff_ram:forget()
			out = diff_ram:forward(input)
			outmemb:copy(out[1])
			outregb:copy(out[2])
			outstopb:copy(out[3])
			weights[5][i][j] = original_in

			outmemb:add(-1, outmema):div(2*perturbation)
			outregb:add(-1, outrega):div(2*perturbation)
			outstopb:add(-1, outstopa):div(2*perturbation)
			reg_err_fd[1][j+(i-1)*M]:copy(outmemb)
			reg_err_fd[2][j+(i-1)*M]:copy(outregb)
			reg_err_fd[3][i]:copy(outstopb)
		end
	end

	mytester:eq(arg1_err_fd[1], arg1_err_backward[1], precision, "Invalid gradInput on arg1 wrt mem after 1 it for "..name)
	mytester:eq(arg1_err_fd[2], arg1_err_backward[2], precision, "Invalid gradInput on arg1 wrt reg mem after 1 it for "..name)
	mytester:eq(arg1_err_fd[3], arg1_err_backward[3], precision, "Invalid gradInput on arg1 wrt stop mem after 1 it for "..name)
	mytester:eq(arg2_err_fd[1], arg2_err_backward[1], precision, "Invalid gradInput on arg2 wrt mem after 1 it for "..name)
	mytester:eq(arg2_err_fd[2], arg2_err_backward[2], precision, "Invalid gradInput on arg2 wrt reg mem after 1 it for "..name)
	mytester:eq(arg2_err_fd[3], arg2_err_backward[3], precision, "Invalid gradInput on arg2 wrt stop mem after 1 it for "..name)
	mytester:eq(out_err_fd[1], out_err_backward[1], precision, "Invalid gradInput on out wrt mem after 1 it for "..name)
	mytester:eq(out_err_fd[2], out_err_backward[2], precision, "Invalid gradInput on out wrt reg mem after 1 it for "..name)
	mytester:eq(out_err_fd[3], out_err_backward[3], precision, "Invalid gradInput on out wrt stop mem after 1 it for "..name)
	mytester:eq(instr_err_fd[1], instr_err_backward[1], precision, "Invalid gradInput on instr wrt mem after 1 it for "..name)
	mytester:eq(instr_err_fd[2], instr_err_backward[2], precision, "Invalid gradInput on instr wrt reg mem after 1 it for "..name)
	mytester:eq(instr_err_fd[3], instr_err_backward[3], precision, "Invalid gradInput on instr wrt stop mem after 1 it for "..name)
	mytester:eq(reg_err_fd[1], reg_err_backward[1], precision, "Invalid gradInput on reg wrt mem after 1 it for "..name)
	mytester:eq(reg_err_fd[2], reg_err_backward[2], precision, "Invalid gradInput on reg wrt reg mem after 1 it for "..name)
	mytester:eq(reg_err_fd[3], reg_err_backward[3], precision, "Invalid gradInput on reg wrt stop mem after 1 it for "..name)
	
end

function torchtest.testCriterion()
	-- Test parameters
	local M = 5
	local R = 3
	local params = {
		alpha = 10,
		beta = 0,
		gamma = 0,
		delta = 0,
	}

	-- We are testing that the criterion effectively ignore the masked output of the memory
	local criterion = layers.AlgCrit(params) -- We're not yet testing the time-pressure.

	-- Generate examples
	local target_memory = distUtils.toDistTensor(torch.Tensor{1,1,1,0,0}, M)
	local incorrect_prediction = distUtils.toDistTensor(torch.Tensor{1,0,1,0,0}, M)
	local _registers = distUtils.toDistTensor(torch.Tensor{0,0,0}, M) -- Should have no impact
	local _stop_tensor = torch.Tensor({0,1}) -- Should have no impact

	local everything_counts_mask = torch.ones(M,M)
	local ignore_the_error_mask = everything_counts_mask:clone()
	ignore_the_error_mask[2]:fill(0)

	-- Generate what would be the machine output.
	-- A machine output is a table, and the output of the ram is all the states
	-- which is why we have a table of table
	local correct_machine_output = {{target_memory, _registers, _stop_tensor}}
	local incorrect_machine_output = {{incorrect_prediction, _registers, _stop_tensor}}
	local empty_gradients = {{torch.zeros(M,M), torch.zeros(R,M), torch.zeros(2)}}

	--------------------------------
    -- No mask, everything counts --
    --------------------------------

	-- Assert that we don't get a loss in the case of a correct answer
	local err_time, err_response = criterion:forward(correct_machine_output,
											   {target_memory, everything_counts_mask})
	local gradOutput = criterion:backward(correct_machine_output,
									{target_memory, everything_counts_mask})

	mytester:eq(err_response, 0, "The loss for a correct answer is not zero")
	mytester:eq(gradOutput, empty_gradients, "The gradients for a correct answer are not zero")

	-- Assert that we actually get a loss and gradients in the case of an incorrect answer
	local err_time, err_response = criterion:forward(incorrect_machine_output,
													 {target_memory, everything_counts_mask})
	local gradOutput = criterion:backward(incorrect_machine_output,
										  {target_memory, everything_counts_mask})
	mytester:ne(err_response,0, "The loss for an incorrect answer is zero")
	mytester:ne(gradOutput, empty_gradients, "There is no gradient for an incorrect answer")




	-----------------------
    -- Masking the error --
    -----------------------

	-- Assert that we don't get a loss in the case of a correct answer
	err_time, err_response = criterion:forward(correct_machine_output,
											   {target_memory, ignore_the_error_mask})
	gradOutput = criterion:backward(correct_machine_output,
									{target_memory, ignore_the_error_mask})
	mytester:eq(err_response, 0, "The loss for a correct answer is not zero")
	mytester:eq(gradOutput, empty_gradients, "The gradients for a correct answer are not zero")
	-- Assert that we don't get a loss in the case of a bad answer that the masks want to ignore
	err_time, err_response = criterion:forward(incorrect_machine_output,
											   {target_memory, ignore_the_error_mask})
	gradOutput = criterion:backward(incorrect_machine_output,
									{target_memory, ignore_the_error_mask})
	mytester:eq(err_response, 0, "The loss for an error that the masks says should be ignored is not zero")
	mytester:eq(gradOutput, empty_gradients, "The gradients for an error that the masks says should be ignored are not zero")

end

function torchtest.finDiffCriterion()
	local name = "Criterion"
	local M = 10
	local R = 5
	local R_inside = R + 1
	local alpha = 10 -- This test only looks at err_reponse
	local nInputs = 3
	local params = {
		alpha = alpha,
		beta = 0,
		gamma = 0,
		delta = 0,
	}

	local mod = layers.AlgCrit(params)

	-- Get a random set of inputs and targets
	local input = {}
	for i=1,nInputs do
		local single_mem = torch.rand(M, M)
		for i=1,M do
			single_mem[i]:copy(single_mem[i] / single_mem[i]:sum())
		end
		local single_reg = torch.rand(R_inside, M)
		for i=1,R_inside do
			single_reg[i]:copy(single_reg[i] / single_reg[i]:sum())
		end
		local single_stop = torch.rand(2)
		table.insert(input, {single_mem, single_reg, single_stop})
	end
	local target = {}
	do
		local gt = torch.rand(M, M)
		for i=1,M do
			gt[i]:copy(gt[i] / gt[i]:sum())
		end
		target = {gt, torch.ones(M, M)}
	end

	-- Tables with gradients
	local loss_err_fd = {}
	local loss_err_backward = {}

	-- Compute the *_fd versions
	-- For each inputs
	for i=1,nInputs do
		loss_err_fd[i] = {torch.zeros(M, M), torch.zeros(R_inside, M), torch.zeros(2)}
		for j=1,M do
			for k=1,M do
				local original_in = input[i][1][j][k]
				input[i][1][j][k] = original_in + perturbation
				local _, lossa = mod:forward(input, target)

				input[i][1][j][k] = original_in - perturbation
				local _, lossb = mod:forward(input, target)

				input[i][1][j][k] = original_in

				lossb = (lossa - lossb) / (2*perturbation)
				loss_err_fd[i][1][j][k] = lossb
			end
		end
		for j=1,R_inside do
			for k=1,M do
				local original_in = input[i][2][j][k]
				input[i][2][j][k] = original_in + perturbation
				local _, lossa = mod:forward(input, target)

				input[i][2][j][k] = original_in - perturbation
				local _, lossb = mod:forward(input, target)

				input[i][2][j][k] = original_in

				lossb = (lossa - lossb) / (2*perturbation)
				loss_err_fd[i][2][j][k] = lossb
			end
		end
		for j=1,2 do
			local original_in = input[i][3][j]
			input[i][3][j] = original_in + perturbation
			local _, lossa = mod:forward(input, target)

			input[i][3][j] = original_in - perturbation
			local _, lossb = mod:forward(input, target)

			input[i][3][j] = original_in

			lossb = (lossa - lossb) / (2*perturbation)
			loss_err_fd[i][3][j] = lossb
		end
	end

	-- Compute the *_backward versions
	-- For loss
	mod:forward(input, target)
	local gradInput = mod:backward(input, target)
	loss_err_backward = gradInput

	mytester:eq(loss_err_fd, loss_err_backward, precision, "Invalid gradInput for "..name)
end

torchtest.unrolledRecurrentTest = function()
	local name = "dRAM weight"
	local M = 10
	local R = 5
	local R_inside = R + 1
	local nb_instructions = 11
	local max_rec = 3
	local params = {
		alpha = 10,
		beta = 0,
		gamma = 0,
		delta = 0,
	}

	-- Model used for recurrent
	local diff_ram = layers.DRAM(R, M)
	local rec_criterion = layers.AlgCrit(params)
	-- Model used for manual
	local first_module = diff_ram.startModule:clone()
	local single_step_ram = diff_ram.singleStepRAM:clone()
	local man_criterion = rec_criterion:clone()

	local input = torch.rand(M, M)
	local target
	do
		local gt = torch.rand(M,M)
		for i=1,M do
			input[i]:copy(input[i] / input[i]:sum())
			gt[i]:copy(gt[i] / gt[i]:sum())
		end
		local mask = torch.ones(M, M)
		target = {gt, mask}
	end

	local function add_table(t1, t2)
		for k,v in ipairs(t1) do
			t1[k]:add(t2[k])
		end
	end

	-- Forward with the recurent
	diff_ram:forget()
	local rec_outputs = diff_ram:forwardProgram(input, max_rec)
	local _, rec_loss = rec_criterion:forward(rec_outputs, target)

	-- Manual forward
	local man_outputs = {}
	local rams = {}
	man_outputs[1] = first_module:forward(input)
	for i=2,max_rec+1 do
		rams[i] = single_step_ram:clone()
		man_outputs[i] = rams[i]:forward(man_outputs[i-1])
	end
	local _, man_loss = man_criterion:forward(man_outputs, target)

	mytester:eq(rec_outputs, man_outputs, "Invalid forward")
	mytester:eq(rec_loss, man_loss, "Invalid loss")


	-- Backward with the recurent
	diff_ram:zeroGradParameters()
	local rec_go = rec_criterion:backward(rec_outputs, target)
	local rec_gradInput = diff_ram:backwardProgram(input, rec_go)

	for i=1,max_rec do
		rec_go[i] = diff_ram._gradOutputs[i]
	end

	-- Manual backward
	local man_go = man_criterion:backward(man_outputs, target)
	for i=max_rec+1,2,-1 do
		rams[i]:zeroGradParameters()
		local gi = rams[i]:backward(man_outputs[i-1], man_go[i])
		add_table(man_go[i-1], gi)
	end
	local man_gradInput = first_module:backward(input, man_go[1])

	for i=1,max_rec+1 do
		mytester:eq(rec_go[i], man_go[i], "Invalid intermediate gradOutput at step "..i)
	end
	mytester:eq(rec_gradInput, man_gradInput, "Invalid gradInput")

end



return function(tests)
	torch.setnumthreads(2) --For better performances
	print("Testing with seed: "..seed)
	torch.manualSeed(seed)
	math.randomseed(seed)
	mytester = torch.Tester()
	mytester:add(torchtest)
	mytester:run(tests)
	return mytester
end
