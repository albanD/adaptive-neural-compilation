local distUtils = {}

distUtils.addDist = function(dist1, dist2, output)
	assert(dist1:size(1)==dist2:size(1), "Distribution should have same size")
	if output == nil then
		output = dist1.new():resizeAs(dist1)
	end
	output:resize(dist1:size(1)):zero()

	local max_val = dist1:size(1)

	for a=0,max_val-1 do
		local Pa = dist1[a+1]
		for b=0,max_val-1 do
			local Pb = dist2[b+1]
			output[(a+b)%max_val+1] = output[(a+b)%max_val+1] + Pa * Pb
		end
	end

	return output
end

distUtils.addDistTensor = function(dist1, dist2, output)
	assert(dist1:dim()==2, "Add tensor only for 1D tensor of distributions")
	assert(dist2:dim()==2, "Add tensor only for 1D tensor of distributions")
	if output == nil then
		output = dist1.new()
	end
	output:resize(dist1:size()):zero()

	for i=1,dist1:size(1) do
		distUtils.addDist(dist1[i], dist2[i], output[i])
	end
	return output
end

distUtils.subDist = function(dist1, dist2, output)
	assert(dist1:size(1)==dist2:size(1), "Distribution should have same size")
	if output == nil then
		output = dist1.new()
	end
	output:resizeAs(dist1):zero()

	local max_val = dist1:size(1)

	for a=0,max_val-1 do
		local Pa = dist1[a+1]
		for b=0,max_val-1 do
			local Pb = dist2[b+1]
			output[(a-b)%max_val+1] = output[(a-b)%max_val+1] + Pa * Pb
		end
	end

	return output
end

distUtils.maxDist = function(dist1, dist2, output)
	if output == nil then
		output = dist1.new()
	end
	output:resizeAs(dist1):zero()

	local cum1 = torch.cumsum(dist1)
	local cum2 = torch.cumsum(dist2)

	output:add(torch.cmul(dist1, cum2))
	output:add(torch.cmul(dist2, cum1))
	output:add(-1, torch.cmul(dist1, dist2))

	return output
end

distUtils.reverseTensor = function(t)
	-- #groshackbiensale
	t:cdata().storageOffset = t:size(1)-1
	t:cdata().stride[0] = -1
end

distUtils.minDist = function(dist1, dist2, output)
	if output == nil then
		output = dist1.new()
	end
	output:resizeAs(dist1):zero()

	local cum1 = torch.Tensor(dist1:storage())
	distUtils.reverseTensor(cum1)
	cum1 = torch.cumsum(cum1)
	distUtils.reverseTensor(cum1)
	local cum2 = torch.Tensor(dist2:storage())
	distUtils.reverseTensor(cum2)
	cum2 = torch.cumsum(cum2)
	distUtils.reverseTensor(cum2)

	output:add(torch.cmul(dist1, cum2))
	output:add(torch.cmul(dist2, cum1))
	output:add(-1, torch.cmul(dist1, dist2))

	return output
end

distUtils.toNumber = function(dist)
	assert(torch.isTensor(dist), "Distribution should be a Tensor")
	assert(dist:dim()==1, "Distribution should be a 1D Tensor")

	local v,p = dist:max(1)

	return p[1]-1 -- Our distribution are zero-based and Lua is 1-based
end

distUtils.toNumberTensor = function(input)
	assert(input:dim()==2, "Supports only 1D distribution tensors")

	local output = torch.Tensor(input:size(1))

	for i=1,input:size(1) do
		output[i] = distUtils.toNumber(input[i])
	end

	return output
end

-- Creates a tensor containing a distribution with the given number
distUtils.toDist = function(number, max_val)
	assert(max_val, "The maximum value should be provided")
	local output = torch.zeros(max_val)
	-- If number != distUtils.flatDist (see below)
	if number - number == 0 then
		output[number%max_val+1] = 1
	else
		-- Special case: the infinity will give back a uniform distribution
		-- If number - number!=0, this is this case and wouldn't happen otherwise.
		output:fill(1/max_val)
	end
	return output
end

-- Add an extra dimension to the tensor such that everything becomes a distribution
distUtils.toDistTensor = function(input, max_val)
	assert(max_val, "The maximum value should be provided")
	local init_size = input:size()
	local ndims = init_size:size(1)
	local new_size = torch.LongStorage(ndims+1)
	for d=1,ndims do
		new_size[d] = init_size[d]
	end
	new_size[ndims+1] = max_val

	local output = torch.zeros(new_size)
	if ndims==1 then
		for i=1,init_size[1] do
			-- If number != distUtils.flatDist (see below)
			if input[i]-input[i]==0 then
				output[i][input[i]%max_val+1] = 1
			else
				output[i]:fill(1/max_val)
			end
		end
	elseif ndims==2 then
		for i=1,init_size[1] do
			for j=1,init_size[2] do
				-- If number != distUtils.flatDist (see below)
				if input[i][j]-input[i][j]==0 then
					output[i][j][input[i][j]%max_val+1] = 1
				else
					output[i][j]:fill(1/max_val)
				end
			end
		end
	elseif ndims==3 then
		for i=1,init_size[1] do
			for j=1,init_size[2] do
				for k=1,init_size[2] do
					-- If number != distUtils.flatDist (see below)
					if input[i][j][k]-input[i][j][k]==0 then
						output[i][j][k][input[i][j][k]%max_val+1] = 1
					else
						output[i][j][k]:fill(1/max_val)
					end
				end
			end
		end
	else
		error("Not implemented")
	end

	return output
end

-- Create a new table where numbers or tensors become distributions
-- Can take either a table or single number for maximum values
distUtils.toDistTable = function(table, max_vals)
	local max_val
	local output = {}
	for k,v in pairs(table) do
		if type(max_vals) == "table" then
			max_val = max_vals[k]
		else
			max_val = max_vals
		end
		if torch.isTensor(v) then
			output[k] = distUtils.toDistTensor(v, max_val)
		else
			output[k] = distUtils.toDist(v, max_val)
		end
	end
	return output
end

-- Convert a distribution to a string for the maximum value
-- If the distribution is too flat, returns "-"
local low_precision = function(a)
	return math.floor(a*100)/100
end
local softmax_layer = nn.SoftMax()
distUtils.toString = function(dist, softmax, add_one)
	if softmax then
		dist = softmax_layer:forward(dist:clone():view(-1))
	end
	local val, ind = dist:max(1)
	if add_one then
		ind[1] = ind[1] + 1
	end
	if val[1] > 0.5 then
		return tostring(low_precision(ind[1]-1)), tostring(low_precision(val[1]))
	else
		return "-", tostring(low_precision(val[1]))
	end
end

-- Can be given to the toDist functions to get flat distributions
distUtils.flatDist = 1/0

return distUtils
