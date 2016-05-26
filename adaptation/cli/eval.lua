local torch = require 'torch'
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
local lapp = require 'pl.lapp'
local nc = require 'nc'
local distUtils = require 'nc.distUtils'
local compiler = require 'nc.decompiler'

math.randomseed(os.time())


local opt = lapp [[
Neural Compiler evaluation CLI tool
Main options
    <config>        (string)         Config file path
    --sample                         Should generate a sample
    --biased_sample                  Should generate a biased sample
    --biased_sample_id  (default 1)  Which of the biased sample to use (if there is several)
    --dr                (default "") Path to pretrained differentiable RAM
    --max_rec           (default 50) Maximum number of recurent iterations
    --sharp             (default 5)  Sharpness of initial program
    --trace                          Print the program trace while running
]]

local config = dofile(opt.config)

if opt.sample and opt.biased_sample then
    print("What do you want? Unbiased or Biased samples? Make up your mind")
    os.exit(1)
end

if opt.biased_sample and type(config.gen_biased_sample)=="table" then
    config.gen_biased_sample = config.gen_biased_sample[opt.biased_sample_id]
end

local init_mem, final_mem, loss_mask
if opt.sample then
    init_mem, final_mem, loss_mask = config.gen_sample()
elseif opt.biased_sample then
    init_mem, final_mem, loss_mask = config.gen_biased_sample()
else
    init_mem, final_mem, loss_mask= config.example_input, config.example_output, config.example_loss_mask
end

if opt.dr == "" then
    neural_ram = nc.layers.DRAM(config.nb_registers, config.memory_size)
    neural_ram:flashHardProgram(config, opt.sharp)
else
    neural_ram = torch.load(opt.dr)
end

print("Initial mem")
print(distUtils.toNumberTensor(init_mem):view(1,-1))

local final, nb_iter_done
if opt.trace then
    final, nb_iter_done = neural_ram:traceProgram(init_mem, opt.max_rec)
else
    final, nb_iter_done = neural_ram:executeProgram(init_mem, opt.max_rec)
end

print("Executed in " .. nb_iter_done .. " iterations")
if (final[1] - final_mem):cmul(loss_mask):abs():max() < 1e-10 then
    print("The output is correct :)")
    print(distUtils.toNumberTensor(final[1]):view(1,-1))
else
    print("Error in the output :(")
    if (distUtils.toNumberTensor(final[1]) - distUtils.toNumberTensor(final_mem)):cmul(loss_mask[1]):abs():max() == 0 then
        print("The projected version is still correct! :)")
        print("Expected:")
        print(distUtils.toNumberTensor(final_mem):view(1,-1))
        print("Confidence:")
        print(final[1]:max(2):view(1,-1))
    else
        print("Even the projected is not correct :(")
        print("Expected:")
        print(distUtils.toNumberTensor(final_mem):view(1,-1))
        print("Result:")
        print(distUtils.toNumberTensor(final[1]):view(1,-1))
    end
end

print(compiler.toASM(neural_ram))
