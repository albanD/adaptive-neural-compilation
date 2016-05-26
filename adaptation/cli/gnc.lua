local torch = require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local nc = require 'nc'
local compiler = require 'nc.decompiler'
local lapp = require 'pl.lapp'

local opt = lapp [[
Neural Compiler compilation CLI tool

Decompilation
    --decompile      (default "")    Where to put the decompiled version
    --dr             (default "")    Path to existing dRAM
]]

assert(opt.dr~="", "Need a dRAM to decompile it")
local dRAM = torch.load(opt.dr)

if opt.decompile~= "" then
    compiler.toASM_file(dRAM, opt.decompile)
else
    print(compiler.toASM(dRAM))
end
