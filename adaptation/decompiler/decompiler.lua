local distUtils = require 'nc.distUtils'

local decompiler = {}

decompiler.instrToName = {
    "STOP",
    "ZERO",
    "INC ",
    "ADD ",
    "SUB ",
    "DEC ",
    "MIN ",
    "MAX ",
    "READ",
    "WRIT",
    "JEZ "
}

decompiler.dump_registers = function(registers, nb_registers, softmax)
    local str = ""
    for register = 1,nb_registers do
        local score, proba = distUtils.toString(registers[register+1], softmax)
        str = str .. "R"..register.." = "..score.." ("..proba..")\n"
    end
    return str
end

decompiler.dump_initial_state = function(RI, softmax)
    local score, proba = distUtils.toString(RI, softmax)
    return "Initial State: " ..score.." ("..proba..")\n"
end

decompiler.dump_line = function(arg1, arg2, out, instr, state, softmax)
    local arg1_score, arg1_proba = distUtils.toString(arg1, softmax, true)
    local arg2_score, arg2_proba = distUtils.toString(arg2, softmax, true)
    local out_score, out_proba = distUtils.toString(out, softmax, true)
    local instr_score, instr_proba = distUtils.toString(instr, softmax, false)
    str = (state-1)..": \t"
    str = str.. "R"..out_score.." ("..out_proba..") "
    str = str.." \t= "
    local instr_str
    if tonumber(instr_score) then
        instr_str = decompiler.instrToName[tonumber(instr_score)+1]
    else
        instr_str = "NOP "
    end
    str = str..instr_str.." ("..instr_proba..") "
    str = str.." \t[ "
    str = str.."R"..arg1_score.." ("..arg1_proba..") "
    str = str.." \t, "
    str = str.."R"..arg2_score.." ("..arg2_proba..") "
    str = str.." \t] "
    str = str .. "\n"
    return str
end

-- Takes a differentiable ram and creates a string containing the assembly representation of it.
decompiler.toASM = function(dRAM)
    local str = ""
    -- Dump registers initial values
    str = str .. decompiler.dump_registers(
        dRAM.startModule.weight,
        dRAM.nb_registers,
        true
    )
    str = str .. "\n\n"
    str = str .. decompiler.dump_initial_state(dRAM.startModule.weight[1], true)
    str = str .. "\n\n"
    -- Dump all lines in the program
    for state = 1,dRAM.firstArgLinear:get(1).weight:size(2) do
        str = str .. decompiler.dump_line(
            dRAM.firstArgLinear:get(1).weight:select(2,state),
            dRAM.secondArgLinear:get(1).weight:select(2,state),
            dRAM.OutputLinear:get(1).weight:select(2,state),
            dRAM.instructionLinear:get(1).weight:select(2,state),
            state,
            true
        )
    end
    return str
end

decompiler.toASM_file = function(dRAM, output_file)
    local file = io.open(output_file, 'w')-- Dump registers initial values
    local content = decompiler.toASM(dRAM)
    file:write(content)
end


return decompiler
