local torch = require 'torch'
require 'trepl'
torch.setdefaulttensortype('torch.FloatTensor')
local nc = require 'nc'
local trainer = require 'nc.trainer'
local plotter = require 'nc.plotter'
local paths = require 'paths'
local threads = require 'threads'

-- Define all the experiments to be ran
-- The main key is the name of the experiment.
-- "times" is the number of times this experience should be ran
-- "output_base" is the main output folder, this will be appended with the xp params
-- "args" is the specific arguments. Each argument either take a single value or
-- a table. in the case of the table, the experiment will be ran multiple times
-- for each possible value of each parameters.
-- Here the example show how to run different experiments with the `access`
-- and `swap` tasks
local todo = {
    access = {
        times = 3,
        output_base = "experiments/",
        args = {
            max_rec = 15,
            it = 1000,
            decompile = true,
            save_it = 100,
            print_val = 100,
            batch = 1,
            optim = {"adam"},
            biased_sample = true,
            lr = {0.1, 1},
            alpha = {1, 10},
            delta = {1, 10},
            sharp = {2, 3, 4}
        },
    },
    swap = {
        times = 3,
        output_base = "experiments/",
        args = {
            max_rec = 20,
            it = 1000,
            decompile = true,
            save_it = 100,
            print_val = 100,
            batch = 1,
            optim = {"adam"},
            biased_sample = true,
            lr = {0.1, 1},
            alpha = {1, 10},
            beta = {0, 1},
            gamma = {0, 1},
            delta = {1, 10},
            sharp = {2, 3, 4}
        },
    },
}
-- Wheter or not to redo experiments that have already been done
local force = false
-- Number of threads that can be used
local n_threads = 8

-- Preload all the experiments config to allow modification of the install
-- while the experiment is running
print("Loading experiments config.")
for name, params in pairs(todo) do
    params["config"] = dofile("examples/"..name..".lua")
end
print("Experiments loaded.")

-- Get the default options as specified in trainer
local tmp = {}
tmp[1] = "dummy"
local default_opt = trainer.parse_opt(tmp)
local clone_tab = function(t)
    local out = {}
    for k,v in pairs(t) do
        out[k] = v
    end
    return out
end

local print_new = print_new

local pool = threads.Threads(
                n_threads,
                function()
                    local torch = require 'torch'
                    torch.setdefaulttensortype('torch.FloatTensor')
                    torch.setnumthreads(1)
                    -- trepl as a whole is not thread safe, we just want the print
                    print = print_new
                    local nc = require 'nc'
                end
            )

-- Recursive function that will run all experiments
local run_xp
run_xp = function(config, opt, out_folder, times)
    -- Check if there are still multiple arguments
    -- If there are, go though them in lexicographic order
    local multi_arg = {}
    for arg, val in pairs(opt) do
        if type(val) == "table" then
            table.insert(multi_arg, arg)
        end
    end
    table.sort(multi_arg)
    for i, arg in ipairs(multi_arg) do
        local val = opt[arg]
        for _, arg_val in ipairs(val) do
            local tmp_opt = clone_tab(opt)
            tmp_opt[arg] = arg_val
            run_xp(config, tmp_opt, out_folder.."_"..arg.."-"..tostring(arg_val), times)
        end
        return
    end

    -- We got here only if each arg has a single value
    -- So run the experiment the required number of times
    for time=1,times do
        -- Check if this has already been done
        opt.save_name = out_folder.."/"..time.."/"
        if force or (not paths.dirp(opt.save_name .. '/figure')) then
            pool:addjob(
            function()
                local paths = require 'paths'
                local trainer = require 'nc.trainer'
                local plotter = require 'nc.plotter'

                print("Running "..opt.save_name)
                -- Prevent flooding stdout
                local stdout = io.output()
                if not paths.dirp(opt.save_name) then
                    paths.mkdir(opt.save_name)
                end
                io.output(opt.save_name.."log.txt")

                -- Run the training
                print("opt")
                print(opt)
                print("config")
                print(config)
                trainer.train(clone_tab(opt), config)

                -- Generate the images associated with this training
                local plot_opt = {
                    path_to_csv = opt.save_name.."plot.csv",
                    output_path = opt.save_name.."figure/",
                    file_only = true,
                }
                plotter.plot(plot_opt)

                -- Put stdout back
                io.output(stdout)
                return opt.save_name
            end,
            function(output)
                print("Finished "..output)
            end)
        sys.sleep(2)
        else
            print("Skipping "..opt.save_name)
        end
    end
end

-- Run all the experiments
for xp_name, params in pairs(todo) do
    local out_folder = params.output_base..xp_name.."/"
    local opt = clone_tab(default_opt)
    for arg, val in pairs(params.args) do
        opt[arg] = val
    end
    run_xp(params.config, opt, out_folder, params.times)
end

pool:terminate()
