local utils = require 'nc.utils'
local optim = require 'nc.optim'
local layers = require 'nc.layers'
local decompiler = require 'nc.decompiler'
local lapp = require 'pl.lapp'
local pl_data = require 'pl.data'
local paths = require 'paths'

local trainer = {}
utils["trainer"] = trainer

trainer.parse_opt = function(arg)
    local opt = lapp([[
    Neural Compiler training CLI tool
    Main options
        <config>        (string)                    Config file path
        --dr            (default "")                Path to pretrained differentiable RAM
        --rand_dr                                   Should use random initialisation

    Dataset options
        --sample                                    Should generate a sample
        --biased_sample                             Should generate a biased sample
        --biased_sample_id  (default 1)             Which of the biased sample to use (if there is several)
        --val           (default 100)               Size of the validation set

    Training options
        --alpha         (default 10)                Alpha parameter for the criterion
        --beta          (default 0)                 Beta parameter for the criterion
        --gamma         (default 0)                 Gamma parameter for the criterion
        --delta         (default 0)                 Delta parameter for the criterion
        --lr            (default 1)                 Learning rate used for training
        --it            (default 1000)              Number of training epochs
        --max_rec       (default 10)                Maximum number of recurent iterations
        --batch         (default 1)                 Batch size
        --train_size    (default 0)                 Number of training samples in the training data
        --val_is_train                              Use the same data as training set and test set
        --optim         (default "sgd")             Which optimisation algorithm to use: SGD, Adam
        --sharp         (default 5)                 Sharpening factor to use
        --prior_inf_ini (default 0)                 Initial weight of the Infinity prior over correct program
        --prior_inf_slp (default 1)                 Slope at which the weight of the Infinity prior decreases
        --prior_soft_ini (default 0)                Initial weight of the Softmax prior over correct program
        --prior_soft_slp (default 0)                Slope at which the weight of the Softmax prior decreases


    Saving options
        --decompile                                 Save decompiled code
        --save_name     (default "output/")         Name for saving the DRAM
        --save_it       (default -1)                Iteration at which to save the DRAM
        --print_err     (default -1)                Print mean error since last print
        --print_val     (default 10)                Print stat on the validation set
        --csv_name      (default "plot.csv")        Name of the csv in the save_name folder

    Other
        --seed          (default -1)                The random seed, -1 will use os.time
    ]],
    arg)

    if opt.sample and opt.biased_sample then
        print("What do you want? Unbiased or Biased samples? Make up your mind")
        os.exit(1)
    end

    return opt
end


-- Takes as input:
--  opt coming from trainer.parse_opt
--  config similar to what can be loaded from a examples/*.lua script
trainer.train = function(opt, config)
    -- Seeding of random generators
    if opt.seed == -1 then
        opt.seed = os.time()
    end
    print("Using "..opt.seed.." as random seed")
    torch.manualSeed(opt.seed)
    math.randomseed(opt.seed)

    if opt.biased_sample and type(config.gen_biased_sample)=="table" then
        config.gen_biased_sample = config.gen_biased_sample[opt.biased_sample_id]
    end

    -- Creating the directories where we need to write
    if not paths.dirp(opt.save_name) then
        paths.mkdir(opt.save_name)
    end
    if opt.decompile and not paths.dirp(opt.save_name .. "decompiled/") then
        paths.mkdir(opt.save_name .. "decompiled/")
    end

    local optimiser = optim.optimisers[opt.optim]
    local priors = {}
    if opt.prior_inf_ini > 0 then
        print("Using Infinity prior")
        priors[#priors+1] = optim.priors.Infinity_prior(config,
                                                       opt.prior_inf_ini, opt.prior_inf_slp)
    end
    if opt.prior_soft_ini> 0 then
        print("Using Softmaxed prior")
        priors[#priors+1] = optim.priors.Softmaxed_prior(config,
                                                       opt.prior_soft_ini, opt.prior_soft_slp)
    end

    local state = {
        learning_rate = opt.lr
    }

    -- Initialise the DRam, do we load, flash with a program or randomly initialise?
    local neural_ram
    if opt.dr == "" then
        neural_ram = layers.DRAM(config.nb_registers, config.memory_size)
        if not opt.rand_dr then
            neural_ram:flashHardProgram(config, opt.sharp)
        end
    else
        neural_ram = torch.load(opt.dr)
    end

    -- Load the criterion with the correct parameters
    local params = {
        alpha = opt.alpha,
        beta = opt.beta,
        gamma = opt.gamma,
        delta = opt.delta,
    }
    local criterion = layers.AlgCrit(params)

    -- Generate the validation data on which we evaluate the performance
    -- if we have requested to measure the performance
    local val_data, val_gt, val_mask
    if opt.val > 0 then
        val_data = torch.Tensor(opt.val, config.memory_size, config.memory_size)
        val_gt = torch.Tensor(opt.val, config.memory_size, config.memory_size)
        val_mask = torch.Tensor(opt.val, config.memory_size, config.memory_size)

        for i=1,opt.val do
            local init_mem, final_mem, loss_mask
            if opt.sample then
                init_mem, final_mem, loss_mask = config.gen_sample()
            elseif opt.biased_sample then
                init_mem, final_mem, loss_mask = config.gen_biased_sample()
            else
                init_mem, final_mem, loss_mask = config.example_input, config.example_output, config.example_loss_mask
            end
            val_data[i]:copy(init_mem)
            val_gt[i]:copy(final_mem)
            val_mask[i]:copy(loss_mask)
        end
    end
    -- If we want to work on a limited training set and not generate it on the fly
    -- during training, generate it now
    local train_data, train_gt, train_mask, train_sample_ordering
    if opt.train_size > 0 then
        train_sample_ordering = torch.randperm(opt.train_size)
        train_data = torch.Tensor(opt.train_size, config.memory_size, config.memory_size)
        train_gt = torch.Tensor(opt.train_size, config.memory_size, config.memory_size)
        train_mask = torch.Tensor(opt.train_size, config.memory_size, config.memory_size)

        for i=1,opt.train_size do
            local init_mem, final_mem
            if opt.sample then
                init_mem, final_mem, loss_mask = config.gen_sample()
            elseif opt.biased_sample then
                init_mem, final_mem, loss_mask = config.gen_biased_sample()
            else
                init_mem, final_mem, loss_mask = config.example_input, config.example_output, config.example_loss_mask
            end
            train_data[i]:copy(init_mem)
            train_gt[i]:copy(final_mem)
            train_mask[i]:copy(loss_mask)
        end
    end

    if opt.val_on_train then
        print("Using training set for evaluation")
        val_data = train_data:clone()
        val_gt = train_gt:clone()
        val_mask = train_mask:clone()
    end

    -- Initialise all the values / data structures that we use during training
    local csv_data_val = {}
    local fieldnames = {}
    table.insert(fieldnames, "it")
    table.insert(fieldnames, "val_err_time")
    table.insert(fieldnames, "val_err_response")
    table.insert(fieldnames, "val_nb_it")
    table.insert(fieldnames, "grad_sum")
    csv_data_val.fieldnames = fieldnames

    local runnning_err_time, runnning_err_response = 0, 0
    local running_nb_it = 0
    local err_time, err_response
    local div_running = (opt.print_err*opt.batch-1)
    local pos_in_training_set = 1

    local parameters_table, grad_parameters_table = neural_ram:parameters()

    -- Training Loop
    for it=0,opt.it do
        neural_ram:zeroGradParameters()

        -- Accumulate gradients for one batch
        for batch=1,opt.batch do
            -- Get sample
            -- Either from your fixed training set,
            -- or on the fly
            local init_mem, final_mem, loss_mask
            if opt.train_size>0 then  -- Fixed training set
                if pos_in_training_set > opt.train_size then
                    train_sample_ordering = torch.randperm(opt.train_size)
                    pos_in_training_set = 1
                end
                local next_sample = train_sample_ordering[pos_in_training_set]
                init_mem = train_data[next_sample]
                final_mem = train_gt[next_sample]
                loss_mask = train_mask[next_sample]
                pos_in_training_set = pos_in_training_set + 1
            else                      -- Dynamically generated training set
                if opt.sample then
                    init_mem, final_mem, loss_mask = config.gen_sample()
                elseif opt.biased_sample then
                    init_mem, final_mem, loss_mask = config.gen_biased_sample()
                else
                    init_mem, final_mem, loss_mask = config.example_input, config.example_output, config.example_loss_mask
                end
            end

            -- Do the forward pass
            -- This computes the loss, wrt this sample
            local outputs = neural_ram:forwardProgram(init_mem, opt.max_rec)
            err_time, err_response = criterion:forward(outputs, {final_mem, loss_mask})

            -- Print loss
            -- Useful to track convergence
            if batch==1 and opt.print_err == 0 then
                print("Train err at iteration "..it..":\t"..err_time.."\t"..err_response)
                print("Train nb of recurent step at iteration "..it..":\t"..#outputs)
            else
                if batch==1 and opt.print_err > 0 and it%opt.print_err == 0 then
                    print("Mean train err at iteration "..it..":\t"..runnning_err_time/div_running.."\t"..runnning_err_response/div_running)
                    print("Mean train nb of recurent step at iteration "..it..":\t"..running_nb_it/div_running)
                    runnning_err_time = 0
                    runnning_err_response = 0
                    running_nb_it = 0
                else
                    runnning_err_time = runnning_err_time + err_time
                    runnning_err_response = runnning_err_response + err_response
                    running_nb_it = running_nb_it + #outputs
                end
            end

            -- Compute the gradients by doing the backward pass
            -- They are automatically going to be accumulated into grad_parameters_table
            local gradOutputs = criterion:backward(outputs, {final_mem, loss_mask})
            neural_ram:backwardProgram(init_mem, gradOutputs)
        end

        -- Modify the gradients to also take into account our priors
        for idx, prior in ipairs(priors) do
            prior:update_gradients(parameters_table, grad_parameters_table)
        end


        -- We now have gradients in grad_parameters_table, and the parameters in parameters_table.
        optimiser.update_parameters(parameters_table, grad_parameters_table, state)
        -- Compute the sum of gradients for plotting
        local gradsum = 0
        for param_set=1, #parameters_table do
            gradsum = gradsum + grad_parameters_table[param_set]:norm()
        end

        -- If you have requested to measure the validation error along,
        -- Sum the errors over the whole validation set and write it down in a csv
        if opt.val > 0 and it%opt.print_val==0 then
            local val_err_time = 0
            local val_err_response = 0
            local val_nb_it = 0

            for i=1,opt.val do
                local outputs = neural_ram:forwardProgram(val_data[i], opt.max_rec)
                local err_time, err_response = criterion:forward(outputs, {val_gt[i], val_mask[i]})
                val_err_time = val_err_time + err_time
                val_err_response = val_err_response + err_response
                val_nb_it = val_nb_it + #outputs
            end
            table.insert(csv_data_val,
                {it,
                val_err_time/opt.val,
                val_err_response/opt.val,
                val_nb_it/opt.val,
                gradsum
            })
            pl_data.write(csv_data_val, opt.save_name..opt.csv_name, fieldnames, ',')
            print("Val results")
            print("Val err at iteration "..it..":\t"..val_err_time/opt.val.."\t"..val_err_response/opt.val)
            print("Val nb of recurent step at iteration "..it..":\t"..val_nb_it/opt.val)
        end

        -- If you have requested to dump the intermediary models,
        -- Write the weights down.
        -- Also potentially "decompile" the program and write its text version
        if opt.save_it > 0 and it%opt.save_it == 0 then
            local file_name = "it-"..string.format("%06d", it)..".t7"
            torch.save(opt.save_name .. file_name, neural_ram)
            if opt.decompile then
                decompiler.toASM_file(neural_ram, opt.save_name .. "decompiled/" .. file_name ..".dump")
            end
        end
    end

end

return trainer
