local utils = require 'nc.utils'
local path = require 'pl.path'
local csvigo = require 'csvigo'
local gnuplot = require 'gnuplot'

local plotter = {}
utils["plotter"] = plotter


-- Keys that will be plotted
plotter.to_plot = {
    "val_err_response",
    "val_err_time",
    "val_nb_it",
    "grad_sum",
}

-- Utility functions
local to_number
to_number = function(table)
    for k,v in pairs(table) do
        if type(v) == "table" then
            table[k] = to_number(v)
        elseif type(v) == "string" then
            table[k] = tonumber(v)
        else
            error("Invalid content in table")
        end
    end
    return table
end
local to_tensor
to_tensor = function(table)
    if type(table) ~= "table" then
        return nil
    end
    local result = {}
    if #table > 0 then
        result = torch.Tensor(#table)
        for i=1,#table do
            result[i] = table[i]
        end
    else
        for k,v in pairs(table) do
            result[k] = to_tensor(v)
        end
    end
    return result
end

-- Take as input
--  The path to the csv
--  The output path
--  The file_only flag
plotter.plot = function(opt)
    if not path.exists(opt.output_path) then
        path.mkdir(opt.output_path)
    end

    local csv_data = csvigo.load{path=opt.path_to_csv, verbose=false}
    csv_data = to_number(csv_data)
    local data = to_tensor(csv_data)

    for id, name in ipairs(plotter.to_plot) do
        if torch.isTensor(data[name]) then
            if opt.file_only then
                gnuplot.pngfigure(opt.output_path..name..".png")
                gnuplot.title(name)
                gnuplot.plot(data[name])
                gnuplot.axis({"","",0,""})
                gnuplot.plotflush()
                gnuplot.close()
            else
                -- Update the figure
                gnuplot.figure(id)
                gnuplot.title(name)
                gnuplot.plot(data[name])
                gnuplot.axis({"","",0,""})
                if output_path ~= "" then
                    -- Plot to a file
                    gnuplot.figprint(opt.output_path..name..".png")
                end
            end
        end
    end
end

return plotter
