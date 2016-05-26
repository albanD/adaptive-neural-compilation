local lapp = require 'pl.lapp'
local nc = require 'nc'
local plotter = require('nc.utils')["plotter"]

local opt = lapp [[
Generating plot from the csv
Options
     --single-run                                    Generate plot for current csv and exit
     --path-to-csv  (default "output/plot.csv")      Path to the input csv
     --file-only                                     Do not create windows
     --output-path  (default "figure/")              Folder where to output graph
     --refresh-time (default 1)                      Plot frequency
]]

-- The infinite plotting loop
while true do
    pcall(plotter.plot, opt)
    sys.sleep(opt.refresh_time)
    if opt.single_run then
        break
    end
end

