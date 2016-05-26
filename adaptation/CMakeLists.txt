CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

FIND_PACKAGE(Torch REQUIRED)

# Specify your lua code
# All ops
file(GLOB ops_files ops/*.lua)
file(GLOB optim_files optim/*.lua)
file(GLOB layers_files layers/*.lua)
file(GLOB decompiler_files decompiler/*.lua)
file(GLOB utils_files utils/*.lua)
SET(luasrc init.lua test.lua ops.lua ${ops_files} optim.lua ${optim_files} layers.lua ${layers_files} utils.lua ${utils_files} ${decompiler_files})

ADD_TORCH_PACKAGE(nc "" "${luasrc}")
