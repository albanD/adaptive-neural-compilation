# Neural Compilation

This folder contains the deterministic compiler that transform any program into a configuration file for the [`adaptation`](../adaptation) part of this repo.

## Install

#### Haskell

The simplest solution to get the dependency would be to install [Haskell Platform](https://www.haskell.org/platform/).
Alternatively, if you want to get directly a binary of the compiler, you can find it on the [project page](http://www.robots.ox.ac.uk/~rudy/publications/2016-05-21-anc.html)


#### Compilation

Running `cabal install` should create the executable in `./dist/build/neulang-compiler/neulang-compiler`


## Usage

The input program should be provided in stdin to the executable.
For example to compile the dijkstra's implementation, run ` cat tests/dijkstra.nl | ./dist/build/neulang-compiler/neulang-compiler`.
This will print the compiled program to stdout and create a configuration file `./dram.lua` containing this program that can be used directly by the ANC code.
