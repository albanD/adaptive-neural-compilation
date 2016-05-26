{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
module TorchNetWriter (torch_file_content) where
import Data.List (findIndex)
import Data.Maybe (fromJust)
import qualified Data.Text.Lazy as L
import qualified Data.Text.Lazy.Builder as TLB (toLazyText)
import NeulangTransformerAST (Statement(..),
                              InitialState(..),
                              Program(..))
import Text.Shakespeare.Text


instr_to_nb :: String -> Int
instr_to_nb "STOP" = 0
instr_to_nb "ZERO" = 1
instr_to_nb "INC" = 2
instr_to_nb "ADD" = 3
instr_to_nb "SUB" = 4
instr_to_nb "DEC" = 5
instr_to_nb "MIN" = 6
instr_to_nb "MAX" = 7
instr_to_nb "READ" = 8
instr_to_nb "WRITE" = 9
instr_to_nb "JEZ" = 10
instr_to_nb _ = error "Invalide Function"

find_register:: (Statement -> String) -> (Statement -> [InitialState] -> Int)
find_register field_to_check st init_states = fromJust(findIndex (\var -> (variable_name var)==(field_to_check st)) init_states)

get_initial_registers::[InitialState] -> [Int]
get_initial_registers = map (\st -> (initial_value st))

torch_table::[Int]->String
torch_table elts = "{" ++ (foldl (\acc x->acc++","++(show x)) (show $ head elts) (tail elts)) ++ "}"

get_registers_init_string::[InitialState] -> String
get_registers_init_string sts = torch_table $ get_initial_registers sts

get_registers_list::(Statement -> String) -> Program -> String
get_registers_list field_to_check (Program init_states stts) = let first_arg_finder = (\st -> find_register field_to_check st init_states)
                                                                 in torch_table $ (map (\st -> first_arg_finder st) stts)

get_instruction_string::[Statement] -> String
get_instruction_string stts = torch_table $ (map (\st -> instr_to_nb (function_name st)) stts)



torch_file_content::Program -> String
torch_file_content prog = let init_string = get_registers_init_string (initial_states prog)
                              first_arg_string = get_registers_list arg_1 prog
                              second_arg_string = get_registers_list arg_2 prog
                              target_string = get_registers_list target_variable prog
                              instruction_string = get_instruction_string (statements prog)
                          in L.unpack $ TLB.toLazyText $ $(textFile "src/DRam-template.lua") ""
