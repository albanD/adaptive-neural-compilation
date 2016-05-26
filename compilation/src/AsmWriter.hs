module AsmWriter (write_asm) where
import Data.List (findIndex)
import Data.Maybe (fromJust)
import NeulangTransformerAST (Statement(..),
                              InitialState(..),
                              Program(..))

write_one_init_register::InitialState -> Int -> String
write_one_init_register st first = "R" ++ (show first) ++ " = " ++ (show (initial_value st)) ++ "\n"


write_initial_registers::[InitialState] -> Int -> String
write_initial_registers (st:sts) first_reg_idx = (write_one_init_register st first_reg_idx) ++ write_initial_registers sts (first_reg_idx+1)
write_initial_registers [] first_reg_idx = ""


find_registers:: (Statement -> String) -> (Statement -> [InitialState] -> Int)
find_registers field_to_check st init_states = 1 + fromJust(findIndex (\var -> (variable_name var)==(field_to_check st)) init_states)

write_one_statement::Statement -> [InitialState] -> String
write_one_statement st all_vars = "R" ++ (show (find_registers target_variable st all_vars)) ++
  " = " ++ (function_name st) ++ "(" ++
  "R" ++ (show (find_registers arg_1 st all_vars)) ++ ", "++
  "R" ++ (show (find_registers arg_2 st all_vars)) ++ ")"


write_asm_program::Program -> String
write_asm_program (Program initial_states (stt:stts)) = (write_one_statement stt initial_states) ++ "\n" ++ (write_asm_program (Program initial_states stts))
write_asm_program (Program initial_states []) = ""

write_asm::Program -> String
write_asm prog =
  write_initial_registers (initial_states prog) 1 ++ "\n\n" ++
  write_asm_program prog
