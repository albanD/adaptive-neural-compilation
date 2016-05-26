module NeulangTransformerAST(transformAST,
                             Statement(..),
                             InitialState(..),
                             Program(..)
                            )
       where
import NeulangParser ( StatementContent (..),
                     Variable (..),
                     Value (..))

junk_var_name = "junk_var"
junk_var = ExistingVariable junk_var_name


data Statement = Statement{target_variable :: String,
                           function_name   :: String,
                           arg_1           :: String,
                           arg_2           :: String
                          }
                 deriving (Eq, Show)

data InitialState = InitialState{ variable_name  :: String,
                                  initial_value  :: Int
                                }
                    deriving (Eq, Show)

data Program = Program { initial_states :: [InitialState],
                         statements :: [Statement]
                       }
               deriving (Eq)

instance Show Program where
  show program =
    (foldl (\acc x -> acc ++ "\n" ++ (show x)) "Initial States"  (initial_states program))
    ++ "\n\n" ++
    (foldl (\acc x -> acc ++ "\n" ++ (show x)) "Program States"   (statements program))


-- Used when a new variable is created,
-- Return 0 if we don't know the value initialised with,
-- otherwise returns the value
predictStatementValue :: StatementContent -> Int
predictStatementValue (EffectStatement (Constant str_value)) = read str_value :: Int
predictStatementValue (EffectStatement _) = 0 -- The default initial value is 0
predictStatementValue (AssigningStatement var state) = predictStatementValue state
predictStatementValue (LabeledStatement label state) = predictStatementValue state

-- The constant can only be given a constant value as initial state.
generateInstrStatement :: Variable -> StatementContent -> [Statement] -> [Statement]
generateInstrStatement _ (EffectStatement (Constant _ )) states = states
generateInstrStatement var (EffectStatement (NoArgFun fun_name)) states = (Statement (var_name var) fun_name junk_var_name junk_var_name):states
generateInstrStatement var (EffectStatement (SingleArgFun fun_name arg1)) states = (Statement (var_name var) fun_name arg1 junk_var_name):states
generateInstrStatement var (EffectStatement (DoubleArgFun fun_name arg1 arg2)) states = (Statement (var_name var) fun_name arg1 arg2):states


transformStatement :: Program -> StatementContent -> Program
transformStatement prog statement_cnt =
   case statement_cnt of LabeledStatement label_name state
                           -> transformStatement
                           (Program
                            ((InitialState label_name (length(statements prog))):(initial_states prog))
                            (statements prog))
                           state
                         AssigningStatement var state
                           -> Program (case var of ExistingVariable var_name
                                                   -- Assert that the variables was defined before?
                                                     -> (initial_states prog)
                                                   NewVariable var_name
                                                     -> ((InitialState var_name (predictStatementValue state)):(initial_states prog))
                                      )
                           (generateInstrStatement var state (statements prog))
                         EffectStatement value
                           -> Program
                           (initial_states prog)
                           (generateInstrStatement junk_var statement_cnt (statements prog))



transformAST::[StatementContent] -> Program
transformAST tree =
  let empty_program = Program [InitialState junk_var_name 0] []
  in  (\prog -> Program (initial_states prog) (reverse (statements prog))) (foldl transformStatement empty_program tree)
