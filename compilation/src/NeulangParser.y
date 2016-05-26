{
  module NeulangParser (parse,
                        StatementContent(..),
                        Variable(..),
                        Value(..)) where
import qualified NeulangLexer as Lexer
}

%name rev_parse
%tokentype { Lexer.Token }
%error  { parseError }

%token
  "="          {Lexer.TAssign}
  var          {Lexer.TVarkw}
  ":"          {Lexer.TLabelSep}
  "("          {Lexer.TOpenParen}
  ")"          {Lexer.TCloseParen}
  ","          {Lexer.TSep}
  ";"          {Lexer.TEnd}
  num          {Lexer.TNum $$}
  id           {Lexer.TId  $$}
  FUN          {Lexer.TFun $$}
%%

statementList:: { [StatementContent]}
statementList: statement                                               { [$1]}
             | statementList statement                              { $2 : $1}

statement:: {StatementContent}
statement: statementContent ";"                                         { $1 }

statementContent:: {StatementContent}
statementContent: id ":" statementContent             {LabeledStatement $1 $3}
  | assignement_target "=" statementContent         {AssigningStatement $1 $3}
  | value                                                 {EffectStatement $1}

value:: {Value}
value: num                                                       {Constant $1}
  | FUN "(" ")"                                                  {NoArgFun $1}
  | FUN "(" id ")"                                        {SingleArgFun $1 $3}
  | FUN "(" id "," id ")"                              {DoubleArgFun $1 $3 $5}

assignement_target:: {Variable}
assignement_target: var id                                    {NewVariable $2}
  | id                                                   {ExistingVariable $1}



{

data StatementContent
  = LabeledStatement String StatementContent
  | AssigningStatement Variable StatementContent
  | EffectStatement Value
  deriving (Show, Eq)

data Value
  = Constant String
  | NoArgFun String
  | SingleArgFun String String
  | DoubleArgFun String String String
  deriving (Show, Eq)

data Variable
  = NewVariable {var_name::String}
  | ExistingVariable {var_name::String}
  deriving (Show, Eq)


parseError :: [Lexer.Token] -> a
parseError _ = error "Parse error"


-- We need to reverse the list due to the way the statement are concatenated
parse = reverse . rev_parse

}
