{
module NeulangLexer
(tokenize, Token(..)) where
}

%wrapper "basic"

$letter =        [a-z]
$upletter =      [A-Z]
$digit =        [0-9]
$idfirstchar =   [_ $letter]
$idchar =        [_ $letter $digit]


tokens :-
"="                                    { \s -> TAssign }
"var"                                  { \s -> TVarkw }
":"                                    { \s -> TLabelSep }
"("                                    { \s -> TOpenParen }
")"                                    { \s -> TCloseParen }
","                                    { \s -> TSep }
";"                                    { \s -> TEnd }
$digit+                                { \s -> TNum s }
$idfirstchar $idchar*                  { \s -> TId s }
$upletter+                             { \s -> TFun s }
$white+                                ;




{
data Token =
     TVarkw
     | TId String
     | TAssign
     | TNum String
     | TFun String
     | TLabelSep
     | TOpenParen
     | TCloseParen
     | TSep
     | TEnd
     deriving (Eq, Show)

tokenize :: String -> [Token]
tokenize s = alexScanTokens s
}
