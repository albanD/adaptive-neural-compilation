import NeulangLexer (tokenize)
import NeulangParser (parse)
import NeulangTransformerAST (transformAST)
import AsmWriter (write_asm)
import TorchNetWriter (torch_file_content)
import System.IO

main::IO ()
main = do
  s <- getContents
  let tokens = tokenize(s)
  let ast = parse(tokens)
  let prog =  transformAST ast
  putStr (write_asm prog)
  writeFile "dram.lua" (torch_file_content prog)
