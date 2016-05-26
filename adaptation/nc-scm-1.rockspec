package = "nc"
version = "scm-1"

source = {
	url = "https://github.com/albanD/adaptive-neural-compilation/tree/master/adaptation"
}

description = {
	summary = "Neural compiler for given algorithm and improving methods",
	homepage = "https://github.com/albanD/adaptive-neural-compilation/tree/master/adaptation",
}

dependencies = {
	"torch >= 7.0",
	"nn",
	"rnn",
  "penlight",
  "csvigo",
  "gnuplot",
  "autograd",
  "threads"
}

build = {
   type = "command",
   build_command = [[cmake -E make_directory build && cd build && cmake .. -DLUALIB=$(LUALIB) -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)]],
   install_command = "cd build && $(MAKE) install"
}
