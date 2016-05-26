var addr=1;
var always_zero=0;

var ref = READ(always_zero);
loop_start:var val = READ(addr);
val = SUB(val, ref);
JEZ(val, termination);
addr = INC(addr);
JEZ(always_zero,loop_start);

termination:WRITE(always_zero, addr);
STOP();