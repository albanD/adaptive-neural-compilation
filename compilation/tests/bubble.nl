var addr = 0;
var swapped = 1;
var always_zero = 0;

var first = READ(addr);
addr = INC(addr);
var second = READ(addr);
JEZ(second, should_terminate);
var diff = MIN(first, second);
diff = SUB(second, diff);
JEZ(diff, swap_elements);
JEZ(always_zero, always_zero);

swap_elements:diff = SUB(first, second);
JEZ(diff, always_zero);
WRITE(addr, first);
addr = DEC(addr);
WRITE(addr,second);
addr = INC(addr);
swapped = ZERO();
JEZ(always_zero, always_zero);

should_terminate:JEZ(swapped, re_init);
STOP();

re_init:addr=ZERO();
swapped=INC(swapped);
JEZ(always_zero, always_zero);