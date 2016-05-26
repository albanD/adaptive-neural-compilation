var head = 0;
var nb_jump = 1;
var out_write = 2;
var always_zero = 0;

head = READ(head);
nb_jump = READ(nb_jump);
out_write = READ(out_write);

loop_start:head = READ(head);
nb_jump = DEC(nb_jump);
JEZ(nb_jump, program_end);
JEZ(always_zero, loop_start);
program_end: head = INC(head);
head = READ(head);
WRITE(out_write, head);
STOP();
