var head = 0;
var nb_jump = 1;
var out_write = 2;

head = READ(head);
nb_jump = READ(nb_jump);
out_write = READ(out_write);

nb_jump = ADD(nb_jump, nb_jump);
head = ADD(nb_jump, head);
head = INC(head);
head = READ(head);
WRITE(out_write, head);
STOP();