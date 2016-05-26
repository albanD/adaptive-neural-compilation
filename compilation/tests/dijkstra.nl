var min = 0;
var argmin = 0;

var p_out = 0;
var p_out_temp = 0;
var p_in = 1;
var p_in_temp = 1;

var nnodes = 0;

var zero = 0;
var big = 99;

var tmp_node = 0;
var tmp_weight = 0;
var tmp_current = 0;
var tmp = 0;

var didsmth = 0;

p_out = READ(p_out);
p_out_temp = ADD(p_out, zero);

tmp_current = INC(zero);
loop_nnodes:tmp = READ(p_in_temp);
JEZ(tmp, found_nnodes);
WRITE(p_out_temp, big);
p_out_temp = INC(p_out_temp);
WRITE(p_out_temp, tmp_current);
p_out_temp = INC(p_out_temp);
p_in_temp = INC(p_in_temp);
nnodes = INC(nnodes);
JEZ(zero, loop_nnodes);

found_nnodes:WRITE(p_out, zero);
JEZ(zero, find_min);
min_return:p_in_temp = ADD(p_in, argmin);
p_in_temp = READ(p_in_temp);

loop_sons:tmp_node = READ(p_in_temp);
JEZ(tmp_node, find_min);
tmp_node = DEC(tmp_node);
p_in_temp = INC(p_in_temp);
tmp_weight = READ(p_in_temp);
p_in_temp = INC(p_in_temp);

p_out_temp = ADD(p_out, tmp_node);
p_out_temp = ADD(p_out_temp, tmp_node);
tmp_current = READ(p_out_temp);
tmp_weight = ADD(min, tmp_weight);

tmp = MIN(tmp_current, tmp_weight);
tmp = SUB(tmp_current, tmp);
JEZ(tmp, loop_sons);
WRITE(p_out_temp, tmp_weight);
JEZ(zero, loop_sons);


find_min:p_out_temp = DEC(p_out);
tmp_node = DEC(zero);
min = ADD(big, zero);
argmin = DEC(zero);

loop_min:p_out_temp = INC(p_out_temp);
tmp_node = INC(tmp_node);
tmp = SUB(tmp_node, nnodes);
JEZ(tmp, min_found);

tmp_weight = READ(p_out_temp);

p_out_temp = INC(p_out_temp);
tmp = READ(p_out_temp);
JEZ(tmp, loop_min);

tmp = MAX(min, tmp_weight);
tmp = SUB(tmp, tmp_weight);
JEZ(tmp, loop_min);
min = ADD(tmp_weight, zero);
argmin = ADD(tmp_node, zero);
JEZ(zero, loop_min);

min_found:tmp = SUB(min, big);
JEZ(tmp, end_of_prog);
p_out_temp = ADD(p_out, argmin);
p_out_temp = ADD(p_out_temp, argmin);
p_out_temp = INC(p_out_temp);
WRITE(p_out_temp, zero);
JEZ(zero, min_return);

end_of_prog:STOP();
