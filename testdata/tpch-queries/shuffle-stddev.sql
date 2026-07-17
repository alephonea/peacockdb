select
    l_returnflag,
    l_linestatus,
    stddev_samp(l_quantity) as std_samp_qty,
    stddev_pop(l_quantity) as std_pop_qty,
    var_samp(l_quantity) as var_samp_qty,
    var_pop(l_quantity) as var_pop_qty
from lineitem
group by l_returnflag, l_linestatus;
