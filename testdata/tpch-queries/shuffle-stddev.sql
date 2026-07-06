select
    l_returnflag,
    l_linestatus,
    stddev(l_quantity) as std_qty
from lineitem
group by l_returnflag, l_linestatus;
