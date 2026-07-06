select
    l_returnflag,
    l_linestatus,
    count(*) as cnt,
    avg(l_quantity) as avg_qty,
    sum(l_quantity) as sum_qty
from lineitem
group by l_returnflag, l_linestatus;
