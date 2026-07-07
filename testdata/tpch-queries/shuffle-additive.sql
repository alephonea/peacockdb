select
    l_returnflag,
    l_linestatus,
    count(*) as cnt,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price
from lineitem
group by l_returnflag, l_linestatus;
