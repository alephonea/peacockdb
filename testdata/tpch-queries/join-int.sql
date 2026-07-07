select
    c_nationkey,
    count(*) as cnt
from orders
join customer on o_custkey = c_custkey
group by c_nationkey;
