# TSP-D C++ reproduction: standard DP + A* third pass

This project implements a small, clarity-first C++ reproduction of the three-pass dynamic programming method for the Traveling Salesman Problem with Drone (TSP-D):

1. `DT(S,v,w)`: first pass, shortest truck paths.
2. `DOP(S,v,w)`: second pass, efficient operations with at most one drone node.
3. Third pass has two implementations:
   - `--third dp`: standard dynamic programming over states `(covered_set, truck_position)`.
   - `--third astar`: A* on the same state graph using an MST-based lower bound.

Node `0` is the depot. Truck travel time is Euclidean distance. Drone travel time is Euclidean distance divided by `alpha`, so `--alpha 2` means the drone is twice as fast as the truck.

## Build

```bash
g++ -O3 -std=c++17 tspd_dp.cpp -o tspd_dp
```

## Solve one instance

Standard DP:

```bash
./tspd_dp solve --n 10 --kind uniform --seed 0 --alpha 2 --k inf --third dp
```

A*:

```bash
./tspd_dp solve --n 10 --kind uniform --seed 0 --alpha 2 --k inf --third astar
```

Restricted operations:

```bash
./tspd_dp solve --n 12 --kind uniform --seed 0 --alpha 2 --k 0 --third astar
./tspd_dp solve --n 12 --kind uniform --seed 0 --alpha 2 --k 1 --third dp
```

Here `k` is the maximum number of truck-only nodes per operation. Use `--k inf` for the unrestricted version.

## Run reproduction experiments

```bash
./tspd_dp experiment \
  --ns 8,9,10 \
  --ks 0,1,2,inf \
  --thirds dp,astar \
  --kinds uniform \
  --seeds 0,1,2 \
  --alpha 2 \
  --out results.csv
```

The CSV columns include objective value, gap versus unrestricted, number of generated operations, number of reached/expanded states, and runtime split by pass.

## Notes

This is not the original optimized Java implementation. It is intended for learning and small-scale reproduction. Start with `n <= 10`. For larger cases, use small `k` such as `0` or `1`.
