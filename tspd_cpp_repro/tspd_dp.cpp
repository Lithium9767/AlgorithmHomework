#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace std;

static constexpr double INF = numeric_limits<double>::infinity();

struct Point { double x, y; };

struct Operation {
    uint32_t opmask{};      // nodes covered by this operation; includes end node, may include start if start==end
    int start{};
    int end{};
    double cost{};
    int drone{-1};          // -1 means no drone node
    uint32_t truck_mask{};  // nodes visited by truck in this operation, excluding start unless start==end
};

struct Pred {
    uint32_t prev_mask{};
    int prev_pos{};
    int op_index{-1};
    bool valid{false};
};

struct SolveResult {
    double value{INF};
    vector<Operation> path;
    size_t generated_ops{};
    size_t reached_states{};
    double time_first{}, time_second{}, time_third{}, time_total{};
};

static inline int popcount(uint32_t x) { return __builtin_popcount(x); }

vector<int> bits(uint32_t mask) {
    vector<int> out;
    while (mask) {
        uint32_t lsb = mask & -mask;
        int b = __builtin_ctz(mask);
        out.push_back(b);
        mask ^= lsb;
    }
    return out;
}

string mask_to_string(uint32_t mask) {
    ostringstream os;
    os << "[";
    bool first = true;
    for (int b : bits(mask)) {
        if (!first) os << ",";
        first = false;
        os << b;
    }
    os << "]";
    return os.str();
}

void gen_masks_rec(int n, int start, int left, uint32_t cur, vector<uint32_t>& res) {
    if (left == 0) { res.push_back(cur); return; }
    for (int i = start; i <= n - left; ++i) {
        gen_masks_rec(n, i + 1, left - 1, cur | (1u << i), res);
    }
}

vector<uint32_t> masks_of_size(int n, int size) {
    vector<uint32_t> res;
    if (size < 0 || size > n) return res;
    gen_masks_rec(n, 0, size, 0, res);
    return res;
}

vector<Point> generate_points(int n, const string& kind, int seed, double scale = 100.0) {
    mt19937 rng(seed);
    uniform_real_distribution<double> uni(0.0, scale);
    vector<Point> p;
    p.push_back({0.0, 0.0}); // depot at a corner

    if (kind == "uniform") {
        for (int i = 1; i < n; ++i) p.push_back({uni(rng), uni(rng)});
    } else if (kind == "1-center") {
        normal_distribution<double> nd(scale * 0.55, scale * 0.12);
        for (int i = 1; i < n; ++i) {
            double x = min(scale, max(0.0, nd(rng)));
            double y = min(scale, max(0.0, nd(rng)));
            p.push_back({x, y});
        }
    } else if (kind == "2-center") {
        normal_distribution<double> noise(0.0, scale * 0.09);
        Point c1{scale * 0.33, scale * 0.33}, c2{scale * 0.75, scale * 0.75};
        for (int i = 1; i < n; ++i) {
            Point c = (i % 2 ? c1 : c2);
            double x = min(scale, max(0.0, c.x + noise(rng)));
            double y = min(scale, max(0.0, c.y + noise(rng)));
            p.push_back({x, y});
        }
    } else {
        throw runtime_error("unknown kind: " + kind);
    }
    return p;
}

vector<vector<double>> distance_matrix(const vector<Point>& pts, double divisor = 1.0) {
    int n = (int)pts.size();
    vector<vector<double>> d(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double dx = pts[i].x - pts[j].x;
            double dy = pts[i].y - pts[j].y;
            d[i][j] = hypot(dx, dy) / divisor;
        }
    }
    return d;
}

struct DTTable {
    int n{};
    unordered_map<uint64_t, double> val;

    uint64_t key(uint32_t mask, int v, int w) const {
        return ((uint64_t)mask * (uint64_t)n + (uint64_t)v) * (uint64_t)n + (uint64_t)w;
    }
    double get(uint32_t mask, int v, int w) const {
        auto it = val.find(key(mask, v, w));
        return it == val.end() ? INF : it->second;
    }
    void set(uint32_t mask, int v, int w, double x) {
        if (isfinite(x)) val[key(mask, v, w)] = x;
    }
};

DTTable first_pass_truck_paths(const vector<vector<double>>& c, int max_mask_size) {
    int n = (int)c.size();
    DTTable dt;
    dt.n = n;
    if (max_mask_size < 0 || max_mask_size > n) max_mask_size = n;
    dt.val.reserve((size_t)min(1 << min(n, 20), 1 << 20) * (size_t)n);

    for (int v = 0; v < n; ++v) {
        for (int w = 0; w < n; ++w) {
            dt.set(1u << w, v, w, c[v][w]);
        }
    }

    for (int sz = 2; sz <= max_mask_size; ++sz) {
        for (uint32_t mask : masks_of_size(n, sz)) {
            auto nodes = bits(mask);
            for (int w : nodes) {
                uint32_t prev = mask ^ (1u << w);
                auto prev_nodes = bits(prev);
                for (int v = 0; v < n; ++v) {
                    double best = INF;
                    for (int u : prev_nodes) {
                        double prev_cost = dt.get(prev, v, u);
                        if (isfinite(prev_cost)) best = min(best, prev_cost + c[u][w]);
                    }
                    dt.set(mask, v, w, best);
                }
            }
        }
    }
    return dt;
}

pair<vector<Operation>, vector<vector<int>>> second_pass_operations(
    const DTTable& dt,
    const vector<vector<double>>& cd,
    int k
) {
    int n = dt.n;
    uint32_t all = (1u << n) - 1u;
    vector<Operation> ops;
    vector<vector<int>> by_start(n);

    auto add_op = [&](const Operation& op) {
        int idx = (int)ops.size();
        ops.push_back(op);
        by_start[op.start].push_back(idx);
    };

    int max_op_size = (k < 0 ? n : min(n, k + 2)); // end + up to k truck-only + one drone

    for (int start = 0; start < n; ++start) {
        for (int end = 0; end < n; ++end) {
            // Efficient no-drone operation: only direct/singleton end is needed.
            uint32_t singleton = 1u << end;
            double direct = dt.get(singleton, start, end);
            if (isfinite(direct)) {
                add_op(Operation{singleton, start, end, direct, -1, singleton});
            }

            // Operations with one drone node. We enumerate covered node sets S = opmask.
            for (int sz = 2; sz <= max_op_size; ++sz) {
                for (uint32_t opmask : masks_of_size(n, sz)) {
                    if (!(opmask & (1u << end))) continue;
                    if (start != end && (opmask & (1u << start))) continue;

                    double best = INF;
                    int best_drone = -1;
                    uint32_t best_truck = 0;

                    for (int d : bits(opmask)) {
                        if (d == start || d == end) continue;
                        uint32_t truck_mask = opmask ^ (1u << d);
                        if (!(truck_mask & (1u << end))) continue;
                        int truck_only = popcount(truck_mask) - 1; // exclude combined end/start node
                        if (k >= 0 && truck_only > k) continue;

                        double truck_time = dt.get(truck_mask, start, end);
                        if (!isfinite(truck_time)) continue;
                        double drone_time = cd[start][d] + cd[d][end];
                        double val = max(truck_time, drone_time);
                        if (val < best) {
                            best = val;
                            best_drone = d;
                            best_truck = truck_mask;
                        }
                    }
                    if (isfinite(best)) {
                        add_op(Operation{opmask, start, end, best, best_drone, best_truck});
                    }
                }
            }
        }
    }
    return {ops, by_start};
}

bool applicable(uint32_t mask, int pos, const Operation& op, int n, bool allow_final_return) {
    if (op.start != pos) return false;
    uint32_t allowed_overlap = (1u << pos) | (1u << op.end);
    if ((op.opmask & mask & ~allowed_overlap) != 0) return false;
    uint32_t newmask = mask | op.opmask;
    if (newmask == mask) {
        uint32_t all = (1u << n) - 1u;
        return allow_final_return && mask == all && op.end == 0 && pos != 0;
    }
    return true;
}

SolveResult third_pass_standard_dp(int n, const vector<Operation>& ops, const vector<vector<int>>& by_start, bool reconstruct) {
    auto t0 = chrono::steady_clock::now();
    uint32_t all = (1u << n) - 1u;
    size_t N = (size_t)(1u << n) * (size_t)n;
    vector<double> D(N, INF);
    vector<Pred> pred(reconstruct ? N : 0);
    auto id = [n](uint32_t mask, int pos) { return (size_t)mask * (size_t)n + (size_t)pos; };

    D[id(1u, 0)] = 0.0;

    for (int sz = 1; sz <= n; ++sz) {
        for (uint32_t mask : masks_of_size(n, sz)) {
            if (!(mask & 1u)) continue;
            for (int pos : bits(mask)) {
                double cur = D[id(mask, pos)];
                if (!isfinite(cur)) continue;
                for (int oi : by_start[pos]) {
                    const Operation& op = ops[oi];
                    if (!applicable(mask, pos, op, n, false)) continue;
                    uint32_t newmask = mask | op.opmask;
                    int newpos = op.end;
                    double val = cur + op.cost;
                    size_t nid = id(newmask, newpos);
                    if (val < D[nid]) {
                        D[nid] = val;
                        if (reconstruct) pred[nid] = Pred{mask, pos, oi, true};
                    }
                }
            }
        }
    }

    // Final return to depot may cover no new customer, so handle explicitly.
    double best = D[id(all, 0)];
    int best_pos = 0;
    int final_op = -1;
    for (int pos = 0; pos < n; ++pos) {
        double cur = D[id(all, pos)];
        if (!isfinite(cur)) continue;
        for (int oi : by_start[pos]) {
            const Operation& op = ops[oi];
            if (op.end == 0 && op.opmask == 1u) {
                double val = cur + op.cost;
                if (val < best) {
                    best = val;
                    best_pos = pos;
                    final_op = oi;
                }
            }
        }
    }

    vector<Operation> path;
    if (reconstruct && isfinite(best)) {
        uint32_t mask = all;
        int pos = best_pos;
        while (!(mask == 1u && pos == 0)) {
            Pred p = pred[id(mask, pos)];
            if (!p.valid) break;
            path.push_back(ops[p.op_index]);
            mask = p.prev_mask;
            pos = p.prev_pos;
        }
        reverse(path.begin(), path.end());
        if (final_op >= 0) path.push_back(ops[final_op]);
    }

    size_t reached = 0;
    for (double x : D) if (isfinite(x)) ++reached;
    auto t1 = chrono::steady_clock::now();
    SolveResult r;
    r.value = best;
    r.path = move(path);
    r.generated_ops = ops.size();
    r.reached_states = reached;
    r.time_third = chrono::duration<double>(t1 - t0).count();
    return r;
}

double mst_lower_bound(uint32_t mask, int pos, int n, const vector<vector<double>>& c, double alpha) {
    uint32_t all = (1u << n) - 1u;
    vector<int> nodes;
    uint32_t remain = all & ~mask;
    for (int x : bits(remain)) nodes.push_back(x);
    nodes.push_back(pos);
    nodes.push_back(0);
    sort(nodes.begin(), nodes.end());
    nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());
    int m = (int)nodes.size();
    if (m <= 1) return 0.0;

    vector<double> min_edge(m, INF);
    vector<char> used(m, 0);
    min_edge[0] = 0.0;
    double total = 0.0;
    for (int it = 0; it < m; ++it) {
        int v = -1;
        for (int i = 0; i < m; ++i) if (!used[i] && (v == -1 || min_edge[i] < min_edge[v])) v = i;
        used[v] = 1;
        total += min_edge[v];
        for (int to = 0; to < m; ++to) {
            if (!used[to]) min_edge[to] = min(min_edge[to], c[nodes[v]][nodes[to]]);
        }
    }
    return total / (2.0 + alpha);
}

SolveResult third_pass_astar(
    int n,
    const vector<Operation>& ops,
    const vector<vector<int>>& by_start,
    const vector<vector<double>>& c,
    double alpha,
    bool reconstruct
) {
    auto t0 = chrono::steady_clock::now();
    uint32_t all = (1u << n) - 1u;
    size_t N = (size_t)(1u << n) * (size_t)n;
    auto id = [n](uint32_t mask, int pos) { return (size_t)mask * (size_t)n + (size_t)pos; };

    vector<double> dist(N, INF);
    vector<Pred> pred(reconstruct ? N : 0);
    vector<char> closed(N, 0);

    struct Node { double f, g; uint32_t mask; int pos; };
    struct Cmp { bool operator()(const Node& a, const Node& b) const { return a.f > b.f; } };
    priority_queue<Node, vector<Node>, Cmp> pq;

    dist[id(1u, 0)] = 0.0;
    pq.push(Node{mst_lower_bound(1u, 0, n, c, alpha), 0.0, 1u, 0});

    size_t expanded = 0;
    double best = INF;
    uint32_t goal_mask = all;
    int goal_pos = 0;

    while (!pq.empty()) {
        Node cur = pq.top(); pq.pop();
        size_t sid = id(cur.mask, cur.pos);
        if (closed[sid]) continue;
        if (cur.g != dist[sid]) continue;
        closed[sid] = 1;
        ++expanded;

        if (cur.mask == all && cur.pos == 0) {
            best = cur.g;
            break;
        }

        for (int oi : by_start[cur.pos]) {
            const Operation& op = ops[oi];
            if (!applicable(cur.mask, cur.pos, op, n, true)) continue;
            uint32_t newmask = cur.mask | op.opmask;
            int newpos = op.end;
            size_t nid = id(newmask, newpos);
            if (closed[nid]) continue;
            double ng = cur.g + op.cost;
            if (ng < dist[nid]) {
                dist[nid] = ng;
                if (reconstruct) pred[nid] = Pred{cur.mask, cur.pos, oi, true};
                double h = mst_lower_bound(newmask, newpos, n, c, alpha);
                pq.push(Node{ng + h, ng, newmask, newpos});
            }
        }
    }

    vector<Operation> path;
    if (reconstruct && isfinite(best)) {
        uint32_t mask = goal_mask;
        int pos = goal_pos;
        while (!(mask == 1u && pos == 0)) {
            Pred p = pred[id(mask, pos)];
            if (!p.valid) break;
            path.push_back(ops[p.op_index]);
            mask = p.prev_mask;
            pos = p.prev_pos;
        }
        reverse(path.begin(), path.end());
    }

    auto t1 = chrono::steady_clock::now();
    SolveResult r;
    r.value = best;
    r.path = move(path);
    r.generated_ops = ops.size();
    r.reached_states = expanded;
    r.time_third = chrono::duration<double>(t1 - t0).count();
    return r;
}

SolveResult solve_tspd(const vector<Point>& pts, double alpha, int k, const string& third, bool reconstruct) {
    auto t0 = chrono::steady_clock::now();
    int n = (int)pts.size();
    auto c = distance_matrix(pts, 1.0);
    auto cd = distance_matrix(pts, alpha); // drone time = truck distance / alpha

    int max_dt_size = (k < 0 ? n : min(n, k + 1));

    auto a = chrono::steady_clock::now();
    DTTable dt = first_pass_truck_paths(c, max_dt_size);
    auto b = chrono::steady_clock::now();
    auto [ops, by_start] = second_pass_operations(dt, cd, k);
    auto cc = chrono::steady_clock::now();

    SolveResult r;
    if (third == "astar") {
        r = third_pass_astar(n, ops, by_start, c, alpha, reconstruct);
    } else if (third == "dp") {
        r = third_pass_standard_dp(n, ops, by_start, reconstruct);
    } else {
        throw runtime_error("third must be dp or astar");
    }
    auto d = chrono::steady_clock::now();

    r.generated_ops = ops.size();
    r.time_first = chrono::duration<double>(b - a).count();
    r.time_second = chrono::duration<double>(cc - b).count();
    r.time_total = chrono::duration<double>(d - t0).count();
    return r;
}

string op_to_string(const Operation& op) {
    ostringstream os;
    os << op.start << "->" << op.end
       << " cover=" << mask_to_string(op.opmask)
       << " cost=" << fixed << setprecision(3) << op.cost;
    if (op.drone >= 0) {
        os << " drone=" << op.drone << " truck_mask=" << mask_to_string(op.truck_mask);
    } else {
        os << " no_drone";
    }
    return os.str();
}

vector<string> split(const string& s, char sep) {
    vector<string> out;
    string item;
    stringstream ss(s);
    while (getline(ss, item, sep)) if (!item.empty()) out.push_back(item);
    return out;
}

int parse_k(const string& s) {
    if (s == "inf" || s == "INF" || s == "none" || s == "unrestricted") return -1;
    return stoi(s);
}

void usage() {
    cerr << "Usage:\n"
         << "  tspd_dp solve --n 10 --kind uniform --seed 0 --alpha 2 --k inf --third dp|astar\n"
         << "  tspd_dp experiment --ns 8,9,10 --ks 0,1,2,inf --thirds dp,astar --seeds 0,1,2 --out results.csv\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    string cmd = argv[1];

    int n = 10, seed = 0;
    string kind = "uniform";
    double alpha = 2.0;
    int k = -1;
    string third = "dp";
    string ns_s = "8,9,10", ks_s = "0,1,2,inf", thirds_s = "dp,astar", seeds_s = "0,1,2";
    string kinds_s = "uniform";
    string out = "results.csv";

    for (int i = 2; i < argc; ++i) {
        string a = argv[i];
        auto need = [&](const string& name) -> string {
            if (i + 1 >= argc) throw runtime_error("missing value for " + name);
            return argv[++i];
        };
        if (a == "--n") n = stoi(need(a));
        else if (a == "--seed") seed = stoi(need(a));
        else if (a == "--kind") kind = need(a);
        else if (a == "--alpha") alpha = stod(need(a));
        else if (a == "--k") k = parse_k(need(a));
        else if (a == "--third") third = need(a);
        else if (a == "--ns") ns_s = need(a);
        else if (a == "--ks") ks_s = need(a);
        else if (a == "--thirds") thirds_s = need(a);
        else if (a == "--seeds") seeds_s = need(a);
        else if (a == "--kinds") kinds_s = need(a);
        else if (a == "--out") out = need(a);
        else { cerr << "Unknown option: " << a << "\n"; usage(); return 1; }
    }

    try {
        if (cmd == "solve") {
            auto pts = generate_points(n, kind, seed);
            auto r = solve_tspd(pts, alpha, k, third, true);
            cout << fixed << setprecision(6);
            cout << "value=" << r.value << "\n";
            cout << "third=" << third << " k=" << (k < 0 ? string("inf") : to_string(k))
                 << " operations=" << r.path.size()
                 << " generated_ops=" << r.generated_ops
                 << " reached_or_expanded_states=" << r.reached_states << "\n";
            cout << "time_first=" << r.time_first << "s time_second=" << r.time_second
                 << "s time_third=" << r.time_third << "s total=" << r.time_total << "s\n";
            cout << "operation sequence:\n";
            for (size_t i = 0; i < r.path.size(); ++i) {
                cout << "  " << setw(2) << (i + 1) << ". " << op_to_string(r.path[i]) << "\n";
            }
        } else if (cmd == "experiment") {
            vector<int> ns, seeds, ks;
            for (auto& x : split(ns_s, ',')) ns.push_back(stoi(x));
            for (auto& x : split(seeds_s, ',')) seeds.push_back(stoi(x));
            for (auto& x : split(ks_s, ',')) ks.push_back(parse_k(x));
            vector<string> thirds = split(thirds_s, ',');
            vector<string> kinds = split(kinds_s, ',');

            ofstream f(out);
            f << "kind,n,seed,alpha,k,third,value,gap_vs_unrestricted_pct,generated_ops,reached_or_expanded_states,time_first,time_second,time_third,time_total\n";
            for (const string& kd : kinds) {
                for (int nn : ns) {
                    for (int sd : seeds) {
                        auto pts = generate_points(nn, kd, sd);
                        unordered_map<string, double> unrestricted;
                        // solve all; compute gap versus unrestricted for each third method
                        struct Row { string kind, third, kstr; int n, seed; double alpha, value; size_t ops, states; double t1,t2,t3,tt; };
                        vector<Row> rows;
                        for (const string& th : thirds) {
                            for (int kk : ks) {
                                cerr << "solving kind=" << kd << " n=" << nn << " seed=" << sd
                                     << " k=" << (kk < 0 ? string("inf") : to_string(kk))
                                     << " third=" << th << "\n";
                                auto r = solve_tspd(pts, alpha, kk, th, false);
                                string kstr = kk < 0 ? string("inf") : to_string(kk);
                                if (kk < 0) unrestricted[th] = r.value;
                                rows.push_back(Row{kd, th, kstr, nn, sd, alpha, r.value, r.generated_ops, r.reached_states,
                                                   r.time_first, r.time_second, r.time_third, r.time_total});
                            }
                        }
                        for (auto& row : rows) {
                            double gap = numeric_limits<double>::quiet_NaN();
                            if (unrestricted.count(row.third) && isfinite(unrestricted[row.third])) {
                                gap = (row.value - unrestricted[row.third]) / unrestricted[row.third] * 100.0;
                            }
                            f << row.kind << ',' << row.n << ',' << row.seed << ',' << row.alpha << ',' << row.kstr << ',' << row.third << ','
                              << setprecision(12) << row.value << ',' << gap << ',' << row.ops << ',' << row.states << ','
                              << row.t1 << ',' << row.t2 << ',' << row.t3 << ',' << row.tt << "\n";
                        }
                    }
                }
            }
            cout << "wrote " << out << "\n";
        } else {
            usage(); return 1;
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
