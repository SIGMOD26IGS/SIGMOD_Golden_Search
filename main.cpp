#pragma comment(linker, "/STACK:102400000,102400000")
#include "bits-stdc++.h"

using namespace std;

typedef pair <int, int> P;
typedef pair <double, int> Pd;


const int Maxn = 5e5 + 5;
const int Maxh = 25;
const int inf = 0x3f3f3f3f;
const int Maxk = 10;
const double Golden_ratio = 1.61803398875;
int K = 1;
int B = 20;
int Cas_num = 100;

struct tree{
    vector <int> edge[Maxn], heavy[Maxn];
    int root, sz[Maxn], dep[Maxn], son[Maxn], top[Maxn], fa[Maxn];
    void dfs1(int x){
        sz[x] = 1;
        if (x != root) dep[x] = dep[fa[x]] + 1;
        int ma = 0, heavy_son = x;
        for (int i = 0; i < edge[x].size(); i++){
            int y = edge[x][i];
            if (y == fa[x]) continue;
            dfs1(y);
            sz[x] += sz[y];
            if (sz[y] > ma){
                ma = sz[y], heavy_son = y;
            }
        }
        son[x] = heavy_son;
    }
    void dfs2(int x, int tp){
        top[x] = tp;
        heavy[tp].push_back(x);
        if (son[x] != x) dfs2(son[x], tp);
        for (int i = 0; i < edge[x].size(); i++){
            int y = edge[x][i];
            if (y == fa[x] || y == son[x]) continue;
            dfs2(y, y);
        }
    }
    void get_heavy(){
        dfs1(root);
        dfs2(root, root);
    }
};

struct graph{
    vector <int> edge[Maxn], anc[Maxn], des[Maxn], edge2[Maxn], Res;
    int n, m, root, du[Maxn], dep[Maxn], sz[Maxn], l[Maxn], par[Maxn];
    bool vis[Maxn], is_yes[Maxn], is_can[Maxn], visdp[Maxn];
    double p[Maxn], pr[Maxn], val[Maxn], g[Maxn], pyes[Maxn], pno[Maxn];
    double dp[Maxn][Maxh][Maxk], N[Maxn][Maxh][Maxk], g2[Maxn];
    double gyes[Maxn], gno[Maxn], sum_val[Maxn], fsum[Maxn], f1[Maxn];
    double Gyes[Maxn][3], Gno[Maxn][3], Pyes[Maxn][3], Pno[Maxn][3];
    bool ans[Maxn];
    tree heavy_tree;
    void input(){
        scanf("%d %d", &n, &m);
        //printf("%d %d\n", n, m);
        Cas_num = n;
        for (int i = 1; i <= n*5; i++) par[i] = 0;
        for (int i = 1; i <= m; i++){
            int x, y;
            scanf("%d%d", &x, &y);
            //printf("%d %d\n", x, y);
            if (par[y] != 0) continue;
            // if (edge[x].size() >= 2){
            //     n++;
            //     du[n]++;
            //     edge[n].push_back(edge[x][n&1]);
            //     par[edge[x][n&1]] = n;
            //     edge[n].push_back(y);
            //     par[y] = n;
            //     du[y]++;
            //     edge[x][n&1] = n;
            //     par[n] = x;
            // }else
            {
                edge[x].push_back(y);
                par[y] = x;
                du[y]++;
            }
        }
        //for (int i = 1; i <= n; i++) printf("%d %d\n", par[i], i);
        // for (int i = 1; i <= n; i++) 
        //     for (int j = 0; j < edge[i].size(); j++)
        //         printf("%d %d\n", i, edge[i][j]);
    }
    void set_root(int x){
        root = x;
        par[x] = -1;
    }
    void get_anc(){
        queue <int> q;
        for (int i = 1; i <= n; i++) dep[i] = inf;
        q.push(root); l[root] = dep[root] = 0;
        while(!q.empty()){
            int x = q.front();
            anc[x].push_back(x);
            des[x].push_back(x);
            q.pop();
            for (int i = 0; i < edge[x].size(); i++){
                int y = edge[x][i];
                du[y]--;
                //if (dep[y] != inf && dep[y] != dep[x] + 1) cout << dep[y] << " " << dep[x] + 1 << endl;
                l[y] = dep[y] = min(dep[y], dep[x] + 1);
                for (int j = 0; j < anc[x].size(); j++)
                    anc[y].push_back(anc[x][j]), des[anc[x][j]].push_back(y);
                if (du[y] == 0) q.push(y);
            }
        }
    }
    void dfs(int x){
        vis[x] = true;
        for (int i = 0; i < edge2[x].size(); i++){
            int y = edge2[x][i];
            if (vis[y]) continue;
            dfs(y);
            heavy_tree.edge[x].push_back(y);
        }
    }
    void get_heavy_tree(){
        heavy_tree.root = root;
        for (int i = 1; i <= n; i++){
            vector <P> tmp;
            vis[i] = false;
            for (int j = 0; j < edge[i].size(); j++){
                int y = edge[i][j];
                tmp.push_back(P(-des[y].size(), y));
            }
            sort(tmp.begin(), tmp.end());
            for (int j = 0; j < tmp.size(); j++){
                edge2[i].push_back(tmp[j].second);
            }
        }
        dfs(root);
        heavy_tree.get_heavy();
    }
    void update_ans(vector <int> T){
        for (int i = 1; i <= n; i++) ans[i] = false;
        for (int i = 0; i < T.size(); i++){
            int x = T[i];
            for (int j = 0; j < anc[x].size(); j++){
                ans[anc[x][j]] = true;
                //printf("%d\n", anc[x][j]);
            }
        }
    }
    void init(){
        input();
        //cout << "ok" << endl;
        set_root(1);
        get_anc();
        //cout << "ok" << endl;
        get_heavy_tree();
        //cout << "ok" << endl;
    }
    void init_cas(vector <int> T){
        update_ans(T);
        return ;
    }
    void init_question(int type){
        for (int i = 1; i <= n; i++){
            pyes[i] = Pyes[i][type], pno[i] = Pno[i][type],
            gyes[i] = Gyes[i][type], gno[i] = Gno[i][type];
            g2[i] = inf;
        }
        for (int i = 1; i <= n; i++){
            is_yes[i] = false, is_can[i] = true;
        }
        is_yes[root] = true;
    }
    void init_dp(int x){
        for (int h = l[root]; h <= l[x]; h++)
            for (int k = 0; k < Maxk; k++)
                dp[x][h][k] = N[x][h][k] = 0;
    }
    bool question(int x){
        return ans[x];
    }
    int question_IGS_dfs(int tmp, int &cnt){
        //if (cnt >= B) return tmp;
        //cout << tmp << endl;
        int len = heavy_tree.heavy[tmp].size();
        //cout << len << endl;
        int l = 0, r = len - 1;
        while(l < r){
            //cout << l << " " << r << endl;
            int mid = (l + r + 1) >> 1, mid_node = heavy_tree.heavy[tmp][mid];
            if (question(mid_node)) l = mid;
            else r = mid - 1;
            cnt++;
            //if (cnt >= B) return heavy_tree.heavy[tmp][l];
        }
        bool tag = true;
        for (int j = r; j >= 0; j--){
            int x = heavy_tree.heavy[tmp][j];
            //cout << x << endl;
            for (int i = 1; i < heavy_tree.edge[x].size(); i++){
                int y = heavy_tree.edge[x][i];
                cnt++;
                if (question(y)){
                    //cnt++;
                    //if (cnt >= B) return y;
                    return question_IGS_dfs(y, cnt);
                    tag = false;
                    //return;
                }
                //cnt++;
                //if (cnt >= B) return x;
            }
            if (j == r && tag){
                //Res.push_back(x);
                return x;
            }
            //cout << x << endl;
        }
    }
    int single_question_IGS(){
        int cnt = 0;
        //Res.clear();
        int x = question_IGS_dfs(root, cnt);
        //printf("%d\n", x);
        return cnt;
    }
    double single_calp(int x, int r){
        if (!is_can[x]) return 0;
        pyes[x] = 1.0 / n;
        double sum_r = 1 * (l[x] - l[r]);
        for (int i = 0; i < edge[x].size(); i++){
            int y = edge[x][i];
            if (!is_can[y]) continue;
            sum_r += single_calp(y, r);
            pyes[x] += pyes[y];
        }
        //cout << x << " " << pyes[x] << endl;
        return sum_r;
    }
    Pd BinG_calg(int x, double psum){
        double ma = min(pyes[x], psum - pyes[x]);
        int id = x;
        for (int i = 0; i < edge[x].size(); i++){
            int y = edge[x][i];
            if (!is_can[y]) continue;
            Pd tmp = BinG_calg(y, psum);
            if (tmp.first > ma){
                ma = tmp.first, id = tmp.second;
            }
        }
        return Pd(ma, id);
    }
    int single_question_BinG(){
        int r = root, cnt = 0;
        double psum = 1;
        for (int i = 1; i <= n; i++) is_can[i] = true;
        while (psum * n > 1.0 + 1e-8){
            cnt++;
            single_calp(r, r);
            int x = BinG_calg(r, psum).second;
            if (question(x)) {r = x, psum = pyes[r];}
            else {is_can[x] = false, psum = pyes[r] - pyes[x];}
            //cout << cnt << " " << x << " " << psum << endl;
        }
        return cnt;
    }
    Pd Golden_calg(int x, double psum){
        double ma = min(pyes[x], (psum - pyes[x]) / Golden_ratio);
        int id = x;
        for (int i = 0; i < edge[x].size(); i++){
            int y = edge[x][i];
            if (!is_can[y]) continue;
            Pd tmp = Golden_calg(y, psum);
            if (tmp.first > ma){
                ma = tmp.first, id = tmp.second;
            }
        }
        return Pd(ma, id);
    }
    int single_question_Golden(){
        int r = root, cnt = 0;
        double psum = 1;
        for (int i = 1; i <= n; i++) is_can[i] = true;
        while (psum * n > 1.0 + 1e-8){
            cnt++;
            single_calp(r, r);
            int x = Golden_calg(r, psum).second;
            if (question(x)) {r = x, psum = pyes[r];}
            else {is_can[x] = false, psum = pyes[r] - pyes[x];}
            //cout << cnt << " " << x << " " << psum << endl;
        }
        return cnt;
    }
    Pd method2_calg(int x, double sum_r, double psum, int r){
        double mi = inf;
        int id = 0;
        gyes[x] = 0, sum_val[x] = 1;
        for (int i = 0; i < edge[x].size(); i++){
            int y = edge[x][i];
            if (!is_can[y]) continue;
            Pd tmp = method2_calg(y, sum_r, psum, r);
            if (tmp.first < mi){
                mi = tmp.first, id = tmp.second;
            }
            gyes[x] += gyes[y], sum_val[x] += sum_val[y];
        }
        gyes[x] += sum_val[x] - 1;
        gno[x] = sum_r - gyes[x] - (l[x] - l[r]) * sum_val[x];
        double tmp = gyes[x] * pyes[x] + gno[x] * (psum - pyes[x]);
        //cout << x << " " << tmp << endl;
        if (tmp < mi){
            mi = tmp, id = x;
        }
        return Pd(mi, id);
    }
    int single_question_method2(){
        int r = root, cnt = 0;
        double psum = 1;
        for (int i = 1; i <= n; i++) is_can[i] = true;
        while (psum * n > 1.0 + 1e-8){
            cnt++;
            double sum_r = single_calp(r, r);
            int x = method2_calg(r, sum_r, psum, r).second;
            if (question(x)) {r = x, psum = pyes[r];}
            else {is_can[x] = false, psum = pyes[r] - pyes[x];}
            //cout << cnt << " " << x << " " << psum << endl;
        }
        return cnt;
    }
    double single_question_HGS_dfs(int x, double sum_max, int B, vector <int> &ans){
        double sum = pr[x];
        vis[x] = true;
        vector <Pd> tmpvec;
        for (int i = 0; i < edge[x].size(); i++){
            int y = edge[x][i];
            if (vis[y]) continue;
            double tmp = single_question_HGS_dfs(y, sum_max, B, ans);
            tmpvec.push_back(Pd(tmp, y));
            sum += tmp;
        }
        //if (sum > sum_max) cout << x << " " << sum << " " << ans.size() << endl;
        sort(tmpvec.begin(), tmpvec.end());
        for (int i = tmpvec.size() - 1; i >= 0; i--){
            if (sum <= sum_max || ans.size() > B) break;
            sum -= tmpvec[i].first;
            //cout << x << " " << sum << " " << ans.size() << endl;
            ans.push_back(tmpvec[i].second);
        }
        return sum;
    }
    int single_question_HGS(){
        double L = 0, R = 1;
        vector <int> ans;
        while (L + 1e-8 <= R){
            double M = (L + R) / 2;
            for (int i = 1; i <= n; i++) vis[i] = false;
            ans.clear();
            single_question_HGS_dfs(1, M, B, ans);
            //cout << M << endl;
            if (ans.size() <= B) R = M;
            else L = M;
        }
        int Ans = 1;
        for (int i = 0; i < ans.size(); i++){
            if (question(ans[i]) && dep[ans[i]] > dep[Ans]) Ans = ans[i];
        }
        return Ans;
    }
    // Pd Golden_calg(int x, int r, double psum){
    //     //if (r == 72) cout << r << " " << x << " " << psum << endl;
    //     double mi = (psum - pyes[x]) / psum, ma_pyes = 0;
    //     int id = x;
    //     vector <double> tmp_vec;
    //     for (int i = 0; i < edge[x].size(); i++){
    //         int y = edge[x][i];
    //         if (!is_can[y]) continue;
    //         tmp_vec.push_back(pyes[y]);
    //     }
    //     sort(tmp_vec.begin(), tmp_vec.end());
    //     // if (edge[x].size() > 5){
    //     //     mi = 1.01;
    //     // }else{
    //         for (int i = 0; i < edge[x].size(); i++){
    //             mi = max(mi, pow(tmp_vec[i] / psum, 1.0 / (edge[x].size() - i)));
    //         }
    //     //}
    //     if (x == r) mi = 1.01;
    //     for (int i = 0; i < edge[x].size(); i++){
    //         int y = edge[x][i];
    //         if (!is_can[y]) continue;
    //         Pd tmp = Golden_calg(y, r, psum);
    //         if (tmp.first < mi){
    //             mi = tmp.first, id = tmp.second;
    //         }
    //     }
    //     //if (x == r) cout << r << " " << mi << " " << id << endl;
    //     return Pd(mi, id);
    // }
    Pd Geo_calg(int x, int r, double psum){
        double mi = (psum - pyes[x]) / psum, ma_pyes = 0;
        int id = x;
        vector <double> tmp_g; 
        for (int i = 0; i < edge[x].size(); i++){
            int y = edge[x][i];
            if (!is_can[y]) continue;
            tmp_g.push_back(pyes[y]);
            //ma_pyes = max(ma_pyes, pyes[y]);
        }
        sort(tmp_g.begin(), tmp_g.end(), greater<double>());
        if (tmp_g.size() == 0) return Pd(mi, x);
        else if (tmp_g.size() == 1) mi = max(mi, tmp_g[0] / psum);
        else{
            for (int i = 0; i < tmp_g.size() - 2; i++){
                mi = max(mi, pow(tmp_g[i] / psum, (double) 1.0 / (i + 2)));
            }
            mi = max(mi, pow(max(tmp_g[tmp_g.size() - 2], tmp_g[tmp_g.size() - 1] + 1.0 / n) / psum, (double) 1.0 / tmp_g.size()));
        }
        if (x == r) mi = 1.01;
        for (int i = 0; i < edge[x].size(); i++){
            int y = edge[x][i];
            if (!is_can[y]) continue;
            Pd tmp = Geo_calg(y, r, psum);
            if (tmp.first < mi){
                mi = tmp.first, id = tmp.second;
            }
        }
        return Pd(mi, id);
    }
    int single_question_Geo(){
        int r = root, cnt = 0;
        double psum = 1;
        for (int i = 1; i <= n; i++) is_can[i] = true;
        while (psum * n > 1.0 + 1e-8){
            //cout << psum << " " << n << endl;
            cnt++;
            single_calp(r, r);
            int x = Geo_calg(r, r, psum).second;
            //cout << Golden_calg(r, r, psum).first << " " << x << endl;
            if (question(x)) {r = x, psum = pyes[r];}
            else {is_can[x] = false, psum = pyes[r] - pyes[x];}
        }
        return cnt;
    }
    void update_dist(int root, int dis[]){
        queue <P> q;
        q.push(P(root, 0));
        bool vis[Maxn] = {};
        vis[root] = true;
        dis[root] = 0;
        while(!q.empty()){
            P tmp = q.front();
            q.pop();
            int x = tmp.first, dep = tmp.second + 1;
            //cout << x << " " << dis[x] << endl;
            for (int i = 0; i < edge[x].size(); i++){
                int y = edge[x][i];
                if (vis[y] || dis[y] <= dep) continue;
                dis[y] = dep;
                q.push(P(y, dep));
                vis[y] = true;
            }
        }
    }
    double get_score(int root, int dis[]){
        queue <P> q;
        q.push(P(root, 0));
        bool vis[Maxn] = {};
        vis[root] = true;
        int score = 0;
        while(!q.empty()){
            P tmp = q.front();
            q.pop();
            int x = tmp.first, dep = tmp.second + 1;
            score += (dis[x] - dep) * p[x];
            for (int i = 0; i < edge[x].size(); i++){
                int y = edge[x][i];
                if (vis[y] || dis[y] <= dep) continue;
                q.push(P(y, dep));
                vis[y] = true;
            }
        }
        //cout << root << " " << score << endl;
        return score;
    }

    int get_score(vector <int> S, vector <int> T){
        int score = 0, reach_num = 0;
        queue <int> q;
        int vis[Maxn] = {};
        for (int i = 1; i <= n; i++) vis[i] = dep[i];
        for (int i = 0; i < S.size(); i++) q.push(S[i]), vis[S[i]] = 0;
        while(!q.empty()){
            int x = q.front();
            q.pop();
            for (int i = 0 ; i < edge[x].size(); i++){
                int y = edge[x][i];
                if (vis[y] > vis[x] + 1){
                    vis[y] = vis[x] + 1;
                    q.push(y);
                }
            }
        }
        for (int i = 0; i < T.size(); i++){
            int x = T[i];
            score += vis[x];
        }
        return score;
    }
    void solve_single(int x, int &score1, int &score2, int &score3, int &score4, int &score5,
                      double &t1, double &t2, double &t3, double &t4, double &t5){
        vector <int> vec;
        vec.push_back(x);
        update_ans(vec);
        clock_t t;
        t = clock();
        score1 = single_question_IGS();
        t1 = (clock() - t) * 1.0 / CLOCKS_PER_SEC;
        score2 = single_question_BinG();
        t2 = (clock() - t) * 1.0 / CLOCKS_PER_SEC - t1;
        score3 = single_question_method2();
        t3 = (clock() - t) * 1.0 / CLOCKS_PER_SEC - t2 - t1;
        score4 = single_question_Geo();
        t4 = (clock() - t) * 1.0 / CLOCKS_PER_SEC - t3 - t2 - t1;
        score5 = single_question_Golden();
        t5 = (clock() - t) * 1.0 / CLOCKS_PER_SEC - t4 - t3 - t2 - t1;
        // score4 = l[x] - l[single_question_HGS()];
        // t4 = (clock() - t) * 1.0 / CLOCKS_PER_SEC - t3 - t2 - t1;


        //printf("%.3f %.3f %.3f\n", t1, t2, t3);
        //question_method1();
        //int score2 = get_score(recommend2(), T);
        //vector <int> tmp_vec;
        //printf("%d %d %d %d %d\n", l[x], score1, score2, score3, score4);
        //cout << get_score(tmp_vec, T) << " " << score1 << " " << score2 << endl;
        return ;
    }
}G;
int main(int argc, char *argv[])
{
    //B = atoi(argv[2]);
    //Cas_num = atoi(argv[3]);
    char file_name[100], output_name[100];
    //sprintf(file_name, "%s.txt", argv[1]);
    //sprintf(output_name, "single_%s_%d_%d.out", argv[1], B, Cas_num);
    //printf("%s\n%s\n", file_name, output_name);
    //freopen("../ACM_CCS.txt", "r", stdin);
    freopen("../imgnet_v1.txt", "r", stdin);
    //freopen(file_name, "r", stdin);
    //freopen("../adder.txt", "r", stdin);
    freopen("../output.txt", "w", stdout);

    srand(19951107);
    G.init();
    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sumt1 = 0, sumt2 = 0, sumt3 = 0, sumt4 = 0, sumt5 = 0;
    int max1 = 0, max2 = 0, max3 = 0, max4 = 0, max5 = 0;
    Cas_num = min(1000, Cas_num - 1);
    for (int i = 2; i <= Cas_num + 1; i++){
        int score1, score2, score3, score4, score5;
        double t1, t2, t3, t4, t5;
        G.solve_single(i, score1, score2, score3, score4, score5, t1, t2, t3, t4, t5);
        //printf("%.2f\n", score);
        sum1 += score1;
        sum2 += score2;
        sum3 += score3;
        sum4 += score4;
        sum5 += score5;
        sumt1 += t1;
        sumt2 += t2;
        sumt3 += t3;
        sumt4 += t4;
        sumt5 += t5;
        max1 = max(max1, score1);
        max2 = max(max2, score2);
        max3 = max(max3, score3);
        max4 = max(max4, score4);
        max5 = max(max5, score5);
        printf("Case %d: %d %d %d %d %d\n", i, score1, score2, score3, score4, score5);
    }
    printf("%.2f %.2f %.2f %.2f %.2f\n", 1.0 * sum1 / Cas_num, 1.0 * sum2 / Cas_num, 1.0 * sum3 / Cas_num, 1.0 * sum4 / Cas_num, 1.0 * sum5 / Cas_num);
    printf("%d %d %d %d %d\n", max1, max2, max3, max4, max5);
    printf("%.4f %.4f %.4f %.4f %.4f\n", 1.0 * sumt1 / Cas_num, 1.0 * sumt2 / Cas_num, 1.0 * sumt3 / Cas_num, 1.0 * sumt4 / Cas_num, 1.0 * sumt5 / Cas_num);

    return 0;
}
/*
10
1 2
1 3
2 4
2 5
2 6
4 7
4 8
4 9
5 10
10
3
6
10
4
7
5
*/
