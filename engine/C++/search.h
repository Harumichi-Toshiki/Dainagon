#pragma once
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <iostream>
#include <thread>
#include <mutex>
#include <future>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <fstream>
#include <random>
#include <cstdlib>

#include "shogi_utils.h"
#include "vocab.h"
#include "dfpn.h"
#include <onnxruntime_cxx_api.h>

const float C_PUCT = 1.0f;
const int SEARCH_THREADS = 64;
const int FIXED_BATCH_SIZE = 64;
const float VIRTUAL_LOSS = 0.05f;

inline int get_action_index(const Move& m) {
    int dir_idx = -1;
    if (m.is_drop) {
        // PAWN=1, LANCE=2, KNIGHT=3, SILVER=4, GOLD=5, BISHOP=6, ROOK=7
        // これから1を引けば 0~6 になり、Python側の drop_dict と完全一致する
        dir_idx = 20 + (m.type - 1);
    }
    else {
        int from_x = m.from % 9;
        int from_y = m.from / 9;
        int to_x = m.to % 9;
        int to_y = m.to / 9;
        int dx = to_x - from_x;
        int dy = to_y - from_y;

        if (dy < 0 && dx == 0) dir_idx = 0;
        else if (dy < 0 && dx < 0 && dy == dx) dir_idx = 1;
        else if (dy < 0 && dx > 0 && -dy == dx) dir_idx = 2;
        else if (dy == 0 && dx < 0) dir_idx = 3;
        else if (dy == 0 && dx > 0) dir_idx = 4;
        else if (dy > 0 && dx == 0) dir_idx = 5;
        else if (dy > 0 && dx < 0 && dy == -dx) dir_idx = 6;
        else if (dy > 0 && dx > 0 && dy == dx) dir_idx = 7;
        else if (dy == -2 && dx == -1) dir_idx = 8;
        else if (dy == -2 && dx == 1) dir_idx = 9;

        // 成る場合は +10
        if (m.is_promote) {
            dir_idx += 10;
        }
    }

    // 万が一不正な手なら -1 を返す
    if (dir_idx < 0 || dir_idx > 26) return -1;

    // [方向(27) × マス(81)] の1次元配列のインデックスを返す
    return dir_idx * 81 + m.to;
}

struct SearchResult {
    std::string best;
    std::string ponder;
};

void dump_features(const std::vector<Ort::Float16_t>& features, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (ofs) {
        ofs.write(reinterpret_cast<const char*>(features.data()), features.size() * sizeof(Ort::Float16_t));
    }
}

inline void atomic_add_float(std::atomic<float>& atom, float val) {
    float current = atom.load(std::memory_order_relaxed);
    while (!atom.compare_exchange_weak(current, current + val, std::memory_order_relaxed));
}

using EvalResult = std::pair<std::shared_ptr<const std::vector<float>>, float>;

class BatchEvaluator {
private:
    Ort::Session* session;
    std::queue<std::pair<const std::vector<Ort::Float16_t>*, std::promise<EvalResult>>> queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<int> queue_count{ 0 };
    bool running = true;
    std::thread worker;
    const char* input_names[1] = { "input" };
    const char* output_names[2] = { "policy", "value" };

public:
    BatchEvaluator(Ort::Session* sess) : session(sess) {
        worker = std::thread(&BatchEvaluator::loop, this);
    }
    ~BatchEvaluator() {
        running = false;
        cv.notify_all();
        if (worker.joinable()) worker.join();
    }

    std::future<EvalResult> predict_async(const std::vector<Ort::Float16_t>* features) {
        std::promise<EvalResult> prom;
        auto fut = prom.get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            queue.push({ features, std::move(prom) }); // ★ポインタ（8バイト）を渡すだけで済む
            queue_count.fetch_add(1, std::memory_order_release);
        }
        cv.notify_one();
        return fut;
    }

private:
    void loop() {
        std::vector<std::pair<const std::vector<Ort::Float16_t>*, std::promise<EvalResult>>> batch;
        batch.reserve(FIXED_BATCH_SIZE);

        std::vector<Ort::Float16_t> input_flat;
        input_flat.reserve(FIXED_BATCH_SIZE * 55 * 81);
        std::vector<Ort::Float16_t> zeros(55 * 81, Ort::Float16_t(0.0f));

        while (running) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv.wait_for(lock, std::chrono::microseconds(500), [this] {
                return !running || queue_count.load(std::memory_order_acquire) >= FIXED_BATCH_SIZE;
                });

            if (!running && queue_count.load() == 0) break;

            while (!queue.empty() && batch.size() < FIXED_BATCH_SIZE) {
                batch.push_back(std::move(queue.front()));
                queue.pop();
                queue_count.fetch_sub(1, std::memory_order_release);
            }
            lock.unlock();

            if (batch.empty()) continue;

            try {
                size_t actual_bs = batch.size();
                input_flat.clear();

                for (const auto& req : batch) {
                    input_flat.insert(input_flat.end(), req.first->begin(), req.first->end());
                }

                while (input_flat.size() < FIXED_BATCH_SIZE * 55 * 81) {
                    input_flat.insert(input_flat.end(), zeros.begin(), zeros.end());
                }

                std::vector<int64_t> input_shape = { FIXED_BATCH_SIZE, 55, 9, 9 };
                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                    memory_info, input_flat.data(), input_flat.size(), input_shape.data(), input_shape.size()
                );

                auto output_tensors = session->Run(
                    Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 2
                );

                Ort::Float16_t* policy_ptr = output_tensors[0].GetTensorMutableData<Ort::Float16_t>();
                Ort::Float16_t* value_ptr = output_tensors[1].GetTensorMutableData<Ort::Float16_t>();
                size_t policy_dim = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1];

                for (size_t i = 0; i < actual_bs; ++i) {
                    std::vector<float> p(policy_dim);
                    for (size_t j = 0; j < policy_dim; ++j) {
                        p[j] = policy_ptr[i * policy_dim + j].ToFloat();
                    }

                    float max_logit = -999999.0f;
                    for (float val : p) {
                        if (val > max_logit) max_logit = val;
                    }
                    float sum_exp = 0.0f;
                    for (float& val : p) {
                        val = std::exp(val - max_logit);
                        sum_exp += val;
                    }
                    for (float& val : p) {
                        val /= sum_exp;
                    }

                    auto p_ptr = std::make_shared<const std::vector<float>>(std::move(p));
                    batch[i].second.set_value({ p_ptr, value_ptr[i].ToFloat() });
                }
            }
            catch (...) {
                for (auto& req : batch) try { req.second.set_value({ nullptr, 0.0f }); }
                catch (...) {}
            }
            batch.clear();
        }
    }
};

struct Node {
    std::atomic<int> visit_count{ 0 };
    std::atomic<float> value_sum{ 0.0f };
    float policy_prob = 0.0f;
    float raw_value = 0.5f;
    Move move_obj;
    std::atomic<bool> is_expanded{ false };
    std::atomic<bool> is_evaluating{ false };
    std::atomic<bool> has_noise{ false };

    // ★ ベクトルを廃止し、数珠繋ぎポインタに変更
    Node* first_child = nullptr;
    Node* sibling = nullptr;

    Node() {}
    Node(float p, Move m) : policy_prob(p), move_obj(m) {}
    ~Node() {}

    void init(float p, Move m) {
        visit_count.store(0, std::memory_order_relaxed);
        value_sum.store(0.0f, std::memory_order_relaxed);
        policy_prob = p;
        raw_value = 0.5f;
        move_obj = m;
        is_expanded.store(false, std::memory_order_relaxed);
        is_evaluating.store(false, std::memory_order_relaxed);
        has_noise.store(false, std::memory_order_relaxed);

        first_child = nullptr;
        sibling = nullptr;
    }

    float get_score(int total_parent_visits, float fpu_value) {
        int n = visit_count.load(std::memory_order_relaxed);
        float v = value_sum.load(std::memory_order_relaxed);
        float q = (n == 0) ? fpu_value : (v / n);

        float c_base = 19652.0f;
        float c_init = 1.25f;
        float dynamic_c_puct = c_init + std::log((total_parent_visits + c_base + 1.0f) / c_base);

        float puct = dynamic_c_puct * policy_prob * (std::sqrt((float)total_parent_visits) / (1 + n));
        return q + puct;
    }
};

class NodeAllocator {
private:
    Node* pool;
    std::atomic<size_t> counter{ 0 };
    size_t capacity;
    std::mutex fallback_mtx;
    std::vector<std::unique_ptr<Node>> fallback_nodes;

public:
    NodeAllocator(size_t cap = 30000000) : capacity(cap) {
        pool = (Node*)std::malloc(capacity * sizeof(Node));
        for (size_t i = 0; i < capacity; ++i) {
            new (&pool[i]) Node();
        }
    }
    ~NodeAllocator() {
        for (size_t i = 0; i < capacity; ++i) pool[i].~Node();
        std::free(pool);
    }

    Node* alloc(float p, Move m) {
        size_t idx = counter.fetch_add(1, std::memory_order_relaxed);
        if (idx < capacity) {
            pool[idx].init(p, m);
            return &pool[idx];
        }
        Node* n = new Node();
        n->init(p, m);
        std::lock_guard<std::mutex> lock(fallback_mtx);
        fallback_nodes.emplace_back(n);
        return n;
    }

    void reset() {
        counter.store(0, std::memory_order_relaxed);
        fallback_nodes.clear();
    }
};

const int MCTS_TT_SIZE = 131072;
struct MctsTTEntry {
    uint64_t key = 0;
    float value = 0.0f;
    std::shared_ptr<const std::vector<float>> policy;
};

class MctsSearcher {
public:
    Node* root = nullptr;
    BatchEvaluator* evaluator = nullptr;
    std::atomic<bool> searching{ false };
    std::chrono::steady_clock::time_point start_time;
    int time_limit_ms = 0;
    int initial_visits = 0;
    std::atomic<int> current_target_ms{ 0 };
    std::atomic<int> current_max_ms{ 0 };
    std::atomic<bool> is_ponder_mode{ false };

    std::vector<MctsTTEntry> nn_cache;
    std::atomic_flag cache_lock[1024];
    DfpnSearcher* dfpn = nullptr;

    DfpnSearcher* anti_dfpn = nullptr;
    std::atomic<bool> anti_searching{ true };

    NodeAllocator allocators[2];
    int current_alloc_idx = 0;

    NodeAllocator& current_alloc() { return allocators[current_alloc_idx]; }
    NodeAllocator& next_alloc() { return allocators[1 - current_alloc_idx]; }

    MctsSearcher() {
        nn_cache.resize(MCTS_TT_SIZE);
        for (int i = 0; i < 1024; ++i) cache_lock[i].clear();
        dfpn = new DfpnSearcher(&searching);
        anti_dfpn = new DfpnSearcher(&anti_searching);
    }
    ~MctsSearcher() {
        if (root) delete root; // poolの再利用時には注意が必要ですが今回は問題なし
        if (dfpn) delete dfpn;
    }

    void set_evaluator(BatchEvaluator* ev) { evaluator = ev; }

    void clear_cache() {
        for (int i = 0; i < MCTS_TT_SIZE; ++i) {
            nn_cache[i].key = 0;
            nn_cache[i].policy.reset();
        }
    }

    void reset() {
        current_alloc().reset();
        Move null_move = { -1, -1, 0, false, false };
        root = current_alloc().alloc(1.0f, null_move);
    }

    Node* clone_tree(Node* old_node, NodeAllocator& target_alloc) {
        if (!old_node) return nullptr;
        Node* new_n = target_alloc.alloc(old_node->policy_prob, old_node->move_obj);

        int visits = old_node->visit_count.load(std::memory_order_relaxed);
        new_n->visit_count.store(visits, std::memory_order_relaxed);
        new_n->value_sum.store(old_node->value_sum.load(std::memory_order_relaxed), std::memory_order_relaxed);
        new_n->raw_value = old_node->raw_value;
        new_n->has_noise.store(old_node->has_noise.load(std::memory_order_relaxed), std::memory_order_relaxed);

        // ★ ここがNPS低下を救う究極のガベージコレクション
        // 訪問回数が「1回以下（＝一度展開されただけで深く読まれていない）」のノードは、
        // 80個以上ある子供ノードをコピーせず、ただの「未展開の葉ノード」に戻します。
        // これにより、強さに直結する本筋の読みだけを残し、消費メモリを90%以上削減します。
        if (visits < 2) {
            new_n->is_expanded.store(false, std::memory_order_relaxed);
            new_n->is_evaluating.store(false, std::memory_order_relaxed);
            new_n->first_child = nullptr;
            return new_n;
        }

        new_n->is_expanded.store(old_node->is_expanded.load(std::memory_order_relaxed), std::memory_order_relaxed);
        new_n->is_evaluating.store(false, std::memory_order_relaxed);

        Node* curr_old_child = old_node->first_child;
        Node* prev_new_child = nullptr;

        while (curr_old_child != nullptr) {
            Node* new_child = clone_tree(curr_old_child, target_alloc);
            if (new_child != nullptr) {
                if (prev_new_child == nullptr) {
                    new_n->first_child = new_child;
                }
                else {
                    prev_new_child->sibling = new_child;
                }
                prev_new_child = new_child;
            }
            curr_old_child = curr_old_child->sibling;
        }
        return new_n;
    }

    /*
    void advance_root(const std::string& move_usi) {
        if (!root) {
            reset();
            return;
        }
        Node* next_root = nullptr;
        for (Node* c = root->first_child; c != nullptr; c = c->sibling) {
            if (move_to_usi(c->move_obj) == move_usi) {
                next_root = c;
                break;
            }
        }

        if (next_root) {
            NodeAllocator& target = next_alloc();
            target.reset();
            root = clone_tree(next_root, target);

            current_alloc_idx = 1 - current_alloc_idx;
            next_alloc().reset();
        }
        else {
            reset();
        }
    }*/
    void advance_root(const std::string& move_usi) {
        // ツリーのコピー（引越し）による容量圧迫と速度低下を防ぐため、
        // 毎回ツリーをまっさらに消去（リセット）します。
        // nn_cache（評価値キャッシュ）が生きているため、強さは全く落ちません！
        reset();
    }


    void print_info() {
        if (!root || root->visit_count == 0 || root->first_child == nullptr) return;

        std::string pv_str = "";
        Node* curr = root;
        int depth = 0;

        while (curr && curr->first_child != nullptr) {
            if (depth > 30) break;
            Node* best = nullptr;
            int max_n = -1;
            for (Node* c = curr->first_child; c != nullptr; c = c->sibling) {
                if (c->visit_count > max_n) { max_n = c->visit_count; best = c; }
            }
            if (!best || best->visit_count == 0) break;
            pv_str += move_to_usi(best->move_obj) + " ";
            curr = best;
            depth++;
        }

        Node* best_child = nullptr;
        int max_visits = -1;
        for (Node* child = root->first_child; child != nullptr; child = child->sibling) {
            if (child->visit_count > max_visits) {
                max_visits = child->visit_count;
                best_child = child;
            }
        }

        if (best_child) {
            float win_rate = best_child->value_sum / (float)best_child->visit_count;
            win_rate = (std::max)(0.0001f, (std::min)(0.9999f, win_rate));
            int score = int(-800.0f * std::log(1.0f / win_rate - 1.0f));

            long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
            long long added_nodes = root->visit_count - initial_visits;
            long long nps = (elapsed > 0) ? (added_nodes * 1000 / elapsed) : 0;

            std::cout << "info depth " << depth << " nodes " << root->visit_count
                << " nps " << nps << " score cp " << score
                << " time " << elapsed << " pv " << pv_str << std::endl;
        }
    }

    void add_dirichlet_noise(Node* root_node) {
        if (!root_node || root_node->first_child == nullptr) return;

        float alpha = 0.15f;
        float epsilon = 0.25f;

        std::mt19937 rng(std::random_device{}());
        std::gamma_distribution<float> gamma(alpha, 1.0f);

        int child_count = 0;
        for (Node* c = root_node->first_child; c != nullptr; c = c->sibling) child_count++;

        std::vector<float> noise;
        float noise_sum = 0.0f;
        for (int i = 0; i < child_count; ++i) {
            float n = gamma(rng);
            noise.push_back(n);
            noise_sum += n;
        }

        int i = 0;
        for (Node* c = root_node->first_child; c != nullptr; c = c->sibling) {
            float n = noise[i] / noise_sum;
            c->policy_prob = (1.0f - epsilon) * c->policy_prob + epsilon * n;
            i++;
        }
    }

    SearchResult search(Board& current_board, int target_ms, int max_ms, bool ponder) {
        is_ponder_mode.store(ponder);
        current_target_ms.store(target_ms);
        current_max_ms.store(max_ms);
        start_time = std::chrono::steady_clock::now();

        // 1. 合法手をスタック配列に生成
        Move root_moves[MAX_LEGAL_MOVES];
        Move* root_end = generate_moves(current_board.bit_pos, root_moves);

        // 合法手がない場合は投了
        if (root_moves == root_end) {
            std::cout << "info string I have no legal moves. Resigning." << std::endl;
            return { "resign", "" };
        }

        // 2. 1手詰み（即詰み）チェックのループ（ポインタ方式）
        for (Move* m = root_moves; m != root_end; ++m) {
            StateInfo st;
            current_board.push_move(*m, st);

            Move next_moves[MAX_LEGAL_MOVES];
            // 相手に合法手がない（＝詰み）かチェック
            if (generate_moves(current_board.bit_pos, next_moves) == next_moves) {
                std::string usi_str = move_to_usi(*m);
                std::cout << "info depth 1 score mate 1 pv " << usi_str << std::endl;

                current_board.undo_move(*m, st); // 忘れずに戻す

                while (searching.load()) {
                    if (!is_ponder_mode.load()) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                return { usi_str, "" };
            }
            current_board.undo_move(*m, st);
        }

        Move null_move = { -1, -1, 0, false, false };
        if (!root) root = current_alloc().alloc(1.0f, null_move);
        initial_visits = root->visit_count.load();
        time_limit_ms = max_ms;

        // 3. ルートノードの初期展開
        if (!root->is_expanded.load()) {
            std::vector<Ort::Float16_t> features(55 * 81, Ort::Float16_t(0.0f));
            make_features(current_board, features);
            auto result = evaluator->predict_async(&features).get();

            // ★ 引数を5つ（root_endを含む）渡す
            expand_node_with_moves(root, current_board, root_moves, root_end, result.first);

            root->raw_value = result.second;
            root->is_expanded.store(true);
        }

        bool expected = false;
        if (!is_ponder_mode.load() && root->has_noise.compare_exchange_strong(expected, true)) {
            add_dirichlet_noise(root);
        }

        std::vector<std::thread> threads;
        for (int i = 0; i < SEARCH_THREADS; ++i) {
            threads.emplace_back(&MctsSearcher::worker, this, current_board);
        }

        std::string mate_move = "";
        std::thread dfpn_thread([&]() {
            if (!current_board.in_check(current_board.turn)) {
                if (dfpn->search_mate(current_board)) {
                    mate_move = dfpn->get_best_mate();
                    if (!is_ponder_mode.load()) {
                        searching = false;
                    }
                }
            }
            });

        auto last_print_time = start_time;

        while (searching) {
            bool expected_noise = false;
            if (!is_ponder_mode.load() && root->has_noise.compare_exchange_strong(expected_noise, true)) {
                add_dirichlet_noise(root);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            auto now = std::chrono::steady_clock::now();
            auto elapsed = (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

            if (!is_ponder_mode.load()) {
                if (elapsed >= current_max_ms.load()) { searching = false; break; }

                int n1 = 0, n2 = 0;
                for (Node* c = root->first_child; c != nullptr; c = c->sibling) {
                    int v = c->visit_count.load(std::memory_order_relaxed);
                    if (v > n1) { n2 = n1; n1 = v; }
                    else if (v > n2) { n2 = v; }
                }

                if (n1 > 0) {
                    float t = (float)current_target_ms.load();
                    if (t < 500) {
                        if (elapsed >= t) { searching = false; break; }
                    }
                    else {
                        if (elapsed >= t * 0.5f && n1 >= n2 * 1.5f) { searching = false; break; }
                        if (elapsed >= t * 0.9f && n1 >= n2 * 1.2f) { searching = false; break; }
                        if (elapsed >= t * 1.1f && n1 >= n2 * 1.05f) { searching = false; break; }
                        if (elapsed >= t * 1.5f) { searching = false; break; }
                    }
                }
            }

            auto ms_since_print = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_time).count();
            if (ms_since_print >= 500) {
                print_info();
                last_print_time = now;
            }
        }
        print_info();

        for (auto& t : threads) t.join();

        searching = false;
        if (dfpn_thread.joinable()) dfpn_thread.join();

        if (!mate_move.empty()) {
            std::vector<std::string> pv = dfpn->get_mate_pv(current_board);
            std::string pv_str = "";
            for (const auto& m : pv) pv_str += m + " ";
            int mate_ply = pv.empty() ? 1 : pv.size();
            std::cout << "info depth " << mate_ply << " score mate " << mate_ply << " pv " << pv_str << std::endl;
            return { mate_move, "" };
        }

        Node* b1 = nullptr;

        if (root->first_child != nullptr) {
            // 1. 候補手を「訪問回数順」に並べる
            std::vector<Node*> candidates;
            for (Node* c = root->first_child; c != nullptr; c = c->sibling) {
                candidates.push_back(c);
            }
            std::sort(candidates.begin(), candidates.end(), [](Node* a, Node* b) {
                return a->visit_count.load() > b->visit_count.load();
                });

            // 2. 上位3手について「指した瞬間に自玉が詰まないか」をDF-PNで検問
            for (size_t i = 0; i < candidates.size(); ++i) {
                Node* top_node = candidates[i];
                Board next_board = current_board;
                StateInfo st;
                next_board.push_move(top_node->move_obj, st); // 候補手を指してみる

                anti_searching.store(true);
                anti_dfpn->set_max_nodes(50000); // 0.05秒程度で切り上げる制限

                if (anti_dfpn->search_mate(next_board)) {
                    // 💥 詰み発見！この手は危ないので、次の候補へ
                    std::cout << "info string Anti-Mate! " << move_to_usi(top_node->move_obj) << " is dangerous." << std::endl;
                    continue;
                }
                else {
                    // ✅ 安全確認完了。これを最終手とする
                    b1 = top_node;
                    break;
                }
            }
            // 全候補が全滅（絶望的状況）なら、仕方ないので1番手を選ぶ
            if (b1 == nullptr) b1 = candidates[0];
        }

        // 3. Ponder（先読み）用の相手の予測手を取得して返す（既存の機能を維持）
        std::string best_move = b1 ? move_to_usi(b1->move_obj) : "resign";
        std::string ponder_move = "";
        if (b1) {
            int max_v2 = -1;
            for (Node* c2 = b1->first_child; c2 != nullptr; c2 = c2->sibling) {
                int v2 = c2->visit_count.load();
                if (v2 > max_v2) { max_v2 = v2; ponder_move = move_to_usi(c2->move_obj); }
            }
        }
        return { best_move, ponder_move };
    }

private:
    void worker(Board root_board) {
        int iter_count = 0;
        std::vector<Ort::Float16_t> features_buffer(55 * 81);
        while (searching) {
            if ((++iter_count & 63) == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                if (!is_ponder_mode.load() && elapsed >= current_max_ms.load()) {
                    searching = false; break;
                }
            }
            run_simulation(root_board, features_buffer);
        }
    }

    void revert_virtual_loss(Node** path, int path_len) {
        for (int i = path_len - 1; i >= 1; --i) {
            Node* n = path[i];
            n->visit_count.fetch_sub(1, std::memory_order_relaxed);
            atomic_add_float(n->value_sum, VIRTUAL_LOSS);
        }
    }

    void run_simulation(Board sim_board, std::vector<Ort::Float16_t>& features_buffer) {
        Node* node = root;

        // ★変更: 動的確保の vector をやめ、高速なスタック配列にする
        Node* path[256];
        int path_len = 0;
        path[path_len++] = node;

        int loop_count = 0;

        // ★変更: history もスタック配列にする
        uint64_t sim_history[256];
        int history_len = 0;
        sim_history[history_len++] = sim_board.get_zobrist_key();

        while (true) {
            if (loop_count++ > 200) break;

            if (!node->is_expanded.load(std::memory_order_acquire)) {
                bool expected = false;
                if (node->is_evaluating.compare_exchange_strong(expected, true)) {
                    break;
                }
                else {
                    revert_virtual_loss(path, path_len); // ★引数変更
                    return;
                }
            }

            if (node->first_child == nullptr) break;

            Node* best_child = nullptr;
            float best_score = -999999.0f;

            int parent_visits = node->visit_count.load(std::memory_order_relaxed);
            float q_val = (parent_visits > 0) ? (node->value_sum.load(std::memory_order_relaxed) / (float)parent_visits) : node->raw_value;
            float parent_q = (std::max)(0.0f, (std::min)(1.0f, q_val));
            float fpu_val = parent_q - 0.05f;

            for (Node* child = node->first_child; child != nullptr; child = child->sibling) {
                float score = child->get_score(parent_visits, fpu_val);

                if (!child->is_expanded.load(std::memory_order_relaxed) && child->is_evaluating.load(std::memory_order_relaxed)) {
                    score -= 1000.0f;
                }

                if (score > best_score) { best_score = score; best_child = child; }
            }

            if (best_child) {
                best_child->visit_count.fetch_add(1, std::memory_order_relaxed);
                atomic_add_float(best_child->value_sum, -VIRTUAL_LOSS);

                node = best_child;
                StateInfo st;
                sim_board.push_move(node->move_obj, st);
                path[path_len++] = node; // ★配列への追加に変更

                uint64_t current_key = sim_board.get_zobrist_key();

                // ★変更: std::find をやめてシンプルな for ループ探索に変更
                bool found_in_history = false;
                for (int i = 0; i < history_len; ++i) {
                    if (sim_history[i] == current_key) {
                        found_in_history = true;
                        break;
                    }
                }

                if (found_in_history) { // 千日手検知
                    bool is_checking = sim_board.in_check(sim_board.turn);
                    float loop_value = is_checking ? 0.0f : 0.5f;
                    float value = 1.0f - loop_value;

                    for (int i = path_len - 1; i >= 0; --i) { // ★ path.size() を path_len に変更
                        Node* n = path[i];
                        if (i != 0) {
                            n->visit_count.fetch_sub(1, std::memory_order_relaxed);
                            atomic_add_float(n->value_sum, VIRTUAL_LOSS);
                        }
                        n->visit_count.fetch_add(1, std::memory_order_relaxed);
                        atomic_add_float(n->value_sum, value);
                        value = 1.0f - value;
                    }
                    return;
                }
                sim_history[history_len++] = current_key; // ★配列への追加に変更
            }
            else {
                break;
            }
        }

        float value = 0.0f;

        if (loop_count > 200) {
            value = 0.5f;
        }
        else if (!node->is_expanded.load() && node->is_evaluating.load()) {
            // 1. スタック上に600手分の領域を確保
            Move l_moves[MAX_LEGAL_MOVES];
            // 2. 配列の先頭を渡し、書き込み終わりの位置（end）を受け取る
            Move* end_moves = generate_moves(sim_board.bit_pos, l_moves);

            if (l_moves == end_moves) {
                node->is_expanded.store(true, std::memory_order_release);
                node->is_evaluating.store(false, std::memory_order_release);
                value = 0.0f;
            }
            else {
                uint64_t z_key = sim_board.get_zobrist_key();
                if (sim_board.last_to_sq >= 0 && sim_board.last_to_sq < 81) {
                    z_key ^= ZOBRIST_LAST_TO[sim_board.last_to_sq];
                }
                int idx = z_key & (MCTS_TT_SIZE - 1);
                int lock_idx = idx & 1023;

                bool cache_hit = false;
                std::shared_ptr<const std::vector<float>> cached_policy;
                float cached_value = 0.0f;

                {
                    while (cache_lock[lock_idx].test_and_set(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }

                    if (nn_cache[idx].key == z_key) {
                        cached_policy = nn_cache[idx].policy;
                        cached_value = nn_cache[idx].value;
                        cache_hit = true;
                    }
                    cache_lock[lock_idx].clear(std::memory_order_release);
                }

                if (cache_hit) {
                    if (!node->is_expanded.load()) {
                        // ★変更：引数に開始ポインタと終了ポインタを渡す
                        expand_node_with_moves(node, sim_board, l_moves, end_moves, cached_policy);
                        node->raw_value = cached_value;
                        node->is_expanded.store(true, std::memory_order_release);
                    }
                    node->is_evaluating.store(false, std::memory_order_release);
                    value = cached_value;
                }
                else {
                    // ★重要：ヒープ確保（std::vector）をやめ、スタック配列（または事前確保済みバッファ）を使う
                    // Ort::Float16_t features_buffer[55 * 81]; // スタック確保
                    // predict_async が vector を要求する場合は、使い回し用の thread_local な vector を検討してください
                    static thread_local std::vector<Ort::Float16_t> features_buffer(55 * 81);

                    make_features(sim_board, features_buffer);
                    auto result = evaluator->predict_async(&features_buffer).get();

                    if (result.first == nullptr) {
                        node->is_evaluating.store(false, std::memory_order_release);
                        revert_virtual_loss(path, path_len);
                        return;
                    }

                    {
                        int lock_idx = idx & 1023;
                        while (cache_lock[lock_idx].test_and_set(std::memory_order_acquire)) {
                            std::this_thread::yield();
                        }
                        nn_cache[idx].key = z_key;
                        nn_cache[idx].policy = result.first;
                        nn_cache[idx].value = result.second;
                        cache_lock[lock_idx].clear(std::memory_order_release);
                    }

                    // ★変更：ここでもポインタ形式で展開
                    expand_node_with_moves(node, sim_board, l_moves, end_moves, result.first);
                    node->raw_value = result.second;
                    value = result.second;

                    node->is_expanded.store(true, std::memory_order_release);
                    node->is_evaluating.store(false, std::memory_order_release);
                }
            }
        }
        else {
            revert_virtual_loss(path, path_len);
            return;
        }

        value = 1.0f - value;

        for (int i = path_len - 1; i >= 0; --i) {
            Node* n = path[i];
            if (i != 0) {
                n->visit_count.fetch_sub(1, std::memory_order_relaxed);
                atomic_add_float(n->value_sum, VIRTUAL_LOSS);
            }
            n->visit_count.fetch_add(1, std::memory_order_relaxed);
            atomic_add_float(n->value_sum, value);
            value = 1.0f - value;
        }
    }

    void expand_node_with_moves(Node* node, const Board& board, Move* moves_start, Move* moves_end, std::shared_ptr<const std::vector<float>> policy) {
        bool is_white = (board.turn != 0);
        size_t num_moves = moves_end - moves_start;
        if (num_moves == 0) return;

        // --- 1. まず Policy の合計値（policy_sum）を計算する ---
        float policy_sum = 0.0f;
        std::vector<float> valid_policies;
        valid_policies.reserve(num_moves);

        for (int i = 0; i < (int)num_moves; ++i) {
            Move check_m = *(moves_start + i);
            // 盤面を先手視点に反転（Python側と同じ条件にする）
            if (is_white) {
                if (!check_m.is_drop) check_m.from = 80 - check_m.from;
                check_m.to = 80 - check_m.to;
            }

            // ★ Vocabを使わずに直接インデックスを計算する
            int p_idx = get_action_index(check_m);

            // 出力サイズは2187なので、その範囲に収まっているかチェック
            float p = (p_idx >= 0 && p_idx < 2187 && policy) ? (*policy)[p_idx] : 0.00001f;

            valid_policies.push_back(p);
            policy_sum += p;
        }

        // ゼロ除算防止
        if (policy_sum < 0.000001f) policy_sum = 1.0f;

        // --- 2. 正規化（合計を1.0にする）しながらノードを確保・連結する ---
        Node* prev_child = nullptr;
        for (int i = 0; i < (int)num_moves; ++i) {
            Move m = *(moves_start + i);
            // ★ここが超重要：生の確率を合計値で割って、全体で100%(1.0)になるようにする
            float normalized_p = valid_policies[i] / policy_sum;

            // 正規化された正しい確率(normalized_p)でノードを作る
            Node* new_child = current_alloc().alloc(normalized_p, m);

            if (prev_child == nullptr) {
                node->first_child = new_child; // 最初の子
            }
            else {
                prev_child->sibling = new_child; // 兄弟として繋ぐ
            }
            prev_child = new_child;
        }
    }
};