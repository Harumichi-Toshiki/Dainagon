#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <fstream> 
#include <unordered_map> 
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include "shogi_utils.h"
#include "search.h"
#include "bitboard.h"
#include "zobrist.h"
#include "vocab.h"

const int INPUT_CHANNELS = 55;
const bool USE_BOOK_FLAG = true;

class Engine {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session* session = nullptr;
    BatchEvaluator* batch_evaluator = nullptr;
    Board board;
    MctsSearcher searcher;

    std::unordered_map<std::string, std::string> book;
    bool book_loaded = false;
    std::vector<std::string> history;

    // スレッド・Ponder管理用フラグ
    std::thread search_thread;
    std::atomic<bool> is_actually_thinking{ false }; // 通常思考(go)中かどうか!!!
    std::atomic<bool> is_pondering_now{ false };     // 先読み(go ponder)中かどうか!

    std::chrono::steady_clock::time_point ponder_start_time;
    //bool ponder_hit_active = false;     // Ponder Hitが発生したかのフラグ
    //int stored_ponder_elapsed = 0;      // 先読みで得した時間(ms)(

public:
    Engine(const std::wstring& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "Chunagon") {
        session_options.DisableMemPattern();
        session_options.SetExecutionMode(ORT_SEQUENTIAL);

        // ==========================================
        // ★追加：ONNX Runtimeの計算グラフ最適化を「最大」に設定
        // ==========================================
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        try {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
            std::cout << "info string GPU (DirectML) Init Success! [Device ID: 0]" << std::endl;
        }
        catch (...) {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 1));
            std::cout << "info string GPU (DirectML) Init Success! [Device ID: 1]" << std::endl;
        }
        session = new Ort::Session(env, model_path.c_str(), session_options);
        batch_evaluator = new BatchEvaluator(session);
        searcher.set_evaluator(batch_evaluator);
        board.reset();
    }

    ~Engine() {
        stop_search();
        if (batch_evaluator) delete batch_evaluator;
        if (session) delete session;
    }

    void load_book(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs) return;
        std::string line;
        int count = 0;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            size_t last_space = line.find_last_of(' ');
            if (last_space == std::string::npos) continue;
            std::string key = line.substr(0, last_space);
            std::string move = line.substr(last_space + 1);
            if (!move.empty() && move.back() == '\r') move.pop_back();
            book[key] = move;
            count++;
        }
        book_loaded = true;
        std::cout << "info string Book loaded: " << count << " entries." << std::endl;
    }

    void new_game() {
        stop_search();
        board.reset();
        history.clear();
        searcher.clear_cache();
        searcher.reset();
    }

    // ★ Ponder中だろうが通常中だろうが、確実に探索を殺す
    void stop_search() {
        searcher.searching = false;
        if (search_thread.joinable()) search_thread.join();
        is_actually_thinking = false;
        is_pondering_now = false;
    }

    void update_position(const std::string& sfen, const std::vector<std::string>& moves) {
        stop_search(); // 盤面を触る前に確実に探索を止める

        board.set_sfen(sfen);
        if (moves.size() >= history.size() && std::equal(history.begin(), history.end(), moves.begin())) {
            for (size_t i = history.size(); i < moves.size(); ++i) searcher.advance_root(moves[i]);
        }
        else {
            searcher.reset();
        }
        for (const auto& m : moves) board.push_usi(m);
        history = moves;
        board.update_bitboard();
    }

    // ★ GUIから ponderhit が来たときの処理
    void ponderhit() {
        if (is_pondering_now.load()) {
            is_pondering_now = false;
            is_actually_thinking = true;
            searcher.is_ponder_mode = false;

            // ★【最重要】Ponder的中の恩恵：目標時間と最大時間を一気に縮めて即指しさせる
            int t = searcher.current_target_ms.load();
            searcher.current_target_ms.store(t / 4);
            searcher.current_max_ms.store(t / 3);

            std::cout << "info string Ponder hit! Transitioning to real thinking." << std::endl;
        }
    }

    void calculate_thinking_time(const std::string& go_cmd, int& out_target, int& out_max) {
        int time_ms = 0, inc_ms = 0, byoyomi_ms = 0;
        std::stringstream ss(go_cmd);
        std::string token;
        while (ss >> token) {
            if (board.turn == BLACK) {
                if (token == "btime") ss >> time_ms;
                if (token == "binc") ss >> inc_ms;
            }
            else {
                if (token == "wtime") ss >> time_ms;
                if (token == "winc") ss >> inc_ms;
            }
            if (token == "byoyomi") ss >> byoyomi_ms;
        }

        // 1. 基本の考慮時間 (target) を計算
        int divisor = (std::max)(10, 45 - ((int)history.size() / 2));

        // ★修正：フィッシャールール等の加算(inc)の使い込みも、0.8倍から0.9倍に増やして少し贅沢に使う1
        out_target = (time_ms / divisor) + (int)(inc_ms * 0.9);

        // 2. ★修正：秒読み(byoyomi)がある場合の賢いマージ
        if (byoyomi_ms > 0) {
            if (time_ms > 0) {
                // 持ち時間＋秒読みの場合：基本時間に秒読みの1/4程度を足して余裕を持たせる
                out_target += byoyomi_ms / 3;
            }
            else {
                // 持ち時間なし（秒読みのみ）の場合：秒読みの 60% を目標時間にする
                out_target = byoyomi_ms * 0.8;
            }
        }

        // 3. ★修正：長考の天井（上限）を目標時間の「2倍」に厳格に設定
        out_max = out_target * 3;

        // 4. 秒読みの絶対限界をオーバーしないためのセーフティ
        if (byoyomi_ms > 0) {
            int byoyomi_limit = (std::max)(100, byoyomi_ms - 500);
            if (out_max > byoyomi_limit && time_ms == 0) {
                out_max = byoyomi_limit;
            }
        }

        // 5. 持ち時間の安全装置 (残り時間が少ない時は強制的に削る)
        int safe = (std::max)(0, time_ms - 1000);
        if (time_ms > 0 && out_max > safe) {
            out_max = safe;
            if (out_target > safe) out_target = safe;
        }
    }

    // ★ "go" も "go ponder" もここで受ける
    void go(const std::string& go_cmd_line) {
        bool is_ponder = (go_cmd_line.find("ponder") != std::string::npos);

        if (USE_BOOK_FLAG && book_loaded && !is_ponder) {
            std::string key = board.to_sfen_key();
            if (book.count(key)) {
                std::cout << "bestmove " << book[key] << std::endl;
                return;
            }
        }

        int target_ms, max_ms;
        calculate_thinking_time(go_cmd_line, target_ms, max_ms);

        stop_search(); // 古い探索を確実に止める11

        searcher.current_target_ms = target_ms;
        searcher.current_max_ms = max_ms;
        searcher.is_ponder_mode = is_ponder;
        is_actually_thinking = !is_ponder;
        is_pondering_now = is_ponder;

        searcher.searching = true;

        search_thread = std::thread([this]() {
            // 探索実行
            SearchResult res = searcher.search(board, searcher.current_target_ms, searcher.current_max_ms, searcher.is_ponder_mode);

            // どんな理由で終わっても【必ず】bestmove を返す
            std::cout << "bestmove " << res.best;
            if (!res.ponder.empty()) std::cout << " ponder " << res.ponder;
            std::cout << std::endl;

            // 状態をリセット
            is_actually_thinking = false;
            is_pondering_now = false;
            });
    }
};

int main() {
    // ★AIのモデル（脳）の場所を指定
    std::wstring model_path = L"C:\\Users\\harum\\Desktop\\MyProject\\DainagonProject\\model\\chunagonfp16_v24.onnx";
    //std::wstring model_path = L".\\model\\model.onnx";

    try {
        init_bitboards();
        init_zobrist(); //
        init_vocab();
        Engine engine(model_path);

        std::string line;
        // 将棋所（GUI）からの命令を待ち続ける無限ループ
        while (std::getline(std::cin, line)) {
            if (line == "quit") {
                break;
            }
            else if (line == "usi") {
                std::cout << "id name Chunagon_CPP_e16.13m24" << std::endl;
                std::cout << "id author Harumichi-Toshiki" << std::endl;
                std::cout << "usiok" << std::endl;
            }
            else if (line == "isready") {
                // 定跡ファイルがあれば読み込む（無ければスキップされます）
                engine.load_book("C:\\Users\\harum\\Desktop\\MyProject\\DainagonProject\\data\\478book.txt");
                //engine.load_book(".\\book.txt");
                std::cout << "readyok" << std::endl;
            }
            else if (line == "usinewgame") {
                engine.stop_search();
                engine.new_game();
            }
            else if (line.find("position") == 0) {
                engine.stop_search();
                std::string sfen = "startpos";
                std::vector<std::string> moves;

                size_t moves_pos = line.find("moves");
                if (line.find("sfen") != std::string::npos) {
                    size_t sfen_start = line.find("sfen") + 5;
                    if (moves_pos != std::string::npos) {
                        sfen = line.substr(sfen_start, moves_pos - sfen_start - 1);
                    }
                    else {
                        sfen = line.substr(sfen_start);
                    }
                }

                if (moves_pos != std::string::npos) {
                    std::stringstream ss(line.substr(moves_pos + 6));
                    std::string m;
                    while (ss >> m) moves.push_back(m);
                }
                // 盤面を更新（ここで履歴の一致を見て、木を引き継ぐか判定します）
                engine.update_position(sfen, moves);
            }
            else if (line.find("go") == 0) {
                engine.go(line);
            }
            else if (line == "stop") {
                engine.stop_search();
            }
            else if (line == "ponderhit") {
                engine.ponderhit();
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in main: " << e.what() << std::endl;
    }
    return 0;

}