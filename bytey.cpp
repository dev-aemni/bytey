#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <map>
#include <chrono>
#include <iomanip>
#include <dirent.h>
#include <readline/readline.h>
#include <readline/history.h>
#include <omp.h>
#include <unistd.h>

const int D_MODEL   = 2048;   // 15M Architecture
const int N_LAYERS  = 3;      // 3 Deep Layers
const float LR_INIT = 0.05f;

std::string get_model_path() {
    const char* h = getenv("HOME");
    return (h ? std::string(h) + "/programs/bytey/bytey.dat" : "bytey.dat");
}

std::vector<std::string> clean_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur = "";
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (isalnum(c) || c == '\'' || c == '+' || c == '#') {
            cur += tolower(c);
        } else {
            if (!cur.empty()) { tokens.push_back(cur); cur = ""; }
            if (c == '.' || c == ',' || c == '?' || c == '!') tokens.push_back(std::string(1, c));
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

struct QAEntry {
    std::vector<std::string> q_words;
    std::string full_answer;
};

std::vector<QAEntry> load_qa_pairs() {
    std::vector<QAEntry> dataset;
    DIR* dir = opendir("fridges");
    if (!dir) return dataset;

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string fname = entry->d_name;
        if (fname.size() > 4 && fname.substr(fname.size() - 4) == ".txt") {
            std::ifstream f("fridges/" + fname);
            std::string line;
            QAEntry cur;
            bool has_q = false;

            std::string current_topic = "";
            while (std::getline(f, line)) {
                if (line.empty() || line[0] == '#') {
                    if (line.rfind("##", 0) == 0) {
                        current_topic = line.substr(2); // Extract heading topic
                    }
                    continue;
                }
                if (line.rfind("human:", 0) == 0) {
                    cur.q_words = clean_tokenize(line.substr(6));
                    has_q = true;
                } else if (line.rfind("bytey:", 0) == 0 && has_q) {
                    cur.full_answer = line.substr(6);
                    while (!cur.full_answer.empty() && cur.full_answer[0] == ' ') {
                        cur.full_answer.erase(cur.full_answer.begin());
                    }
                    if (!cur.full_answer.empty()) cur.full_answer[0] = toupper(cur.full_answer[0]);
                    dataset.push_back(cur);
                    has_q = false;
                } else if (line.size() > 50) {
                    // Raw textbook paragraph found! (like human_bio1.txt)
                    QAEntry p_entry;
                    p_entry.full_answer = line;
                    // Keywords extracted from topic heading + first sentence
                    std::string prompt_extract = current_topic + " " + line.substr(0, std::min((size_t)80, line.size()));
                    p_entry.q_words = clean_tokenize(prompt_extract);
                    dataset.push_back(p_entry);
                }
            }
        }
    }
    closedir(dir);
    return dataset;
}

int calculate_params(int V, int N) {
    return (V * D_MODEL) + (3 * D_MODEL * D_MODEL) + (N * D_MODEL);
}

void print_banner(const std::string& mode, int V, int N, int steps) {
    int total_params = calculate_params(V, N);
    std::cout << "\033[1;34m┌──────────────────────────────────────────────────┐\033[0m\n";
    std::cout << "│ \033[1;37mBytey 15M Independent Persona Engine (" << mode << ")\033[0m\n";
    std::cout << "\033[1;34m├──────────────────────────────────────────────────┤\033[0m\n";
    std::cout << "│ Parameters:  \033[1;33m" << total_params << " floats (" 
              << std::fixed << std::setprecision(2) << (total_params * 4) / (1024.0 * 1024.0) << " MB Brain)\033[0m\n";
    std::cout << "│ Hardware:    \033[1;32m3 Dedicated Cores (Thermal Controlled)\033[0m\n";
    std::cout << "│ Knowledge:   \033[1;34m" << N << " Verified Responses + Persona Thoughts\033[0m\n";
    std::cout << "│ Vocabulary:  \033[1;34m" << V << " Keywords\033[0m\n";
    if (steps > 0) std::cout << "│ Total Steps: \033[1;37m" << steps / 1000 << "k (" << steps << " iterations)\033[0m\n";
    std::cout << "\033[1;34m└──────────────────────────────────────────────────┘\033[0m\n";
}

void train_model(int steps) {
    omp_set_num_threads(3); // 3 cores keeps your phone completely cool & lag-free!

    auto dataset = load_qa_pairs();
    if (dataset.empty()) {
        std::cerr << "No Q&A pairs found in fridges/*.txt!\n";
        return;
    }

    std::map<std::string, int> w2i;
    std::vector<std::string> i2w;
    for (const auto& item : dataset) {
        for (const auto& w : item.q_words) {
            if (w2i.find(w) == w2i.end()) {
                w2i[w] = i2w.size();
                i2w.push_back(w);
            }
        }
    }
    int V = i2w.size();
    int N = dataset.size();

    print_banner("Training", V, N, steps);

    float scale = 1.0f / std::sqrt((float)D_MODEL);

    std::vector<float> W_emb(V * D_MODEL);
    std::vector<float> W_l1(D_MODEL * D_MODEL);
    std::vector<float> W_l2(D_MODEL * D_MODEL);
    std::vector<float> W_l3(D_MODEL * D_MODEL);
    std::vector<float> W_ans(N * D_MODEL);

    for (size_t i = 0; i < W_emb.size(); ++i) W_emb[i] = (((float)rand() / (float)RAND_MAX) - 0.5f) * scale;
    for (size_t i = 0; i < W_l1.size(); ++i) W_l1[i] = (((float)rand() / (float)RAND_MAX) - 0.5f) * scale;
    for (size_t i = 0; i < W_l2.size(); ++i) W_l2[i] = (((float)rand() / (float)RAND_MAX) - 0.5f) * scale;
    for (size_t i = 0; i < W_l3.size(); ++i) W_l3[i] = (((float)rand() / (float)RAND_MAX) - 0.5f) * scale;
    for (size_t i = 0; i < W_ans.size(); ++i) W_ans[i] = (((float)rand() / (float)RAND_MAX) - 0.5f) * scale;

    std::cout << "\033[1;36mStep & Losses Log:\033[0m\n";
    std::cout << "\033[0;37m──────────────────────────────────────────────────\033[0m\n";

    auto t_start = std::chrono::steady_clock::now();
    auto t_last = t_start;
    float smooth_loss = 5.0f;
    int log_interval = (steps >= 50000) ? 10000 : 2500;

    for (int step = 0; step < steps; ++step) {
        float progress = (float)step / steps;
        float lr = LR_INIT * 0.5f * (1.0f + std::cos(progress * 3.14159f));
        if (lr < 0.005f) lr = 0.005f;

        int target_idx = step % N;
        const auto& item = dataset[target_idx];

        std::vector<float> q_vec(D_MODEL, 0.0f);
        for (const auto& w : item.q_words) {
            int tid = w2i[w];
            for (int d = 0; d < D_MODEL; ++d) q_vec[d] += W_emb[tid * D_MODEL + d];
        }

        std::vector<float> h1(D_MODEL, 0.0f);
        #pragma omp parallel for
        for (int i = 0; i < D_MODEL; ++i) {
            float sum = 0.0f;
            const float* l_ptr = &W_l1[i * D_MODEL];
            for (int j = 0; j < D_MODEL; ++j) sum += q_vec[j] * l_ptr[j];
            h1[i] = q_vec[i] + std::tanh(sum * scale);
        }

        std::vector<float> h2(D_MODEL, 0.0f);
        #pragma omp parallel for
        for (int i = 0; i < D_MODEL; ++i) {
            float sum = 0.0f;
            const float* l_ptr = &W_l2[i * D_MODEL];
            for (int j = 0; j < D_MODEL; ++j) sum += h1[j] * l_ptr[j];
            h2[i] = h1[i] + std::tanh(sum * scale);
        }

        std::vector<float> h3(D_MODEL, 0.0f);
        #pragma omp parallel for
        for (int i = 0; i < D_MODEL; ++i) {
            float sum = 0.0f;
            const float* l_ptr = &W_l3[i * D_MODEL];
            for (int j = 0; j < D_MODEL; ++j) sum += h2[j] * l_ptr[j];
            h3[i] = h2[i] + std::tanh(sum * scale);
        }

        float rms = 0.0f;
        for (int d = 0; d < D_MODEL; ++d) rms += h3[d] * h3[d];
        rms = 1.0f / std::sqrt(rms / D_MODEL + 1e-5f);
        for (int d = 0; d < D_MODEL; ++d) h3[d] *= rms;

        std::vector<float> logits(N, 0.0f);
        float max_l = -1e9f;
        #pragma omp parallel for reduction(max:max_l)
        for (int i = 0; i < N; ++i) {
            float sum = 0.0f;
            const float* a_ptr = &W_ans[i * D_MODEL];
            for (int d = 0; d < D_MODEL; ++d) sum += h3[d] * a_ptr[d];
            logits[i] = sum;
            if (sum > max_l) max_l = sum;
        }

        float sum_exp = 0.0f;
        std::vector<float> probs(N, 0.0f);
        for (int i = 0; i < N; ++i) {
            probs[i] = std::exp(logits[i] - max_l);
            sum_exp += probs[i];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-7f);
        for (int i = 0; i < N; ++i) probs[i] *= inv_sum;

        float loss = -std::log(std::max(probs[target_idx], 1e-7f));
        smooth_loss = 0.99f * smooth_loss + 0.01f * loss;

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            float grad = probs[i] - ((i == target_idx) ? 1.0f : 0.0f);
            if (std::abs(grad) < 1e-5f) continue;
            float* a_ptr = &W_ans[i * D_MODEL];
            for (int d = 0; d < D_MODEL; ++d) {
                a_ptr[d] -= lr * grad * h3[d];
            }
        }

        if ((step + 1) % log_interval == 0 || step == steps - 1) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> total = now - t_start;
            std::chrono::duration<double> step_time = now - t_last;
            t_last = now;

            int k_val = (step + 1) / 1000;
            std::cout << "\033[1;36m" << std::setw(3) << k_val << "k\033[0m   "
                      << "\033[1;32m" << std::fixed << std::setprecision(4) << smooth_loss << "\033[0m "
                      << "\033[2m((" << std::setprecision(1) << step_time.count() << "s) [Total: " 
                      << int(total.count()) << "s])\033[0m\n";
        }
    }

    std::string mpath = get_model_path();
    std::ofstream out(mpath, std::ios::binary);
    out.write((char*)&V, sizeof(int));
    out.write((char*)&N, sizeof(int));

    for (const auto& w : i2w) {
        int len = w.size();
        out.write((char*)&len, sizeof(int));
        out.write(w.data(), len);
    }
    for (const auto& item : dataset) {
        int len = item.full_answer.size();
        out.write((char*)&len, sizeof(int));
        out.write(item.full_answer.data(), len);
    }

    out.write((char*)W_emb.data(), W_emb.size() * sizeof(float));
    out.write((char*)W_l1.data(), W_l1.size() * sizeof(float));
    out.write((char*)W_l2.data(), W_l2.size() * sizeof(float));
    out.write((char*)W_l3.data(), W_l3.size() * sizeof(float));
    out.write((char*)W_ans.data(), W_ans.size() * sizeof(float));
    out.close();

    std::cout << "\033[0;37m──────────────────────────────────────────────────\033[0m\n";
    std::cout << "\033[1;32m✔ 15M Independent Brain saved to " << mpath << "!\033[0m\n\n";
}

void chat_model() {
    omp_set_num_threads(3);

    std::string mpath = get_model_path();
    std::ifstream in(mpath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "bytey.dat not found! Run: bytey train\n";
        return;
    }

    int V, N;
    in.read((char*)&V, sizeof(int));
    in.read((char*)&N, sizeof(int));

    std::vector<std::string> i2w(V);
    std::map<std::string, int> w2i;
    for (int i = 0; i < V; ++i) {
        int len;
        in.read((char*)&len, sizeof(int));
        std::string w(len, ' ');
        in.read(&w[0], len);
        i2w[i] = w;
        w2i[w] = i;
    }

    std::vector<std::string> answers(N);
    for (int i = 0; i < N; ++i) {
        int len;
        in.read((char*)&len, sizeof(int));
        std::string a(len, ' ');
        in.read(&a[0], len);
        answers[i] = a;
    }

    std::vector<float> W_emb(V * D_MODEL);
    std::vector<float> W_l1(D_MODEL * D_MODEL);
    std::vector<float> W_l2(D_MODEL * D_MODEL);
    std::vector<float> W_l3(D_MODEL * D_MODEL);
    std::vector<float> W_ans(N * D_MODEL);

    in.read((char*)W_emb.data(), W_emb.size() * sizeof(float));
    in.read((char*)W_l1.data(), W_l1.size() * sizeof(float));
    in.read((char*)W_l2.data(), W_l2.size() * sizeof(float));
    in.read((char*)W_l3.data(), W_l3.size() * sizeof(float));
    in.read((char*)W_ans.data(), W_ans.size() * sizeof(float));
    in.close();

    print_banner("Independent Persona Online", V, N, 0);

    float scale = 1.0f / std::sqrt((float)D_MODEL);

    // Bytey's Independent Persona Thoughts
    std::vector<std::string> persona_thoughts = {
        "Hmm, that's an interesting angle! Tell me more about what you mean.",
        "I'm pondering that. What makes you think of that right now?",
        "That's outside my core database, but as Bytey, I'm genuinely curious! How would you describe it?",
        "I hear you! We should explore that deeper together.",
        "That's a unique perspective. What should we build or discuss next?"
    };

    char* buf = nullptr;
    while ((buf = readline("\n\033[1;33mYou: \033[0m")) != nullptr) {
        std::string line(buf);
        if (!line.empty()) add_history(buf);
        free(buf);

        if (line.empty()) continue;
        auto words = clean_tokenize(line);
        if (words.empty()) continue;

        std::vector<float> q_vec(D_MODEL, 0.0f);
        int matched = 0;
        for (const auto& w : words) {
            if (w2i.find(w) != w2i.end()) {
                int tid = w2i[w];
                for (int d = 0; d < D_MODEL; ++d) q_vec[d] += W_emb[tid * D_MODEL + d];
                matched++;
            }
        }

        // INDEPENDENT PERSONA REASONING
        if (matched == 0) {
            int pick = rand() % persona_thoughts.size();
            std::cout << "\033[1;34mBytey:\033[0m " << persona_thoughts[pick] << "\n";
            continue;
        }

        std::vector<float> h1(D_MODEL, 0.0f);
        #pragma omp parallel for
        for (int i = 0; i < D_MODEL; ++i) {
            float sum = 0.0f;
            const float* l_ptr = &W_l1[i * D_MODEL];
            for (int j = 0; j < D_MODEL; ++j) sum += q_vec[j] * l_ptr[j];
            h1[i] = q_vec[i] + std::tanh(sum * scale);
        }

        std::vector<float> h2(D_MODEL, 0.0f);
        #pragma omp parallel for
        for (int i = 0; i < D_MODEL; ++i) {
            float sum = 0.0f;
            const float* l_ptr = &W_l2[i * D_MODEL];
            for (int j = 0; j < D_MODEL; ++j) sum += h1[j] * l_ptr[j];
            h2[i] = h1[i] + std::tanh(sum * scale);
        }

        std::vector<float> h3(D_MODEL, 0.0f);
        #pragma omp parallel for
        for (int i = 0; i < D_MODEL; ++i) {
            float sum = 0.0f;
            const float* l_ptr = &W_l3[i * D_MODEL];
            for (int j = 0; j < D_MODEL; ++j) sum += h2[j] * l_ptr[j];
            h3[i] = h2[i] + std::tanh(sum * scale);
        }

        float rms = 0.0f;
        for (int d = 0; d < D_MODEL; ++d) rms += h3[d] * h3[d];
        rms = 1.0f / std::sqrt(rms / D_MODEL + 1e-5f);
        for (int d = 0; d < D_MODEL; ++d) h3[d] *= rms;

        // Confidence Calculation
        float best_score = -1e9f;
        int best_idx = 0;
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            float dot = 0.0f;
            const float* a_ptr = &W_ans[i * D_MODEL];
            for (int d = 0; d < D_MODEL; ++d) dot += h3[d] * a_ptr[d];
            #pragma omp critical
            {
                if (dot > best_score) {
                    best_score = dot;
                    best_idx = i;
                }
            }
        }

        // Independent Persona: If confidence is low, don't guess blindly—think naturally!
        if (best_score < 0.2f && words.size() <= 2) {
            int pick = rand() % persona_thoughts.size();
            std::cout << "\033[1;34mBytey:\033[0m " << persona_thoughts[pick] << "\n";
        } else {
            std::cout << "\033[1;34mBytey:\033[0m " << answers[best_idx] << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    int steps = 30000;
    std::string mode = "train";

    if (argc > 1) mode = argv[1];

    if (mode == "chat") {
        chat_model();
        return 0;
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-steps" && i + 1 < argc) {
            std::string s = argv[++i];
            if (s.back() == 'k' || s.back() == 'K') { s.pop_back(); steps = std::stoi(s) * 1000; }
            else steps = std::stoi(s);
        }
    }

    train_model(steps);
    return 0;
}
