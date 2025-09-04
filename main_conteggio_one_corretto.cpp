#include "ArducamTOFCamera.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <open3d/Open3D.h>
#include <filesystem>
#include <unordered_map>
#include <set>
#include <algorithm> // max_element, nth_element, sort
#include <numeric>   // accumulate, iota
#include <cmath>     // fabs
#include <random>
#include <deque>

// ONNX Runtime
#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;
using namespace Arducam;

// =========================
// Parametri / Costanti
// =========================

static constexpr int   MAX_DISTANCE     = 4000; // mm
static constexpr int   CONFIDENCE_VALUE = 60;   // soglia confidenza
static constexpr size_t kNumPoints      = 1024; // input alla rete

// =========================
// Stato globale conteggio
// =========================
int in_count = 0, out_count = 0, occ = 0;
int frame_id = 0;
bool line_calibrated = true;
float line_pos = 0.0f;          // m, aggiornata dopo calibrazione
bool positive_is_out = true;      // verso OUT: neg->pos se true

// =========================
// Tracce
// =========================


int next_track_id = 0;

// LineCounter leggero
struct LCState {
    std::deque<float> hist; // ultimi kSmoothW valori su asse
    char armed_side = 0;    // 'n' neg, 'p' pos, 0 none
    float armed_pos = 0.f;
    int cooldown = 0;
};

std::unordered_map<int, LCState> lc_states;

// =========================
// Camera helpers
// =========================
bool getControl(ArducamTOFCamera& tof, Control mode, float& val, float alpha = 1.0) {
    int tmp = 0;
    if (tof.getControl(mode, &tmp) != 0) return false;
    val = tmp / alpha;
    return true;
}

bool initCamera(ArducamTOFCamera& tof, const char* cfg_path) {
    if (cfg_path && tof.openWithFile(cfg_path)) return false;
    if (!cfg_path && tof.open(Connection::CSI)) return false;
    if (tof.start(FrameType::DEPTH_FRAME)) return false;
    tof.setControl(Control::RANGE, MAX_DISTANCE);
    return true;
}

ArducamFrameBuffer* acquireFrame(ArducamTOFCamera& tof) {
    return tof.requestFrame(500);
}

// =========================
// Open3D → Point Cloud (identica allo scheletro)
// =========================
std::shared_ptr<open3d::geometry::PointCloud>
generatePointCloud(ArducamTOFCamera& tof, ArducamFrameBuffer* frame, const Eigen::Matrix4d& transform) {
    Arducam::FrameFormat format;
    frame->getFormat(FrameType::DEPTH_FRAME, format);

    float* depth = (float*)frame->getData(FrameType::DEPTH_FRAME);
    float* conf  = (float*)frame->getData(FrameType::CONFIDENCE_FRAME);

    // 1) Filtro confidenza → buffer float32 (non cambiamo unità qui)
    std::vector<float> filtered(format.width * format.height);
    for (int i = 0; i < format.width * format.height; ++i)
        filtered[i] = (conf[i] >= CONFIDENCE_VALUE) ? depth[i] : 0.f;

    // 2) Copia nel depth image
    open3d::geometry::Image depth_img;
    depth_img.Prepare(format.width, format.height, 1, 4);
    std::memcpy(depth_img.data_.data(), filtered.data(), filtered.size() * sizeof(float));

    // 3) Intrinseci dallo scheletro (divisi per 100)
    float fx, fy, cx, cy;
    getControl(tof, Control::INTRINSIC_FX, fx, 100);
    getControl(tof, Control::INTRINSIC_FY, fy, 100);
    getControl(tof, Control::INTRINSIC_CX, cx, 100);
    getControl(tof, Control::INTRINSIC_CY, cy, 100);

    open3d::camera::PinholeCameraIntrinsic intr(
        format.width, format.height, fx, fy, cx, cy);

    // 4) depth_scale=1000.0, depth_trunc=2.5 → cloud in METRI
    auto cloud = open3d::geometry::PointCloud::CreateFromDepthImage(
        depth_img, intr, Eigen::Matrix4d::Identity(), 1000.0, 2.5);

    // 5) Flip assi (Y=-Y, Z=-Z) come dataset
    cloud->Transform(transform);

    // 6) (Stesso filtro "inutile ma innocuo" dello scheletro)
    std::vector<size_t> keep;
    keep.reserve(cloud->points_.size());
    for (size_t i = 0; i < cloud->points_.size(); ++i)
        if (cloud->points_[i][2]  > -2000 and cloud->points_[i][2]  < -20) keep.push_back(i); 

    return cloud->SelectByIndex(keep);
}

// =========================
// Sampler 1024 punti (coerente al dataset)
// =========================
std::vector<Eigen::Vector3f>
samplePoints(const std::shared_ptr<open3d::geometry::PointCloud>& cloud, size_t N, uint32_t seed) {
    std::vector<Eigen::Vector3f> out;
    const auto& P = cloud->points_;
    const size_t M = P.size();

    if (M == 0) return out;
    out.reserve(N);

    std::mt19937 rng(seed);
    if (M >= N) {
        // senza rimpiazzo
        std::vector<size_t> idx(M);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        for (size_t i = 0; i < N; ++i) {
            const auto& v = P[idx[i]];
            out.emplace_back((float)v(0), (float)v(1), (float)v(2)); // METRI
        }
    } else {
        // con rimpiazzo
        std::uniform_int_distribution<size_t> uni(0, M - 1);
        for (size_t i = 0; i < N; ++i) {
            const auto& v = P[uni(rng)];
            out.emplace_back((float)v(0), (float)v(1), (float)v(2)); // METRI
        }
    }
    return out;
}

Eigen::Vector3f
geometricCentroidM(const std::shared_ptr<open3d::geometry::PointCloud>& cloud) {
    Eigen::Vector3f c(0,0,0);
    if (cloud->points_.empty()) return c;
    for (const auto& p : cloud->points_)
        c += Eigen::Vector3f((float)p(0),(float)p(1),(float)p(2));
    c /= (float)cloud->points_.size(); // METRI
    return c;
}

// =========================
// ONNX Runner
// =========================
struct OnnxRunner {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "tofnet"};
    Ort::SessionOptions so;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<const char*> input_names{"pts"};
    std::vector<const char*> output_names{"class_logits","reg_coords"};
    float last_probs[3] = {0,0,0};  // salva le ultime probabilità softmax

    // Se i punti sono in METRI: 1.0
    // Se fossero in MILLIMETRI: 0.001
    float input_unit_scale = 1.0f;

    explicit OnnxRunner(const std::string& model_path) {
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), so);
    }

    inline void set_input_unit_scale(float s) { input_unit_scale = s; }


    int infer(const std::vector<Eigen::Vector3f>& pts_native_units, float reg_out[2][3]) {
        if (pts_native_units.empty()) return 0;

        const int64_t num_pts = static_cast<int64_t>(pts_native_units.size());
        std::vector<int64_t> shape{1, num_pts, 3};

        std::vector<float> input;
        input.reserve(static_cast<size_t>(num_pts) * 3);
        for (const auto& p : pts_native_units) {
            input.push_back(p.x() * input_unit_scale);
            input.push_back(p.y() * input_unit_scale);
            input.push_back(p.z() * input_unit_scale);
        }

        auto input_tensor = Ort::Value::CreateTensor<float>(
            mem, input.data(), input.size(), shape.data(), shape.size());

        // qui finalmente creo outputs
        auto outputs = session->Run(
            Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
            output_names.data(), output_names.size());

        // logits
        float* cls = outputs[0].GetTensorMutableData<float>(); // [1,3]
        int n = static_cast<int>(std::max_element(cls, cls + 3) - cls); // 0/1/2

        // regressione
        float* rc = outputs[1].GetTensorMutableData<float>(); // [1,2,3]
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                reg_out[i][j] = rc[i * 3 + j];

        // debug input bounds
        static int dbg_ctr = 0;
        if ((dbg_ctr++ % 30) == 0) {
            float xmin=1e9f,xmax=-1e9f, ymin=1e9f,ymax=-1e9f, zmin=1e9f,zmax=-1e9f;
            for (size_t i=0;i<input.size(); i+=3) {
                xmin = std::min(xmin, input[i+0]);
                xmax = std::max(xmax, input[i+0]);
                ymin = std::min(ymin, input[i+1]);
                ymax = std::max(ymax, input[i+1]);
                zmin = std::min(zmin, input[i+2]);
                zmax = std::max(zmax, input[i+2]);
            }
            std::cout << "[NET IN] scale=" << input_unit_scale
                    << " X:[" << xmin << "," << xmax << "]"
                    << " Y:[" << ymin << "," << ymax << "]"
                    << " Z:[" << zmin << "," << zmax << "]\n";
        }

        std::cout << "[ONNX LOGITS] " << cls[0] << " " << cls[1] << " " << cls[2] << std::endl;

        // softmax
        float m = std::max(cls[0], std::max(cls[1], cls[2]));
        double e0 = std::exp(cls[0]-m), e1 = std::exp(cls[1]-m), e2 = std::exp(cls[2]-m);
        double s = e0+e1+e2;
        last_probs[0] = e0/s;
        last_probs[1] = e1/s;
        last_probs[2] = e2/s;

        std::cout << "[PROB] n0=" << last_probs[0]
                << " n1=" << last_probs[1]
                << " n2=" << last_probs[2] << "\n";

        return n;
    }



};



// =====================================================
// Kalman Filter semplice 3D (posizione+velocità)
// =====================================================
struct KalmanTrack {
    Eigen::Matrix<float, 6, 1> x;   // stato: [x,y,z,vx,vy,vz]
    Eigen::Matrix<float, 6, 6> P;   // covarianza
    int age = 0;                    // frame consecutivi senza update

    // Costruttore di default
    KalmanTrack() {
        x.setZero();
        P = Eigen::Matrix<float,6,6>::Identity();
    }

    // Costruttore con posizione iniziale
    KalmanTrack(const Eigen::Vector3f& p) {
        x << p.x(), p.y(), p.z(), 0,0,0;
        P = Eigen::Matrix<float,6,6>::Identity();
    }

    void predict(float dt=1.0f) {
        Eigen::Matrix<float,6,6> F = Eigen::Matrix<float,6,6>::Identity();
        F(0,3)=dt; F(1,4)=dt; F(2,5)=dt;
        x = F * x;
        P = F * P * F.transpose() + Eigen::Matrix<float,6,6>::Identity()*0.01f;
    }

    void update(const Eigen::Vector3f& z) {
        Eigen::Matrix<float,3,6> H = Eigen::Matrix<float,3,6>::Zero();
        H(0,0)=H(1,1)=H(2,2)=1.0f;
        Eigen::Vector3f y = z - H*x;
        Eigen::Matrix3f R = Eigen::Matrix3f::Identity()*0.05f; // rumore misura
        Eigen::Matrix<float,3,3> S = H*P*H.transpose() + R;
        Eigen::Matrix<float,6,3> K = P*H.transpose()*S.inverse();
        x = x + K*y;
        P = (Eigen::Matrix<float,6,6>::Identity() - K*H)*P;
        age = 0;
    }

    Eigen::Vector3f pos() const { return x.head<3>(); }
};

// =====================================================
// MultiPersonTracker: Kalman + Hungarian
// =====================================================
struct MultiPersonTracker {
    struct Track {
        KalmanTrack kf;
        int age = 0; // frame consecutivi senza update
    };

    std::unordered_map<int, Track> tracks;
    int next_id = 0;
    int max_age;
    float dist_thresh;

    MultiPersonTracker(int max_age_=10, float dist_thresh_=1.0f)
        : max_age(max_age_), dist_thresh(dist_thresh_) {}

    std::unordered_map<int, Eigen::Vector3f> update(const std::vector<Eigen::Vector3f>& detections) {
        // 1) Predici e invecchia tutti
        std::vector<int> tids;
        std::vector<Eigen::Vector3f> preds;
        tids.reserve(tracks.size());
        preds.reserve(tracks.size());

        for (auto& kv : tracks) {
            kv.second.kf.predict();
            kv.second.age++;
            tids.push_back(kv.first);
            preds.push_back(kv.second.kf.pos());
        }

        int M = preds.size();
        int N = detections.size();

        // 2) Cost matrix con gating
        Eigen::MatrixXf cost = Eigen::MatrixXf::Constant(M, N, 1e6f);
        for (int i=0;i<M;i++) {
            for (int j=0;j<N;j++) {
                float d = (preds[i] - detections[j]).norm();
                if (d <= dist_thresh) cost(i,j) = d;
            }
        }

        // 3) Hungarian assignment
        std::set<int> matched_tids;
        std::set<int> matched_dets;

        if (M>0 && N>0) {
            // implementazione semplice Hungarian "manuale"
            // (puoi sostituirla con libreria, ma così resta chiaro)
            for (int i=0;i<M;i++) {
                float best=1e6; int best_j=-1;
                for (int j=0;j<N;j++) {
                    if (matched_dets.count(j)) continue;
                    if (cost(i,j) < best) {best=cost(i,j); best_j=j;}
                }
                if (best_j!=-1 && cost(i,best_j) < 1e6) {
                    int tid = tids[i];
                    tracks[tid].kf.update(detections[best_j]);
                    tracks[tid].age = 0;
                    matched_tids.insert(tid);
                    matched_dets.insert(best_j);
                }
            }
        }

        // 4) Nuove tracce
        for (int j=0;j<N;j++) {
            if (!matched_dets.count(j)) {
                Track t;
                t.kf = KalmanTrack(detections[j]);
                t.age = 0;
                tracks[next_id] = t;
                matched_tids.insert(next_id);
                next_id++;
            }
        }

        // 5) Cancella tracce troppo vecchie
        std::vector<int> to_del;
        for (auto& kv : tracks) {
            if (kv.second.age > max_age) to_del.push_back(kv.first);
        }
        for (int tid : to_del) tracks.erase(tid);

        // 6) Output → mappa tid → posizione
        std::unordered_map<int,Eigen::Vector3f> out;
        for (auto& kv : tracks) {
            out[kv.first] = kv.second.kf.pos();
        }
        return out;
    }
};


// =====================================================
// LineCounter robusto
// =====================================================
struct LineCounter {
    float line;
    int axis;
    float deadzone;
    float min_cross;
    int cooldown_frames;
    int smooth_w;
    bool positive_is_out;
    static int global_cooldown;   // dichiara variabile

    int in_count = 0, out_count = 0;
    int last_cross_frame;   // membro normale

    struct State {
        std::deque<float> hist;
        std::string armed_side;
        float armed_pos = 0.f;
        int cooldown = 0;
    };
    std::unordered_map<int, State> states;

    LineCounter(float l, int ax, float dz, float mc, int cd, int sw, bool pos_out)
        : line(l), axis(ax), deadzone(dz), min_cross(mc),
          cooldown_frames(cd), smooth_w(sw), positive_is_out(pos_out),
          last_cross_frame(-1000) {}  // inizializzato qui

    std::string sideOf(float v) {
        if (v <= line - deadzone) return "neg";
        if (v >= line + deadzone) return "pos";
        return "mid";
    }

    void update(const std::unordered_map<int,Eigen::Vector3f>& tracks, int frame_id) {
        // scala cooldown
        for (auto& kv : states) {
            if (kv.second.cooldown > 0) kv.second.cooldown--;
        }

        for (auto& kv : tracks) {
            int tid = kv.first;
            float coord = kv.second[axis];

            // smoothing con mediana
            auto& st = states[tid];
            st.hist.push_back(coord);
            if ((int)st.hist.size() > smooth_w) st.hist.pop_front();
            std::vector<float> tmp(st.hist.begin(), st.hist.end());
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
            float s_coord = tmp[tmp.size()/2];

            std::string side = sideOf(s_coord);

            if (st.armed_side.empty()) {
                if (side=="neg" || side=="pos") {
                    st.armed_side = side;
                    st.armed_pos  = s_coord;
                }
            } else {
                if ((side=="neg" || side=="pos") && side != st.armed_side) {
                    float travel = std::fabs(s_coord - st.armed_pos);

                    if (travel >= min_cross && st.cooldown==0 && global_cooldown==0) {
                        std::string direction;
                        if (side=="pos" && st.armed_side=="neg")
                            direction = positive_is_out ? "OUT" : "IN";
                        else if (side=="neg" && st.armed_side=="pos")
                            direction = positive_is_out ? "IN" : "OUT";

                        if (!direction.empty()) {
                            if (direction=="IN")  { in_count++;  occ++; }
                            if (direction=="OUT") { out_count++; occ = std::max(0, occ-1); }

                            std::cout << "[CROSS] frame="<<frame_id
                                    << " tid="<<tid
                                    << " " << st.armed_side << "->" << side
                                    << " travel="<<travel
                                    << " dir="<<direction << "\n";

                            st.cooldown = cooldown_frames;
                            st.armed_side = side;
                            st.armed_pos  = s_coord;
                            global_cooldown = 60;   // blocca nuovi cross per ~2s a 30FPS
                        }
                    }
                } else if (side == st.armed_side) {
                    st.armed_pos = s_coord;
                }
            }
            states[tid] = st;
        }

        if (global_cooldown > 0) global_cooldown--;
    }

};

int LineCounter::global_cooldown = 0;



// =========================
// MAIN
// =========================
int main(int argc, char* argv[]) {
    open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Error);


    // all'inizio del main()
    auto t_start = std::chrono::high_resolution_clock::now();
    int fps_counter = 0;



    // --- argomenti
    std::string onnx_path = "tofnet.onnx";
    const char* cfg_path = nullptr;
    for (int i=1;i<argc;i++) {
        std::string a(argv[i]);
        if (a.rfind("--onnx=",0)==0) onnx_path = a.substr(7);
        else cfg_path = argv[i];
    }

    // --- camera
    ArducamTOFCamera tof;
    if (!initCamera(tof, cfg_path)) {
        std::cerr << "Errore apertura camera\n";
        return -1;
    }

    // Flip assi come nel dataset: Y=-Y, Z=-Z
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m << 1, 0, 0, 0,
         0,-1, 0, 0,
         0, 0,-1, 0,
         0, 0, 0, 1;

    // --- ONNX
    OnnxRunner runner(onnx_path);
    runner.set_input_unit_scale(1.0f);

    // --- viewer
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("ToFNet RT", 1280, 720);
    vis.AddGeometry(open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.5));
    vis.AddGeometry(pcd);

    int frame_idx = 0;
    while (true) {
        auto* frame = acquireFrame(tof);
        if (!frame) continue;
        auto cloud = generatePointCloud(tof, frame, m);
        tof.releaseFrame(frame);
        if (!cloud || cloud->points_.empty()) continue;

        // === Input per la RETE: campiona dalla cloud "piena" (come nel dataset) ===
        auto pts_m = samplePoints(cloud, kNumPoints, (uint32_t)frame_idx);

        // Diagnostica: range in metri
        if ((frame_idx % 30) == 0 && !pts_m.empty()) {
            float xmin=1e9f,xmax=-1e9f, ymin=1e9f,ymax=-1e9f, zmin=1e9f,zmax=-1e9f;
            for (auto &p : pts_m) {
                xmin = std::min(xmin, p.x());
                xmax = std::max(xmax, p.x());
                ymin = std::min(ymin, p.y());
                ymax = std::max(ymax, p.y());
                zmin = std::min(zmin, p.z());
                zmax = std::max(zmax, p.z());
            }
            std::cout << "[CHK METERS] X:["<<xmin<<","<<xmax<<"] "
                    << "Y:["<<ymin<<","<<ymax<<"] "
                    << "Z:["<<zmin<<","<<zmax<<"]\n";
        }

        // Inferenza
        float reg[2][3] = {{0}};
        int n = runner.infer(pts_m, reg);

        // === costruzione detections ===
        std::vector<Eigen::Vector3f> dets;
        auto reg_is_zero = [&](int i){
            return std::fabs(reg[i][0]) + std::fabs(reg[i][1]) + std::fabs(reg[i][2]) < 1e-4f;
        };

        if (n <= 0) {
            // none
        } else if (n == 1) {
            if (!reg_is_zero(0))
                dets.emplace_back(reg[0][0], reg[0][1], reg[0][2]);
            else
                dets.push_back(geometricCentroidM(cloud));
        } else {
            if (!reg_is_zero(0)) dets.emplace_back(reg[0][0], reg[0][1], reg[0][2]);
            if (!reg_is_zero(1)) dets.emplace_back(reg[1][0], reg[1][1], reg[1][2]);
        }

        // === filtro probabilità rete ===
        float prob_one = runner.last_probs[1];
        if (n == 1 && prob_one < 0.88f) {
            dets.clear();  // scarta detection poco sicura
        }

        // === Calibrazione rapida linea ===


        // === Tracking + filtro movimento ===
        static MultiPersonTracker tracker;
        static LineCounter counter(
            0.0f,    // linea x=0
            0,       // asse X
            0.05f,   // deadzone 3cm
            0.20f,   // min_cross 10cm
            20,       // cooldown 8 frame
            3,       // smoothing finestra 3
            true     // positive_is_out
        );
        static std::deque<float> recent_x;

        for (auto& d : dets) {
            recent_x.push_back(d.x());
            if (recent_x.size() > 10) recent_x.pop_front();
        }


        bool moving = false;
        if (recent_x.size() >= 5) {
            float delta = fabs(recent_x.back() - recent_x.front());
            moving = (delta > 0.05f); // almeno 5 cm
        }

        if (moving) {
            auto trk = tracker.update(dets);
            counter.update(trk, frame_idx);
        }

        // Viewer
        pcd->points_ = cloud->points_;
        pcd->colors_.assign(pcd->points_.size(), {0.6,0.6,0.6});

        if ((frame_idx % 30) == 0) {
            std::cout << "[RT] frame="<<frame_idx
                    << " n="<<n
                    << " IN="<<in_count
                    << " OUT="<<out_count
                    << " OCC="<<occ<<"\n";
        }

        frame_idx++;

        // dentro il while(true), alla fine del loop:
        fps_counter++;
        if (fps_counter >= 30) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double fps = fps_counter / elapsed;
            std::cout << "[FPS] " << fps << std::endl;
            fps_counter = 0;
            t_start = t_now;
        }

    }


    vis.DestroyVisualizerWindow();
    tof.stop();
    tof.close();
    return 0;
}
