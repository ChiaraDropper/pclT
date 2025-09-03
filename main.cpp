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
static constexpr int    MAX_DISTANCE     = 4000;
static constexpr int    CONFIDENCE_VALUE = 60;

static constexpr int    kAxis     = 0;     // X
static constexpr float  kDeadzone = 0.03f; // 3 cm

static constexpr size_t kNumPoints = 1024; // input alla rete
static constexpr float  kDistGate  = 1.0f; // gating (m) tracker
static constexpr int    kMaxAge    = 6;    // frame "persi" prima di cancellare track
static constexpr float  kMinCross  = 0.12f;// 10 cm min travel
static constexpr int    kCooldown  = 4;    // frame di cooldown anti-doppio
static constexpr int    kSmoothW   = 3;    // finestra smoothing (mediana)

// =========================
// Stato globale conteggio
// =========================
int in_count = 0, out_count = 0, occ = 0;
int frame_id = 0;
bool line_calibrated = false;
float line_pos = -0.20f;      // m, aggiornata dopo calibrazione
bool  positive_is_out = true; // verso OUT: neg->pos se true

// =========================
// Tracce
// =========================
struct Track {
    int id;
    Eigen::Vector3f pos;  // in metri
    int age = 0;
    int since_update = 0;
};
int next_track_id = 0;
std::unordered_map<int, Track> tracks;

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

    open3d::camera::PinholeCameraIntrinsic intr(format.width, format.height, fx, fy, cx, cy);

    // 4) depth_scale=1000.0, depth_trunc=2.5 → cloud in METRI
    auto cloud = open3d::geometry::PointCloud::CreateFromDepthImage(
        depth_img, intr, Eigen::Matrix4d::Identity(), 1000.0, 2.5);

    // 5) Flip assi (Y=-Y, Z=-Z) come dataset
    cloud->Transform(transform);

    // 6) (Stesso filtro "inutile ma innocuo" dello scheletro)
    std::vector<size_t> keep;
    keep.reserve(cloud->points_.size());
    for (size_t i = 0; i < cloud->points_.size(); ++i)
        if (cloud->points_[i](2) > -2000)
            keep.push_back(i);

    return cloud->SelectByIndex(keep);
}

// =========================
// Sampler 1024 punti (coerente al dataset)
// =========================
std::vector<Eigen::Vector3f>
samplePoints(const std::shared_ptr<open3d::geometry::PointCloud>& cloud,
             size_t N, uint32_t seed) {
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

Eigen::Vector3f geometricCentroidM(const std::shared_ptr<open3d::geometry::PointCloud>& cloud) {
    Eigen::Vector3f c(0,0,0);
    if (cloud->points_.empty()) return c;
    for (const auto& p : cloud->points_) c += Eigen::Vector3f((float)p(0),(float)p(1),(float)p(2));
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
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<const char*> input_names{"pts"};
    std::vector<const char*> output_names{"class_logits","reg_coords"};

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

        auto outputs = session->Run(Ort::RunOptions{nullptr},
                                    input_names.data(), &input_tensor, 1,
                                    output_names.data(), output_names.size());

        float* cls = outputs[0].GetTensorMutableData<float>(); // [1,3]
        int n = static_cast<int>(std::max_element(cls, cls + 3) - cls); // 0/1/2

        float* rc = outputs[1].GetTensorMutableData<float>();   // [1,2,3]
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                reg_out[i][j] = rc[i * 3 + j]; // già in METRI

        static int dbg_ctr = 0;
        if ((dbg_ctr++ % 30) == 0) {
            float xmin=1e9f,xmax=-1e9f, ymin=1e9f,ymax=-1e9f, zmin=1e9f,zmax=-1e9f;
            for (size_t i=0;i<input.size(); i+=3) {
                xmin = std::min(xmin, input[i+0]); xmax = std::max(xmax, input[i+0]);
                ymin = std::min(ymin, input[i+1]); ymax = std::max(ymax, input[i+1]);
                zmin = std::min(zmin, input[i+2]); zmax = std::max(zmax, input[i+2]);
            }
            std::cout << "[NET IN] scale=" << input_unit_scale
                      << " X:[" << xmin << "," << xmax << "]"
                      << " Y:[" << ymin << "," << ymax << "]"
                      << " Z:[" << zmin << "," << zmax << "]\n";
        }

        std::cout << "[ONNX LOGITS] " << cls[0] << " " << cls[1] << " " << cls[2] << std::endl;
        // debug: softmax
        {
            float m = std::max(cls[0], std::max(cls[1], cls[2]));
            double e0 = std::exp(cls[0]-m), e1 = std::exp(cls[1]-m), e2 = std::exp(cls[2]-m);
            double s = e0+e1+e2;
            std::cout << "[PROB] n0=" << e0/s << " n1=" << e1/s << " n2=" << e2/s << "\n";
        }
        return n;
    }
};

// =========================
// Tracker semplice (NN + gating)
// =========================
std::unordered_map<int, Eigen::Vector3f>
updateTracks(const std::vector<Eigen::Vector3f>& detections) {
    std::set<int> used;
    std::unordered_map<int, Eigen::Vector3f> out;

    for (auto& kv : tracks) kv.second.since_update++;

    for (const auto& det : detections) {
        float best = 1e9f; int best_id = -1;
        for (auto& kv : tracks) {
            if (used.count(kv.first)) continue;
            float d = (kv.second.pos - det).norm();
            if (d < best) { best = d; best_id = kv.first; }
        }
        if (best_id != -1 && best <= kDistGate) {
            auto& t = tracks[best_id];
            t.pos = det; t.since_update = 0; t.age++;
            used.insert(best_id);
            out[best_id] = t.pos;
        } else {
            int nid = next_track_id++;
            tracks[nid] = Track{nid, det, 1, 0};
            used.insert(nid);
            out[nid] = det;
        }
    }

    std::vector<int> to_del;
    for (auto& kv : tracks) if (kv.second.since_update > kMaxAge) to_del.push_back(kv.first);
    for (int id : to_del) tracks.erase(id);

    return out;
}

// =========================
// LineCounter
// =========================
inline char sideOf(float v, float line) {
    if (v <= line - kDeadzone) return 'n';
    if (v >= line + kDeadzone) return 'p';
    return 'm';
}

void updateLineCounter(const std::unordered_map<int, Eigen::Vector3f>& trk, int frame_idx) {
    for (auto& kv : trk) {
        int id = kv.first;
        float coord = kv.second[kAxis]; // X
        auto& st = lc_states[id];

        st.hist.push_back(coord);
        if ((int)st.hist.size() > kSmoothW) st.hist.pop_front();
        std::vector<float> tmp(st.hist.begin(), st.hist.end());
        std::nth_element(tmp.begin(), tmp.begin()+tmp.size()/2, tmp.end());
        float s_coord = tmp[tmp.size()/2];

        if (st.cooldown > 0) st.cooldown--;

        char sd = sideOf(s_coord, line_pos);

        if (st.armed_side == 0) {
            if (sd == 'n' || sd == 'p') { st.armed_side = sd; st.armed_pos = s_coord; }
        } else {
            if ((sd == 'n' || sd == 'p') && sd != st.armed_side) {
                float travel = std::fabs(s_coord - st.armed_pos);
                if (travel >= kMinCross && st.cooldown == 0) {
                    bool out_evt = (sd=='p' && st.armed_side=='n') ? positive_is_out
                                  : (sd=='n' && st.armed_side=='p') ? !positive_is_out
                                  : false;
                    if (out_evt) { out_count++; occ = std::max(0, occ - 1);
                        std::cout << "[CROSS OUT] frame="<<frame_idx<<" id="<<id
                                  <<" x="<<s_coord<<" line="<<line_pos<<" occ="<<occ<<"\n";
                    } else { in_count++; occ++;
                        std::cout << "[CROSS  IN] frame="<<frame_idx<<" id="<<id
                                  <<" x="<<s_coord<<" line="<<line_pos<<" occ="<<occ<<"\n";
                    }
                    st.cooldown = kCooldown;
                    st.armed_side = sd;
                    st.armed_pos = s_coord;
                }
            } else if (sd == st.armed_side) {
                st.armed_pos = s_coord;
            }
        }
    }
}

// =========================
// MAIN
// =========================
int main(int argc, char* argv[]) {
    open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Error);

    // --- argomenti
    std::string onnx_path = "tofnet.onnx";
    const char* cfg_path  = nullptr;
    for (int i=1;i<argc;i++) {
        std::string a(argv[i]);
        if (a.rfind("--onnx=",0)==0) onnx_path = a.substr(7);
        else cfg_path = argv[i];
    }

    // --- camera
    ArducamTOFCamera tof;
    if (!initCamera(tof, cfg_path)) {
        std::cerr << "Errore apertura camera\n"; return -1;
    }

    // Flip assi come nel dataset: Y=-Y, Z=-Z
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m << 1, 0, 0, 0,
         0,-1, 0, 0,
         0, 0,-1, 0,
         0, 0, 0, 1;

    // --- ONNX
    OnnxRunner runner(onnx_path);
    runner.set_input_unit_scale(1.0f); // <<<<< PUNTI IN METRI

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
                xmin = std::min(xmin, p.x()); xmax = std::max(xmax, p.x());
                ymin = std::min(ymin, p.y()); ymax = std::max(ymax, p.y());
                zmin = std::min(zmin, p.z()); zmax = std::max(zmax, p.z());
            }
            std::cout << "[CHK METERS] X:["<<xmin<<","<<xmax<<"] Y:["<<ymin<<","<<ymax
                      <<"] Z:["<<zmin<<","<<zmax<<"]\n";
        }

        // Inferenza (punti assoluti, nessun centering)
        float reg[2][3] = {{0}};
        int n = runner.infer(pts_m, reg);

        // Detections (in METRI)
        std::vector<Eigen::Vector3f> dets;
        auto reg_is_zero = [&](int i){
            return std::fabs(reg[i][0]) + std::fabs(reg[i][1]) + std::fabs(reg[i][2]) < 1e-4f;
        };
        if (n <= 0) {
            // none
        } else if (n == 1) {
            if (!reg_is_zero(0)) dets.emplace_back(reg[0][0], reg[0][1], reg[0][2]);
            else                 dets.push_back( geometricCentroidM(cloud) );
        } else {
            if (!reg_is_zero(0)) dets.emplace_back(reg[0][0], reg[0][1], reg[0][2]);
            if (!reg_is_zero(1)) dets.emplace_back(reg[1][0], reg[1][1], reg[1][2]);
        }

        // Calibrazione rapida
        if (!line_calibrated) {
            if (!dets.empty()) {
                float x = dets.size()==1 ? dets[0].x()
                                         : dets[ std::max_element(dets.begin(), dets.end(),
                                           [](auto&a, auto&b){return std::fabs(a.y())<std::fabs(b.y());}) - dets.begin() ].x();
                static std::vector<float> xs;
                xs.push_back(x);
                if ((int)xs.size() >= 25) {
                    std::vector<float> v = xs;
                    std::sort(v.begin(), v.end());
                    auto p20 = v[(int)(0.2f*v.size())];
                    auto p80 = v[(int)(0.8f*v.size())];
                    line_pos = 0.5f*(p20+p80);
                    int k = std::max(1, (int)v.size()/5);
                    float start_med = std::accumulate(v.begin(), v.begin()+k, 0.f) / k;
                    float end_med   = std::accumulate(v.end()-k,   v.end(),   0.f) / k;
                    positive_is_out = (end_med > start_med);
                    line_calibrated = true;
                    std::cout << "[CAL] axis="<<kAxis<<" line="<<line_pos<<" m  dir="<<(positive_is_out?"OUT=+":"OUT=-")<<"\n";
                }
            }
        }

        // Tracking
        auto trk = updateTracks(dets);

        // Line counter
        if (line_calibrated) updateLineCounter(trk, frame_idx);

        // Viewer (cloud piena, colori neutri)
        pcd->points_ = cloud->points_;
        pcd->colors_.assign(pcd->points_.size(), {0.6,0.6,0.6});
        vis.UpdateGeometry(pcd);
        static bool did_fit = false;
        if (!did_fit && !pcd->IsEmpty()) {
            vis.ResetViewPoint(true);
            did_fit = true;
        }
        vis.PollEvents();
        vis.UpdateRender();

        if ((frame_idx % 30) == 0) {
            std::cout << "[RT] frame="<<frame_idx<<" n="<<n
                      <<" IN="<<in_count<<" OUT="<<out_count<<" OCC="<<occ<<"\n";
        }

        frame_idx++;
    }

    vis.DestroyVisualizerWindow();
    tof.stop();
    tof.close();
    return 0;
}
