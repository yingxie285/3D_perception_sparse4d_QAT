// Detection-only Sparse4D, four fisheye cameras, J6M UCP.
// UCP RunInfer copied from SparseDrive sparsedrive_run.cc.
// Fisheye preprocessing/decoder retained from validated Sparse4D adaptation.
#include "detection.hpp"
#include "tensor_io.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "path_io.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cfenv>
#include <cctype>
#include <cstdlib>
namespace fs=sparse4d::files;
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
using namespace sparse4d;
static double ms(Clock::time_point t) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}
static bool trace_enabled=false;
static const auto process_start=Clock::now();
static void trace(const std::string& message) {
    if(trace_enabled) std::cerr<<"[stage +"<<std::fixed<<std::setprecision(1)
                              <<ms(process_start)<<" ms] "<<message<<std::endl;
}
template<class F> static int traced_call(const std::string& name,F&& fn) {
    trace(name+" ENTER");
    auto begin=Clock::now();
    int rc=fn();
    trace(name+" RETURN rc="+std::to_string(rc)+" elapsed_ms="+std::to_string(ms(begin)));
    return rc;
}
static void require(bool ok,const std::string& msg) { if (!ok) throw std::runtime_error(msg); }
static void save_json(const fs::path& p,const json& j) {
    std::ofstream out(p.string()); require(bool(out),"Cannot write "+p.string());
    out<<std::setw(2)<<j<<"\n"; require(bool(out),"Write failed "+p.string());
}
static json read_json(const fs::path& p) {
    std::ifstream in(p.string()); require(bool(in),"Cannot open "+p.string());
    json j; in>>j; return j;
}
struct Options {
    fs::path model,frame,out="output";
    bool inspect=false,dump=false,preprocess_only=false,trace=false;
    int warmup=0,repeat=1,top_k=300;
    float threshold=.5f;
    std::string model_name;
};
#include "gflags/gflags.h"
DEFINE_string(model_file, "model.hbm", "Path to .hbm model");
DEFINE_string(manifest, "", "Single frame JSON or JSON list of frame JSON paths");
DEFINE_string(dump_dir, "./output", "Output directory");
DEFINE_string(model_name, "forward", "HBM graph name");
DEFINE_string(model_io, "", "Expected model interface JSON");
DEFINE_string(packed_input_dir, "", "Optional raw padded input_0.bin ... input_6.bin");
DEFINE_bool(preprocess_only, false, "Dump logical inputs without loading HBM");
DEFINE_bool(inspect_only, false, "Inspect HBM without reading images");
DEFINE_bool(save_outputs, true, "Save model and detection outputs");
DEFINE_bool(dump_inputs, false, "Save logical and packed inputs");
DEFINE_bool(visualize, true, "Save camera overlays and BEV");
DEFINE_bool(trace, true, "Print execution stages");
DEFINE_int32(num_frames, 1, "Frames to run, -1 for all");
DEFINE_int32(core_id, 0, "0=any, 1=core0 (J6M)");
DEFINE_int32(task_timeout_ms, 0, "UCP wait timeout; 0 follows reference infinite wait");
DEFINE_int32(warmup, 0, "Warmup iterations");
DEFINE_int32(repeat, 1, "Measured iterations per frame");
DEFINE_int32(top_k, 300, "Global detection top-k");
DEFINE_double(score_threshold, 0.0, "Saved detection threshold");
DEFINE_double(visualize_score_threshold, 0.5, "Display threshold only");
static Options options(const std::string& frame, const std::string& output) {
    Options o; o.model=FLAGS_model_file; o.frame=frame; o.out=output;
    o.model_name=FLAGS_model_name; o.inspect=FLAGS_inspect_only;
    o.preprocess_only=FLAGS_preprocess_only; o.dump=FLAGS_dump_inputs;
    o.trace=FLAGS_trace; o.warmup=FLAGS_warmup; o.repeat=FLAGS_repeat;
    o.top_k=FLAGS_top_k; o.threshold=float(FLAGS_score_threshold);
    require(o.repeat>0 && o.warmup>=0 && o.top_k>0,"Invalid iteration/top-k settings");
    require(o.threshold>=0 && o.threshold<=1,"Invalid score threshold");
    require(FLAGS_core_id==0 || FLAGS_core_id==1,"J6M core_id must be 0 or 1");
    require(FLAGS_task_timeout_ms>=0,"Negative task timeout");
    return o;
}
template<size_t N> static std::array<double,N> numbers(const json& j) {
    std::vector<double> flat;
    for(const auto& row:j) {
        if(row.is_array()) for(const auto& x:row) flat.push_back(x.get<double>());
        else flat.push_back(row.get<double>());
    }
    require(flat.size()==N,"Calibration matrix has wrong element count.");
    std::array<double,N> a{};
    for(size_t i=0;i<N;++i) {require(std::isfinite(flat[i]),"Non-finite calibration.");a[i]=flat[i];}
    return a;
}
struct Camera {
    std::string name; fs::path path; cv::Mat image;
    std::array<double,9> k;
    std::array<double,16> e,lidar2img;
    std::array<double,4> distortion;
};
struct LogicalTensor {std::vector<int> shape;std::vector<float> data;};
struct Frame {
    json manifest;
    std::vector<Camera> cameras;
    std::map<std::string,LogicalTensor> inputs;
};
static constexpr const char* kInputs[]={
    "img","cam_extrinsic","cam_intrinsic","cam_distortion","lidar2img","image_wh","time_interval"};
static void append(std::vector<float>& dst,const double* src,size_t n) {
    for(size_t i=0;i<n;++i) dst.push_back(float(src[i]));
}
static Frame preprocess(const fs::path& path) {
    Frame f; f.manifest=read_json(path);
    const auto& j=f.manifest;
    require(j.at("schema_version")==1,"Unsupported manifest schema.");
    require(j.at("cameras").size()==4,"Exactly four cameras required.");
    const auto& p=j.at("preprocess");
    require(!p.value("flip",false) && p.value("rotate",0.)==0.,
            "Only deterministic test augmentation supported.");
    double resize=p.at("resize").get<double>();
    auto crop=p.at("crop").get<std::vector<int>>();
    require(crop.size()==4 && crop[2]-crop[0]==704 && crop[3]-crop[1]==256,"Expected 704x256 crop.");
    require(std::isfinite(resize) && resize>0 && resize<=10,"Invalid resize.");
    auto mean=numbers<3>(p.at("mean")),stddev=numbers<3>(p.at("std"));
    // Python stores config mean/std as float32 before promoting them for OpenCV.
    for(double& v:mean) v=double(float(v));
    for(double& v:stddev) v=double(float(v));
    for(double v:stddev) require(v>0,"Normalization std must be positive.");
    f.inputs["img"].shape={1,4,3,256,704};
    f.inputs["cam_extrinsic"].shape={1,4,4,4};
    f.inputs["cam_intrinsic"].shape={1,4,3,3};
    f.inputs["cam_distortion"].shape={1,4,4};
    f.inputs["lidar2img"].shape={1,4,4,4};
    f.inputs["image_wh"]={{1,2},{256.f,704.f}};
    f.inputs["time_interval"]={{1},{float(j.value("time_interval",1.0))}};
    require(f.inputs["time_interval"].data[0]==1.f,"Non-temporal export requires time_interval=1.");
    std::set<std::string> seen;
    for(const auto& cj:j.at("cameras")) {
        Camera c;
        c.name=cj.at("name").get<std::string>();
        require(seen.insert(c.name).second,"Duplicate camera name.");
        c.path=fs::path(cj.at("image").get<std::string>());
        if(c.path.is_relative()) c.path=path.parent_path()/c.path;
        c.k=numbers<9>(cj.at("cam_intrinsic")); c.e=numbers<16>(cj.at("cam_extrinsic"));
        c.distortion=numbers<4>(cj.at("cam_distortion"));
        c.lidar2img=numbers<16>(cj.at("lidar2img"));
        c.image=cv::imread(c.path.string(),cv::IMREAD_COLOR);
        require(!c.image.empty(),"Cannot load "+c.path.string());
        if(!f.cameras.empty()) require(c.image.size()==f.cameras[0].image.size(),"Camera dimensions differ.");
        int rw=int(c.image.cols*resize),rh=int(c.image.rows*resize);
        require(rw>0 && rh>0,"Empty resize.");
        std::cout<<"camera["<<f.cameras.size()<<"] "<<c.name<<" "<<c.image.cols<<"x"<<c.image.rows
                 <<" -> "<<rw<<"x"<<rh<<" -> crop "<<crop[0]<<","<<crop[1]<<",704,256\n";
        cv::Mat resized,patch=cv::Mat::zeros(256,704,CV_8UC3);
        // _img_transform recomputes dsize from actual image dimensions.
        cv::resize(c.image,resized,cv::Size(rw,rh),0,0,cv::INTER_LINEAR);
        cv::Rect region(crop[0],crop[1],704,256),available(0,0,rw,rh);
        cv::Rect intersection=region & available;
        // PIL.crop fills out-of-bounds pixels with zero.
        if(intersection.area()>0) resized(intersection).copyTo(patch(cv::Rect(
            intersection.x-region.x,intersection.y-region.y,intersection.width,intersection.height)));
        cv::Mat normalized;patch.convertTo(normalized,CV_32FC3);
        if(p.value("to_rgb",true)) cv::cvtColor(normalized,normalized,cv::COLOR_BGR2RGB);
        cv::subtract(normalized,cv::Scalar(mean[0],mean[1],mean[2]),normalized);
        cv::multiply(normalized,cv::Scalar(1/stddev[0],1/stddev[1],1/stddev[2]),normalized);
        auto& img=f.inputs["img"].data;
        for(int ch=0;ch<3;++ch) for(int y=0;y<256;++y) {
            const auto* row=normalized.ptr<cv::Vec3f>(y);
            for(int x=0;x<704;++x) img.push_back(row[x][ch]);
        }
        append(f.inputs["cam_extrinsic"].data,c.e.data(),16);
        auto inference_k=c.k;
        // Exact repo behavior: scale entire 3x3 including K[2,2]; no crop subtraction in K.
        for(auto& v:inference_k) v*=resize;
        append(f.inputs["cam_intrinsic"].data,inference_k.data(),9);
        append(f.inputs["cam_distortion"].data,c.distortion.data(),4);
        auto l2i=c.lidar2img;
        for(int col=0;col<4;++col) {
            l2i[col]=resize*c.lidar2img[col]-crop[0]*c.lidar2img[8+col];
            l2i[4+col]=resize*c.lidar2img[4+col]-crop[1]*c.lidar2img[8+col];
        }
        append(f.inputs["lidar2img"].data,l2i.data(),16);
        // image_wh is one [H,W] pair per batch, NOT one pair per camera.
        f.cameras.push_back(std::move(c));
    }
    return f;
}
static void dump_floats(const fs::path& path,const std::vector<float>& data) {
    std::ofstream out(path.string(),std::ios::binary);require(bool(out),"Cannot write "+path.string());
    out.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()*sizeof(float)));
    require(bool(out),"Write failed "+path.string());
}
static void dump_inputs(const Frame& f,const fs::path& out) {
    fs::create_directories(out);json info=json::array();
    for(const auto* name:kInputs) {
        const auto& t=f.inputs.at(name);
        dump_floats(out/(std::string(name)+".bin"),t.data);
        info.push_back({{"semantic",name},{"shape",t.shape},{"dtype","float32"},{"path",std::string(name)+".bin"}});
    }
    save_json(out/"reference_inputs.json",{{"schema_version",1},{"inputs",info}});
}
static int RunInfer(std::vector<hbDNNTensor>& inputs, std::vector<hbDNNTensor>& outputs,
                    hbDNNHandle_t handle, uint64_t backend) {
  // Submit the asynchronous HBM task, wait for completion, and refresh output
  // properties before the buffers are consumed or dumped.
  hbUCPTaskHandle_t task = nullptr;
  int ret = hbDNNInferV2(&task, outputs.data(), inputs.data(), handle);
  if (ret) return ret;
  hbUCPSchedParam sp;
  HB_UCP_INITIALIZE_SCHED_PARAM(&sp);
  sp.backend = backend;
  ret = hbUCPSubmitTask(task, &sp);
  if (!ret) ret = hbUCPWaitTaskDone(task, FLAGS_task_timeout_ms);
  if (!ret) {
    for (size_t i = 0; i < outputs.size(); ++i) {
      ret = hbDNNGetTaskOutputTensorProperties(&outputs[i].properties, task, 0, i);
      if (ret) break;
    }
  }
  int release_ret = hbUCPReleaseTask(task);
  if (ret) return ret;
  if (release_ret) return release_ret;
  return 0;
}


struct Model {
    hbDNNPackedHandle_t packed=nullptr;hbDNNHandle_t handle=nullptr;
    std::vector<hbDNNTensor> inputs,outputs;
    std::vector<std::string> input_names,output_names;
    Model()=default;Model(const Model&)=delete;Model& operator=(const Model&)=delete;
    ~Model() {
        for(auto& t:inputs) if(t.sysMem.virAddr) hbUCPFree(&t.sysMem);
        for(auto& t:outputs) if(t.sysMem.virAddr) hbUCPFree(&t.sysMem);
        if(packed) hbDNNRelease(packed);
    }
    void load(const Options& o) {
        std::string filename=o.model.string();const char* path=filename.c_str();
        check(hbDNNInitializeFromFiles(&packed,&path,1),"hbDNNInitializeFromFiles");
        const char** names=nullptr;int count=0;
        check(hbDNNGetModelNameList(&names,&count,packed),"hbDNNGetModelNameList");
        require(count>0,"HBM contains no model.");
        require(count==1 || !o.model_name.empty(),"Multi-graph HBM needs --model_name.");
        check(hbDNNGetModelHandle(&handle,packed,o.model_name.empty()?names[0]:o.model_name.c_str()),"hbDNNGetModelHandle");
        int ni=0,no=0;
        check(hbDNNGetInputCount(&ni,handle),"hbDNNGetInputCount");
        check(hbDNNGetOutputCount(&no,handle),"hbDNNGetOutputCount");
        require(ni>0 && no>0,"Empty IO contract.");
        inputs.resize(ni);outputs.resize(no);
        for(int i=0;i<ni;++i) {
            const char* name=nullptr;
            check(hbDNNGetInputName(&name,handle,i),"hbDNNGetInputName");
            check(hbDNNGetInputTensorProperties(&inputs[i].properties,handle,i),"input properties");
            input_names.emplace_back(name);
        }
        for(int i=0;i<no;++i) {
            const char* name=nullptr;
            check(hbDNNGetOutputName(&name,handle,i),"hbDNNGetOutputName");
            check(hbDNNGetOutputTensorProperties(&outputs[i].properties,handle,i),"output properties");
            output_names.emplace_back(name);
        }
    }
    json describe() const {
        json all={{"dnn_version",hbDNNGetVersion()},{"inputs",json::array()},{"outputs",json::array()}};
        for(int side=0;side<2;++side) {
            const auto& ts=side?outputs:inputs;const auto& names=side?output_names:input_names;
            for(size_t i=0;i<ts.size();++i) {
                const auto& p=ts[i].properties;
                json t={{"name",names[i]},{"shape",shape_of(p)},{"dtype",dtype_name(p.tensorType)},
                        {"aligned_bytes",p.alignedByteSize},{"quantize_axis",p.quantizeAxis},
                        {"quanti_type",p.quantiType==SCALE?"SCALE":"NONE"}};
                t["strides_bytes"]=json::array();
                for(int d=0;d<p.validShape.numDimensions;++d) t["strides_bytes"].push_back(p.stride[d]);
                t["scales"]=json::array();t["zero_points"]=json::array();
                if(p.quantiType==SCALE) {
                    require(p.scale.scaleLen<=0 || p.scale.scaleData,"Missing scale pointer.");
                    require(p.scale.zeroPointLen<=0 || p.scale.zeroPointData,"Missing zero point pointer.");
                    for(int q=0;q<p.scale.scaleLen;++q) t["scales"].push_back(p.scale.scaleData[q]);
                    for(int q=0;q<p.scale.zeroPointLen;++q) t["zero_points"].push_back(p.scale.zeroPointData[q]);
                }
                all[side?"outputs":"inputs"].push_back(t);
                std::cout<<(side?"output ":"input ")<<i<<" "<<t.dump()<<"\n";
            }
        }
        return all;
    }
    void allocate() {
        for(auto* ts:{&inputs,&outputs}) for(size_t i=0;i<ts->size();++i) {
            auto& t=ts->at(i);
            size_t bytes=resolve_strides(t.properties);
            std::string label="hbUCPMallocCached "+std::string(ts==&inputs?"input[":"output[")+
                              std::to_string(i)+"] bytes="+std::to_string(bytes);
            check(traced_call(label,[&]{return hbUCPMallocCached(&t.sysMem,bytes,0);}),"hbUCPMallocCached");
        }
    }
    double infer() {
        auto start=Clock::now();
        trace("RunInfer ENTER");
        int rc=RunInfer(inputs,outputs,handle,FLAGS_core_id==0?HB_UCP_BPU_CORE_ANY:HB_UCP_BPU_CORE_0);
        trace("RunInfer RETURN rc="+std::to_string(rc));
        check(rc,"SparseDrive RunInfer");
        return ms(start);
    }

};
static std::vector<int> squeezed(std::vector<int> dims) {
    dims.erase(std::remove(dims.begin(),dims.end(),1),dims.end());return dims;
}
static size_t binding(const std::vector<std::string>& names,const json& j,
                      const std::string& section,const std::string& semantic,int index) {
    std::string name=std::string(section=="input_bindings"?"_input_":"_output_")+std::to_string(index);
    if(std::find(names.begin(),names.end(),semantic)!=names.end()) name=semantic;
    if(j.contains(section) && j.at(section).contains(semantic)) name=j.at(section).at(semantic).get<std::string>();
    auto it=std::find(names.begin(),names.end(),name);
    require(it!=names.end(),"Missing "+name+"; check hbm_interface.json and manifest "+section);
    return size_t(it-names.begin());
}
static std::vector<std::vector<float>> bind_inputs(const Model& m,const Frame& f) {
    require(m.inputs.size()==7 && m.outputs.size()==2,"Expected detection-only 7-input 2-output HBM.");
    std::vector<std::vector<float>> result(7);std::set<size_t> used;
    for(int i=0;i<7;++i) {
        std::string semantic=kInputs[i];
        size_t n=binding(m.input_names,f.manifest,"input_bindings",semantic,i);
        require(used.insert(n).second,"Duplicate binding.");
        const auto& logical=f.inputs.at(semantic);
        auto actual=squeezed(shape_of(m.inputs[n].properties)),expected=squeezed(logical.shape);
        if(actual==expected) result[n]=logical.data;
        else if(semantic=="img" && actual==std::vector<int>({4,256,704,3})) {
            result[n].resize(logical.data.size());
            for(size_t cam=0;cam<4;++cam) for(size_t y=0;y<256;++y)
                for(size_t x=0;x<704;++x) for(size_t c=0;c<3;++c)
                    result[n][((cam*256+y)*704+x)*3+c]=logical.data[((cam*3+c)*256+y)*704+x];
        } else throw std::runtime_error("Unsupported "+semantic+" shape "+json(actual).dump()+
                                        "; expected "+json(expected).dump());
    }
    return result;
}
static void pack_and_clean(Model& m,const std::vector<std::vector<float>>& values) {
    for(size_t i=0;i<values.size();++i) {
        trace("pack input["+std::to_string(i)+"] ENTER");
        if(FLAGS_packed_input_dir.empty()) pack_tensor(m.inputs[i],values[i]);
        else {
            auto& t=m.inputs[i];
            std::ifstream raw((fs::path(FLAGS_packed_input_dir)/("input_"+std::to_string(i)+".bin")).string(),std::ios::binary|std::ios::ate);
            require(bool(raw) && raw.tellg()==std::streamoff(t.properties.alignedByteSize),"Packed input byte count mismatch");
            raw.seekg(0); raw.read(static_cast<char*>(t.sysMem.virAddr),t.properties.alignedByteSize);
            require(bool(raw),"Packed input read failed");
        }
        trace("pack input["+std::to_string(i)+"] DONE");
        check(traced_call("input cache clean ["+std::to_string(i)+"]",
            [&]{return hbUCPMemFlush(&m.inputs[i].sysMem,HB_SYS_MEM_CACHE_CLEAN);}),"input cache clean");
    }
}
static void dump_packed_inputs(const Model& m,const fs::path& out) {
    fs::create_directories(out);
    json items=json::array();
    for(size_t i=0;i<m.inputs.size();++i) {
        const auto& t=m.inputs[i];
        std::string filename="input_"+std::to_string(i)+".bin";
        std::ofstream stream((out/filename).string(),std::ios::binary);
        stream.write(static_cast<const char*>(t.sysMem.virAddr),std::streamsize(t.properties.alignedByteSize));
        require(bool(stream),"Cannot write packed input "+filename);
        items.push_back({{"index",i},{"name",m.input_names[i]},{"path",filename},
                         {"bytes",t.properties.alignedByteSize},{"dtype",dtype_name(t.properties.tensorType)}});
    }
    save_json(out/"manifest.json",{{"description","Raw quantized HBM buffers WITH padding; for hrt_model_exec --input_file, not logical float32 inputs"},{"inputs",items}});
}
static json detection_json(const std::vector<Detection>& detections) {
    json a=json::array();
    for(const auto& d:detections) a.push_back({{"query",d.query},{"label",d.label},
        {"class_name",kClasses[d.label]},{"score",d.score},{"box_lidar_xyz_wlh_yaw",d.box}});
    return a;
}
static std::string safe_name(std::string t) {
    for(char& c:t) if(!std::isalnum(static_cast<unsigned char>(c)) && c!='-' && c!='_') c='_';
    return t.empty()?"camera":t;
}
static cv::Point image_point(const std::array<double,2>& p) {
    return {int(std::clamp(p[0],-1e4,1e5)),int(std::clamp(p[1],-1e4,1e5))};
}
static bool inside(const cv::Mat& im,const cv::Point& p) {
    return p.x>=0 && p.y>=0 && p.x<im.cols && p.y<im.rows;
}
static void save_image(const fs::path& p,const cv::Mat& im) {
    require(cv::imwrite(p.string(),im),"Cannot save "+p.string());
}
static void visualize(const Frame& f,const std::vector<Detection>& detections,const fs::path& out) {
    cv::Scalar green(0,255,0);
    cv::Mat mosaic(800,1280,CV_8UC3,cv::Scalar(25,25,25));
    for(size_t cam=0;cam<f.cameras.size();++cam) {
        const auto& c=f.cameras[cam];cv::Mat overlay=c.image.clone();
        for(const auto& d:detections) {
            auto lidar=corners(d);std::array<cv::Point,8> points{};
            bool front=false,finite=true;
            for(int v=0;v<8;++v) {
                Point3 cp{};
                for(int r=0;r<3;++r) cp[r]=c.e[r*4]*lidar[v][0]+c.e[r*4+1]*lidar[v][1]+c.e[r*4+2]*lidar[v][2]+c.e[r*4+3];
                front|=cp[2]>0;auto uv=fisheye(cp,c.k,c.distortion);
                finite&=std::isfinite(uv[0]) && std::isfinite(uv[1]);
                if(finite) points[v]=image_point(uv);
            }
            if(!front || !finite) continue;
            bool drawn=false;
            for(const auto& edge:kEdges) {
                const auto& a=points[edge[0]];const auto& b=points[edge[1]];
                if(!inside(overlay,a) && !inside(overlay,b)) continue;
                cv::line(overlay,a,b,green,3,cv::LINE_AA);drawn=true;
            }
            if(drawn) {
                std::ostringstream label;label<<kClasses[d.label]<<" "<<std::fixed<<std::setprecision(2)<<d.score;
                cv::Point origin((points[2].x+points[3].x)/2,(points[0].y+points[1].y)/2);
                origin.x=std::clamp(origin.x,0,std::max(0,overlay.cols-200));
                origin.y=std::clamp(origin.y,20,std::max(20,overlay.rows-5));
                cv::putText(overlay,label.str(),origin,cv::FONT_HERSHEY_SIMPLEX,.6,green,2,cv::LINE_AA);
            }
        }
        cv::putText(overlay,c.name,{20,40},cv::FONT_HERSHEY_SIMPLEX,1.,{0,255,255},2);
        save_image(out/(std::to_string(cam)+"_"+safe_name(c.name)+".jpg"),overlay);
        cv::Mat small;cv::resize(overlay,small,cv::Size(640,400));
        small.copyTo(mosaic(cv::Rect(int(cam%2)*640,int(cam/2)*400,640,400)));
    }
    save_image(out/"four_cameras.jpg",mosaic);
    cv::Mat bev(1000,1000,CV_8UC3,cv::Scalar(28,28,28));
    auto pt=[](double x,double y) {
        return cv::Point(int(std::clamp(500+x*12.,-1e4,1e4)),int(std::clamp(500-y*12.,-1e4,1e4)));
    };
    for(int v=-40;v<=40;v+=5) {
        cv::line(bev,pt(v,-40),pt(v,40),{55,55,55});cv::line(bev,pt(-40,v),pt(40,v),{55,55,55});
    }
    cv::arrowedLine(bev,pt(0,0),pt(5,0),{0,0,255},2);
    cv::arrowedLine(bev,pt(0,0),pt(0,5),{255,150,0},2);
    cv::putText(bev,"LiDAR BEV: +x right, +y up; grid 5 m",{15,25},cv::FONT_HERSHEY_SIMPLEX,.65,{255,255,255},1);
    for(const auto& d:detections) {
        auto cs=corners(d);constexpr int ids[4]={0,3,7,4};
        for(int k=0;k<4;++k) cv::line(bev,pt(cs[ids[k]][0],cs[ids[k]][1]),
            pt(cs[ids[(k+1)%4]][0],cs[ids[(k+1)%4]][1]),green,2,cv::LINE_AA);
        cv::putText(bev,kClasses[d.label],pt(d.box[0],d.box[1]),cv::FONT_HERSHEY_SIMPLEX,.4,green,1);
    }
    save_image(out/"bev.jpg",bev);
}
static int run_frame(const Options& o, Model& m) {
    auto frame_start=Clock::now();
    try {
        std::fesetround(FE_TONEAREST);
        trace_enabled=o.trace;
        trace("sparse4d_J6M diagnostic revision 2; warmup="+std::to_string(o.warmup)+" repeat="+std::to_string(o.repeat));
        fs::create_directories(o.out);json timing;Frame f;
        if(!o.inspect) {
            auto t=Clock::now();f=preprocess(o.frame);timing["load_images_and_preprocess_ms"]=ms(t);
            if(o.dump || o.preprocess_only) dump_inputs(f,o.out/"inputs");
            if(o.preprocess_only) {save_json(o.out/"timing.json",timing);return 0;}
        }
        auto load_start=Clock::now();if(!m.packed) m.load(o);timing["model_load_ms"]=ms(load_start);
        auto interface=m.describe();
        if(!FLAGS_model_io.empty()) {
            auto expected=read_json(FLAGS_model_io);
            for(const auto* side:{"inputs","outputs"}) {
                require(expected.at(side).size()==interface.at(side).size(),"HBM IO count mismatch");
                for(size_t i=0;i<expected.at(side).size();++i)
                    for(auto it=expected.at(side)[i].begin();it!=expected.at(side)[i].end();++it)
                        require(interface.at(side)[i].at(it.key())==it.value(),"HBM IO property mismatch: "+it.key());
            }
        }
        trace("write hbm_interface.json ENTER");
        save_json(o.out/"hbm_interface.json",interface);
        trace("write hbm_interface.json DONE");
        if(o.inspect) return 0;
        trace("bind inputs ENTER");
        auto values=bind_inputs(m,f);
        trace("bind inputs DONE");
        size_t cls=binding(m.output_names,f.manifest,"output_bindings","classification",0);
        size_t box=binding(m.output_names,f.manifest,"output_bindings","prediction",1);
        require(cls!=box,"Duplicate output binding.");
        for(size_t n:{cls,box}) require(is_float_type(m.outputs[n].properties.tensorType) ||
            m.outputs[n].properties.quantiType==SCALE,
            "Integer detection output has no quantization scale; cannot decode physical boxes/logits.");
        auto cs=squeezed(shape_of(m.outputs[cls].properties)),bs=squeezed(shape_of(m.outputs[box].properties));
        require(cs.size()==2 && bs.size()==2 && cs[0]==bs[0] && cs[1]==13 && bs[1]==8,
                "Expected class [1,Q,13] and boxes [1,Q,8]; inspect hbm_interface.json.");
        if(!m.inputs[0].sysMem.virAddr) m.allocate();
        auto pack_start=Clock::now();pack_and_clean(m,values);timing["input_pack_and_clean_ms"]=ms(pack_start);
        if(o.dump) {
            trace("dump packed inputs ENTER");
            dump_packed_inputs(m,o.out/"packed_inputs");
            trace("dump packed inputs DONE");
        }
        std::vector<double> times;
        for(int i=0;i<o.warmup+o.repeat;++i) {
            if(i>0) pack_and_clean(m,values); // restore inputs potentially modified by in-place ops
            double elapsed=m.infer();
            if(i>=o.warmup) times.push_back(elapsed);
            std::cout<<"inference "<<i+1<<"/"<<o.warmup+o.repeat<<": "<<elapsed<<" ms"
                     <<(i<o.warmup?" (warmup)":"")<<"\n";
        }
        auto read_start=Clock::now();
        for(auto& t:m.outputs) check(traced_call("output cache invalidate",
            [&]{return hbUCPMemFlush(&t.sysMem,HB_SYS_MEM_CACHE_INVALIDATE);}),"output cache invalidate");
        trace("output unpack ENTER");
        auto logits=unpack_tensor(m.outputs[cls]),boxes=unpack_tensor(m.outputs[box]);
        if(FLAGS_save_outputs) {
            auto metadata=m.describe();
            for(size_t i=0;i<m.outputs.size();++i) {
                const auto& t=m.outputs[i];
                auto name="output_"+std::to_string(i)+".bin";
                std::ofstream raw((o.out/name).string(),std::ios::binary);
                for(size_t n=0;n<elements(t.properties);++n)
                    raw.write(static_cast<const char*>(t.sysMem.virAddr)+byte_offset(t.properties,n),element_bytes(t.properties.tensorType));
                require(bool(raw),"Cannot save compact raw output");
                metadata["outputs"][i]["path"]=name;
                metadata["outputs"][i]["storage"]="compact C-order native dtype, no padding";
            }
            save_json(o.out/"raw_outputs.json",metadata);
        }
        trace("output unpack DONE");
        timing["output_invalidate_and_unpack_ms"]=ms(read_start);
        if(FLAGS_save_outputs) {
        dump_floats(o.out/"classification_f32.bin",logits);dump_floats(o.out/"prediction_f32.bin",boxes);
        save_json(o.out/"outputs.json",{{"inputs",json::array({
            {{"semantic","classification"},{"shape",shape_of(m.outputs[cls].properties)},{"dtype","float32"},{"path","classification_f32.bin"}},
            {{"semantic","prediction"},{"shape",shape_of(m.outputs[box].properties)},{"dtype","float32"},{"path","prediction_f32.bin"}}})}});
        }
        trace("decode ENTER");
        auto dec_start=Clock::now();auto detections=decode(logits,boxes,o.top_k,o.threshold);timing["decode_ms"]=ms(dec_start);
        trace("decode DONE");
        if(FLAGS_save_outputs) save_json(o.out/"detections.json",{{"frame_id",f.manifest.value("frame_id",std::string(""))},
            {"classes",kClasses},{"score_threshold",o.threshold},{"top_k",o.top_k},
            {"box_convention","LiDAR center x,y,z; w,l,h in meters; yaw radians"},
            {"detections",detection_json(detections)}});
        trace("visualization ENTER");
        auto vis_start=Clock::now();if(FLAGS_visualize) {
            auto displayed=detections;
            displayed.erase(std::remove_if(displayed.begin(),displayed.end(),[](const Detection& d){return d.score<FLAGS_visualize_score_threshold;}),displayed.end());
            visualize(f,displayed,o.out);
        }timing["visualize_and_save_images_ms"]=ms(vis_start);
        trace("visualization DONE");
        auto sorted=times;std::sort(sorted.begin(),sorted.end());
        timing["inference_ms_samples"]=times;
        timing["inference_mean_ms"]=std::accumulate(times.begin(),times.end(),0.)/times.size();
        timing["inference_median_ms"]=(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])/2.;
        timing["inference_p95_ms"]=sorted[size_t(std::ceil(.95*sorted.size()))-1];
        timing["warmup"]=o.warmup;timing["repeat"]=o.repeat;
        timing["inference_scope"]="hbDNNInferV2 + submit + wait; excludes pack, invalidate, decode, visualization";
#if defined(__x86_64__) || defined(_M_X64)
        timing["execution"]="x86 simulation: NOT J6M board latency";
#else
        timing["execution"]="native target; record board clocks/load separately";
#endif
        timing["end_to_end_before_timing_write_ms"]=ms(frame_start);
        save_json(o.out/"timing.json",timing);
        std::cout<<"Saved "<<detections.size()<<" detections, camera overlays and BEV in "<<fs::absolute(o.out).string()<<"\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"sparse4d_J6M: "<<e.what()<<"\n";return 1;}
}


int Sparse4DMain(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc,&argv,true);
    try {
        Model model;
        if(FLAGS_inspect_only) return run_frame(options("",FLAGS_dump_dir),model);
        require(!FLAGS_manifest.empty(),"--manifest is required");
        auto spec=read_json(FLAGS_manifest);
        if(!spec.is_array()) return run_frame(options(FLAGS_manifest,FLAGS_dump_dir),model);
        require(FLAGS_num_frames==-1 || FLAGS_num_frames>0,"Invalid num_frames");
        int count=0;
        for(const auto& item:spec) {
            if(FLAGS_num_frames!=-1 && count>=FLAGS_num_frames) break;
            auto frame=fs::path(FLAGS_manifest).parent_path()/item.get<std::string>();
            int rc=run_frame(options(frame.string(),(fs::path(FLAGS_dump_dir)/std::to_string(count)).string()),model);
            if(rc) return rc;
            ++count;
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
