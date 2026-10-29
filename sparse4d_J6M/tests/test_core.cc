#include "../src/detection.hpp"
#include "../src/tensor_io.hpp"
#include <cfenv>
#include <iostream>
using namespace sparse4d;
static void test(bool ok,const char* message) {
    if(!ok) throw std::runtime_error(message);
}
int main() {
    std::fesetround(FE_TONEAREST);
    // Every finite half, including subnormals and signed zero, must round-trip.
    for(unsigned h=0;h<65536;++h) {
        if((h&0x7c00)==0x7c00 && (h&1023)) continue;
        test(float_to_half(half_to_float(uint16_t(h)))==h,"half round-trip");
    }
    test(float_to_half(1.00048828125f)==0x3c00,"half tie-even down");
    test(float_to_half(1.00146484375f)==0x3c02,"half tie-even up");
    test(float_to_half(65520.f)==0x7c00,"half overflow");
    std::vector<float> logits(26,-10),boxes(16,0);
    logits[2]=3;logits[1]=2;logits[13+12]=1;
    boxes[3]=std::log(2.f);boxes[4]=std::log(4.f);boxes[5]=std::log(6.f);
    boxes[6]=1;boxes[7]=0;boxes[15]=1;
    auto ds=decode(logits,boxes,3,.5f);
    test(ds.size()==3 && ds[0].query==0 && ds[1].query==0 && ds[2].query==1,"global class-query topk");
    test(ds[0].label==2 && ds[1].label==1 && ds[2].label==12,"label order");
    test(std::abs(ds[0].box[3]-2)<1e-6 && std::abs(ds[0].box[6]-std::acos(-1.)/2)<1e-6,"box decode");
    test(decode(logits,boxes,3,.99f).empty(),"score threshold");
    auto cs=corners(ds[0]);
    test(std::abs(cs[0][0]-2)<1e-5 && std::abs(cs[0][1]+1)<1e-5 && std::abs(cs[0][2]+3)<1e-5,"corners rotation");
    std::array<double,9> k{{400,0,960,0,400,600,0,0,1}};
    std::array<double,4> d{{0,0,0,0}};
    auto uv=fisheye({{0,0,5}},k,d);
    test(uv[0]==960 && uv[1]==600,"optical axis projection");
    uv=fisheye({{1,0,1}},k,d);
    test(std::abs(uv[0]-(960+400*std::atan2(1.,1.000001)))<1e-9,"fisheye projection");

    hbDNNTensor t{};auto& p=t.properties;
    p.tensorType=HB_DNN_TENSOR_TYPE_S8;p.validShape.numDimensions=2;
    p.validShape.dimensionSize[0]=2;p.validShape.dimensionSize[1]=3;
    p.stride[0]=8;p.stride[1]=1;p.alignedByteSize=16;p.quantiType=SCALE;p.quantizeAxis=1;
    float scales[]={.5f,1.f,2.f};int32_t zeros[]={1,2,3};
    p.scale={3,scales,3,zeros};
    std::vector<uint8_t> mem(resolve_strides(p),255);t.sysMem.virAddr=mem.data();
    pack_tensor(t,{.25f,.5f,1.f,500,-500,6});
    // nearbyint(x/scale + zp), including nonzero odd zero points.
    test(int8_t(mem[0])==2 && int8_t(mem[1])==2 && int8_t(mem[2])==4,"per-axis round-even");
    test(int8_t(mem[8])==127 && int8_t(mem[9])==-128 && int8_t(mem[10])==6,"integer saturation");
    test(mem[3]==0 && mem[7]==0 && mem[11]==0,"padding initialized");
    auto unpacked=unpack_tensor(t);
    test(unpacked[0]==.5f && unpacked[1]==0 && unpacked[2]==2 && unpacked[5]==6,"per-axis dequantize");
    p.tensorType=HB_DNN_TENSOR_TYPE_F32;p.quantiType=NONE;
    p.stride[0]=-1;p.stride[1]=-1;p.alignedByteSize=0;
    test(resolve_strides(p)==24 && p.stride[0]==12 && p.stride[1]==4,"float dynamic byte strides");
    mem.resize(24);t.sysMem.virAddr=mem.data();
    std::vector<float> original{.1f,-3,0,8,1e-4f,9};
    pack_tensor(t,original);test(unpack_tensor(t)==original,"float pack/unpack");
    p.stride[0]=4;p.stride[1]=8;p.alignedByteSize=24;
    resolve_strides(p);
    pack_tensor(t,original);test(unpack_tensor(t)==original,"permuted physical strides");
    test(read_scalar<float>(mem.data()+4)==original[3],"transpose memory placement");
    // Regression against the user's actual HBM runtime interface.
    const std::vector<std::vector<int>> shapes={{4,3,256,704},{1,4,4,4},{1,4,3,3},
        {1,4,1,1,4},{1,4,4,4},{1,2},{},{1,384,13},{1,384,8}};
    const std::vector<std::vector<int64_t>> strides={{589824,196608,768,1},{128,32,8,2},
        {128,32,8,2},{32,8,8,8,2},{256,64,16,4},{4,2},{},{24576,64,4},{6144,16,2}};
    const int types[]={HB_DNN_TENSOR_TYPE_S8,HB_DNN_TENSOR_TYPE_S16,HB_DNN_TENSOR_TYPE_S16,
        HB_DNN_TENSOR_TYPE_S16,HB_DNN_TENSOR_TYPE_F32,HB_DNN_TENSOR_TYPE_S16,
        HB_DNN_TENSOR_TYPE_F32,HB_DNN_TENSOR_TYPE_S32,HB_DNN_TENSOR_TYPE_F16};
    const int64_t sizes[]={2359296,128,128,128,256,128,64,24576,6144};
    for(size_t n=0;n<shapes.size();++n) {
        hbDNNTensor actual{};auto& a=actual.properties;
        a.tensorType=types[n];a.validShape.numDimensions=int(shapes[n].size());
        a.alignedByteSize=sizes[n];
        for(size_t d=0;d<shapes[n].size();++d) {
            a.validShape.dimensionSize[d]=shapes[n][d];a.stride[d]=strides[n][d];
        }
        test(resolve_strides(a)==size_t(sizes[n]),"actual HBM allocation size");
        std::vector<uint8_t> storage(size_t(a.alignedByteSize));actual.sysMem.virAddr=storage.data();
        std::vector<float> data(elements(a),1.f);
        pack_tensor(actual,data);test(unpack_tensor(actual)==data,"actual HBM physical layout roundtrip");
        if(n==0) test(storage[704]==0 && storage[767]==0 && storage[768]==1,"image row padding 704 to 768");
    }
    std::cout<<"PASS: half exhaustive, decoder, fisheye, quantization, strides, padding\n";
}

