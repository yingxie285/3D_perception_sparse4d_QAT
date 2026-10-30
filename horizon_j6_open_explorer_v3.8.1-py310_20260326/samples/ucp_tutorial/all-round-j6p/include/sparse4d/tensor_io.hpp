#pragma once
#include "hobot/dnn/hb_dnn.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sparse4d {
inline void check(int rc, const char* operation) {
    if (rc != 0) throw std::runtime_error(std::string(operation)+" failed, code "+std::to_string(rc));
}
inline size_t element_bytes(int type) {
    switch(type) {
        case HB_DNN_TENSOR_TYPE_S8: case HB_DNN_TENSOR_TYPE_U8:
        case HB_DNN_TENSOR_TYPE_BOOL8: return 1;
        case HB_DNN_TENSOR_TYPE_F16: case HB_DNN_TENSOR_TYPE_S16:
        case HB_DNN_TENSOR_TYPE_U16: return 2;
        case HB_DNN_TENSOR_TYPE_F32: case HB_DNN_TENSOR_TYPE_S32:
        case HB_DNN_TENSOR_TYPE_U32: return 4;
        case HB_DNN_TENSOR_TYPE_F64: return 8;
        default: throw std::runtime_error("Unsupported HBM IO dtype (4-bit/64-bit integer IO is not implemented).");
    }
}
inline const char* dtype_name(int t) {
    static const char* names[]={"s4","u4","s8","u8","f16","s16","u16",
                               "f32","s32","u32","f64","s64","u64","bool8"};
    return t>=0 && t<14 ? names[t] : "unknown";
}
inline std::vector<int> shape_of(const hbDNNTensorProperties& p) {
    if (p.validShape.numDimensions<0 || p.validShape.numDimensions>10)
        throw std::runtime_error("Invalid HBM rank.");
    return {p.validShape.dimensionSize,p.validShape.dimensionSize+p.validShape.numDimensions};
}
inline size_t elements(const hbDNNTensorProperties& p) {
    size_t n=1;
    for (int d : shape_of(p)) {
        if (d<=0 || n>std::numeric_limits<size_t>::max()/size_t(d))
            throw std::runtime_error("Dynamic/invalid/overflowed HBM shape; export a fixed batch-1 model.");
        n*=size_t(d);
    }
    return n;
}
inline size_t resolve_strides(hbDNNTensorProperties& p) {
    const size_t b=element_bytes(p.tensorType);
    auto shape=shape_of(p);
    elements(p);
    uint64_t dense=b;
    for (int d=int(shape.size())-1; d>=0; --d) {
        // SDK -1 means caller-defined byte stride. Choose dense at that axis.
        if (p.stride[d]==-1) p.stride[d]=static_cast<int64_t>(dense);
        if (p.stride[d]<0 || (p.stride[d]==0 && shape[d]>1) || p.stride[d]%b)
            throw std::runtime_error("Invalid HBM byte stride.");
        if (p.stride[d]>0 && uint64_t(shape[d])>uint64_t(std::numeric_limits<int64_t>::max())/uint64_t(p.stride[d]))
            throw std::runtime_error("Tensor stride overflow.");
        dense=std::max(uint64_t(b),uint64_t(p.stride[d])*uint64_t(shape[d]));
    }
    // Static external strides can describe NCHW axes stored physically as NHWC.
    // Validate disjoint elements in physical stride order, not logical axis order.
    std::vector<int> axes;
    for (int d=0; d<int(shape.size()); ++d) if (shape[d]>1) axes.push_back(d);
    std::sort(axes.begin(),axes.end(),[&](int a,int c){return p.stride[a]<p.stride[c];});
    uint64_t extent=b;
    for (int d:axes) {
        if (uint64_t(p.stride[d])<extent) throw std::runtime_error("Overlapping HBM strides.");
        uint64_t extra=uint64_t(p.stride[d])*uint64_t(shape[d]-1);
        if (extra>uint64_t(std::numeric_limits<int64_t>::max())-extent)
            throw std::runtime_error("Tensor extent overflow.");
        extent+=extra;
    }
    for (int d=0; d<int(shape.size()); ++d)
        extent=std::max(extent,uint64_t(p.stride[d])*uint64_t(shape[d]));
    // alignedByteSize can cover a padded buffer larger than dense extents.
    if (p.alignedByteSize>0) extent=std::max(extent,uint64_t(p.alignedByteSize));
    if (extent>std::numeric_limits<size_t>::max()) throw std::runtime_error("Tensor allocation overflow.");
    p.alignedByteSize=static_cast<int64_t>(extent);
    return size_t(extent);
}
inline size_t byte_offset(const hbDNNTensorProperties& p,size_t flat) {
    size_t offset=0;
    for (int d=p.validShape.numDimensions-1;d>=0;--d) {
        size_t dim=size_t(p.validShape.dimensionSize[d]);
        offset+=(flat%dim)*size_t(p.stride[d]); flat/=dim;
    }
    return offset;
}
inline size_t axis_index(const hbDNNTensorProperties& p,size_t flat) {
    int axis=p.quantizeAxis;
    if (axis<0) axis+=p.validShape.numDimensions;
    if (axis<0 || axis>=p.validShape.numDimensions)
        throw std::runtime_error("Invalid quantizeAxis for per-axis quantization.");
    for (int d=p.validShape.numDimensions-1;d>axis;--d)
        flat/=size_t(p.validShape.dimensionSize[d]);
    return flat%size_t(p.validShape.dimensionSize[axis]);
}
inline std::pair<double,double> quant_params(const hbDNNTensorProperties& p,size_t i) {
    if (p.quantiType==NONE) return {1,0};
    if (p.quantiType!=SCALE || p.scale.scaleLen<=0 || !p.scale.scaleData)
        throw std::runtime_error("Invalid HBM quantization metadata.");
    size_t axis=0;
    if (p.scale.scaleLen>1 || p.scale.zeroPointLen>1) axis=axis_index(p,i);
    int scale_index=p.scale.scaleLen==1 ? 0 : int(axis);
    int zero_index=p.scale.zeroPointLen<=1 ? 0 : int(axis);
    if (scale_index>=p.scale.scaleLen || zero_index>=std::max(1,p.scale.zeroPointLen))
        throw std::runtime_error("Quantization scale/zero point axis length mismatch.");
    double s=p.scale.scaleData[scale_index];
    if (!(s>0) || !std::isfinite(s)) throw std::runtime_error("Invalid quantization scale.");
    if (p.scale.zeroPointLen>0 && !p.scale.zeroPointData) throw std::runtime_error("Missing zero points.");
    return {s,p.scale.zeroPointLen ? double(p.scale.zeroPointData[zero_index]) : 0.};
}
template<class T> inline T read_scalar(const uint8_t* p) { T v; std::memcpy(&v,p,sizeof(v)); return v; }
template<class T> inline void write_scalar(uint8_t* p,T v) { std::memcpy(p,&v,sizeof(v)); }
inline float half_to_float(uint16_t h) {
    const int sign=(h>>15)&1, exponent=(h>>10)&31;
    const int mantissa=h&1023;
    float v;
    if (exponent==0) v=std::ldexp(float(mantissa),-24);
    else if (exponent==31) v=mantissa ? std::numeric_limits<float>::quiet_NaN()
                                   : std::numeric_limits<float>::infinity();
    else v=std::ldexp(float(1024+mantissa),exponent-25);
    return sign ? -v : v;
}
inline uint16_t float_to_half(float v) {
    uint32_t x; std::memcpy(&x,&v,4);
    uint16_t sign=uint16_t((x>>16)&0x8000);
    uint32_t mantissa=x&0x7fffff, e=(x>>23)&255;
    if (e==255) return uint16_t(sign|0x7c00|(mantissa ? 0x200:0));
    int he=int(e)-127+15;
    if (he>=31) return uint16_t(sign|0x7c00);
    if (he<=0) {
        if (he < -10) return sign;
        mantissa|=0x800000;
        int shift=14-he;
        uint32_t value=mantissa>>shift, rest=mantissa&((1u<<shift)-1), mid=1u<<(shift-1);
        value+=(rest>mid || (rest==mid && (value&1)));
        return uint16_t(sign|value);
    }
    uint32_t value=(uint32_t(he)<<10)|(mantissa>>13), rest=mantissa&0x1fff;
    value+=(rest>0x1000 || (rest==0x1000 && (value&1)));
    return uint16_t(sign|value);
}
inline bool is_float_type(int t) {
    return t==HB_DNN_TENSOR_TYPE_F32 || t==HB_DNN_TENSOR_TYPE_F16 || t==HB_DNN_TENSOR_TYPE_F64;
}
template<class T> inline void write_integer(uint8_t* p,double v) {
    double rounded=std::nearbyint(v);
    rounded=std::max(double(std::numeric_limits<T>::lowest()),
                     std::min(double(std::numeric_limits<T>::max()),rounded));
    write_scalar<T>(p,static_cast<T>(rounded));
}
inline void pack_tensor(hbDNNTensor& t,const std::vector<float>& values) {
    const auto& p=t.properties;
    if (values.size()!=elements(p)) throw std::runtime_error("HBM input element count mismatch.");
    auto* base=static_cast<uint8_t*>(t.sysMem.virAddr);
    std::memset(base,0,size_t(p.alignedByteSize));
    for (size_t i=0;i<values.size();++i) {
        double v=values[i]; auto q=quant_params(p,i);
        if (!std::isfinite(v)) throw std::runtime_error("Non-finite model input.");
        if (is_float_type(p.tensorType) && p.quantiType!=NONE)
            throw std::runtime_error("Unexpected SCALE on float HBM tensor.");
        if (!is_float_type(p.tensorType)) {
            if (p.quantiType==NONE && v!=std::nearbyint(v))
                throw std::runtime_error("Integer IO lacks scale; cannot silently quantize float input.");
            v=v/q.first+q.second;
        }
        uint8_t* ptr=base+byte_offset(p,i);
        switch(p.tensorType) {
            case HB_DNN_TENSOR_TYPE_F32: write_scalar<float>(ptr,float(v)); break;
            case HB_DNN_TENSOR_TYPE_F16: write_scalar<uint16_t>(ptr,float_to_half(float(v))); break;
            case HB_DNN_TENSOR_TYPE_F64: write_scalar<double>(ptr,v); break;
            case HB_DNN_TENSOR_TYPE_S8: write_integer<int8_t>(ptr,v); break;
            case HB_DNN_TENSOR_TYPE_U8: case HB_DNN_TENSOR_TYPE_BOOL8: write_integer<uint8_t>(ptr,v); break;
            case HB_DNN_TENSOR_TYPE_S16: write_integer<int16_t>(ptr,v); break;
            case HB_DNN_TENSOR_TYPE_U16: write_integer<uint16_t>(ptr,v); break;
            case HB_DNN_TENSOR_TYPE_S32: write_integer<int32_t>(ptr,v); break;
            case HB_DNN_TENSOR_TYPE_U32: write_integer<uint32_t>(ptr,v); break;
            default: throw std::runtime_error("Unsupported input dtype.");
        }
    }
}
inline std::vector<float> unpack_tensor(const hbDNNTensor& t) {
    const auto& p=t.properties;
    const auto* base=static_cast<const uint8_t*>(t.sysMem.virAddr);
    std::vector<float> result(elements(p));
    if (is_float_type(p.tensorType) && p.quantiType!=NONE)
        throw std::runtime_error("Unexpected SCALE on float HBM output.");
    for (size_t i=0;i<result.size();++i) {
        const uint8_t* ptr=base+byte_offset(p,i);
        double v;
        switch(p.tensorType) {
            case HB_DNN_TENSOR_TYPE_F32: v=read_scalar<float>(ptr); break;
            case HB_DNN_TENSOR_TYPE_F16: v=half_to_float(read_scalar<uint16_t>(ptr)); break;
            case HB_DNN_TENSOR_TYPE_F64: v=read_scalar<double>(ptr); break;
            case HB_DNN_TENSOR_TYPE_S8: v=read_scalar<int8_t>(ptr); break;
            case HB_DNN_TENSOR_TYPE_U8: case HB_DNN_TENSOR_TYPE_BOOL8: v=read_scalar<uint8_t>(ptr); break;
            case HB_DNN_TENSOR_TYPE_S16: v=read_scalar<int16_t>(ptr); break;
            case HB_DNN_TENSOR_TYPE_U16: v=read_scalar<uint16_t>(ptr); break;
            case HB_DNN_TENSOR_TYPE_S32: v=read_scalar<int32_t>(ptr); break;
            case HB_DNN_TENSOR_TYPE_U32: v=read_scalar<uint32_t>(ptr); break;
            default: throw std::runtime_error("Unsupported output dtype.");
        }
        auto q=quant_params(p,i);
        result[i]=float((v-q.second)*q.first);
    }
    return result;
}
} // namespace sparse4d

