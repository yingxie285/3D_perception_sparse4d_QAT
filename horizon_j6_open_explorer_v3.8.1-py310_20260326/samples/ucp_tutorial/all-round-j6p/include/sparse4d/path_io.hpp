// Use the SDK's OpenCV filesystem helpers; no extra stdc++fs dependency.
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <stdexcept>
#include <string>
namespace sparse4d { namespace files {
class path {
    std::string value_;
public:
    path()=default;
    path(const char* s):value_(s){}
    path(std::string s):value_(std::move(s)){}
    const std::string& string() const {return value_;}
    bool empty() const {return value_.empty();}
    bool is_relative() const {
        return value_.empty() || (value_[0]!='/' && value_[0]!='\\' &&
                                  !(value_.size()>1 && value_[1]==':'));
    }
    path parent_path() const {
        auto pos=value_.find_last_of("/\\");
        if(pos==std::string::npos) return ".";
        return value_.substr(0,pos==0?1:pos);
    }
    path operator/(const path& rhs) const {
        if(!rhs.is_relative()) return rhs;
        return std::string(cv::utils::fs::join(value_,rhs.value_));
    }
};
inline void create_directories(const path& p) {
    if(!cv::utils::fs::isDirectory(p.string()) && !cv::utils::fs::createDirectories(p.string()))
        throw std::runtime_error("Cannot create directory "+p.string());
}
inline path absolute(const path& p) {
    return std::string(cv::utils::fs::canonical(p.string()));
}
}}
