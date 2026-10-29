#include "detection.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
using json=nlohmann::json;
int main(int argc,char** argv) {
    try {
        if(argc!=3) return 2;
        std::ifstream in(argv[1]);json j;in>>j;
        auto detections=sparse4d::decode(j.at("logits").get<std::vector<float>>(),
            j.at("boxes").get<std::vector<float>>(),j.at("top_k"),j.at("threshold"));
        json out=json::array();
        for(const auto& d:detections) out.push_back({{"label",d.label},{"query",d.query},
            {"score",d.score},{"box",d.box},{"corners",sparse4d::corners(d)}});
        std::ofstream file(argv[2]);file<<out.dump();return file?0:1;
    } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}

