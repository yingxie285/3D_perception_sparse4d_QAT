import os
import json
from collections import defaultdict

# ================== 数据根目录 ==================
root_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/data/fisheye_data_2w_frame"

# ================== 保存路径（你指定的） ==================
save_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/tool/data_converter"
os.makedirs(save_dir, exist_ok=True)  # 自动创建目录（如果不存在）
save_path = os.path.join(save_dir, "bbox_stats.txt")

# ================== 类别映射 ==================
class_names_parking_mappings = {
    '普通轿车': 'car','敞篷轿车': 'car','SUV': 'car','MPV': 'car','面包车': 'car',
    '皮卡': 'car','警车': 'car','救护车': 'car',

    '房车': 'truck','普通小型货车': 'truck','箱式小型货车': 'truck',
    '普通大型货车': 'truck','轿运车': 'truck',

    '客车': 'bus','校车': 'bus','普通公交车': 'bus','铰链公交车': 'bus',
    '有轨电车': 'bus','无轨电车': 'bus',

    '消防车': 'construction_vehicle','清洁车': 'construction_vehicle',
    '工程车': 'construction_vehicle','拖车': 'construction_vehicle',
    '拖拉机': 'construction_vehicle','叉车': 'construction_vehicle',
    '油罐车': 'construction_vehicle',

    '成人': 'pedestrian','儿童': 'pedestrian','交警': 'pedestrian',
    '环卫工人': 'pedestrian','道路施工人员': 'pedestrian',

    '两轮摩托': 'motor','三轮摩托': 'motor',

    '两轮电动车': 'bicycle','三轮电动车': 'bicycle','两轮自行车': 'bicycle',
    '三轮自行车': 'bicycle','滑板车': 'bicycle','平衡车': 'bicycle',
    '婴儿车': 'bicycle','轮椅': 'bicycle','平板小推车': 'bicycle',
    '超市购物车': 'bicycle','手推车': 'bicycle',
    '其他非机动车': 'bicycle','未知非机动车': 'bicycle','非机动车组': 'bicycle',

    '小型动物': 'animal','大型动物': 'animal',

    '锥桶': 'traffic_cone',

    '防撞桶': 'barrier','路桩': 'barrier','石墩': 'barrier',
    '水马': 'barrier',

    '车位停止器': 'stopper','车位锁': 'stopper','减速带': 'stopper',

    '垃圾桶': 'trash_bin','灭火器': 'trash_bin','箱子': 'trash_bin',

    'A字牌': 'sign','三角牌': 'sign','施工警示牌': 'sign',
    '道闸杆': 'sign','construcion_sign': 'sign',

    '柱子': 'barrier','石块': 'barrier','树枝/树杈': 'barrier',
    '空中漂浮物': 'barrier','路坑/水洼': 'barrier',
    '其他静态障碍物': 'barrier',

    '其他机动车': 'car','未知机动车': 'car',
    '未知机动车车轮': 'car','未知机动车车灯': 'car',
}

# ================== 目标13类 ==================
target_classes = [
    'car','pedestrian','truck','bus','motor',
    'construction_vehicle','bicycle','barrier',
    'stopper','animal','trash_bin','sign','traffic_cone'
]

# ================== 统计 ==================
counter = defaultdict(int)
total_bbox = 0

for scene in os.listdir(root_dir):
    result_dir = os.path.join(root_dir, scene, "result")
    if not os.path.isdir(result_dir):
        continue

    for file in os.listdir(result_dir):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(result_dir, file)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            print(f"读取失败: {json_path}")
            continue

        for obj in data.get("objects", []):
            cls = obj.get("className", None)
            if cls is None:
                continue

            mapped_cls = class_names_parking_mappings.get(cls, None)
            if mapped_cls is None:
                continue

            counter[mapped_cls] += 1
            total_bbox += 1

# ================== 输出 ==================
output_lines = []
header = f"{'Object Class':<25} {'GT':<15} {'percentage(%)'}"
print(header)
output_lines.append(header)

sum_check = 0

for cls in target_classes:
    count = counter[cls]
    sum_check += count
    ratio = (count / total_bbox * 100) if total_bbox > 0 else 0.0

    line = f"{cls:<25} {count:<15} {ratio:.2f}%"
    print(line)
    output_lines.append(line)

# ================== Total ==================
total_line = f"{'Total':<25} {total_bbox:<15} 100.00%"
print(total_line)
output_lines.append(total_line)

# ================== 校验 ==================
if sum_check != total_bbox:
    print("⚠️ 警告：存在未映射类别，已被忽略")

# ================== 保存 ==================
with open(save_path, "w") as f:
    f.write("\n".join(output_lines))

print(f"\n结果已保存到: {save_path}")