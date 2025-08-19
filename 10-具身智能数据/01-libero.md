# Libero数据集统计分析报告

```python
#!/usr/bin/env python3
"""
分析Libero数据集，统计每个大任务下的小任务数量和样本数量
"""

import pandas as pd
import os
from pathlib import Path
import json
from collections import defaultdict

def analyze_libero_dataset():
    """分析Libero数据集"""
  
    # 数据集路径
    datasets = [
        "libero_10_no_noops_lerobot_v21",
        "libero_goal_no_noops_lerobot_v21", 
        "libero_object_no_noops_lerobot_v21",
        "libero_spatial_no_noops_lerobot_v21"
    ]
  
    base_path = Path("/data1/DATA")
  
    results = {}
  
    for dataset_name in datasets:
        print(f"\n分析数据集: {dataset_name}")
        print("=" * 50)
      
        dataset_path = base_path / dataset_name
        data_path = dataset_path / "data" / "chunk-000"
      
        if not data_path.exists():
            print(f"数据路径不存在: {data_path}")
            continue
      
        # 获取所有parquet文件
        parquet_files = list(data_path.glob("episode_*.parquet"))
        print(f"总episode数量: {len(parquet_files)}")
      
        # 分析每个episode
        task_stats = defaultdict(lambda: {"episodes": 0, "total_samples": 0})
        episode_info = []
      
        for parquet_file in parquet_files:
            try:
                # 读取parquet文件
                df = pd.read_parquet(parquet_file)
              
                # 获取任务信息
                task_name = "unknown"
                if 'task_index' in df.columns and len(df) > 0:
                    task_index = df['task_index'].iloc[0]
                    task_name = f"task_{task_index}"
                elif 'task' in df.columns:
                    task_name = df['task'].iloc[0] if len(df) > 0 else "unknown"
                elif 'task_name' in df.columns:
                    task_name = df['task_name'].iloc[0] if len(df) > 0 else "unknown"
                else:
                    # 打印前几列看看数据结构
                    if len(episode_info) < 3:  # 只打印前3个文件的信息
                        print(f"文件 {parquet_file.name} 的列名: {list(df.columns)}")
                        if len(df) > 0:
                            print(f"样本数: {len(df)}")
                            print(f"前几行数据:")
                            print(df.head(2))
                            print("-" * 30)
              
                # 统计信息
                sample_count = len(df)
                task_stats[task_name]["episodes"] += 1
                task_stats[task_name]["total_samples"] += sample_count
              
                episode_info.append({
                    "file": parquet_file.name,
                    "task": task_name,
                    "samples": sample_count
                })
              
            except Exception as e:
                print(f"处理文件 {parquet_file.name} 时出错: {e}")
      
        # 输出统计结果
        print(f"\n{dataset_name} 统计结果:")
        print(f"{'任务名称':<15} {'Episode数量':<12} {'总样本数':<12} {'平均样本数':<12}")
        print("-" * 60)
      
        total_episodes = 0
        total_samples = 0
      
        # 按任务索引排序
        sorted_tasks = sorted(task_stats.items(), key=lambda x: x[0])
      
        for task_name, stats in sorted_tasks:
            avg_samples = stats["total_samples"] / stats["episodes"] if stats["episodes"] > 0 else 0
            print(f"{task_name:<15} {stats['episodes']:<12} {stats['total_samples']:<12} {avg_samples:<12.1f}")
            total_episodes += stats["episodes"]
            total_samples += stats["total_samples"]
      
        print("-" * 60)
        print(f"{'总计':<15} {total_episodes:<12} {total_samples:<12} {total_samples/total_episodes if total_episodes > 0 else 0:<12.1f}")
      
        # 保存结果
        results[dataset_name] = {
            "task_stats": dict(task_stats),
            "total_episodes": total_episodes,
            "total_samples": total_samples,
            "episode_info": episode_info
        }
  
    return results

if __name__ == "__main__":
    results = analyze_libero_dataset()
  
    # 保存详细结果到JSON文件
    with open("libero_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
  
    print("\n" + "=" * 80)
    print("总体统计:")
    print("=" * 80)
  
    for dataset_name, result in results.items():
        print(f"\n{dataset_name}:")
        print(f"  - 任务类型数量: {len(result['task_stats'])}")
        print(f"  - 总Episode数量: {result['total_episodes']}")
        print(f"  - 总样本数量: {result['total_samples']}")
        print(f"  - 平均每Episode样本数: {result['total_samples']/result['total_episodes'] if result['total_episodes'] > 0 else 0:.1f}")
      
        # 显示每个任务的详细统计
        print(f"  - 任务分布:")
        sorted_tasks = sorted(result['task_stats'].items(), key=lambda x: x[0])
        for task_name, stats in sorted_tasks:
            print(f"    * {task_name}: {stats['episodes']} episodes, {stats['total_samples']} samples") 
```

上面的是lerobot格式的数据分析

下面的是openvla格式的

```python
#!/usr/bin/env python3
"""
分析Libero数据集，统计每个大任务下的小任务数量和样本数量
"""

import os
import tensorflow_datasets as tfds
import tensorflow as tf

def analyze_dataset(dataset_path):
    """
    Analyzes a single dataset to extract unique language instructions using tfds.
    """
    try:
        # Suppress verbose logging
        tf.get_logger().setLevel('ERROR')

        builder = tfds.builder_from_directory(builder_dir=dataset_path)
        dataset = builder.as_dataset(split='train')

        unique_instructions = set()
      
        for episode in dataset:
            # The language instruction is the same for all steps in an episode
            # We need to iterate through the steps dataset to get the first step
            for step in episode['steps'].take(1):
                instruction = step['language_instruction'].numpy().decode('utf-8')
                unique_instructions.add(instruction)

        return sorted(list(unique_instructions))

    except Exception as e:
        print(f"Error analyzing {dataset_path}: {e}")
        return []

def main():
    base_dir = "/data/DATA/modified_libero_rlds"
    task_categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]

    for category in task_categories:
        print(f"--- Task Category: {category} ---")
      
        version_dir = os.path.join(base_dir, category, "1.0.0") # Assuming version 1.0.0
      
        if not os.path.exists(version_dir):
            print(f"Version directory not found for {category}")
            continue

        subtasks = analyze_dataset(version_dir)
      
        print(f"  Found {len(subtasks)} subtasks:")
        for i, task in enumerate(subtasks):
            print(f"    {i+1}. {task}")
        print("\n")

if __name__ == "__main__":
    main() 
```

数据集概况

本次分析了4个Libero数据集，每个数据集都包含10个不同的任务类型（task_0到task_9）。



## 详细统计结果

### 1. libero_10_no_noops_lerobot_v21

- **总Episode数量**: 379
- **总样本数量**: 101,469
- **平均每Episode样本数**: 267.7
- **任务类型数量**: 10个任务

**各任务详细统计**:


| 任务编号 | Episode数量 | 总样本数 | 平均样本数 |
| -------- | ----------- | -------- | ---------- |
| task_0   | 38          | 9,807    | 258.1      |
| task_1   | 36          | 9,020    | 250.6      |
| task_2   | 34          | 9,990    | 293.8      |
| task_3   | 41          | 10,866   | 265.0      |
| task_4   | 43          | 11,494   | 267.3      |
| task_5   | 33          | 9,571    | 290.0      |
| task_6   | 29          | 11,808   | 407.2      |
| task_7   | 49          | 12,702   | 259.2      |
| task_8   | 35          | 8,577    | 245.1      |
| task_9   | 41          | 7,634    | 186.2      |

### 2. libero_goal_no_noops_lerobot_v21

- **总Episode数量**: 428
- **总样本数量**: 52,042
- **平均每Episode样本数**: 121.6
- **任务类型数量**: 10个任务

**各任务详细统计**:


| 任务编号 | Episode数量 | 总样本数 | 平均样本数 |
| -------- | ----------- | -------- | ---------- |
| task_0   | 49          | 4,563    | 93.1       |
| task_1   | 36          | 6,224    | 172.9      |
| task_2   | 36          | 7,157    | 198.8      |
| task_3   | 40          | 4,203    | 105.1      |
| task_4   | 47          | 5,004    | 106.5      |
| task_5   | 33          | 4,990    | 151.2      |
| task_6   | 50          | 4,454    | 89.1       |
| task_7   | 48          | 4,854    | 101.1      |
| task_8   | 46          | 4,574    | 99.4       |
| task_9   | 43          | 6,019    | 140.0      |

### 3. libero_object_no_noops_lerobot_v21

- **总Episode数量**: 454
- **总样本数量**: 66,984
- **平均每Episode样本数**: 147.5
- **任务类型数量**: 10个任务

**各任务详细统计**:


| 任务编号 | Episode数量 | 总样本数 | 平均样本数 |
| -------- | ----------- | -------- | ---------- |
| task_0   | 45          | 6,291    | 139.8      |
| task_1   | 45          | 6,919    | 153.8      |
| task_2   | 45          | 6,378    | 141.7      |
| task_3   | 46          | 6,723    | 146.2      |
| task_4   | 44          | 6,867    | 156.1      |
| task_5   | 45          | 6,443    | 143.2      |
| task_6   | 47          | 6,200    | 131.9      |
| task_7   | 45          | 7,102    | 157.8      |
| task_8   | 42          | 6,097    | 145.2      |
| task_9   | 50          | 7,964    | 159.3      |

### 4. libero_spatial_no_noops_lerobot_v21

- **总Episode数量**: 432
- **总样本数量**: 52,970
- **平均每Episode样本数**: 122.6
- **任务类型数量**: 10个任务

**各任务详细统计**:


| 任务编号 | Episode数量 | 总样本数 | 平均样本数 |
| -------- | ----------- | -------- | ---------- |
| task_0   | 46          | 5,775    | 125.5      |
| task_1   | 42          | 6,257    | 149.0      |
| task_2   | 39          | 4,472    | 114.7      |
| task_3   | 35          | 4,747    | 135.6      |
| task_4   | 45          | 4,487    | 99.7       |
| task_5   | 43          | 4,287    | 99.7       |
| task_6   | 47          | 5,570    | 118.5      |
| task_7   | 45          | 5,940    | 132.0      |
| task_8   | 46          | 5,343    | 116.2      |
| task_9   | 44          | 6,092    | 138.5      |

## 总体汇总


| 数据集名称     | 任务数量 | 总Episode数 | 总样本数    | 平均Episode样本数 |
| -------------- | -------- | ----------- | ----------- | ----------------- |
| libero_10      | 10       | 379         | 101,469     | 267.7             |
| libero_goal    | 10       | 428         | 52,042      | 121.6             |
| libero_object  | 10       | 454         | 66,984      | 147.5             |
| libero_spatial | 10       | 432         | 52,970      | 122.6             |
| **总计**       | **40**   | **1,693**   | **273,465** | **161.5**         |

## 主要发现

1. **数据集结构统一**: 所有4个数据集都包含10个任务类型（task_0到task_9）
2. **样本数量差异**: libero_10数据集的样本数量最多（101,469），其次是libero_object（66,984）
3. **Episode长度差异**: libero_10的平均Episode长度最长（267.7），libero_goal和libero_spatial的Episode相对较短（约122）
4. **任务分布**: 各数据集内的任务分布相对均匀，每个任务大约有30-50个episodes
5. **数据规模**: 总共包含1,693个episodes，273,465个样本点

## 数据特点

- **数据格式**: 所有数据都以parquet格式存储，每个episode一个文件
- **数据结构**: 每个样本包含observation.state、action、timestamp等字段
- **任务标识**: 通过task_index字段标识不同的任务类型
- **时序性**: 每个episode代表一个完整的任务执行序列


# lerobot和openvla数据对齐性讨论

```
# OpenVLA vs. LeRobot 数据格式对比分析

## 1. 核心差异总结

| 对比项 | OpenVLA (modified_libero_rlds) | LeRobot (lerobot_v21) | 对齐状态 |
| :--- | :--- | :--- | :--- |
| **存储格式** | `TFRecord` | `Parquet` | 🔴 **不兼容** |
| **特征元数据** | `features.json` (结构不同) | `info.json` | 🟡 **需映射** |
| **特征结构** | 嵌套在 `steps` 和 `episode_metadata` | 扁平化结构 | 🟡 **需转换** |

**结论**: OpenVLA 和 LeRobot 的数据格式**没有直接对齐**。要使其兼容，需要进行数据转换和特征映射。

---

## 2. 详细对比

### 存储格式

-   **OpenVLA**: 使用 `TFRecord` 格式 (`.tfrecord`)。这是一种序列化的二进制格式，在 TensorFlow 生态中广泛使用，适合高效的 `I/O` 操作。
-   **LeRobot**: 使用 `Apache Parquet` 格式 (`.parquet`)。这是一种列式存储格式，在 `Pandas` 和 `Spark` 等大数据处理框架中很流行，提供了高效的压缩和编码。

👉 **对齐分析**: 两种格式在二进制层面完全不同，无法直接通用。

### 元数据和特征定义

#### OpenVLA

-   **`dataset_info.json`**: 包含数据集的版本、描述、分片信息等高级元数据。
-   **`features.json`**: 定义了数据的具体结构。从分析脚本的初步运行结果看，其根级别没有`features`键，这表明特征定义可能嵌套在更深层次，或者需要结合 `TensorFlow Datasets (TFDS)` 的加载器来完整解析。通常，RLDS/OpenVLA 格式会将数据组织成 `steps`，每个 `step` 包含 `observation`, `action`, `reward`, `is_first`, `is_last` 等。

#### LeRobot

-   **`info.json`**: 集中包含了数据集的元数据和详细的特征定义。
-   **`features` 键**: 在 `info.json` 中直接定义了所有数据字段，包括：
    -   `observation.images.wrist_image` (视频)
    -   `observation.images.image` (视频)
    -   `observation.state` (机器人状态)
    -   `action` (动作)
    -   `timestamp`, `frame_index`, `episode_index` (元数据)
    -   `task_index` (任务ID)

👉 **对齐分析**:
-   `LeRobot` 的特征是**扁平化**的，每个字段（如 `action`, `observation.state`）都作为 `Parquet` 文件中的一列。
-   `OpenVLA` 的特征是**层级化**的，通常封装在 `steps` 序列中。

---

## 3. 如何对齐？

要将 `LeRobot` 的数据格式对齐到 `OpenVLA`，需要执行以下步骤：

1.  **读取 Parquet 数据**: 使用 `pandas` 或 `pyarrow` 逐个读取 `LeRobot` 的 `episode_*.parquet` 文件。
2.  **重构数据**:
    -   将每个 `episode` 的数据帧（`DataFrame` 的行）转换为 `OpenVLA` 的 `step` 结构。
    -   每个 `step` 应该包含 `observation`（包含图像和状态）、`action` 等字段。
    -   在 `episode` 级别添加 `episode_metadata`，如 `task_id`。
3.  **序列化为 TFRecord**:
    -   将重构后的 `episode` 序列化为 `tf.train.Example` 协议缓冲区。
    -   将序列化后的数据写入 `.tfrecord` 文件。
4.  **创建元数据文件**:
    -   生成符合 `OpenVLA` 规范的 `dataset_info.json` 和 `features.json`。

这个过程本质上是一个 `ETL` (Extract, Transform, Load) 流程，需要编写专门的转换脚本。

如果您需要，我可以帮您编写一个初步的转换脚本。 
```

详细task分析


这个 `openvla` 格式的 `libero` 数据集分为4个大类任务，每个大类下正好包含10个不同的子任务。

### 详细分析结果

**1. `libero_object_no_noops`**

这个类别主要关注于**拾取特定物品并将其放入篮子**。

* **子任务列表:**
  1. 捡起字母汤罐头并放入篮子。
  2. 捡起烧烤酱并放入篮子。
  3. 捡起黄油并放入篮子。
  4. 捡起巧克力布丁并放入篮子。
  5. 捡起奶油芝士并放入篮子。
  6. 捡起番茄酱并放入篮子。
  7. 捡起牛奶并放入篮子。
  8. 捡起橙汁并放入篮子。
  9. 捡起沙拉酱并放入篮子。
  10. 捡起番茄酱并放入篮子。

**2. `libero_goal_no_noops`**

这个类别主要关注于**实现特定的目标状态**，例如开关抽屉、放置物品到特定位置等。

* **子任务列表:**
  1. 打开橱柜的中间抽屉。
  2. 打开顶层抽屉，把碗放进去。
  3. 把盘子推到炉子前面。
  4. 把碗放在盘子上。
  5. 把碗放在炉子上。
  6. 把碗放在橱柜顶部。
  7. 把奶油芝士放进碗里。
  8. 把酒瓶放在酒架上。
  9. 把酒瓶放在橱柜顶部。
  10. 打开炉子。

**3. `libero_10_no_noops`**

这个类别包含了一系列**组合任务**，需要同时操作多个物品。

* **子任务列表:**
  1. 捡起书并将其放入球童的后隔间。
  2. 把两个摩卡壶都放在炉子上。
  3. 把字母汤罐头和奶油芝士盒都放进篮子。
  4. 把字母汤罐头和番茄酱都放进篮子。
  5. 把奶油芝士盒和黄油都放进篮子。
  6. 把黑色的碗放进橱柜的底层抽屉并关上。
  7. 把白色的杯子放在左边的盘子上，把黄白色的杯子放在右边的盘子上。
  8. 把白色的杯子放在盘子上，把巧克力布丁放在盘子右边。
  9. 把黄白色的杯子放进微波炉并关上。
  10. 打开炉子，把摩卡壶放在上面。

**4. `libero_spatial_no_noops`**

这个类别主要关注于**空间关系**，即根据物品之间的相对位置来执行操作。

* **子任务列表:**
  1. 从盘子和模子之间捡起黑碗，然后放在盘子上。
  2. 从桌子中央捡起黑碗，然后放在盘子上。
  3. 从木柜顶层抽屉里捡起黑碗，然后放在盘子上。
  4. 从饼干盒旁边捡起黑碗，然后放在盘子上。
  5. 从盘子旁边捡起黑碗，然后放在盘子上。
  6. 从模子旁边捡起黑碗，然后放在盘子上。
  7. 从饼干盒上捡起黑碗，然后放在盘子上。
  8. 从模子上捡起黑碗，然后放在盘子上。
  9. 从炉子上捡起黑碗，然后放在盘子上。
  10. 从木柜上捡起黑碗，然后放在盘子上。
