

### 预备：sam_vit_h_4b8939.pth 下载

这个文件是 Meta AI 发布的 Segment Anything Model (SAM) 的预训练模型权重文件，具体是 **ViT-H (Huge) 版本**。它是 SAM 提供的三种尺寸模型中最大、也是效果最好的一个。

```
# sam_vit_h_4b8939.pth 是 ViT-H SAM 模型的权重
# 文件大小约为 2.4 GB，请确保有足够的磁盘空间和稳定的网络连接
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 一、 Segment Anything Model (SAM) 原理讲解

#### 1. 核心思想：从“分割什么”到“分割万物”

在 SAM 出现之前，图像分割模型通常是“专家模型”，它们被训练来分割特定的类别，比如“人”、“车”、“猫”等。如果你想分割一个训练时没见过的物体，它们通常会失败。

SAM 的核心思想是创建一个**通用的、可提示的分割模型**。它不关心物体 *是* 什么，只关心 *在哪里*。它的目标是，只要用户给出任何形式的提示（点、框、文本等），它就能准确地分割出对应的物体或区域。这被称为“提示分割” (Promptable Segmentation)。

#### 2. SAM 的三大核心组件

SAM 的架构可以看作由三个部分组成，它们协同工作，实现了高效、灵活的分割：

- **图像编码器 (Image Encoder)**
  
  - **作用**：这是 SAM 中最“重”的部分，负责“理解”整个图像。它采用了一个非常强大的**视觉 Transformer (Vision Transformer, ViT)** 模型（在您下载的模型中是 ViT-H）。
  
  - **过程**：当输入一张图像时，图像编码器会对其进行复杂的计算，将其转换成一个包含丰富空间和语义信息的数字表示（称为特征嵌入）。这个过程只需要对每张图片执行一次，然后结果可以被重复使用。
  
  - **类比**：可以把它想象成一位艺术家在绘画前对整个场景进行的彻底观察和构思，将所有细节和关系都记在脑子里。

- **提示编码器 (Prompt Encoder)**
  
  - **作用**：负责将用户的各种提示转换成模型能理解的数字表示（同样是特征嵌入）。
  
  - **支持的提示类型**：
    
    - **点 (Points)**：在物体上点击一个或多个点（前景/背景）。
    
    - **框 (Bounding Boxes)**：在物体周围画一个大致的方框。
    
    - **掩码 (Masks)**：提供一个粗略的分割区域。
    
    - **文本 (Text)**：虽然原始 SAM 论文主要关注几何提示，但其架构可以扩展，通过与其他模型（如 CLIP）结合来理解文本提示。
  
  - **类比**：如果图像编码器是艺术家，提示编码器就是翻译官，它把你用“点”或“框”下达的简单指令，翻译成艺术家能听懂的语言。

- **掩码解码器 (Mask Decoder)**
  
  - **作用**：这是 SAM 实现实时交互的关键。它非常轻量且高效。
  
  - **过程**：解码器接收来自**图像编码器**的“大脑记忆”（图像特征）和来自**提示编码器**的“指令”（提示特征），然后迅速地计算出最终的、高质量的分割掩码（Mask）。
  
  - **优势**：由于解码器非常快（毫秒级），它可以让你在图像上移动鼠标点或调整框时，实时地看到分割结果的变化，提供了极佳的交互体验。
  
  - **类比**：艺术家（图像编码器）已经构思好了整个画面，当你（用户）通过翻译官（提示编码器）指出“我要这个杯子”时，艺术家（掩码解码器）几乎瞬间就能用画笔精确地勾勒出杯子的轮廓。

#### 3. 训练过程：规模的力量

SAM 的强大通用性来自于其海量的训练数据。Meta AI 创建了一个名为 **SA-1B** 的数据集，其中包含 **1100 万张**图片和超过 **10 亿个**高质量的分割掩码。通过在这个庞大的数据集上进行训练，SAM 学会了识别和分割各种各样、闻所未闻的物体和结构，实现了“分割万物”的能力。

### 二、单目深度估计 (Monocular Depth Estimation) 原理讲解

#### 1. 核心问题：从 2D 图像中恢复 3D 信息

一张普通的照片是三维世界在二维平面上的投影。在这个过程中，一个至关重要的维度——**深度（即物体离相机的远近）**——丢失了。单目深度估计（Monocular Depth Estimation）就是利用人工智能，仅从**单张** RGB 图像中，推断出这个丢失的深度信息。

#### 2. AI 如何“看见”深度：模仿人脑的视觉线索

人脑可以很自然地从 2D 图像中感知深度，因为它利用了许多视觉线索。深度学习模型正是通过学习这些线索来完成任务的：

- **相对大小 (Relative Size)**：同类物体，在视野中看起来越小，通常离得越远。

- **遮挡 (Occlusion)**：如果物体 A 挡住了物体 B，那么 A 比 B 更近。

- **纹理梯度 (Texture Gradient)**：远处的物体表面纹理（如草地、砖墙）会显得更加密集和模糊。

- **线性透视 (Linear Perspective)**：平行的线条（如公路、铁轨）在远处会汇聚到一点。

- **光影和阴影 (Shading and Shadows)**：光照在物体上形成的光影可以揭示物体的形状和相对位置。

#### 3. 深度学习模型的实现方法

目前主流的深度估计模型通常采用**有监督学习**和**编码器-解码器架构**。

- **训练数据**：模型需要在一个大型数据集上进行训练。这个数据集包含成对的**RGB 图像**和它们对应的**真实深度图**。这些真实的深度图通常是用专业设备（如 LiDAR 激光雷达、立体相机）采集的。

- **编码器-解码器架构 (Encoder-Decoder Architecture)**：
  
  - **编码器 (Encoder)**：与 SAM 类似，编码器负责从输入的 RGB 图像中提取特征。它像一个信息漏斗，逐层压缩图像，同时提取出从低级（边缘、角点）到高级（物体部件、纹理）的各种特征。正是在这个过程中，模型学会了识别上述提到的各种深度线索。
  
  - **解码器 (Decoder)**：解码器则与编码器相反。它接收编码器提取出的浓缩特征，然后逐层地将它们放大，最终“绘制”出一张与原图大小相同的深度图。解码器的每一层都会融合编码器对应层的特征，以确保最终的深度图既有丰富的细节，又有准确的全局结构。

- **我们代码中使用的 DPT 模型**：
  
  - 在之前的代码中，我们使用了 `Intel/dpt-large` 模型。**DPT** 的全称是 **Dense Prediction Transformer**。
  
  - 它的独特之处在于其编码器采用了强大的 **Vision Transformer (ViT)**。相比于传统的卷积网络（CNN），Transformer 更擅长捕捉图像中的**全局依赖关系**。例如，它能更好地理解整个场景的透视结构，从而对远处的物体做出更准确的深度判断。这使得 DPT 在深度估计任务上表现非常出色。

#### 4. 理解输出：相对深度 vs. 绝对深度

需要特别注意的是，大多数从单张图像进行深度估计的模型，输出的是**相对深度图**。

- **什么是相对深度？**：输出的深度图中，像素值的大小（或颜色）只表示“远近关系”。例如，一个像素值为 0.8 的点比像素值为 0.2 的点更远，但你不能直接说它就远 10 米。整个场景的深度被归一化到一个固定的范围（如 0 到 1）。

- **为什么不是绝对深度？**：从单张图片无法确定场景的真实尺度。一张小房子的照片和一张大房子的照片可能看起来完全一样。要获得以“米”为单位的绝对深度，通常需要额外的尺度信息或使用立体相机等硬件。



代码：

```python
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

# --- Transformers Imports for Depth Estimation ---
from transformers import DPTImageProcessor, DPTForDepthEstimation

# --- SAM Imports ---
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# --- Global Variables for Models (Load once) ---
sam_model_instance = None # Changed name for clarity
depth_model = None
depth_processor = None

# --------------------------------------------------------------------------
# Model Loading Functions
# --------------------------------------------------------------------------

def load_sam_for_automatic_maskgen(model_path, model_type="vit_h"):
    """
    加载 SAM 模型用于 SamAutomaticMaskGenerator.
    如果成功则返回 SamAutomaticMaskGenerator 实例，否则返回 None.
    """
    global sam_model_instance
    if sam_model_instance is None:
        print(f"正在从 {model_path} (类型: {model_type}) 加载 SAM 模型...")
        if not os.path.exists(model_path):
            print(f"错误: 在 {model_path} 未找到 SAM 模型文件")
            return None
        try:
            sam_model_instance = sam_model_registry[model_type](checkpoint=model_path)
            sam_model_instance.to(device=device)
            print("SAM 模型加载成功。")
        except Exception as e:
            print(f"加载 SAM 模型时出错: {e}")
            sam_model_instance = None
            return None

    if sam_model_instance:
        try:
            mask_generator = SamAutomaticMaskGenerator(sam_model_instance)
            print("SamAutomaticMaskGenerator 创建成功。")
            return mask_generator
        except Exception as e:
            print(f"创建 SamAutomaticMaskGenerator 时出错: {e}")
            return None
    return None

def load_depth_estimation_model(model_name="Intel/dpt-large"):
    """
    加载深度估计模型和处理器 (例如 DPT).
    如果成功则返回 True，否则返回 False.
    """
    global depth_model, depth_processor
    if depth_model is None or depth_processor is None:
        print(f"正在加载深度估计模型 ({model_name})...")
        try:
            depth_processor = DPTImageProcessor.from_pretrained(model_name)
            depth_model = DPTForDepthEstimation.from_pretrained(model_name, torch_dtype=torch_dtype)
            depth_model.to(device)
            depth_model.eval() # 设置为评估模式
            print("深度估计模型加载成功。")
            return True
        except Exception as e:
            print(f"加载深度估计模型时出错: {e}")
            depth_model = None
            depth_processor = None
            return False
    return True

# --------------------------------------------------------------------------
# Perception Functions
# --------------------------------------------------------------------------

def segment_image_fully_sam(image_np, mask_generator):
    """
    使用 SAM AutomaticMaskGenerator 对整个图像进行分割。
    (函数与之前版本相同)
    """
    if mask_generator is None:
        print("错误: SamAutomaticMaskGenerator 未加载。")
        return None
    try:
        print("SAM: 正在为整个图像生成掩码...")
        if image_np.shape[2] != 3:
            print(f"错误: 图像需要是 3 通道 RGB，但得到 {image_np.shape[2]} 通道。")
            return None
        if image_np.dtype != np.uint8:
            print(f"警告: 图像数据类型应为 np.uint8，但得到 {image_np.dtype}。正在尝试转换...")
            image_np = image_np.astype(np.uint8)
        masks = mask_generator.generate(image_np)
        print(f"SAM: 生成了 {len(masks)} 个掩码。")
        if not masks:
            print("SAM: 未生成掩码。")
            return None
        return masks
    except Exception as e:
        print(f"SAM 全图分割过程中出错: {e}")
        return None

def estimate_depth_from_rgb(image_pil):
    """
    从 RGB 图像估计深度。

    Args:
        image_pil (PIL.Image): 输入 RGB 图像。

    Returns:
        np.ndarray or None: 预测的深度图 (H, W)，如果出错则为 None。
    """
    if depth_model is None or depth_processor is None:
        print("错误: 深度估计模型未加载。")
        return None

    try:
        print("深度估计: 正在处理图像并预测深度...")
        inputs = depth_processor(images=image_pil, return_tensors="pt").to(device, dtype=torch_dtype)
        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 将深度图插值到原始图像大小
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1], # (height, width)
            mode="bicubic",
            align_corners=False,
        )
        prediction = prediction.squeeze().cpu().numpy()
        print("深度估计完成。")
        return prediction
    except Exception as e:
        print(f"深度估计过程中出错: {e}")
        return None

# --------------------------------------------------------------------------
# Visualization Functions
# --------------------------------------------------------------------------

def show_sam_anns(anns, image_np, output_filename="sam_segmented_output.png"):
    """
    在图像上显示 SamAutomaticMaskGenerator 生成的注释（掩码）。
    (函数与之前版本类似，增加了保存功能)
    """
    if not anns:
        print("没有可显示的 SAM 注释。")
        plt.figure(figsize=(10, 8))
        plt.imshow(image_np)
        plt.title("原始图像 (无 SAM 掩码)")
        plt.axis('off')
        plt.savefig(output_filename.replace(".png", "_no_anns.png"))
        plt.show()
        return

    plt.figure(figsize=(12, 10))
    plt.imshow(image_np)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img_overlay = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img_overlay[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img_overlay[m] = color_mask
    ax.imshow(img_overlay)
    plt.title("SAM 全图分割结果")
    plt.axis('off')
    plt.savefig(output_filename)
    print(f"SAM 分割结果已保存到 {output_filename}")
    plt.show()

def show_depth_map(depth_map_np, original_image_np, output_filename="depth_estimation_output.png"):
    """
    显示估计的深度图。

    Args:
        depth_map_np (np.ndarray): 估计的深度图 (H, W)。
        original_image_np (np.ndarray): 原始 RGB 图像 (H, W, 3)，用于并排显示。
        output_filename (str): 保存深度图可视化结果的文件名。
    """
    if depth_map_np is None:
        print("没有可显示的深度图。")
        return

    plt.figure(figsize=(12, 6)) # 调整大小以适应两个子图

    plt.subplot(1, 2, 1)
    plt.imshow(original_image_np)
    plt.title("原始 RGB 图像")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(depth_map_np, cmap="plasma") # 使用 'plasma' 或 'viridis' 等 colormap
    plt.colorbar(label="相对深度")
    plt.title("估计的深度图")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"深度图可视化结果已保存到 {output_filename}")
    plt.show()

# --------------------------------------------------------------------------
# Main Pipeline Function
# --------------------------------------------------------------------------

def run_perception_pipeline(
    rgb_image_path,
    sam_model_path=None, # SAM 模型路径变为可选
    sam_model_type="vit_h",
    depth_model_name="Intel/dpt-large",
    run_sam_segmentation=False, # 控制是否运行 SAM
    run_depth_estimation=True,  # 控制是否运行深度估计
    show_visuals=True
):
    """
    运行感知流程，可以选择执行 SAM 分割和/或深度估计。
    """
    start_time = time.time()

    # --- 加载输入数据 ---
    print("--- 加载数据 ---")
    if not os.path.exists(rgb_image_path):
        print(f"错误: RGB 图像未在 {rgb_image_path} 找到")
        return None
    try:
        image_pil = Image.open(rgb_image_path).convert("RGB")
        image_np = np.array(image_pil)
        print(f"已加载 RGB 图像: {image_np.shape}, 类型: {image_np.dtype}")
    except Exception as e:
        print(f"加载图像时出错: {e}")
        return None

    # --- 深度估计 ---
    estimated_depth_map = None
    if run_depth_estimation:
        print("\n--- 运行深度估计 ---")
        if not load_depth_estimation_model(depth_model_name):
            print("深度估计模型加载失败，跳过深度估计。")
        else:
            estimated_depth_map = estimate_depth_from_rgb(image_pil)
            if estimated_depth_map is not None and show_visuals:
                show_depth_map(estimated_depth_map, image_np, output_filename=f"{os.path.splitext(os.path.basename(rgb_image_path))[0]}_depth.png")

    # --- SAM 全图分割 ---
    sam_masks = None
    if run_sam_segmentation:
        print("\n--- 运行 SAM 全图分割 ---")
        if sam_model_path is None:
            print("错误: 未提供 SAM 模型路径，跳过 SAM 分割。")
        else:
            sam_mask_generator = load_sam_for_automatic_maskgen(sam_model_path, sam_model_type)
            if sam_mask_generator is None:
                print("SAM MaskGenerator 加载失败，跳过 SAM 分割。")
            else:
                sam_masks = segment_image_fully_sam(image_np, sam_mask_generator)
                if sam_masks is not None and show_visuals:
                    show_sam_anns(sam_masks, image_np, output_filename=f"{os.path.splitext(os.path.basename(rgb_image_path))[0]}_sam_seg.png")

    end_time = time.time()
    print(f"\n感知流程在 {end_time - start_time:.2f} 秒内完成。")
    
    results = {}
    if estimated_depth_map is not None:
        results["depth_map"] = estimated_depth_map
    if sam_masks is not None:
        results["sam_masks"] = sam_masks
    return results if results else None

# --- Example Usage (for testing this script directly) ---
if __name__ == "__main__":
    print("运行感知流程示例...")

    # --- 配置 ---
    rgb_path = "image_d61af3.jpg" # 确保此图像文件存在
    sam_ckpt_path = "/home/kewei/17robo/01mydemo/01ckpt/sam_vit_h_4b8939.pth" # 您的 SAM 模型路径

    # 检查文件是否存在
    if not os.path.exists(rgb_path):
        print(f"错误: 示例图像 '{rgb_path}' 未找到。请将其放置在脚本目录或更新路径。")
    else:
        # 示例 1: 只运行深度估计
        print("\n--- 示例 1: 仅运行深度估计 ---")
        results_depth_only = run_perception_pipeline(
            rgb_image_path=rgb_path,
            run_depth_estimation=True,
            run_sam_segmentation=False, # 关闭 SAM
            show_visuals=True
        )
        if results_depth_only and "depth_map" in results_depth_only:
            print(f"深度估计成功。深度图形状: {results_depth_only['depth_map'].shape}")
        else:
            print("深度估计失败或未运行。")

        # 示例 2: 运行深度估计和 SAM 分割 (确保 SAM 检查点路径有效)
        if os.path.exists(sam_ckpt_path):
            print("\n--- 示例 2: 运行深度估计和 SAM 分割 ---")
            results_both = run_perception_pipeline(
                rgb_image_path=rgb_path,
                sam_model_path=sam_ckpt_path,
                run_depth_estimation=True,
                run_sam_segmentation=True,
                show_visuals=True
            )
            if results_both:
                if "depth_map" in results_both:
                    print(f"深度估计成功。深度图形状: {results_both['depth_map'].shape}")
                if "sam_masks" in results_both:
                     print(f"SAM 分割成功。生成了 {len(results_both['sam_masks'])} 个掩码。")
            else:
                print("深度估计和/或 SAM 分割失败或未运行。")
        else:
            print(f"\n警告: SAM 检查点 '{sam_ckpt_path}' 未找到。跳过运行 SAM 分割的示例。")

        # 示例 3: 只运行 SAM 分割 (确保 SAM 检查点路径有效)
        if os.path.exists(sam_ckpt_path):
            print("\n--- 示例 3: 仅运行 SAM 分割 ---")
            results_sam_only = run_perception_pipeline(
                rgb_image_path=rgb_path,
                sam_model_path=sam_ckpt_path,
                run_depth_estimation=False, # 关闭深度估计
                run_sam_segmentation=True,
                show_visuals=True
            )
            if results_sam_only and "sam_masks" in results_sam_only:
                print(f"SAM 分割成功。生成了 {len(results_sam_only['sam_masks'])} 个掩码。")
            else:
                print("SAM 分割失败或未运行。")
        else:
            print(f"\n警告: SAM 检查点 '{sam_ckpt_path}' 未找到。跳过仅运行 SAM 分割的示例。")

```





![](assets/2025-07-07-10-25-39-image.png)



![](assets/2025-07-07-10-24-58-image.png)
