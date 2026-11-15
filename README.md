# 视觉作业二：AprilTag 识别与魔方颜色识别

基于 OpenCV 4 的 C++ 程序，用一个 `Detector` 类提供两个 `detect` 重载：

- **静态图片**：识别 AprilTag，绘制边框、角点、中心点和编号。
- **视频**：逐帧识别魔方颜色，判断当前主色并框出魔方位置。

## 环境

| 项目 | 版本 |
|---|---|
| 系统 | Ubuntu 22.04 |
| OpenCV | 4.5.4 |
| C++ | C++17 |
| CMake | 3.16 以上 |

安装 OpenCV 开发包：

```bash
sudo apt install libopencv-dev
```

## 编译

```bash
cmake -S . -B build
cmake --build build -j2
```

生成的可执行文件是 `build/homework_2`。

## 运行

程序根据文件扩展名自动判断模式，不需要额外参数。

**AprilTag 模式**（图片）：

```bash
./build/homework_2 assets/tag36h11_id7.png
```

弹出三个窗口：

| 窗口 | 内容 |
|---|---|
| `Image_raw` | 原图 |
| `Image_binary` | 灰度阈值分割后的二值图 |
| `result` | 绿框、红色角点、蓝色中心点、`ID:7` |

按任意键退出。同时在当前目录生成 `binary_image.png` 和 `result_image.png`。

**魔方模式**（视频）：

```bash
./build/homework_2 "assets/1da6c4bdaad0682d691b0ee6f814bfff(1).mp4"
```

文件名带括号，命令行里必须加引号。

弹出五个窗口：

| 窗口 | 内容 |
|---|---|
| `Image_raw` | 原视频帧 |
| `Image_gray` | 灰度图 |
| `Image_canny` | Canny 边缘图 |
| `Image_mask` | 六色颜色掩膜的并集 |
| `result` | 贴纸彩色小框 + 绿色正方形大框 + 颜色结论 |

按 `q` 或 `Esc` 退出。终端会打印处理的总帧数。

## 文件结构

```text
homework_2/
├── include/
│   └── Detector.hpp        类的声明，两个 detect 重载
├── src/
│   ├── Detector.cpp        检测与绘制实现
│   └── main.cpp            扩展名分流 + 两个模式的入口
├── assets/
│   ├── tag36h11_id7.png    tag36h11 家族、编号 7 的测试标签
│   └── *.mp4               魔方测试视频
├── CMakeLists.txt
└── README.md
```

## 两个 detect 重载

```cpp
// AprilTag 版：第四个参数是 int
int detect(const cv::Mat& image,
           cv::Mat& binary_image,
           cv::Mat& result_image,
           int binary_threshold = 127);

// 魔方版：第四个参数是颜色表，还多一个结果结构体
int detect(const cv::Mat& frame,
           cv::Mat& binary_image,
           cv::Mat& result_image,
           const std::vector<ColorRange>& colors,
           CubeResult& cube);
```

编译器靠参数个数和类型区分，与参数名无关。

## AprilTag 识别流程

```text
读图
  ↓
转灰度 → 阈值分割（仅用于展示二值图）
  ↓
detectMarkers(图像, tag36h11 字典)
  ↓
得到 ids（编号）和 corners（四个角点）
  ↓
在原图副本上画绿框、红角点、蓝中心点、编号
```

## 魔方颜色识别流程

```text
视频帧
  ↓
BGR → HSV
  ↓
六种颜色各做一次 inRange，生成掩膜（红色分两段 H 范围）
  ↓
形态学开运算去噪
  ↓
findContours 找轮廓
  ↓
按面积、填充度、宽高比、与中位边长的偏差四道条件筛选
  ↓
统计各色贴纸数，最多且不少于 6 个的判为当前主色
  ↓
所有贴纸求并集，扩展成轴对齐正方形
  ↓
绘制贴纸框、绿色方框和颜色文字
```

## 已知行为

- 魔方视频中 9.1–12.3 秒是翻转过渡段，没有完整的面朝向镜头，这几帧会判为 `unknown`，属于预期。
- 白面贴纸检出数通常是 13–15 个而非 16 个，原因是相邻白色贴纸在反光处容易粘连成一个轮廓，不影响主色判断。
- 绿色方框是轴对齐正方形，边长取魔方包围盒的长边，所以魔方倾斜时方框会变大，但一定能完整框住。

## 参数调整

六种颜色的 HSV 阈值集中在 `Detector.cpp` 的 `defaultColorRanges()` 里。

| 参数 | 位置 | 作用 |
|---|---|---|
| HSV 上下限 | `defaultColorRanges()` | 控制每种颜色的判定范围 |
| `kMinArea` | 匿名命名空间 | 贴纸最小面积 |
| `kMaxAreaRatio` | 匿名命名空间 | 贴纸最大面积占整图比例 |
| `kMinFillRatio` | 匿名命名空间 | 轮廓填充度下限 |
| `kMinAspectRatio` / `kMaxAspectRatio` | 匿名命名空间 | 宽高比范围 |
| `kSizeTolerance` | 匿名命名空间 | 与中位边长的允许偏差 |
| `kMinStickersForColor` | 匿名命名空间 | 判定主色所需的最少贴纸数 |
