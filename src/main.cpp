#include "Detector.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
// imgproc.hpp：cvtColor、GaussianBlur、Canny
// videoio.hpp：VideoCapture
// 这两个是魔方模式才需要的，之前只跑AprilTag时没有

namespace {

// 匿名命名空间：下面两个函数只在本文件内可见
// 等价于给每个函数加static，避免和其他文件重名

bool isImagePath(const std::string& path)
// 根据文件扩展名判断这是图片还是视频
{
    const std::string image_extensions[] = {".png",  ".jpg",  ".jpeg",
                                            ".bmp",  ".tif",  ".tiff",
                                            ".webp"};
    // 常见图片格式。用std::string数组而不是const char*
    // 这样下面可以直接用==比较，不用写strcmp

    const std::size_t dot = path.find_last_of('.');
    // find_last_of('.')找到最后一个点的位置
    // 用“最后”而不是“第一个”，是因为路径里可能有 ./ 或者目录名带点
    // 比如 ./assets/tag.png 最后一个点才是扩展名的分隔符

    if (dot == std::string::npos)
    // npos是string的“找不到”标记，本质是size_t的最大值
    {
        return false;
    }

    std::string extension = path.substr(dot);
    // substr(dot)从点开始一直取到结尾，得到 ".png" 这样带点的字符串

    for (char& ch : extension)
    // 统一转成小写，这样 .PNG 和 .png 都能识别
    // char& 是引用，改ch就是改extension里的字符
    {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }

    for (const std::string& known : image_extensions)
    // 逐个比较，命中任何一个就认为是图片
    {
        if (extension == known) {
            return true;
        }
    }

    return false;
    // 不是已知图片扩展名，就当视频处理
}

int runAprilTag(const std::string& path)
// AprilTag模式：读一张静态图片，检测并显示三个窗口
// 这部分是原来main里的内容，原样搬了
{
    const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Cannot read image: " << path << std::endl;
        return 1;
    }
    // imread失败不会抛异常，而是返回空Mat，所以要主动检查

    Detector detector;
    // 创建检测器，构造函数里准备好字典和参数

    cv::Mat binary_image;
    cv::Mat result_image;
    const int binary_threshold = 127;
    // 这两个Mat现在是空的，detect里的引用参数会把它们填满

    const int detected_count = detector.detect(image, binary_image,
                                               result_image, binary_threshold);
    // 四个实参，编译器选中的是AprilTag版
    // 因为第四个参数是int，不是vector<ColorRange>

    std::cout << "Detected " << detected_count << " AprilTags." << std::endl;

    cv::imwrite("binary_image.png", binary_image);
    cv::imwrite("result_image.png", result_image);
    // 存成文件，方便写报告时直接用

    cv::imshow("Image_raw", image);
    cv::imshow("Image_binary", binary_image);
    cv::imshow("result", result_image);
    // 题目要求的三窗口

    cv::waitKey(0);
    // 参数0表示无限等待按键。不加这句窗口会一闪而过

    cv::destroyAllWindows();
    return 0;
}

int runRubik(const std::string& path)
// 魔方模式：读视频，逐帧识别，显示五个窗口
{
    cv::VideoCapture capture(path);
    if (!capture.isOpened()) {
        std::cerr << "Cannot open video: " << path << std::endl;
        return 1;
    }
    // VideoCapture既能开摄像头也能开视频文件
    // 传文件路径就是读文件，传0就是开默认摄像头
    // 打不开通常是路径错、文件损坏、缺解码器

    Detector detector;
    const std::vector<ColorRange> colors = defaultColorRanges();
    // 六色阈值表只建一次，不要放进循环里反复构造

    cv::namedWindow("Image_raw", cv::WINDOW_NORMAL);
    cv::namedWindow("Image_gray", cv::WINDOW_NORMAL);
    cv::namedWindow("Image_canny", cv::WINDOW_NORMAL);
    cv::namedWindow("Image_mask", cv::WINDOW_NORMAL);
    cv::namedWindow("result", cv::WINDOW_NORMAL);
    // 显式创建窗口，WINDOW_NORMAL表示窗口可以缩放
    // 不显式创建的话imshow会自动建WINDOW_AUTOSIZE窗口，不能缩放

    cv::resizeWindow("Image_raw", 360, 640);
    cv::resizeWindow("Image_gray", 360, 640);
    cv::resizeWindow("Image_canny", 360, 640);
    cv::resizeWindow("Image_mask", 360, 640);
    cv::resizeWindow("result", 360, 640);
    // 原视频是720*1280竖屏，缩一半正好360*640
    // 五个窗口并排才放得下。注意这只改显示大小，不改图像数据

    int frame_count = 0;
    // 记录处理了多少帧，退出时打印

    while (true) {
        cv::Mat frame;
        capture >> frame;
        // >> 是read的运算符重载写法，效果等同capture.read(frame)

        if (frame.empty()) {
            break;
        }
        // 视频播完、解码出错都会得到空帧
        // 这里是正常退出路径，不是错误

        ++frame_count;

        cv::Mat gray_image;
        cv::cvtColor(frame, gray_image, cv::COLOR_BGR2GRAY);
        // 转灰度，给Image_gray窗口显示

        cv::Mat blurred;
        cv::GaussianBlur(gray_image, blurred, cv::Size(5, 5), 0);
        // 高斯模糊，为Canny做准备
        // 单独存到blurred而不是覆盖gray_image
        // 因为Image_gray窗口要显示未模糊的原灰度图

        cv::Mat canny_image;
        cv::Canny(blurred, canny_image, 50, 150);
        // 50是低阈值，150是高阈值
        // 双阈值让边缘更连续：弱边缘只有连到强边缘才保留

        cv::Mat mask_image;
        cv::Mat result_image;
        CubeResult cube;
        // mask_image、result_image开始是空的，cube里的成员也是默认值

        detector.detect(frame, mask_image, result_image, colors, cube);
        // 五个实参，编译器选中的是魔方版
        // 第四个参数是vector<ColorRange>，和AprilTag版的int不同
        // 调用后mask_image是六色掩膜并集，result_image画好了框

        cv::imshow("Image_raw", frame);
        cv::imshow("Image_gray", gray_image);
        cv::imshow("Image_canny", canny_image);
        cv::imshow("Image_mask", mask_image);
        cv::imshow("result", result_image);
        // Image_raw显示frame而不是result_image
        // 因为frame从头到尾没被改过，所有绘制都在clone出来的副本上

        const int key = cv::waitKey(30) & 0xFF;
        // 等30毫秒，约33fps，接近视频的30fps
        // &0xFF只保留低8位，把特殊值规范成0到255
        // waitKey在视频循环里必须调用，它同时负责刷新窗口和处理按键

        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
        // 27是Esc的ASCII码
    }

    std::cout << "Processed " << frame_count << " frames." << std::endl;

    cv::destroyAllWindows();
    return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <image_or_video_path>" << std::endl;
        return 1;
    }
    // 只接受一个参数：图片或者视频的路径

    const std::string path = argv[1];

    if (isImagePath(path)) {
        return runAprilTag(path);
    }
    // 图片交给AprilTag模式

    return runRubik(path);
    // 不是图片就当视频，交给魔方模式
    // 这里用return直接返回函数的返回值，省掉一个中间变量
}