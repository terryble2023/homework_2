#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <string>
#include <vector>

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>

//魔方部分
struct ColorRange
//运用的是hsv颜色空间，所以需要hsv的上下限
{
    std::string name;
    cv::Scalar lower_bound;
    cv::Scalar upper_bound;
    cv::Scalar draw_color;
    bool has_second = false;
    //红色有第二个范围，因此需要单开一个上下限范围
    cv::Scalar lower_bound_red;
    cv::Scalar upper_bound_red;
};

struct Sticker
//Sticker用来把颜色名和位置绑定成整体
{
    std::string color;
    cv::Rect box;
};

struct CubeResult
//一帧图片检测结果，包含所有检测到的Sticker和主要颜色，以及主要颜色的矩形框位置
{
    std::vector<Sticker> stickers;
    std::string dominant_color;
    cv::Rect square;
};
//魔方部分结束



class Detector
{
public:
    Detector();
    // 静态图片：识别AprilTag
    int detect(const cv::Mat& image,
               cv::Mat& binary_image,
               cv::Mat& result_image,
               int binary_threshold = 127);

    // 视频帧：识别魔方颜色。第四个参数是颜色表，用来和上面的重载区分。
    int detect(const cv::Mat& frame,
               cv::Mat& binary_image,
               cv::Mat& result_image,
               const std::vector<ColorRange>& colors,
               CubeResult& cube);

private:
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Ptr<cv::aruco::DetectorParameters> parameters_;

    // 从一个通道掩膜里找出符合“贴纸”形状的轮廓。
    std::vector<Sticker> findStickers(const cv::Mat& mask,
        const std::string& color_name) const;

    // 把矩形扩展成轴对齐正方形，边长取长边，保证能完整框住倾斜的魔方。
    static cv::Rect toSquare(const cv::Rect& box);
    // 用static关键字声明，因为这个函数不需要访问Detector类的成员变量，
    // 所以可以作为静态成员函数

    // 用边缘信息兜底定位，贴纸太少时使用。
    static cv::Rect locateByEdges(const cv::Mat& frame);



};

// 内置的六种魔方颜色阈值。
std::vector<ColorRange> defaultColorRanges();

#endif
