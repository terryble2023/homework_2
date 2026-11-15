#include "Detector.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace
{
    // 整个视频是一个1280*720的分辨率。贴纸大约是52*52的像素。

    const double kMinArea = 600.0;   
    // 因此，设置600像素，大约1/5的贴纸面积，这样可以过滤掉噪点。
    const double kMaxAreaRatio = 0.02;
    // 超过整幅图2%，2%约18400像素。高于此可以认为是手或者桌子等其他对象，不考虑。
    const double kMinFillRatio = 0.60;
    // 轮廓面积除以外接矩形面积，这个被记为填充度，倾斜的贴纸透视变形后填充度会下降。
    // 填充度大于0.60，就可以认为是贴纸。

    const double kMinAspectRatio = 0.5;
    const double kMaxAspectRatio = 2.0;
    // 宽高比范围，魔方倾斜时透视会让贴纸变扁，宽高比可能到0.6至1.6。
    // 取0.5至2.0的范围，加余量。
    const double kSizeTolerance = 0.5;      
    // 与中位边长的偏差容忍度
    // 同一个面上的贴纸尺寸接近。明显偏大或偏小的轮廓，多半是误检或者被合并的多个贴纸。
    // 因此，这里设置为0.5，即允许边长偏差为50%。

    const int kMorphKernelSize = 5;
    // 开运算 核心大小
    const int kMinStickersForColor = 6;
    // 最小贴纸数量，少于这个数就不判定主色

}//namespace，等价于给每个名字都加了static

std::vector<ColorRange> defaultColorRanges()
{
    std::vector<ColorRange> colors;
    // 动态数组，用来存储颜色范围，因为是动态，所以后面可以push_back来添加对象
    // 添加一个对象会包含其中的所有元素
    
    ColorRange red;
    red.name = "red";
    red.lower_bound = cv::Scalar(0, 110, 0);
    red.upper_bound = cv::Scalar(8, 255, 255);
    red.draw_color = cv::Scalar(0, 0, 255);
    red.has_second = true;
    red.lower_bound_red = cv::Scalar(172, 110, 70);
    red.upper_bound_red = cv::Scalar(179, 255, 255);
    colors.push_back(red);

    ColorRange orange;
    orange.name = "orange";
    orange.lower_bound = cv::Scalar(9, 110, 70);
    orange.upper_bound = cv::Scalar(19, 255, 255);
    orange.draw_color = cv::Scalar(0, 140, 255);
    colors.push_back(orange);

    ColorRange yellow;
    yellow.name = "yellow";
    yellow.lower_bound = cv::Scalar(20, 110, 70);
    yellow.upper_bound = cv::Scalar(35, 255, 255);
    yellow.draw_color = cv::Scalar(0, 255, 255);
    colors.push_back(yellow);

    ColorRange green;
    green.name = "green";
    green.lower_bound = cv::Scalar(40, 80, 50);
    green.upper_bound = cv::Scalar(90, 255, 255);
    green.draw_color = cv::Scalar(0, 200, 0);
    colors.push_back(green);

    ColorRange blue;
    blue.name = "blue";
    blue.lower_bound = cv::Scalar(95, 80, 50);
    blue.upper_bound = cv::Scalar(135, 255, 255);
    blue.draw_color = cv::Scalar(255, 0, 0);
    colors.push_back(blue);

    // 白面靠“高亮度 + 低饱和度”区分，不靠色相。
    ColorRange white;
    white.name = "white";
    white.lower_bound = cv::Scalar(0, 0, 205);
    white.upper_bound = cv::Scalar(179, 45, 255);
    white.draw_color = cv::Scalar(255, 255, 255);
    colors.push_back(white);

    return colors;
}

// 冒号后面初始化的是成员变量，不是“对函数初始化”
Detector::Detector()
    : dictionary_(cv::aruco::getPredefinedDictionary(
          cv::aruco::DICT_APRILTAG_36h11)),
      parameters_(cv::aruco::DetectorParameters::create())
{
}

cv::Rect Detector::toSquare(const cv::Rect& box)
{
    const int side = std::max(box.width, box.height);
    // 取长边作为正方形边长，以此保证框住整个魔方
    const int center_x = box.x + box.width / 2;
    const int center_y = box.y + box.height / 2;
    return cv::Rect(center_x - side / 2, center_y - side / 2, side, side);
    // 返回一个正方形矩形框，框住整个魔方
}

cv::Rect Detector::locateByEdges(const cv::Mat& frame)
//不依赖颜色，用边缘信息找画面里最大的物体，作为魔方位置的兜底。
//贴纸少于6个时，比如翻转过渡帧、严重反光帧，会使用这个方法。
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    // 转灰度图，后面做Canny边缘检测都要用（Canny边缘检测只吃单通道）
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    // Canny对噪声敏感。不模糊的话，桌面纹理、传感器噪声都会变成边缘，干扰轮廓提取

    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);    //低、高阈值设置

    // 把邻近边缘连成块，再取最大的外轮廓。
    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    // 创建一个9*9的矩形结构元素，矩形核，9*9比贴纸之间的缝隙大，可以把相邻边缘连成一片。
    
    cv::dilate(edges, edges, kernel, cv::Point(-1, -1), 2);
    cv::erode(edges, edges, kernel, cv::Point(-1, -1), 1);
    // 这个不是开运算或者闭运算，这个是膨胀-腐蚀特化组合
    // 膨胀2次加腐蚀1次，可以解决细线有断口的问题
    // 连接断口，同时略微扩张

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    // 找到所有外轮廓
    /*CV_EXPORTS_W void findContours(InputArray image,
                              OutputArrayOfArrays contours,
                              OutputArray hierarchy,
                              int mode,
                              int method,
                              Point offset = Point());
        image：输入的二值图
        contours：输出的轮廓，每个轮廓是一串点
        hierarchy：输出的轮廓层级关系，可省略
        mode：检索模式，决定要哪些层级的轮廓
        method：近似方法，决定轮廓点怎么存
        offset：所有有点统一平移，不写默认 (0,0)
    */
    if (contours.empty())
    {
        return cv::Rect();
    }
    // 一条轮廓都没找到，返回空矩形

    const auto largest = std::max_element(
        contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b)
        {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    // 找到最大的轮廓    
    //[]( ... ) { ... }是lambda表达式，表示匿名函数。这里临时定义了一个比较规则，不用单独写一个函数。
    
    if (cv::contourArea(*largest) < 20000.0) {
        return cv::Rect();
    }
    // 最大轮廓面积小于20000，返回空矩形
    // 魔方包围盒约268*285=76000平方像素。20000是它的四分之一左右，作为“至少像魔方”的下限。

    return cv::boundingRect(*largest);
    // 返回最大轮廓的外接矩形
}

std::vector<Sticker> Detector::findStickers(const cv::Mat& mask,
        const std::string& color_name) const
// 从一张颜色掩膜里，找出所有符合贴纸形状的轮廓
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    //找掩膜里所有白色区域的轮廓。

    const double max_area = kMaxAreaRatio * mask.total();
    // 算出面积上限
    std::vector<cv::Rect> candidates;
    // 存储粗筛选后的矩形框

    for (const auto& contour : contours)
    // 遍历所有轮廓
    // const auto&：auto让编译器推导类型，const&避免复制每条轮廓的点序列
    {
        const double area = cv::contourArea(contour);
        // 算轮廓面积
        if (area < kMinArea || area > max_area)
        // 面积太小（噪点）或太大（背景、手）的，跳过
        {
            continue;
        }

        const cv::Rect box = cv::boundingRect(contour);
        // 算轮廓的外接矩形
        if (box.width <= 0 || box.height <= 0)
        // 矩形宽度或高度非正，跳过
        {
            continue;
        }

        const double fill = area / (box.width * box.height);
        // 算填充率
        if (fill < kMinFillRatio)
        // 填充率太小，跳过
        {
            continue;
        }

        const double aspect = box.width / static_cast<double>(box.height);
        // 算宽高比
        // static_cast<double>(box.height)：box.height是int
        // 如果直接写 box.width/box.height，两个int相除会做整数除法，100/200得到0，宽高比就全错了
        // 转成 double 才能得到 0.5
        if (aspect < kMinAspectRatio || aspect > kMaxAspectRatio)
        // 宽高比不在范围内，跳过
        {
            continue;
        }

        candidates.push_back(box);
        // 加入候选矩形框
    }

    if (candidates.empty())
    // 没有候选矩形框，返回空vector
    {
        return std::vector<Sticker>();
    }

    std::vector<int> sides;
    sides.reserve(candidates.size());
    // 存储每个候选矩形框的边长
    // reserve和vector的capacity有关，只是提前分配内存，不改变size

    for (const auto& box : candidates)
    // 第一步：把所有候选的边长收集起来
    // 这一步必须先完整做完，下面才能算中位数
    {
        sides.push_back(std::max(box.width, box.height));
        // 加入候选矩形框的边长
        // std::max(box.width, box.height)：取矩形的长边作为边长
        // 之所以取长边，是因为矩形可能被透视压扁，长边更能代表真实尺寸
    }

    std::sort(sides.begin(), sides.end());
    // 第二步：排序
    // 求中位数需要先排序，所以这一句必须在取中位数之前

    const double median = sides[sides.size() / 2];
    // sides.size()/2是整数除法，偶数个元素时取中间偏右那个
    // size为16时，下标8，属于偏右的中位数，对筛选来说足够

    const double size_low = median * (1.0 - kSizeTolerance);
    const double size_high = median * (1.0 + kSizeTolerance);
    // 算出允许的尺寸区间，kSizeTolerance是0.5，即允许偏差50%
    // 算中位数，不用平均值是因为平均值会被极端值拉偏
    // 这里变量名不用lower_bound/upper_bound，避免和std::lower_bound混淆

    std::vector<Sticker> stickers;
    // 第三步：用刚才算好的区间筛选候选

    for (const auto& box : candidates)
    // 重新遍历一遍全部候选
    // 不能用上面那个循环，因为那时中位数还没算出来
    {
        const double side = std::max(box.width, box.height);
        if (side < size_low || side > size_high)
        // 尺寸明显偏离中位数的候选，跳过
        {
            continue;
        }

        Sticker sticker;
        sticker.color = color_name;
        sticker.box = box;
        stickers.push_back(sticker);
        // 组装一个Sticker并加入结果
        // 这一句每轮只执行一次，不会重复添加
    }

    return stickers;
    // 返回筛过的贴纸列表，可能为空
}
int Detector::detect(const cv::Mat& frame,
                     cv::Mat& binary_image,
                     cv::Mat& result_image,
                     const std::vector<ColorRange>& colors,
                     CubeResult& cube)
{
    cube = CubeResult();
    // 把结果结构体重置为默认状态
    // 不重置的话，上一帧的贴纸会残留。

    if (frame.empty()) {
        return 0;
    }
    
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    binary_image = cv::Mat::zeros(frame.size(), CV_8UC1);
    // 做一张纯黑的二值图，用来存储检测到的贴纸
    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_RECT,
            cv::Size(kMorphKernelSize,kMorphKernelSize));
    
    for (const auto& range : colors) {
        cv::Mat mask;
        cv::inRange(hsv, range.lower_bound, range.upper_bound, mask);

        if (range.has_second) {
            cv::Mat second_mask;
            cv::inRange(hsv, range.lower_bound_red, range.upper_bound_red, second_mask);
            cv::bitwise_or(mask, second_mask, mask);
            // bitwise_or：逐像素按位或。两张掩膜里只要有一张是255，结果就是255。
        }

        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        // 开运算去噪。在mask上进行开运算，把小的噪点去掉。
        const std::vector<Sticker> found = findStickers(mask, range.name);
        // 从掩膜里筛出贴纸
        cube.stickers.insert(cube.stickers.end(), found.begin(), found.end());
        // 把当前颜色的贴纸追加到总列表
        cv::bitwise_or(binary_image, mask, binary_image);
        // 把当前颜色的掩膜累加到总二值图里
    }

    /*
-----------下方继续施工------------------
    */

    // 统计每种颜色的贴纸数，最多的一种就是当前主色。
    std::vector<std::string> names;     //颜色名
    std::vector<int> counts;            //每个颜色的贴纸数 
    for (const auto& sticker : cube.stickers) {
        auto it = std::find(names.begin(), names.end(), sticker.color);
        // 在已有的名字列表里找当前贴纸的颜色，找不到的话跳至names.end()
        if (it == names.end()) {
            names.push_back(sticker.color);
            counts.push_back(1);
        } else {
            ++counts[static_cast<std::size_t>(it - names.begin())];
            // static_cast<std::size_t>使下标非负
        }
    }

    int dominant_count = 0;
    if(!names.empty())
    {
        const auto best = std::max_element(counts.begin(), counts.end());
        // 找到最大贴纸数的颜色
        const std::size_t index = static_cast<std::size_t>(best - counts.begin());
        // 找到最大贴纸数的颜色的下标
        if (*best >= kMinStickersForColor) {
            cube.dominant_color = names[index];
            dominant_count = *best;
        }
        // 数量达到门槛才认定为“当前主色”
    }

    if (!cube.stickers.empty())
    //有贴纸时，用贴纸的位置定位
    {
        cv::Rect union_box = cube.stickers.front().box;
        // 用第一个贴纸的矩形作为并集的初始值
        for (const auto& sticker : cube.stickers)
        // 把所有贴纸的矩形都并集起来
        {
            union_box |= sticker.box;
            // |=：按位或，把union_box和sticker.box的每个像素都取较大的值
        }
        cube.square = toSquare(union_box);
        // 把并集得到的矩形扩展成轴对齐正方形
    } else {
        const cv::Rect edge_box = locateByEdges(frame);
        // 完全不看颜色，只看画面里面最大的物体轮廓
        // 魔方翻转的过渡帧，贴纸朝向侧面，颜色面积太小，检不出来
        // 因此需要这个方法来兜底，防止它边框乱闪
        if (edge_box.width > 0 && edge_box.height > 0)
        {
            cube.square = toSquare(edge_box);
        }
    }

    result_image = frame.clone();

    for (const auto& sticker : cube.stickers)
    {
        cv::Scalar color(200, 200, 200);
        for (const auto& range : colors) {
            if (range.name == sticker.color) {
                color = range.draw_color;
                // 按照名字找到对应的颜色
                break;
            }
        }
        cv::rectangle(result_image, sticker.box, color, 2, cv::LINE_AA);
        // 绘制贴纸的矩形框
    }

    if (cube.square.width > 0 && cube.square.height > 0) {
        // 方框可能超出图像边界，画之前裁剪一下
        const cv::Rect image_area(0, 0, result_image.cols,
            result_image.rows);
        const cv::Rect visible = cube.square & image_area;
        // 取交集，把方框裁剪到图像边界
        if (visible.width > 0 && visible.height > 0) {
            cv::rectangle(result_image, visible,
                cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        }
    }

    const std::string text =
        cube.dominant_color.empty()
            ? std::string("color: unknown")
            : "color: " + cube.dominant_color + " (" +
                  std::to_string(dominant_count) + " stickers)";
    cv::putText(result_image, text, cv::Point(20, 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2,
                cv::LINE_AA);

    return dominant_count;
    // 返回主色贴纸数
}
int Detector::detect(const cv::Mat& image,
                     cv::Mat& binary_image,
                     cv::Mat& result_image,
                     int binary_threshold)
{
    if (image.empty())
    {
        return 0;
    }

    // 第一步：转灰度图，后面做二值化和检测都要用。
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // 第二步：生成展示用二值图。它不影响 detectMarkers 的检测结果。
    cv::threshold(gray_image, binary_image, binary_threshold, 255,
                  cv::THRESH_BINARY);

    // 第三步：检测标签。ids 存编号，corners 存每个标签的四个角点。
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    // 外层vector：每个元素代表一个标签
    // 内层vector：那个标签的四个角点
    // cv::Point2f：一个点，x 和 y 是 float
    cv::aruco::detectMarkers(image, dictionary_, corners, ids, parameters_);
    //调用OpenCV的detectMarkers函数，检测标签。

    // 第四步：复制一份原图，所有绘制都画在副本上。
    result_image = image.clone();

    const int tag_count = static_cast<int>(ids.size());
    //将ids.size()显式转换成int形式，防止后面循环内与int i报警告（ids.size()是size_t类型，i是int类型）

    if (tag_count == 0) 
    {
        cv::putText(result_image, "No AprilTag found", cv::Point(15, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2,
                    cv::LINE_AA);
        return 0;
    }
    //没识别到标签返回文字段。cv::FONT_HERSHEY_SIMPLEX为OpenCV 内置字体。字体不支持中文。
    cv::aruco::drawDetectedMarkers(result_image, corners);
    //绘制检测到的标签，每个标签用一个矩形框表示。
    
    for (int i = 0; i<tag_count;++i) 
    {
        const std::vector<cv::Point2f>& tag_corners = corners[i];
        const int corner_count = static_cast<int>(tag_corners.size());
        //取角点数量

        if (corner_count != 4) 
        {
            continue;
        }
        //如果角点数量不是4，跳过。(纯粹防御性写法，非必要)

        for (int j = 0; j<corner_count;++j) 
        {
            //遍历每个角点
            const cv::Point2f corner = tag_corners[j];
            const cv::Point corner_pixel(cvRound(corner.x), cvRound(corner.y));
            //将角点坐标四舍五入为int类型，防止后面绘制时报警告（corner.x和corner.y是float类型）
            //cvRound(float x)：将浮点数四舍五入为最近的整数。
            //cv::Point(x, y)：用两个整数构造一个点。
            cv::circle(result_image, corner_pixel, 5, cv::Scalar(0, 0, 255),
                       cv::FILLED, cv::LINE_AA);
            //绘制角点，半径为5，颜色为红色，线型为抗锯齿（cv::LINE_AA）
        }
        // 四个角点分别取出来，方便下面一行一行算中心。
        const cv::Point2f p0 = tag_corners[0];
        const cv::Point2f p1 = tag_corners[1];
        const cv::Point2f p2 = tag_corners[2];
        const cv::Point2f p3 = tag_corners[3];

        // 中心点 = 四个角点坐标的平均值。
        const float center_x = (p0.x + p1.x + p2.x + p3.x) / 4.0F;
        const float center_y = (p0.y + p1.y + p2.y + p3.y) / 4.0F;
        const cv::Point center_pixel(cvRound(center_x), cvRound(center_y));
        //见71～73行注释

        cv::circle(result_image, center_pixel, 5, cv::Scalar(255,0,0),
               cv::FILLED, cv::LINE_AA);
        //绘制中心点，半径为5，颜色为蓝色，线型为抗锯齿

        // 在中心点右上方写标签编号。
        const std::string label = "ID:" + std::to_string(ids[i]);
        const cv::Point label_position(center_pixel.x + 12, center_pixel.y - 12);
        cv::putText(result_image, label, label_position,
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2,
                    cv::LINE_AA);
        //绘制标签编号
    }
    
    const std::string status =
        "AprilTag detected: " + std::to_string(tag_count);
    cv::putText(result_image, status, cv::Point(15, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2,
                cv::LINE_AA);
    //绘制状态文本

    return tag_count;
}
