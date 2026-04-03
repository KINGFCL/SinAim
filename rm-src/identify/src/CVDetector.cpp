#include "CVDetector.hpp"
#include <deque>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sys/types.h>
#include <vector>
//#include <cmath>
//#define detectorDebug
//Debug

CVDetector::CVDetector(Light::Color color, cv::Size ROISize): 
                   color(color),
                   ROISize(ROISize)
                   {}

/**
 * @brief 该函数用于检测装甲板
 * @param frame 图像帧
 * @param armors_pattern 装甲板的图像
 * @return 可能的装甲板
 */
std::deque<CVArmor> CVDetector:: operator () (cv::Mat& frame,std::vector<cv::Mat>& armors_pattern) 
{
    this->rgb_img = frame;

    cv::Mat binary_img = preprocessImage(frame); //预处理图像

    std::deque<Light> lights = FindLight(binary_img); //寻找灯条
    #ifdef detectorDebug
    std::cout <<"lights num:" << lights.size() << "\n";
    #endif

    std::deque<CVArmor> armors = FindArmor(lights); //寻找装甲板
    #ifdef detectorDebug
    std::cout <<"possible_armors num:" << armors.size() << "\n";
    cv::Mat show__ = this->rgb_img.clone();
    this->ArmorShow(show__,armors);
    cv::imshow("armors",show__);
    #endif 

    armors_pattern = this->ROIArmor(armors);

    return armors;
}

std::deque<CVArmor> CVDetector::operator () (cv::Mat& frame,std::vector<cv::Mat>& armors_pattern,bool isSmallROI)
{
    this->rgb_img = frame;

    cv::Mat binary_img = preprocessImage(frame); //预处理图像

    std::deque<Light> lights = FindLight(binary_img); //寻找灯条
    #ifdef detectorDebug
    std::cout <<"lights num:" << lights.size() << "\n";
    #endif

    std::deque<CVArmor> armors = FindArmor(lights); //寻找装甲板
    #ifdef detectorDebug
    std::cout <<"possible_armors num:" << armors.size() << "\n";
    cv::Mat show__ = this->rgb_img.clone();
    this->ArmorShow(show__,armors);
    cv::imshow("armors",show__);
    #endif 

    armors_pattern = this->SmallROIArmor(armors);

    return armors;
}

/**
 * @brief 该函数用于对图像进行预处理
 * @param rgb_img  RGB 图像
 * @return  binary_img  二值图像
 * @details 该函数将 RGB 图像转换为灰度图像，然后对其进行二值化处理
 */
cv::Mat CVDetector::preprocessImage(cv::Mat& rgb_img) //图像预处理
{

  cv::cvtColor(rgb_img, this->gray_img, cv::COLOR_RGB2GRAY);

  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, 200, 255, cv::THRESH_BINARY);
  #ifdef detectorDebug
  cv::imshow("binary_img",binary_img);
  cv::waitKey(1);
  #endif

  return binary_img;
}


/**
 * @brief 寻找灯条
 * @param binary_img 二值图像
 * @return std::deque<Light>  可能的灯条
 * @details 该函数使用OpenCV的findContours函数来找到图像中的所有轮廓，然后对每个轮廓进行拟合矩形，过滤掉不符合灯条特征的矩形，最后记录识别到的灯条
 */
std::deque<Light> CVDetector::FindLight(const cv::Mat & binary_img) //寻找灯条
{
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::deque<Light> lights;
  for (const auto & contour : contours) {

    //过滤掉小轮廓
    if (contour.size() < 4) continue; //最小点数
    if(cv::contourArea(contour)<30) continue; //最小面积

    //拟合矩形
    cv::RotatedRect rect = cv::minAreaRect(contour);//最小外接矩形
    Light light(rect);

    //过滤掉不符合灯条特征的矩形
    double aspect_ratio = light.length/light.width; //长宽比
    if (aspect_ratio < 2) continue; //长宽比阈值

    if (light.length < 10 || light.width > 100) continue; //长度，宽度阈值

    auto AngleIsOK = [&light]() ->bool
    {
        double tilt_angle = std::atan2(std::abs(light.top.x - light.bottom.x), std::abs(light.top.y - light.bottom.y));
        tilt_angle = tilt_angle / CV_PI * 180;

        if (tilt_angle > 60) return false;
        return true;
    };
    if (!AngleIsOK()) continue; //角度阈值
    
    auto& rgb_image = this->rgb_img;

    auto getLightColor = [&rect, &rgb_image]() -> Light::Color {
        // 获取正方向的外接矩形，并与原图边界求交集，防止越界访问引发段错误
        cv::Rect bbox = rect.boundingRect();
        cv::Rect safe_bbox = bbox & cv::Rect(0, 0, rgb_image.cols, rgb_image.rows);

        // 提前提取旋转矩形的几何参数，供后续的投影计算使用
        float cx = rect.center.x;
        float cy = rect.center.y;
        // 注意：OpenCV 中角度的单位可能依版本和生成方式不同，确保转为弧度
        float angle = rect.angle * CV_PI / 180.0f; 
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        float half_w = rect.size.width * 0.5f;
        float half_h = rect.size.height * 0.5f;

        int redrate = 0;
        int bluerate = 0;

        //在原图上按行连续遍历外接矩形区域（对 CPU Cache 最友好的方式）
        for (int y = safe_bbox.y; y < safe_bbox.y + safe_bbox.height; ++y) {
            // 获取当前行的行首指针，避免使用缓慢的 .at()
            const cv::Vec3b* row_ptr = rgb_image.ptr<cv::Vec3b>(y);
            float dy = y - cy;
            
            for (int x = safe_bbox.x; x < safe_bbox.x + safe_bbox.width; ++x) {
                float dx = x - cx;
                
                // 将当前坐标 (dx, dy) 投影到旋转矩形的局部 X 轴和 Y 轴上
                float local_x = dx * cosA + dy * sinA;
                float local_y = -dx * sinA + dy * cosA;
                
                // 判断投影后的局部坐标是否落在旋转矩形的宽高范围内
                if (std::abs(local_x) <= half_w && std::abs(local_y) <= half_h) {
                    // 如果在内部，执行原有的红蓝判断逻辑
                    uchar b = row_ptr[x][0];
                    uchar r = row_ptr[x][2];
                    
                    if (r > b) {
                        redrate++;
                    } else if (b > r) {
                        bluerate++;
                    }
                }
            }
        }

        // 返回数量较多的一方
        return (redrate > bluerate) ? Light::Color::Red : Light::Color::Blue;
    };

    if(getLightColor() != this->color) continue; //不是目标颜色
 
    lights.emplace_back(rect);//记录识别到的灯条
    }  

  return lights;
}

/**
 * @brief Finds all armor in the given lights
 * @param lights The deque of lights to find armor in
 * @return A deque of all armor found in the given lights
 * 
 * This function iterates over all lights and checks if any two lights can form an armor
 * It then checks if the two lights are close enough to the center of the image and
 * if the distance between the two lights is less than 3 times the length of the shorter light
 * If all conditions are met, the two lights are added to the result deque
 */
std::deque<CVArmor> CVDetector::FindArmor(const std::deque<Light> & lights)
{
    std::deque<CVArmor> armors;
    std::deque<unsigned long> LightIndex;//记录配对成功的灯条索引
    std::deque<std::array<unsigned long, 2>> armorLightIndex;//记录每个装甲板两个等条的索引
    std::vector<bool> HaxLight(lights.size(),false);//灯条是否配对成功的哈希表
    if(lights.empty()) return armors;

    auto matchIsOk = [](const Light& light1, const Light& light2) -> bool
    {
        //长度匹配
        double biglen = std::max(light1.length, light2.length);
        double smalen = std::min(light1.length, light2.length);
        double rate = smalen / biglen;
        if(rate<0.6) return false;

        //灯条平行匹配
        cv::Point2f L1vec = light1.top-light1.bottom,
                    L2vec = light2.top-light2.bottom;
        double cosAngle = L1vec.dot(L2vec) / (cv::norm(L1vec) * cv::norm(L2vec));
        double Angle = std::abs( std::acos(cosAngle)/CV_PI*180 );
        if(Angle>30) return false;

        //高度匹配
        cv::Point2f toward = L1vec + L2vec;
        cv::Point2f L1ToL2vec = light1.center-light2.center;

        double HighDiff = std::abs(toward.dot(L1ToL2vec)/cv::norm(toward));
        if(HighDiff>(smalen*0.8)) return false;

        //距离匹配
        double distance = cv::norm(L1ToL2vec);
        if(distance>(3*biglen)||distance<(smalen*0.2)) return false;

        //匹配成功
        return true;
    };

    //枚举灯条进行配对并记录配对成功的灯条索引
    for(unsigned long i=0;i<lights.size()-1;i++)
    {
        for(unsigned long j=i+1;j<lights.size();j++)
        {
            if(matchIsOk(lights[i],lights[j]))
            {
                armors.emplace_back(lights[i], lights[j]);
                armorLightIndex.emplace_back(std::array<unsigned long, 2>{i,j});
                if(HaxLight[i]==false) { LightIndex.emplace_back(i); HaxLight[i]=true;}
                if(HaxLight[j]==false) { LightIndex.emplace_back(j); HaxLight[j]= true;}     
            }      
        }
    }

    //筛选配对不准确的装甲板
    auto InMind = [](const std::vector<cv::Point2f>& rect,const cv::Point2f& p) ->bool
    {
        // 检查输入是否为四边形
        if (rect.size() != 4) return false;

        // 计算向量(a,b)和向量(a,p)的叉积
        auto cross_product = [](const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& p) -> float {
        return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);};

        // 计算点p与四条边的叉积
        float cp1 = cross_product(rect[0], rect[1], p);
        float cp2 = cross_product(rect[1], rect[2], p);
        float cp3 = cross_product(rect[2], rect[3], p);
        float cp4 = cross_product(rect[3], rect[0], p);

        // 检查叉积结果的符号是否一致。
        bool has_positive = (cp1 > 0) || (cp2 > 0) || (cp3 > 0) || (cp4 > 0);
        bool has_negative = (cp1 < 0) || (cp2 < 0) || (cp3 < 0) || (cp4 < 0);

        // 如果没有同时出现正数和负数，则点在内部或边界上
        return !(has_positive && has_negative);//返回真，表明内部或边界上
    };

    std::deque<CVArmor> result;
    int num = 0;
    for(auto& armor:armors)
    {
        bool ArmorOK = true;
        for(auto& index:LightIndex)
        {
            if( index==armorLightIndex[num][0] || index==armorLightIndex[num][1]) continue;//如果是自己的灯条跳过
            if( InMind(armor.Lightcorners,lights[index].top) ) {ArmorOK = false;break;}
            if( InMind(armor.Lightcorners,lights[index].center) ) {ArmorOK = false;break;}
            if( InMind(armor.Lightcorners,lights[index].bottom) ) {ArmorOK = false;break;}
        }
        if(ArmorOK) result.emplace_back(armor);
        num++;
    }
    return result;
}



/**
 * @brief      获取装甲板的裁剪后图像
 * @details    输入装甲板std::deque<Armor>，返回裁剪后图像std::vector<cv::Mat>
 * @param      armors 装甲板std::deque<Armor>
 * @return     裁剪后图像std::vector<cv::Mat>
 */
std::vector<cv::Mat> CVDetector::ROIArmor(const std::deque<CVArmor> & armors)
{
    std::vector<cv::Mat> armors_pattern;
    if(armors.empty()) return armors_pattern;
    armors_pattern.reserve(armors.size());
    
    for(const auto& armor:armors)
    {
        const cv::Size roi_sz(112,112); //裁剪后图像大小
        const int extendHei = 28;
        const int contractWid = 18;
        
        std::vector<cv::Point2f> Roi_rect{
            cv::Point2f(-contractWid,extendHei),
            cv::Point2f(roi_sz.width + contractWid - 1, extendHei),
            cv::Point2f(roi_sz.width + contractWid - 1, roi_sz.height - extendHei - 1),
            cv::Point2f(-contractWid, roi_sz.height - extendHei - 1)
        };
        
        // 计算透视变换矩阵
        cv::Mat M = cv::getPerspectiveTransform(armor.Lightcorners, Roi_rect);
        
        // 应用透视变换
        cv::Mat armor_roi;
        cv::warpPerspective(this->rgb_img, armor_roi, M, roi_sz,cv::INTER_LINEAR);
    
        armors_pattern.emplace_back(armor_roi);
    }
    return armors_pattern;
}
std::vector<cv::Mat> CVDetector::SmallROIArmor(const std::deque<CVArmor> & armors)
{
    std::vector<cv::Mat> armors_pattern;
    if(armors.empty()) return armors_pattern;
    armors_pattern.reserve(armors.size());
    
    for(const auto& armor:armors)
    {
        const cv::Size roi_sz(20, 28); //裁剪后图像大小
        const cv::Size armor_sz(32,28);
        const int extendLen = 8;
        const int contractWid = 6;
        
        std::vector<cv::Point2f> aim_rect{
            cv::Point2f(-contractWid,extendLen),
            cv::Point2f(armor_sz.width - contractWid - 1, extendLen),
            cv::Point2f(armor_sz.width - contractWid - 1, armor_sz.height - extendLen - 1),
            cv::Point2f(-contractWid, roi_sz.height - extendLen - 1)
        };
        
        // 计算透视变换矩阵
        cv::Mat M = cv::getPerspectiveTransform(armor.Lightcorners, aim_rect,cv::INTER_NEAREST);
        
        // 应用透视变换
        cv::Mat armor_roi;
        cv::warpPerspective(this->gray_img, armor_roi, M, roi_sz);
        
        cv::threshold(armor_roi, armor_roi, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);//cv::THRESH_OTSU自动计算最优阈值
        
        // cv::imwrite("/home/king/desktop/homework/workindentify/images/roi2.png",armor_roi);
        // cv::waitKey(1);
        armor_roi = armor_roi/255.0;//神经网络输入归一化
        armors_pattern.push_back(armor_roi);
    }
    return armors_pattern;
}

void CVDetector::ArmorShow(cv::Mat & rgb_img, const std::deque<CVArmor> & armors)
{
    for(auto& armor:armors)
    {
        std::vector<cv::Point> Lightcorners;
        Lightcorners.reserve(4);
        for(auto c:armor.Lightcorners) {Lightcorners.push_back(c);}
        std::vector<std::vector<cv::Point>> contours{Lightcorners};

        cv::polylines(rgb_img,contours,1,cv::Scalar(0, 255, 0),3,cv::LINE_AA);
    }
}
void CVDetector::ArmorShow(cv::Mat & rgb_img, const std::vector<CVArmor> & armors)
{
    for(auto& armor:armors)
    {
        std::vector<cv::Point> Lightcorners;
        Lightcorners.reserve(4);
        for(auto c:armor.Lightcorners) {Lightcorners.push_back(c);}
        std::vector<std::vector<cv::Point>> contours{Lightcorners};
        cv::polylines(rgb_img,contours,1,cv::Scalar(0, 255, 0),3,cv::LINE_AA);
        // std::cout<<"id:"<<armor.type<<std::endl;
    }
}               