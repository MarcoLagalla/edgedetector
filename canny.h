//
// Created by marco on 05/09/19.
//

#ifndef EDGEDETECTOR_CANNY_H
#define EDGEDETECTOR_CANNY_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



class canny {

public:

    canny(cv::Mat inputImage, const char *imgName, int size, double sigma);
    void computeCannyEdgeDetector();
    void showOutputImage(char *title);

private:
    cv::Mat inputImage;
    const char* inputImageFileName;
    cv::Mat outputImage;

    cv::Mat anglesMap;
    std::vector<std::vector<double>> gaussianFilter;

    void generateFilter(int size, double sigma);
    cv::Mat applyGaussianFilter();
    cv::Mat sobel(cv::Mat inputImage);
    cv::Mat nonMaximumSupression(cv::Mat inputImage);
    cv::Mat doubleThreshold(cv::Mat inputImage);

};


#endif //EDGEDETECTOR_CANNY_H
