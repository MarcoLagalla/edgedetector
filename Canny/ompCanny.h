//
// Created by marco on 05/09/19.
//

#ifndef EDGEDETECTOR_OMPCANNY_H
#define EDGEDETECTOR_OMPCANNY_H


#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



class ompCanny {

public:

    ompCanny(cv::Mat inputImage, const char *imgName, int size, double sigma);
    cv::Mat computeCannyEdgeDetector_Horizontal();
    cv::Mat computeCannyEdgeDetector_Vertical();
    cv::Mat computeCannyEdgeDetector_Blocks(int numOfBlocks);

    void setChunksNum(int n);
    int getChunksNum();
    void setThreadsNum(int n);
    void showOutputImage(char *title);


private:
    cv::Mat inputImage;
    const char* inputImageFileName;
    cv::Mat outputImage;

    int chunksNum = -1;
    int numThreads = 1;

    cv::Mat anglesMap;
    std::vector<std::vector<double>> gaussianFilter;

    void generateFilter(int size, double sigma);
    cv::Mat applyGaussianFilter(cv::Mat inputImage);
    cv::Mat sobel(cv::Mat inputImage);
    cv::Mat nonMaximumSuppression(cv::Mat inputImage);
    cv::Mat doubleThreshold(cv::Mat inputImage);
    int isPerfect(long n);
};


#endif //EDGEDETECTOR_OMPCANNY_H
