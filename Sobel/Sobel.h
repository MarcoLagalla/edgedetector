//
// Created by marco on 21/08/19.
//

#ifndef EDGEDETECTOR_SOBEL_H
#define EDGEDETECTOR_SOBEL_H

#include <math.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define KERNEL_DIM 3


class Sobel {

public:

    Sobel(cv::Mat inImage, const char *imgName);
    void displayOutputImg(const cv::String title);
    long getComputationTime();
    const char* getInputImageFileName();
    void writeToFile(std::string outputDirectory);

private:

    cv::Mat inputImage;
    cv::Mat outputImage;
    const char *inputImageFileName;
    long executionTime = 0;

    int norm2(int x, int y) {
        return sqrt(x*x + y*y);
    }

    const int xKernel[KERNEL_DIM][KERNEL_DIM] = {
            {1, 0, -1},
            {2, 0, -2},
            {1, 0, -1}
    };

    const int yKernel[KERNEL_DIM][KERNEL_DIM] = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
    };

    typedef struct grad {
        int gradX;
        int gradY;
    };

    grad gradient(cv::Mat I, int x, int y); // deprecated
    grad gradient(int a, int b,int c, int d,int e, int f, int g, int h, int i);
    cv::Mat processMatrix(cv::Mat inputImage);
};



#endif //EDGEDETECTOR_SOBEL_H
