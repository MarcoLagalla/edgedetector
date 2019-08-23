//
// Created by marco on 22/08/19.
//

#ifndef EDGEDETECTOR_OMPSOBEL_H
#define EDGEDETECTOR_OMPSOBEL_H

#include <omp.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define KERNEL_DIM 3

class ompSobel {


public:

    ompSobel(cv::Mat inImage, const char *imgName);

    void computeHorizontal();
    void computeVertical();
    void computeBlocks(int numOfBlocks);

    void displayOutputImg(const cv::String title);
    const char* getInputImageFileName();
    void setChunksNum(int n);
    void setThreadsNum(int n);
    int getThreadsNum();
    int getChunksNum();
    int isPerfect(long n);
    double getComputationTime();

private:

    cv::Mat inputImage;
    cv::Mat outputImage;
    const char *inputImageFileName;
    int chunksNum = -1;
    double executionTime = 0;
    int numThreads = 1;

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
};


#endif //EDGEDETECTOR_OMPSOBEL_H
