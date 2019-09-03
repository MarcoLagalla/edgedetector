//
// Created by marco on 21/08/19.
//

#include <iostream>
#include <chrono>
#include "sobel.h"

Sobel::Sobel(cv::Mat inImage, const char *imgName) {
    inputImage = inImage;
    inputImageFileName = imgName;
    outputImage = processMatrix(inputImage);
};

/***
 * Compute the gradient of the image using pixel values. Both axis
 * @param a
 * @param b
 * @param c
 * @param d
 * @param e
 * @param f
 * @param g
 * @param h
 * @param i
 * @return both x and y gradient
 */
Sobel::grad Sobel::gradient(int a, int b,int c, int d,int e, int f, int g, int h, int i) {

    grad gradient;

    int xGradient = 0, yGradient = 0;

    xGradient = (1 * a) + (2 * b) + (1 * c) - (1 * d) - (2 * e) - (1 * f);
    yGradient = (1 * a) + (2 * g) + (1 * d) - (1 * c) - (2 * h) - (1 * f);

    gradient.gradX = xGradient;
    gradient.gradY = yGradient;

    return gradient;

}
/***
 * [DEPRECATED] Compute the gradient of the image using pixel values. Both axis
 * @param a
 * @param b
 * @param c
 * @param d
 * @param e
 * @param f
 * @param g
 * @param h
 * @param i
 * @return both x and y gradient
 */
Sobel::grad Sobel::gradient(cv::Mat I, int x, int y) {

    grad gradient;

    int Xgradient = 0, Ygradient = 0;

    Xgradient = xKernel[0][0] * I.at<uchar>(x-1,y-1)  +
                xKernel[0][1] * I.at<uchar>(x-1, y)   +
                xKernel[0][2] * I.at<uchar>(x-1, y+1) +
                xKernel[1][0] * I.at<uchar>(x,y-1)    +
                xKernel[1][1] * I.at<uchar>(x,y)      +
                xKernel[1][2] * I.at<uchar>(x, y+1)   +
                xKernel[2][0] * I.at<uchar>(x+1,y-1)    +
                xKernel[2][1] * I.at<uchar>(x+1,y)      +
                xKernel[2][2] * I.at<uchar>(x+1, y+1);

    gradient.gradX = Xgradient;

    Ygradient = yKernel[0][0] * I.at<uchar>(x-1,y-1)  +
                yKernel[0][1] * I.at<uchar>(x-1, y)   +
                yKernel[0][2] * I.at<uchar>(x-1, y+1) +
                yKernel[1][0] * I.at<uchar>(x,y-1)    +
                yKernel[1][1] * I.at<uchar>(x,y)      +
                yKernel[1][2] * I.at<uchar>(x, y+1)   +
                yKernel[2][0] * I.at<uchar>(x+1,y-1)    +
                yKernel[2][1] * I.at<uchar>(x+1,y)      +
                yKernel[2][2] * I.at<uchar>(x+1, y+1);


    gradient.gradY = Ygradient;

    return gradient;

}

/***
 * Starts serial Sobel computation
 * @param inputImage
 * @return
 */
cv::Mat Sobel::processMatrix(cv::Mat inputImage) {


    cv::Mat I = inputImage; // shorter for convolution :)

    auto startTime = std::chrono::high_resolution_clock::now();
    cv::Mat outImage(cv::Size(inputImage.cols, inputImage.rows), CV_8UC1, cv::Scalar(0));


    unsigned char *input = (I.data);
    int step = I.step;

    for(int i=1; i < inputImage.rows -1; i++) {
        for(int j=1; j < inputImage.cols -1; j++) {
            try {

                int a = input[step * (j - 1) + (i - 1)];

                int b = input[step * (j) + (i - 1)];
                int c = input[step * (j + 1) + (i - 1)];

                int d = input[step * (j - 1) + (i + 1)];
                int e = input[step * (j) + (i + 1)];
                int f = input[step * (j + 1) + (i + 1)];

                int g = input[step * (j - 1) + (i)];
                int h1 = input[step * (j + 1) + (i)];
                int ik = input[step * (j) + (i)];

                grad _gradient = gradient(a,b,c,d,e,f,g,h1,ik);

                int gradVal = norm2( _gradient.gradX, _gradient.gradY );

                gradVal = gradVal > 255 ? 255:gradVal;
                gradVal = gradVal < 0 ? 0 : gradVal;
                gradVal = gradVal < 50 ? 0 : gradVal; // threshold

                outImage.at<uchar >(i,j) = gradVal;
                //
            } catch (const cv::Exception& e) {
                std::cerr << e.what() << " in file: " << getInputImageFileName() << std::endl;
            }
        }
    }

    // replicate borders
    outImage.row(0) = outImage.row(1); // extends first row
    outImage.row(outImage.rows - 1) = outImage.row(outImage.rows - 2); // extends last row
    outImage.col(0) = outImage.col(1); // extends first column
    outImage.col(outImage.cols -1 ) = outImage.col(outImage.cols - 2); // extends last column

    // replicate corners
    outImage.at<uchar>(0,0) = outImage.at<uchar>(1,1); // top-left
    outImage.at<uchar>(0,outImage.cols -1) = outImage.at<uchar>(1,outImage.cols -2); // top-right

    outImage.at<uchar>(outImage.rows -1 ,0) = outImage.at<uchar>(outImage.rows -2,1); // bottom-left
    outImage.at<uchar>(outImage.rows -1,outImage.cols -1) = outImage.at<uchar>(outImage.rows -2 , outImage.cols - 2); // bottom-right



    // computation performance
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    executionTime = duration.count();

    return outImage;
}

/***
 * display the results of edges detection
 * @param title
 */
void Sobel::displayOutputImg(const cv::String title) {
    cv::namedWindow(title, CV_WINDOW_NORMAL);
    imshow(title, outputImage);
    cv::waitKey(0);
}

/***
 * return the computation time for the given image
 * @return
 */
long Sobel::getComputationTime() {
    return executionTime;
}

/***
 *
 * @return input image file name
 */
const char* Sobel::getInputImageFileName(){
    return inputImageFileName;
}

/***
 * write image to file
 * @param outputDirectory
 */
void Sobel::writeToFile(std::string outputDirectory) {
    cv::imwrite(outputDirectory,outputImage);
}