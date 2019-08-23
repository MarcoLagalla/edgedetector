//
// Created by marco on 22/08/19.
//

#include "ompSobel.h"
#include <omp.h>
#include <iostream>
#include <algorithm>

ompSobel::ompSobel(cv::Mat inImage, const char *imgName) {
    inputImage = inImage;
    inputImageFileName = imgName;
    outputImage = cv::Mat(cv::Size(inputImage.cols, inputImage.rows), CV_8UC1, cv::Scalar(0)); // create empty black image with same size
}

ompSobel::grad ompSobel::gradient(int a, int b,int c, int d,int e, int f, int g, int h, int i) {

    grad gradient;

    int xGradient = 0, yGradient = 0;

    xGradient = (1 * a) + (2 * b) + (1 * c) - (1 * d) - (2 * e) - (1 * f);
    yGradient = (1 * a) + (2 * g) + (1 * d) - (1 * c) - (2 * h) - (1 * f);

    gradient.gradX = xGradient;
    gradient.gradY = yGradient;

    return gradient;

}
ompSobel::grad ompSobel::gradient(cv::Mat I, int x, int y) {

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

void ompSobel::computeHorizontal() {

    omp_set_dynamic(0);
    omp_set_num_threads(getThreadsNum()); // SET NUMBER OF THREADS

    if (chunksNum == -1) {
        int chunks = 0;
        std::cout << "Please set chunks num: ";
        std::cin >> chunks;
        setChunksNum(chunks);
    }

    int i,j; // for
    cv::Mat I = inputImage; // shorter for convolution :)
    int w = I.cols;
    int h = I.rows;


    int chunks = getChunksNum();

    cv::Mat outImage(cv::Size(I.cols, I.rows), CV_8UC1, cv::Scalar(0));
    //std::cout << "Width: " << outImage.cols << "height: " << outImage.rows << std::endl;

    // compute 3x3 convolution without 1st and last rows and 1st and last column
    // because convolution is not well defined over borders.

    double start = omp_get_wtime( ); //initial time
#pragma omp parallel for private(i,j)shared(w,h,outImage,chunks,I) schedule(static,chunks)
    for(i=1; i < h -1; i++) {
        for(j=1; j < w -1; j++) {
            try {
                grad _gradient = gradient(I.at<uchar>(i-1,j-1), I.at<uchar>(i,j-1), I.at<uchar>(i+1,j-1),
                                          I.at<uchar>(i-1,j+1), I.at<uchar>(i,j+1), I.at<uchar>(i+1,j+1),
                                          I.at<uchar>(i-1,j), I.at<uchar>(i+1, j),I.at<uchar>(i,j));

                //grad _gradient = gradient(inputImage,i,j);
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
    double end = omp_get_wtime( ); //final time
    double duration = (end - start);

    executionTime = duration;

    outputImage = outImage; // save outputImage

}

void ompSobel::computeVertical() {

    omp_set_dynamic(0);
    omp_set_num_threads(getThreadsNum()); // SET NUMBER OF THREADS

    if (chunksNum == -1) {
        int chunks = 0;
        std::cout << "Please set chunks num: ";
        std::cin >> chunks;
        setChunksNum(chunks);
    }

    int i,j; // for
    cv::Mat I = inputImage; // shorter for convolution :)
    int w = I.cols;
    int h = I.rows;


    int chunks = getChunksNum();

    cv::Mat outImage(cv::Size(I.cols, I.rows), CV_8UC1, cv::Scalar(0));
    //std::cout << "Width: " << outImage.cols << "height: " << outImage.rows << std::endl;

    // compute 3x3 convolution without 1st and last rows and 1st and last column
    // because convolution is not well defined over borders.

    double start = omp_get_wtime( ); //initial time

#pragma omp parallel for private(i,j)shared(w,h,outImage,chunks,I) schedule(static,chunks)
    for(j=1; j < w -1; j++) {
        for(i=1; i < h -1; i++) {
            try {
                grad _gradient = gradient(I.at<uchar>(i-1,j-1), I.at<uchar>(i,j-1), I.at<uchar>(i+1,j-1),
                                          I.at<uchar>(i-1,j+1), I.at<uchar>(i,j+1), I.at<uchar>(i+1,j+1),
                                          I.at<uchar>(i-1,j), I.at<uchar>(i+1, j),I.at<uchar>(i,j));

                //grad _gradient = gradient(inputImage,i,j);
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
    double end = omp_get_wtime( ); //final time
    double duration = (end - start);

    executionTime = duration;

    outputImage = outImage; // save outputImage

}

void ompSobel::computeBlocks(int numOfBlocks) {

    omp_set_dynamic(0);
    omp_set_num_threads(getThreadsNum()); // SET NUMBER OF THREADS


    int numOfRows = 1; int numOfCols = 1;

    int nBlocks = numOfBlocks;


    while (!isPerfect((nBlocks))) {
            numOfRows += 1; // add one column --> vertical splitting
            nBlocks = nBlocks / numOfRows;
    }

    numOfCols = nBlocks;


    if (isPerfect(numOfBlocks)) {
        // desired number of blocks is not a perfect square number
        numOfCols = numOfRows = sqrt(numOfBlocks);

    }

    int n;
    int i,j;
    cv::Mat I = inputImage; // shorter for convolution :)
    int w = I.cols;
    int h = I.rows;
    int i1,j1;

    int widthStep = w / numOfCols;
    int heightStep = h / numOfRows;

    cv::Mat outImage(cv::Size(w, h), CV_8UC1, cv::Scalar(0));

    int nCol,nRow;
    int idBlocco = 0;

    double start = omp_get_wtime( ); //initial time

    #pragma omp parallel private(i1,j1,i,j,nCol,nRow) shared(w,h,outImage,I,widthStep,heightStep,numOfCols,numOfRows)
    {
        #pragma omp for schedule(static)

        for (nCol = 0; nCol < numOfCols; nCol++) {

            for (nRow = 0; nRow < numOfRows; nRow++) {

                i1 = (nCol) * widthStep + 1;
                j1 = (nRow) * heightStep + 1;

                for (j = j1; j < std::min(j1 + heightStep, h - 1); j++) {
                    for (i = i1; i < std::min(i1 + widthStep, w - 1); i++) {

                        try {
                            grad _gradient = gradient(I.at<uchar>(i - 1, j - 1), I.at<uchar>(i, j - 1),
                                                      I.at<uchar>(i + 1, j - 1),
                                                      I.at<uchar>(i - 1, j + 1), I.at<uchar>(i, j + 1),
                                                      I.at<uchar>(i + 1, j + 1),
                                                      I.at<uchar>(i - 1, j), I.at<uchar>(i + 1, j), I.at<uchar>(i, j));

                            //grad _gradient = gradient(inputImage,i,j);
                            int gradVal = norm2(_gradient.gradX, _gradient.gradY);

                            gradVal = gradVal > 255 ? 255 : gradVal;
                            gradVal = gradVal < 0 ? 0 : gradVal;
                            gradVal = gradVal < 50 ? 0 : gradVal; // threshold

                            outImage.at<uchar>(i, j) = gradVal;

                            //
                        } catch (const cv::Exception &e) {
                            std::cerr << e.what() << " in file: " << getInputImageFileName() << std::endl;
                        }
                    }
                }
                idBlocco++;
                /* UNCOMMMENT TO SEE BLOCKS
                std::string title = "Blocco " + std::to_string(idBlocco) + " di " + std::to_string(numOfBlocks);
                cv::namedWindow(title.c_str(), CV_WINDOW_NORMAL);
                imshow(title.c_str(), outImage);
                cv::waitKey(0);
                 */
            }


        }
    }


    // computation performance
    double end = omp_get_wtime( ); //final time
    double duration = (end - start);

    executionTime = duration;

    outputImage = outImage; // save outputImage
}

const char* ompSobel::getInputImageFileName(){
    return inputImageFileName;
}

int ompSobel::getChunksNum() {
    return chunksNum;
}

void ompSobel::setChunksNum(int n) {
    chunksNum = n;
}

double ompSobel::getComputationTime() {
    return executionTime;
}

int ompSobel::getThreadsNum() {
    return numThreads;
}

void ompSobel::setThreadsNum(int n) {
    numThreads = n;
}
void ompSobel::displayOutputImg(const cv::String title) {
    cv::namedWindow(title, CV_WINDOW_NORMAL);
    imshow(title, outputImage);
    cv::waitKey(0);
}

int ompSobel::isPerfect(long n)
{
    double xp=sqrt((double)n);
    if(n==(xp*xp))
        return 1;
    else
        return 0;
}