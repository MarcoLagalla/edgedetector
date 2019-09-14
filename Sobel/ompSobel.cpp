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
ompSobel::grad ompSobel::gradient(int a, int b,int c, int d,int e, int f, int g, int h, int i) {

    grad gradient;

    int xGradient = 0;
    xGradient = (1 * a) + (2 * b) + (1 * c) - (1 * d) - (2 * e) - (1 * f);
    gradient.gradX = xGradient;

    int yGradient = 0;
    yGradient = (1 * a) + (2 * g) + (1 * d) - (1 * c) - (2 * h) - (1 * f);
    gradient.gradY = yGradient;

    return gradient;

}

/***
 * starts parallel Sobel computation in horizontal
 *
 */

void ompSobel::computeHorizontal() {

    omp_set_num_threads(getThreadsNum()); // SET NUMBER OF THREADS

    // chunksNum is the number of rows in which we have to split inputImage
    // is set with setChunksNum(chunks)
    if (chunksNum == -1) {
        int chunks = 0;
        std::cout << "Please set chunks num: ";
        std::cin >> chunks;
        setChunksNum(chunks);
    }

    int i, j; // for cycle
    cv::Mat I = inputImage; // shorter for convolution :)
    int w = I.cols; // number of columns
    int h = I.rows; // number of rows


    int chunks = getChunksNum(); // number of chunks
    int chunkSize = h / chunks; // how much each chunk is wide

    // output image filled with 0 (black)
    cv::Mat outImage(cv::Size(I.cols, I.rows), CV_8UC1, cv::Scalar(0));

    // compute 3x3 convolution without 1st and last rows and 1st and last column
    // because convolution is not well defined over borders.

    double start = omp_get_wtime(); //initial time
    unsigned char *input = (I.data); // pointer to input image
    int step = I.step;

    #pragma omp parallel for  private(i, j) shared(w, h, outImage, chunks, I) schedule(static, chunkSize)
    for (i = 1; i < h - 1; i++) {
        for (j = 1; j < w - 1; j++) {
            try {

                int a = input[step * (j - 1) + (i - 1)];  // pointer math is faster than cv::Mat.at<>

                int b = input[step * (j) + (i - 1)];
                int c = input[step * (j + 1) + (i - 1)];

                int d = input[step * (j - 1) + (i + 1)];
                int e = input[step * (j) + (i + 1)];
                int f = input[step * (j + 1) + (i + 1)];

                int g = input[step * (j - 1) + (i)];
                int h1 = input[step * (j + 1) + (i)];
                int ik = input[step * (j) + (i)];

                grad _gradient = gradient(a, b, c, d, e, f, g, h1, ik);


                int gradVal = norm2(_gradient.gradX, _gradient.gradY);

                gradVal = gradVal > 255 ? 255 : gradVal;
                gradVal = gradVal < 0 ? 0 : gradVal;
                gradVal = gradVal < 50 ? 0 : gradVal; // threshold

                //outImage.data[step * j + i] = (unsigned char) gradVal;
                outImage.at<uchar>(i, j) = gradVal;

            } catch (const cv::Exception &e) {
                std::cerr << e.what() << " in file: " << getInputImageFileName() << std::endl;
            }
        }
    }

            outImage.row(0) = outImage.row(1); // extends first row
            outImage.row(outImage.rows - 1) = outImage.row(outImage.rows - 2); // extends last row

            outImage.col(0) = outImage.col(1); // extends first column
            outImage.col(outImage.cols - 1) = outImage.col(outImage.cols - 2); // extends last column

            outImage.at<uchar>(0, 0) = outImage.at<uchar>(1, 1); // top-left
            outImage.at<uchar>(0, outImage.cols - 1) = outImage.at<uchar>(1, outImage.cols - 2); // top-right

            outImage.at<uchar>(outImage.rows - 1, 0) = outImage.at<uchar>(outImage.rows - 2, 1); // bottom-left
            outImage.at<uchar>(outImage.rows - 1, outImage.cols - 1) = outImage.at<uchar>(outImage.rows - 2,
                                                                                          outImage.cols - 2); // bottom-right

    // performance computation
    double end = omp_get_wtime(); //final time
    double duration = (end - start);

    executionTime = duration;

    outputImage = outImage; // save outputImage


}

/***
 * Starts parallel Sobel computation in vertical
 */

void ompSobel::computeVertical() {

    omp_set_num_threads(getThreadsNum()); // SET NUMBER OF THREADS

    // chunksNum is the number of columns in which we have to split inputImage
    // is set with setChunksNum(chunks)
    if (chunksNum == -1) {
        int chunks = 0;
        std::cout << "Please set chunks num: ";
        std::cin >> chunks;
        setChunksNum(chunks);
    }

    int i,j; // for
    cv::Mat I = inputImage; // shorter for convolution :)
    int w = I.cols; // image width
    int h = I.rows; // image height


    int chunks = getChunksNum(); // number of chunks
    int chunkSize = w / chunks; // chunks width

    // output image
    cv::Mat outImage(cv::Size(I.cols, I.rows), CV_8UC1, cv::Scalar(0));

    // compute 3x3 convolution without 1st and last rows and 1st and last column
    // because convolution is not well defined over borders.

    double start = omp_get_wtime( ); //initial time

    unsigned char *input = (I.data);
    int step = I.step;

    #pragma omp parallel for private(i,j) shared(w,h,outImage,chunks,I) schedule(static,chunkSize)
    for(j=1; j < w -1; j++) {
        for(i=1; i < h -1; i++) {
            try {

                int a = input[step * (j-1) + (i-1)];

                int b = input[step * (j) + (i-1)];
                int c = input[step * (j+1) + (i-1)];

                int d = input[step * (j-1) + (i+1)];
                int e = input[step * (j) + (i+1)];
                int f = input[step * (j+1) + (i+1)];

                int g = input[step * (j-1) + (i)];
                int h1 = input[step * (j+1) + (i)];
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



    // performance computation
    double end = omp_get_wtime( ); //final time
    double duration = (end - start);

    executionTime = duration;

    outputImage = outImage; // save outputImage

}

/***
 * Starts parallel Sobel computation block by block.
 * @param numOfBlocks is the number of blocks in which image is splitted
 */
void ompSobel::computeBlocks(int numOfBlocks) {

    omp_set_num_threads(getThreadsNum()); // SET NUMBER OF THREADS


    int numOfRows = 1; int numOfCols = 1;

    int nBlocks = numOfBlocks;


    while (!isPerfect((nBlocks))) {
            numOfRows += 1; // add one more column --> vertical splitting
            nBlocks = nBlocks / numOfRows;
    }

    numOfCols = nBlocks;


    if (isPerfect(numOfBlocks)) {
        // desired number of blocks is not a perfect square number
        numOfCols = numOfRows = sqrt(numOfBlocks);

    }

    int i,j; // for inner cycle
    cv::Mat I = inputImage; // shorter for convolution :)
    int w = I.cols; // image width
    int h = I.rows; // image height

    int i1,j1; // outer for --> for each block

    int widthStep = w / numOfCols; // block width
    int heightStep = h / numOfRows; // block height

    // output image
    cv::Mat outImage(cv::Size(w, h), CV_8UC1, cv::Scalar(0));


    int nCol,nRow; // for each column and row creates the corresponding block
    int blockID = 0;

    unsigned char *input = (I.data);
    int step = I.step;

    double start = omp_get_wtime( ); //initial time

   // #pragma omp parallel private(i1,j1,i,j,nCol,nRow) shared(w,h,outImage,I,widthStep,heightStep,numOfCols,numOfRows)
    {
        #pragma omp parallel for schedule(static)  private(j,i,j1,i1) shared(widthStep,heightStep,I,outImage)
        for (nCol = 0; nCol < numOfCols; nCol++) {
            for (nRow = 0; nRow < numOfRows; nRow++) {

                i1 = (nCol) * widthStep + 1;
                j1 = (nRow) * heightStep + 1;
               // cv::Mat tempBlock = cv::Mat(heightStep,widthStep, CV_8UC1, cvScalar(0));
                // block is scanned in parallel
                #pragma omp parallel for schedule(static)//schedule(dynamic,widthStep) private(j,i) shared(j1,i1,widthStep,heightStep,I,outImage)
                for (j = j1; j < std::min(j1 + heightStep, h - 1); j++) {
                    for (i = i1; i < std::min(i1 + widthStep, w - 1); i++) {

                        try {

                            int a = input[step * (j-1) + (i-1)];

                            int b = input[step * (j) + (i-1)];
                            int c = input[step * (j+1) + (i-1)];

                            int d = input[step * (j-1) + (i+1)];
                            int e = input[step * (j) + (i+1)];
                            int f = input[step * (j+1) + (i+1)];

                            int g = input[step * (j-1) + (i)];
                            int h1 = input[step * (j+1) + (i)];
                            int ik = input[step * (j) + (i)];

                            grad _gradient = gradient(a,b,c,d,e,f,g,h1,ik);

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
                blockID++;

                //UNCOMMMENT TO SEE BLOCKS



            }


        }
    }
        // computation performance
        double end = omp_get_wtime( ); //final time
        double duration = (end - start);

        executionTime = duration;

        outputImage = outImage; // save outputImage


}

/***
 *
 * @return inputImage file name
 */
const char* ompSobel::getInputImageFileName(){
    return inputImageFileName;
}

/***
 *
 * @return chunks number
 */
int ompSobel::getChunksNum() {
    return chunksNum;
}

/***
 * set chunks number
 * @param n
 */
void ompSobel::setChunksNum(int n) {
    chunksNum = n;
}

/***
 * get computation time for the given image
 * @return computation time in microseconds
 */
double ompSobel::getComputationTime() {
    return executionTime;
}

/***
 * get the number of threads
 * @return
 */
int ompSobel::getThreadsNum() {
    return numThreads;
}

/***
 * set the number of threads
 * @param n
 */
void ompSobel::setThreadsNum(int n) {
    numThreads = n;
}

/***
 * display the result of edges detection
 * @param title window title
 */
void ompSobel::displayOutputImg(const cv::String title) {
    cv::namedWindow(title, CV_WINDOW_NORMAL);
    imshow(title, outputImage);
    cv::waitKey(0); // waits for user's input
}

/***
 * write outputImage to file
 * @param outputDirectory
 */
void ompSobel::writeToFile(std::string outputDirectory) {
    cv::imwrite(outputDirectory,outputImage);
}

/***
 * check is a number is a perfect square
 * @param n
 * @return 1 if is a perfect square, 0 otherwise
 */
int ompSobel::isPerfect(long n)
{
    double xp=sqrt((double)n);
    if(n==(xp*xp))
        return 1;
    else
        return 0;
}