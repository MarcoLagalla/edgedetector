//
// Created by marco on 05/09/19.
//

#include "ompCanny.h"
#include <omp.h>

#include <vector>
#include <iostream>


#define HIGH_TRESHOLD 140
#define LOW_THRESHOLD 70


ompCanny::ompCanny(cv::Mat inImage, const char *imgName, int size, double sigma) {
    inputImage = inImage;
    inputImageFileName = imgName;
    generateFilter(size, sigma); // create filter

}

/***
 * creates a gaussian filter of the given size with the specified sigma
 * @param size
 * @param sigma
 * @return returns the gaussian filter
 */
void ompCanny::generateFilter(int size, double sigma) {

    omp_set_num_threads(numThreads); // SET NUMBER OF THREADS

    std::vector<std::vector<double>> filter(size, std::vector<double>(size));; // output filter (size*size)

    double r, s = 2.0 * sigma * sigma;
    double sum = 0; // for filter normalization

    // fill the filter
//#pragma omp parallel for collapse(2) reduction(+:sum) shared(size,filter) schedule(dynamic)
    for (int x = -(size/2) ; x <= (size/2); x++) {
        for (int y = -(size/2); y <= (size/2); y++) {
            r = sqrt(x * x + y * y);
            filter[x + (size/2)][y + (size/2)] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += filter[x + (size/2)][y + (size/2)];
        }
    }

    // normalize elements from 0 to 1
//#pragma omp parallel for collapse(2) shared(size,filter,sum) schedule(dynamic)
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            filter[x][y] /= sum;
        }
    }


    gaussianFilter = filter;
}

/*
void ompCanny::computeCannyEdgeDetector() {

    outputImage = applyGaussianFilter();
    outputImage = sobel(outputImage);
    outputImage = nonMaximumSuppression(outputImage);
    outputImage = doubleThreshold(outputImage);

}
 */

cv::Mat ompCanny::applyGaussianFilter(cv::Mat inputImage) {



    cv::Mat outputImage = cv::Mat(inputImage.rows - 2*((int)gaussianFilter.size()/2), inputImage.cols - 2*((int)gaussianFilter.size()/2), CV_8UC1, cv::Scalar(0)); // creates an empty output image


    // convolution is not well defined over borders


    int size = (int)gaussianFilter.size()/2;

    // first step --> Gaussian filtering of the image
    omp_set_num_threads(numThreads); // SET NUMBER OF THREADS
    int chunkSize = inputImage.cols / numThreads;

    #pragma omp parallel for collapse(2) shared(inputImage, size, gaussianFilter) schedule(dynamic, chunkSize)
    for (int i = size; i < inputImage.rows - size; i++)
    {
        for (int j = size; j < inputImage.cols - size; j++)
        {
            double sum = 0;

          //  #pragma omp parallel for collapse(2) shared(gaussianFilter, inputImage, outputImage) reduction(+:sum) schedule(dynamic)
            for (int x = 0; x < gaussianFilter.size(); x++)
                for (int y = 0; y < gaussianFilter.size(); y++)
                {
                    sum += gaussianFilter[x][y] * (double)(inputImage.at<uchar>(i + x - size, j + y - size));
                }

            outputImage.at<uchar>(i-size, j-size) = sum;
        }
    }

    return outputImage;

}

cv::Mat ompCanny::sobel(cv::Mat inputImage) {

    //Sobel X Filter
    double x1[] = {-1.0, 0, 1.0};
    double x2[] = {-2.0, 0, 2.0};
    double x3[] = {-1.0, 0, 1.0};

    std::vector<std::vector<double>> xFilter(3);
    xFilter[0].assign(x1, x1+3);
    xFilter[1].assign(x2, x2+3);
    xFilter[2].assign(x3, x3+3);

    //Sobel Y Filter
    double y1[] = {1.0, 2.0, 1.0};
    double y2[] = {0, 0, 0};
    double y3[] = {-1.0, -2.0, -1.0};

    std::vector<std::vector<double>> yFilter(3);
    yFilter[0].assign(y1, y1+3);
    yFilter[1].assign(y2, y2+3);
    yFilter[2].assign(y3, y3+3);

    int size = (int)xFilter.size()/2;

    cv::Mat outputImage = cv::Mat(inputImage.rows - 2*size, inputImage.cols - 2*size, CV_8UC1);

    anglesMap = cv::Mat(inputImage.rows - 2*size, inputImage.cols - 2*size, CV_32FC1); //AngleMap

    omp_set_num_threads(numThreads); // SET NUMBER OF THREADS

    int chunkSize = inputImage.cols / numThreads;
    #pragma omp parallel for collapse(2) shared(inputImage, outputImage, size) schedule(dynamic,chunkSize)
    for (int i = size; i < inputImage.rows - size; i++)
    {
        for (int j = size; j < inputImage.cols - size; j++)
        {
            double sumx = 0;
            double sumy = 0;

           // #pragma omp parallel for collapse(2) shared(xFilter,yFilter) reduction(+: sumx, sumy)
            for (int x = 0; x < xFilter.size(); x++)
                for (int y = 0; y < xFilter.size(); y++)
                {
                    sumx += xFilter[x][y] * (double)(inputImage.at<uchar>(i + x - size, j + y - size)); //Sobel_X Filter Value
                    sumy += yFilter[x][y] * (double)(inputImage.at<uchar>(i + x - size, j + y - size)); //Sobel_Y Filter Value
                }

            double sumxsq = sumx*sumx;
            double sumysq = sumy*sumy;

            double sq2 = sqrt(sumxsq + sumysq);

            if(sq2 > 255) //Unsigned Char Fix
                sq2 =255;
            outputImage.at<uchar>(i-size, j-size) = sq2;

            if(sumx==0) //Arctan Fix
                anglesMap.at<float>(i-size, j-size) = 90;
            else
                anglesMap.at<float>(i-size, j-size) = atan(sumy/sumx);
        }
    }

    return outputImage;


}



cv::Mat ompCanny::nonMaximumSuppression(cv::Mat inputImage) {

    cv::Mat outputImage = cv::Mat(inputImage.rows-2, inputImage.cols-2, CV_8UC1);
    omp_set_num_threads(numThreads); // SET NUMBER OF THREADS
    #pragma omp parallel for collapse(2) schedule(dynamic) shared(inputImage,outputImage)
    for (int i = 1; i < inputImage.rows -1 ; i++) {
        for (int j = 1; j < inputImage.cols -1 ; j++) {

            float tan = anglesMap.at<float>(i,j); // corresponding tangent value in angles map

            outputImage.at<uchar>(i-1, j-1) = inputImage.at<uchar>(i,j);


            //Horizontal Edge
            if (((-22.5 < tan) && (tan <= 22.5)) || ((157.5 < tan) && (tan <= -157.5)))
            {
                if ((inputImage.at<uchar>(i,j) < inputImage.at<uchar>(i,j+1)) || (inputImage.at<uchar>(i,j) < inputImage.at<uchar>(i,j-1)))
                    outputImage.at<uchar>(i-1, j-1) = 0;
            }
            //Vertical Edge
            if (((-112.5 < tan) && (tan <= -67.5)) || ((67.5 < tan) && (tan <= 112.5)))
            {
                if ((inputImage.at<uchar>(i,j) < inputImage.at<uchar>(i+1,j)) || (inputImage.at<uchar>(i,j) < inputImage.at<uchar>(i-1,j)))
                    outputImage.at<uchar>(i-1, j-1) = 0;
            }

            //-45 Degree Edge
            if (((-67.5 < tan) && (tan <= -22.5)) || ((112.5 < tan) && (tan <= 157.5)))
            {
                if ((inputImage.at<uchar>(i,j) < inputImage.at<uchar>(i-1,j+1)) || (inputImage.at<uchar>(i,j) < inputImage.at<uchar>(i+1,j-1)))
                    outputImage.at<uchar>(i-1, j-1) = 0;
            }

            //45 Degree Edge
            if (((-157.5 < tan) && (tan <= -112.5)) || ((22.5 < tan) && (tan <= 67.5)))
            {
                if ((inputImage.at<uchar>(i,j) < inputImage.at<uchar>(i+1,j+1)) || (inputImage.at<uchar>(i,j) < inputImage.at<uchar>(i-1,j-1)))
                    outputImage.at<uchar>(i-1, j-1) = 0;
            }
        }
    }

    return outputImage;

}


cv::Mat ompCanny::doubleThreshold(cv::Mat inputImage) {

    cv::Mat outputImage = cv::Mat(inputImage.rows, inputImage.cols, CV_8UC1);

    int chunkSize = inputImage.cols / numThreads;
    #pragma omp parallel for collapse(2) shared(inputImage, outputImage)  schedule(dynamic,chunkSize) num_threads(numThreads)
    for (int i = 0; i < inputImage.rows -1; i++) {
        for (int j = 0; j < inputImage.cols -1; j++) {

            int pixelVal = inputImage.at<uchar>(i,j);

            if (pixelVal > HIGH_TRESHOLD) {
                // strong edge
                outputImage.at<uchar>(i,j) = 255;
                continue; // not interesting
            } else if ( pixelVal < HIGH_TRESHOLD && pixelVal > LOW_THRESHOLD) {

                // is connected to a strong edge?
                // check if region is feasible ( 8-bit neighbours)
                // check neighbours
            //    #pragma omp parallel for
                for (int x = i-1 ; x < i+2; x++) {
                    for (int y = j-1; y < j+2; y++) {

                        if (x <= 0 || y <= 0 || x > inputImage.rows || y > inputImage.cols ) {
                            // out of bounds
                            continue;
                        } else {
                            // region is feasible
                            int pVal = inputImage.at<uchar>(x,y);
                            if (pVal >= HIGH_TRESHOLD) {
                                outputImage.at<uchar>(i,j) = 255; // connected to a strong edge
                                break;

                            } else if( pVal < HIGH_TRESHOLD && pVal > LOW_THRESHOLD) {
                                outputImage.at<uchar>(i,j) = 0;
                                break;
                            }
                        }
                    }
                }


            } else if (pixelVal < LOW_THRESHOLD) {
                outputImage.at<uchar>(i,j) = 0; // suppression
            }


        }
    }
    return outputImage;
}



cv::Mat ompCanny::computeCannyEdgeDetector_Horizontal() {

    // STEP 1: GAUSSIAN FILTER

    cv::Mat gaussianFiltered = cv::Mat(inputImage.rows - 2*((int)gaussianFilter.size()/2), inputImage.cols - 2*((int)gaussianFilter.size()/2), CV_8UC1, cv::Scalar(0)); // creates an empty output image


    // convolution is not well defined over borders

    int size = (int)gaussianFilter.size()/2;
    int chunkSize = inputImage.cols / getChunksNum();

    #pragma omp parallel for collapse(2) shared(inputImage,gaussianFiltered, size, gaussianFilter) schedule(dynamic, chunkSize) num_threads(numThreads)
    for (int i = size; i < inputImage.rows - size; i++)
    {
        for (int j = size; j < inputImage.cols - size; j++)
        {
            double sum = 0;

            //  #pragma omp parallel for collapse(2) shared(gaussianFilter, inputImage, outputImage) reduction(+:sum) schedule(dynamic)
            for (int x = 0; x < gaussianFilter.size(); x++)
                for (int y = 0; y < gaussianFilter.size(); y++)
                {
                    sum += gaussianFilter[x][y] * (double)(inputImage.at<uchar>(i + x - size, j + y - size));
                }

            gaussianFiltered.at<uchar>(i-size, j-size) = sum;
        }
    }


    // STEP 2: SOBEL FILTER

    double x1[] = {-1.0, 0, 1.0};
    double x2[] = {-2.0, 0, 2.0};
    double x3[] = {-1.0, 0, 1.0};

    std::vector<std::vector<double>> xFilter(3);
    xFilter[0].assign(x1, x1+3);
    xFilter[1].assign(x2, x2+3);
    xFilter[2].assign(x3, x3+3);

    //Sobel Y Filter
    double y1[] = {1.0, 2.0, 1.0};
    double y2[] = {0, 0, 0};
    double y3[] = {-1.0, -2.0, -1.0};

    std::vector<std::vector<double>> yFilter(3);
    yFilter[0].assign(y1, y1+3);
    yFilter[1].assign(y2, y2+3);
    yFilter[2].assign(y3, y3+3);

    size = (int)xFilter.size()/2;

    cv::Mat sobelFiltered = cv::Mat(gaussianFiltered.rows - 2*size, gaussianFiltered.cols - 2*size, CV_8UC1);

    anglesMap = cv::Mat(gaussianFiltered.rows - 2*size, gaussianFiltered.cols - 2*size, CV_32FC1); //AngleMap

    chunkSize = gaussianFiltered.cols / getChunksNum();

    #pragma omp parallel for collapse(2) shared(gaussianFiltered, sobelFiltered, size,anglesMap) schedule(dynamic,chunkSize) num_threads(numThreads)
    for (int i = size; i < gaussianFiltered.rows - size; i++)
    {
        for (int j = size; j < gaussianFiltered.cols - size; j++)
        {
            double sumx = 0;
            double sumy = 0;

            // #pragma omp parallel for collapse(2) shared(xFilter,yFilter) reduction(+: sumx, sumy)
            for (int x = 0; x < xFilter.size(); x++)
                for (int y = 0; y < xFilter.size(); y++)
                {
                    sumx += xFilter[x][y] * (double)(gaussianFiltered.at<uchar>(i + x - size, j + y - size)); //Sobel_X Filter Value
                    sumy += yFilter[x][y] * (double)(gaussianFiltered.at<uchar>(i + x - size, j + y - size)); //Sobel_Y Filter Value
                }

            double sumxsq = sumx*sumx;
            double sumysq = sumy*sumy;

            double sq2 = sqrt(sumxsq + sumysq);

            if(sq2 > 255) //Unsigned Char Fix
                sq2 =255;

            sobelFiltered.at<uchar>(i-size, j-size) = sq2;

            if(sumx==0) //Arctan Fix
                anglesMap.at<float>(i-size, j-size) = 90;
            else
                anglesMap.at<float>(i-size, j-size) = atan(sumy/sumx);
        }
    }


    // STEP 3: NON-MAXIMUM SUPRRESSION
    cv::Mat nonMaxSuppressed = cv::Mat(sobelFiltered.rows-2, sobelFiltered.cols-2, CV_8UC1);

    chunkSize = sobelFiltered.cols / getChunksNum();
    #pragma omp parallel for collapse(2) schedule(dynamic,chunkSize) shared(sobelFiltered,nonMaxSuppressed,anglesMap) num_threads(numThreads)
    for (int i = 1; i < sobelFiltered.rows -1 ; i++) {
        for (int j = 1; j < sobelFiltered.cols -1 ; j++) {

            float tan = anglesMap.at<float>(i,j); // corresponding tangent value in angles map

            nonMaxSuppressed.at<uchar>(i-1, j-1) = sobelFiltered.at<uchar>(i,j);


            //Horizontal Edge
            if (((-22.5 < tan) && (tan <= 22.5)) || ((157.5 < tan) && (tan <= -157.5)))
            {
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i,j+1)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i,j-1)))
                    nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
            }
            //Vertical Edge
            if (((-112.5 < tan) && (tan <= -67.5)) || ((67.5 < tan) && (tan <= 112.5)))
            {
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i+1,j)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i-1,j)))
                    nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
            }

            //-45 Degree Edge
            if (((-67.5 < tan) && (tan <= -22.5)) || ((112.5 < tan) && (tan <= 157.5)))
            {
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i-1,j+1)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i+1,j-1)))
                    nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
            }

            //45 Degree Edge
            if (((-157.5 < tan) && (tan <= -112.5)) || ((22.5 < tan) && (tan <= 67.5)))
            {
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i+1,j+1)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i-1,j-1)))
                    nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
            }
        }
    }

    // STEP 4: STRONG EDGES CONCATENATION
    cv::Mat outputImage = cv::Mat(nonMaxSuppressed.rows, nonMaxSuppressed.cols, CV_8UC1);

    chunkSize = inputImage.cols / getChunksNum();

    #pragma omp parallel for collapse(2) shared(nonMaxSuppressed, outputImage)  schedule(dynamic,chunkSize) num_threads(numThreads)
    for (int i = 0; i < nonMaxSuppressed.rows -1; i++) {
        for (int j = 0; j < nonMaxSuppressed.cols -1; j++) {

            int pixelVal = nonMaxSuppressed.at<uchar>(i,j);

            if (pixelVal > HIGH_TRESHOLD) {
                // strong edge
                outputImage.at<uchar>(i,j) = 255;
                continue; // not interesting
            } else if ( pixelVal < HIGH_TRESHOLD && pixelVal > LOW_THRESHOLD) {

                // is connected to a strong edge?
                // check if region is feasible ( 8-bit neighbours)
                // check neighbours
                //    #pragma omp parallel for
                for (int x = i-1 ; x < i+2; x++) {
                    for (int y = j-1; y < j+2; y++) {

                        if (x <= 0 || y <= 0 || x > nonMaxSuppressed.rows || y > nonMaxSuppressed.cols ) {
                            // out of bounds
                            continue;
                        } else {
                            // region is feasible
                            int pVal = nonMaxSuppressed.at<uchar>(x,y);
                            if (pVal >= HIGH_TRESHOLD) {
                                outputImage.at<uchar>(i,j) = 255; // connected to a strong edge
                                break;

                            } else if( pVal < HIGH_TRESHOLD && pVal > LOW_THRESHOLD) {
                                outputImage.at<uchar>(i,j) = 0;
                                break;
                            }
                        }
                    }
                }


            } else if (pixelVal < LOW_THRESHOLD) {
                outputImage.at<uchar>(i,j) = 0; // suppression
            }


        }
    }
    return outputImage;
}

cv::Mat ompCanny::computeCannyEdgeDetector_Vertical() {

    // STEP 1: GAUSSIAN FILTER
    cv::Mat gaussianFiltered = cv::Mat(inputImage.rows - 2*((int)gaussianFilter.size()/2), inputImage.cols - 2*((int)gaussianFilter.size()/2), CV_8UC1, cv::Scalar(0)); // creates an empty output image


    // convolution is not well defined over borders

    int size = (int)gaussianFilter.size()/2;
    int chunkSize = inputImage.rows / getChunksNum();

    #pragma omp parallel for collapse(2) shared(inputImage,gaussianFiltered, size, gaussianFilter) schedule(dynamic, chunkSize) num_threads(numThreads)

    for (int j = size; j < inputImage.cols - size; j++)
    {
        for (int i = size; i < inputImage.rows - size; i++)
        {
            double sum = 0;

            //  #pragma omp parallel for collapse(2) shared(gaussianFilter, inputImage, outputImage) reduction(+:sum) schedule(dynamic)
            for (int x = 0; x < gaussianFilter.size(); x++)
                for (int y = 0; y < gaussianFilter.size(); y++)
                {
                    sum += gaussianFilter[x][y] * (double)(inputImage.at<uchar>(i + x - size, j + y - size));
                }

            gaussianFiltered.at<uchar>(i-size, j-size) = sum;
        }
    }


    // STEP 2: SOBEL FILTER

    double x1[] = {-1.0, 0, 1.0};
    double x2[] = {-2.0, 0, 2.0};
    double x3[] = {-1.0, 0, 1.0};

    std::vector<std::vector<double>> xFilter(3);
    xFilter[0].assign(x1, x1+3);
    xFilter[1].assign(x2, x2+3);
    xFilter[2].assign(x3, x3+3);

    //Sobel Y Filter
    double y1[] = {1.0, 2.0, 1.0};
    double y2[] = {0, 0, 0};
    double y3[] = {-1.0, -2.0, -1.0};

    std::vector<std::vector<double>> yFilter(3);
    yFilter[0].assign(y1, y1+3);
    yFilter[1].assign(y2, y2+3);
    yFilter[2].assign(y3, y3+3);

    size = (int)xFilter.size()/2;

    cv::Mat sobelFiltered = cv::Mat(gaussianFiltered.rows - 2*size, gaussianFiltered.cols - 2*size, CV_8UC1);

    anglesMap = cv::Mat(gaussianFiltered.rows - 2*size, gaussianFiltered.cols - 2*size, CV_32FC1); //AngleMap

    chunkSize = gaussianFiltered.rows / getChunksNum();

    #pragma omp parallel for collapse(2) shared(gaussianFiltered, sobelFiltered, size) schedule(dynamic,chunkSize) num_threads(numThreads)
    for (int j = size; j < gaussianFiltered.cols - size; j++)
    {
        for (int i = size; i < gaussianFiltered.rows - size; i++)
        {
            double sumx = 0;
            double sumy = 0;

            // #pragma omp parallel for collapse(2) shared(xFilter,yFilter) reduction(+: sumx, sumy)
            for (int y = 0; y < xFilter.size(); y++)
                for (int x = 0; x < xFilter.size(); x++)
                {
                    sumx += xFilter[x][y] * (double)(gaussianFiltered.at<uchar>(i + x - size, j + y - size)); //Sobel_X Filter Value
                    sumy += yFilter[x][y] * (double)(gaussianFiltered.at<uchar>(i + x - size, j + y - size)); //Sobel_Y Filter Value
                }

            double sumxsq = sumx*sumx;
            double sumysq = sumy*sumy;

            double sq2 = sqrt(sumxsq + sumysq);

            if(sq2 > 255) //Unsigned Char Fix
                sq2 =255;

            sobelFiltered.at<uchar>(i-size, j-size) = sq2;

            if(sumx==0) //Arctan Fix
                anglesMap.at<float>(i-size, j-size) = 90;
            else
                anglesMap.at<float>(i-size, j-size) = atan(sumy/sumx);
        }
    }


    // STEP 3: NON-MAXIMUM SUPRRESSION
    cv::Mat nonMaxSuppressed = cv::Mat(sobelFiltered.rows-2, sobelFiltered.cols-2, CV_8UC1);

    chunkSize = sobelFiltered.rows / getChunksNum();

    #pragma omp parallel for collapse(2) schedule(dynamic,chunkSize) shared(sobelFiltered,nonMaxSuppressed) num_threads(numThreads)
    for (int j = 1; j < sobelFiltered.cols -1 ; j++) {
        for (int i = 1; i < sobelFiltered.rows -1 ; i++) {

            float tan = anglesMap.at<float>(i,j); // corresponding tangent value in angles map

            nonMaxSuppressed.at<uchar>(i-1, j-1) = sobelFiltered.at<uchar>(i,j);


            //Horizontal Edge
            if (((-22.5 < tan) && (tan <= 22.5)) || ((157.5 < tan) && (tan <= -157.5)))
            {
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i,j+1)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i,j-1)))
                    nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
            }
            //Vertical Edge
            if (((-112.5 < tan) && (tan <= -67.5)) || ((67.5 < tan) && (tan <= 112.5)))
            {
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i+1,j)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i-1,j)))
                    nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
            }

            //-45 Degree Edge
            if (((-67.5 < tan) && (tan <= -22.5)) || ((112.5 < tan) && (tan <= 157.5)))
            {
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i-1,j+1)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i+1,j-1)))
                    nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
            }

            //45 Degree Edge
            if (((-157.5 < tan) && (tan <= -112.5)) || ((22.5 < tan) && (tan <= 67.5)))
            {
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i+1,j+1)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i-1,j-1)))
                    nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
            }
        }
    }

    // STEP 4: STRONG EDGES CONCATENATION
    cv::Mat outputImage = cv::Mat(nonMaxSuppressed.rows, nonMaxSuppressed.cols, CV_8UC1);

    chunkSize = inputImage.cols / getChunksNum();

    #pragma omp parallel for collapse(2) shared(nonMaxSuppressed, outputImage)  schedule(dynamic,chunkSize) num_threads(numThreads)
    for (int j = 0; j < nonMaxSuppressed.cols -1; j++){
        for (int i = 0; i < nonMaxSuppressed.rows -1; i++)  {

            int pixelVal = nonMaxSuppressed.at<uchar>(i,j);

            if (pixelVal > HIGH_TRESHOLD) {
                // strong edge
                outputImage.at<uchar>(i,j) = 255;
                continue; // not interesting
            } else if ( pixelVal < HIGH_TRESHOLD && pixelVal > LOW_THRESHOLD) {

                // is connected to a strong edge?
                // check if region is feasible ( 8-bit neighbours)
                // check neighbours
                //    #pragma omp parallel for
                for (int x = i-1 ; x < i+2; x++) {
                    for (int y = j-1; y < j+2; y++) {

                        if (x <= 0 || y <= 0 || x > nonMaxSuppressed.rows || y > nonMaxSuppressed.cols ) {
                            // out of bounds
                            continue;
                        } else {
                            // region is feasible
                            int pVal = nonMaxSuppressed.at<uchar>(x,y);
                            if (pVal >= HIGH_TRESHOLD) {
                                outputImage.at<uchar>(i,j) = 255; // connected to a strong edge
                                break;

                            } else if( pVal < HIGH_TRESHOLD && pVal > LOW_THRESHOLD) {
                                outputImage.at<uchar>(i,j) = 0;
                                break;
                            }
                        }
                    }
                }


            } else if (pixelVal < LOW_THRESHOLD) {
                outputImage.at<uchar>(i,j) = 0; // suppression
            }


        }
    }
    return outputImage;
}

cv::Mat ompCanny::computeCannyEdgeDetector_Blocks(int numOfBlocks) {

    // block division
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
    int w = inputImage.cols; // image width
    int h = inputImage.rows; // image height

    int i1,j1; // outer for --> for each block

    int widthStep = w / numOfCols; // block width
    int heightStep = h / numOfRows; // block height
    int nCol,nRow; // for each column and row creates the corresponding block
    int blockID = 0;


    //Sobel X Filter
    double x1[] = {-1.0, 0, 1.0};
    double x2[] = {-2.0, 0, 2.0};
    double x3[] = {-1.0, 0, 1.0};

    std::vector<std::vector<double>> xFilter(3);
    xFilter[0].assign(x1, x1+3);
    xFilter[1].assign(x2, x2+3);
    xFilter[2].assign(x3, x3+3);

    //Sobel Y Filter
    double y1[] = {1.0, 2.0, 1.0};
    double y2[] = {0, 0, 0};
    double y3[] = {-1.0, -2.0, -1.0};

    std::vector<std::vector<double>> yFilter(3);
    yFilter[0].assign(y1, y1+3);
    yFilter[1].assign(y2, y2+3);
    yFilter[2].assign(y3, y3+3);

    int size = (int)gaussianFilter.size()/2;

    cv::Mat filtered = cv::Mat(inputImage.rows - 2*((int)gaussianFilter.size()/2), inputImage.cols - 2*((int)gaussianFilter.size()/2), CV_8UC1, cv::Scalar(0)); // creates an empty output image
    cv::Mat sobelled = cv::Mat(filtered.rows, filtered.cols, CV_8UC1, cv::Scalar(0));

    anglesMap = cv::Mat(inputImage.rows - 2*((int)xFilter.size()/2), inputImage.cols - 2*((int)xFilter.size()/2), CV_32FC1); //AngleMap


    for (nCol = 0; nCol < numOfCols; nCol++) {
        for (nRow = 0; nRow < numOfRows; nRow++) {

            i1 = (nCol) * widthStep + size;
            j1 = (nRow) * heightStep + size;

            // block is scanned in parallel
            #pragma omp parallel for schedule(static,widthStep) private(i,j) shared(i1,j1,inputImage,filtered,gaussianFilter,widthStep, heightStep,h,w,size) num_threads(numThreads)
            for (j = j1; j < std::min(j1 + heightStep, h - size); j++) {
                for (i = i1; i < std::min(i1 + widthStep, w - size); i++) {

                    double sum = 0;
                    //  #pragma omp parallel for collapse(2) shared(gaussianFilter, inputImage, outputImage) reduction(+:sum) schedule(dynamic)
                    for (int x = 0; x < gaussianFilter.size(); x++)
                        for (int y = 0; y < gaussianFilter.size(); y++)
                        {
                            sum += gaussianFilter[x][y] * (double)(inputImage.at<uchar>(i + x - size, j + y - size));
                        }

                    filtered.at<uchar>(i-size, j-size) = sum;

                }
            }

            #pragma omp barrier
        {
            i1 = (nCol) * widthStep + 1;
            j1 = (nRow) * heightStep + 1;
        };

            // STEP 2 sobel
            #pragma omp parallel for schedule(static,widthStep) private(i,j) shared(i1,j1,inputImage,filtered,sobelled,anglesMap,widthStep, heightStep,h,w,size) num_threads(numThreads)
            for (j = j1; j < std::min(j1 + heightStep, filtered.rows - 1); j++) {
                for (i = i1; i < std::min(i1 + widthStep, filtered.cols - 1); i++) {



                    double sumx = 0;
                    double sumy = 0;

                    // #pragma omp parallel for collapse(2) shared(xFilter,yFilter) reduction(+: sumx, sumy)
                    for (int x = 0; x < xFilter.size(); x++)
                        for (int y = 0; y < xFilter.size(); y++)
                        {
                            sumx += xFilter[x][y] * (double)(filtered.at<uchar>(i + x - 1, j + y - 1)); //Sobel_X Filter Value
                            sumy += yFilter[x][y] * (double)(filtered.at<uchar>(i + x - 1, j + y - 1)); //Sobel_Y Filter Value
                        }

                    double sumxsq = sumx*sumx;
                    double sumysq = sumy*sumy;

                    double sq2 = sqrt(sumxsq + sumysq);

                    if(sq2 > 255) //Unsigned Char Fix
                        sq2 =255;
                    sobelled.at<uchar>(i, j) = sq2;

                    if(sumx==0) //Arctan Fixdd
                        anglesMap.at<float>(i-1, j-1) = 90;
                    else
                        anglesMap.at<float>(i-1, j-1) = atan(sumy/sumx);


                }
            }

        #pragma omp barrier
        {
            i1 = (nCol) * widthStep + 1;
            j1 = (nRow) * heightStep + 1;
        };

            cv::Mat nonMaxSuppressed = cv::Mat(sobelled.rows, sobelled.cols, CV_8UC1);


            #pragma omp parallel for schedule(static,widthStep) shared(sobelled,nonMaxSuppressed) num_threads(numThreads)
            for (int j = 1; j < sobelled.cols -1 ; j++) {
                for (int i = 1; i < sobelled.rows -1 ; i++) {

                    float tan = anglesMap.at<float>(i,j); // corresponding tangent value in angles map

                    nonMaxSuppressed.at<uchar>(i-1, j-1) = sobelled.at<uchar>(i,j);


                    //Horizontal Edge
                    if (((-22.5 < tan) && (tan <= 22.5)) || ((157.5 < tan) && (tan <= -157.5)))
                    {
                        if ((sobelled.at<uchar>(i,j) < sobelled.at<uchar>(i,j+1)) || (sobelled.at<uchar>(i,j) < sobelled.at<uchar>(i,j-1)))
                            nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
                    }
                    //Vertical Edge
                    if (((-112.5 < tan) && (tan <= -67.5)) || ((67.5 < tan) && (tan <= 112.5)))
                    {
                        if ((sobelled.at<uchar>(i,j) < sobelled.at<uchar>(i+1,j)) || (sobelled.at<uchar>(i,j) < sobelled.at<uchar>(i-1,j)))
                            nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
                    }

                    //-45 Degree Edge
                    if (((-67.5 < tan) && (tan <= -22.5)) || ((112.5 < tan) && (tan <= 157.5)))
                    {
                        if ((sobelled.at<uchar>(i,j) < sobelled.at<uchar>(i-1,j+1)) || (sobelled.at<uchar>(i,j) < sobelled.at<uchar>(i+1,j-1)))
                            nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
                    }

                    //45 Degree Edge
                    if (((-157.5 < tan) && (tan <= -112.5)) || ((22.5 < tan) && (tan <= 67.5)))
                    {
                        if ((sobelled.at<uchar>(i,j) < sobelled.at<uchar>(i+1,j+1)) || (sobelled.at<uchar>(i,j) < sobelled.at<uchar>(i-1,j-1)))
                            nonMaxSuppressed.at<uchar>(i-1, j-1) = 0;
                    }
                }
            }


            #pragma omp barrier
            {
                i1 = (nCol) * widthStep + 1;
                j1 = (nRow) * heightStep + 1;
            };

            // step 4: strong edges concatenation
            cv::Mat outputImage = cv::Mat(nonMaxSuppressed.rows, nonMaxSuppressed.cols, CV_8UC1);

            #pragma omp parallel for shared(nonMaxSuppressed, outputImage) schedule(static,widthStep) num_threads(numThreads)
            for (int i = 0; i < nonMaxSuppressed.rows -1; i++) {
                for (int j = 0; j < nonMaxSuppressed.cols -1; j++){

                    int pixelVal = nonMaxSuppressed.at<uchar>(i,j);

                    if (pixelVal > HIGH_TRESHOLD) {
                        // strong edge
                        outputImage.at<uchar>(i,j) = 255;
                        continue; // not interesting
                    } else if ( pixelVal < HIGH_TRESHOLD && pixelVal > LOW_THRESHOLD) {

                        // is connected to a strong edge?
                        // check if region is feasible ( 8-bit neighbours)
                        // check neighbours
                        //    #pragma omp parallel for
                        for (int x = i-1 ; x < i+2; x++) {
                            for (int y = j-1; y < j+2; y++) {

                                if (x <= 0 || y <= 0 || x > nonMaxSuppressed.rows || y > nonMaxSuppressed.cols ) {
                                    // out of bounds
                                    continue;
                                } else {
                                    // region is feasible
                                    int pVal = nonMaxSuppressed.at<uchar>(x,y);
                                    if (pVal >= HIGH_TRESHOLD) {
                                        outputImage.at<uchar>(i,j) = 255; // connected to a strong edge
                                        break;

                                    } else if( pVal < HIGH_TRESHOLD && pVal > LOW_THRESHOLD) {
                                        outputImage.at<uchar>(i,j) = 0;
                                        break;
                                    }
                                }
                            }
                        }


                    } else if (pixelVal < LOW_THRESHOLD) {
                        outputImage.at<uchar>(i,j) = 0; // suppression
                    }


                }
            }


        }
    }





    /*
     *             cv::namedWindow("blocco", cv::WINDOW_NORMAL);
            imshow("blocco", outputImage);
            cv::waitKey(0);
     */
        // sync
        return outputImage;



}

void ompCanny::showOutputImage(char* title) {
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    imshow(title, outputImage);
    cv::waitKey(0);
}

void ompCanny::setThreadsNum(int n) {
    numThreads = n;
}

/***
 *
 * @return chunks number
 */
int ompCanny::getChunksNum() {
    return chunksNum;
}

/***
 * set chunks number
 * @param n
 */
void ompCanny::setChunksNum(int n) {
    chunksNum = n;
}


/***
 * check is a number is a perfect square
 * @param n
 * @return 1 if is a perfect square, 0 otherwise
 */
int ompCanny::isPerfect(long n)
{
    double xp=sqrt((double)n);
    if(n==(xp*xp))
        return 1;
    else
        return 0;
}