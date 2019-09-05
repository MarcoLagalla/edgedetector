//
// Created by marco on 05/09/19.
//

#include "canny.h"
#include <vector>


#define HIGH_TRESHOLD 140
#define LOW_THRESHOLD 40


canny::canny(cv::Mat inImage, const char *imgName, int size, double sigma) {
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
void canny::generateFilter(int size, double sigma) {

    std::vector<std::vector<double>> filter(size, std::vector<double>(size));; // output filter (size*size)

    double r, s = 2.0 * sigma * sigma;
    double sum = 0; // for filter normalization

    // fill the filter
    for (int x = -(size/2) ; x <= (size/2); x++) {
        for (int y = -(size/2); y <= (size/2); y++) {
            r = sqrt(x * x + y * y);
            filter[x + (size/2)][y + (size/2)] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += filter[x + (size/2)][y + (size/2)];
        }
    }

    // normalize elements from 0 to 1
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            filter[x][y] /= sum;
        }
    }


    gaussianFilter = filter;
}

void canny::computeCannyEdgeDetector() {

    outputImage = applyGaussianFilter();
    outputImage = sobel(outputImage);
    outputImage = nonMaximumSupression(outputImage);
    outputImage = doubleThreshold(outputImage);

}
cv::Mat canny::applyGaussianFilter() {


    cv::Mat outputImage = cv::Mat(inputImage.rows - 2*((int)gaussianFilter.size()/2), inputImage.cols - 2*((int)gaussianFilter.size()/2), CV_8UC1, cv::Scalar(0)); // creates an empty output image


    // convolution is not well defined over borders


    int size = (int)gaussianFilter.size()/2;

    // first step --> Gaussian filtering of the image
    for (int i = size; i < inputImage.rows - size; i++)
    {
        for (int j = size; j < inputImage.cols - size; j++)
        {
            double sum = 0;

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

cv::Mat canny::sobel(cv::Mat inputImage) {

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

    for (int i = size; i < inputImage.rows - size; i++)
    {
        for (int j = size; j < inputImage.cols - size; j++)
        {
            double sumx = 0;
            double sumy = 0;

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



cv::Mat canny::nonMaximumSupression(cv::Mat inputImage) {

    cv::Mat outputImage = cv::Mat(inputImage.rows-2, inputImage.cols-2, CV_8UC1);

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


cv::Mat canny::doubleThreshold(cv::Mat inputImage) {

    cv::Mat outputImage = cv::Mat(inputImage.rows, inputImage.cols, CV_8UC1);

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


void canny::showOutputImage(char* title) {
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    imshow(title, outputImage);
    cv::waitKey(0);
}