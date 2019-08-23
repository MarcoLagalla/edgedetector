#include <iostream>
#include <dirent.h>
#include <ostream>
#include <chrono>
#include "sobel.h"
#include "ompSobel.h"
#include <omp.h>

#define REPETITIONS 1
#define CHUNKS 16
#define THREADS 16
#define BLOCKS 16

double serialExecution(int times);
double parallelHorizontalExecution(int times);
double parallelVerticalExecution(int times);
double parallelBlocksExecution(int times);

char *inputImgFolder = "/home/marco/Scrivania/test/"; // without the last slash will fail!


int main() {


    std::cout << "Welcome to Sobel edges detector :)" << std::endl;

    std::cout << "[SERIAL]" << std::endl;
    double serial = serialExecution(REPETITIONS);
    std::cout << "Computation time: " << serial / 1000 << "[msec]" << std::endl;

    std::cout << "[PARALLEL - HORIZONTAl]" << std::endl;
    double horizontal = parallelHorizontalExecution(REPETITIONS);
    std::cout << "Computation time: " << horizontal / 1000 << "[msec]" << std::endl;

    std::cout << "[PARALLEL - VERTICAL]" << std::endl;
    double vertical = parallelVerticalExecution(REPETITIONS);
    std::cout << "Computation time: " << vertical / 1000 << "[msec]" << std::endl;

    std::cout << "[PARALLEL - BLOCKS] --> " << BLOCKS << " BLOCKS" << std::endl;
    double blocks = parallelBlocksExecution(REPETITIONS);
    std::cout << "Computation time: " << blocks / 1000 << "[msec]" << std::endl;

    std::cout << "Done, bye :)";

    return 0;
}



double serialExecution(int times) {

    double executionTime = 0;

    for (int k = 0; k < times; k++) {

        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (inputImgFolder)) != NULL) {

            //std::cout << "Working directory: " << inputImgFolder << std::endl;
            int i = 0;

            /* print all the files and directories within directory */

            auto startSerialExecutionTime = std::chrono::high_resolution_clock::now();
            while ((ent = readdir (dir)) != NULL) {
                if (ent->d_type == 8) { // 8 stands for image
                    std::string buf(inputImgFolder);
                    buf.append(ent->d_name);

                    // buf.c_str() contains the current file in the directory
                    try {

                        cv::Mat inputImage = imread(buf.c_str() , cv::IMREAD_GRAYSCALE);
                        Sobel mySobel(inputImage,buf.c_str());

                        //mySobel.displayOutputImg("Result of EdgeDetection");
                        //std::cout << "Input image: " << buf.c_str() << "\t-\tComputation time: " << mySobel.getComputationTime() << "[us]" << std::endl;
                        i++;

                    } catch (const cv::Exception& e) {
                        std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                    }

                }

            }

            auto endSerialExecutionTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endSerialExecutionTime - startSerialExecutionTime);
            closedir (dir);

            //std::cout << "Total processed files: " << i << "\nTotal execution time: " << duration.count() << "[us] -> " << duration.count() / 1000 << "[ms] -> " << duration.count() / 1000000 << "[s]" <<  std::endl;

            executionTime += duration.count();
        } else {
            /* could not open directory */
            perror ("");
        }
    }

    return executionTime / times; // average time over #times repetitions
}

double parallelHorizontalExecution(int times) {

    double executionTime = 0;

    for (int k = 0; k < times; k++) {

        DIR *dir;
        struct dirent *ent;

        if ((dir = opendir(inputImgFolder)) != NULL) {

            //std::cout << "Working directory: " << inputImgFolder << std::endl;
            int i = 0;
            /* print all the files and directories within directory */
            auto startParallelExecutionTime_Horizontal = std::chrono::high_resolution_clock::now();
            while ((ent = readdir(dir)) != NULL) {
                if (ent->d_type == 8) { // 8 stands for image
                    std::string buf(inputImgFolder);
                    buf.append(ent->d_name);

                    // buf.c_str() contains the current file in the directory
                    try {

                        cv::Mat inputImage = imread(buf.c_str(), cv::IMREAD_GRAYSCALE);
                        ompSobel mySobel(inputImage, buf.c_str());
                        mySobel.setThreadsNum(THREADS);
                        mySobel.setChunksNum(CHUNKS);
                        mySobel.computeHorizontal();

                        //mySobel.displayOutputImg("Result of EdgeDetection");
                        //std::cout << "Input image: " << buf.c_str() << "\t-\tComputation time: " << mySobel.getComputationTime() << "[us]" << std::endl;
                        i++;

                    } catch (const cv::Exception &e) {
                        std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                    }

                }

            }
            auto endParallelExecutionTime_Horizontal = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    endParallelExecutionTime_Horizontal - startParallelExecutionTime_Horizontal);
            closedir(dir);

            //std::cout << "Total processed files: " << i << "\nTotal execution time: " << duration.count() << "[us] -> " << duration.count() / 1000 << "[ms] -> " << duration.count() / 1000000 << "[s]" << std::endl;

            executionTime += duration.count();
        } else {
            /* could not open directory */
            perror("");
        }
    }

    return executionTime / times; // average time
}

double parallelVerticalExecution(int times) {

    double executionTime = 0;

    for (int k = 0; k < times; k++) {

        DIR *dir;
        struct dirent *ent;

        if ((dir = opendir(inputImgFolder)) != NULL) {

            //std::cout << "Working directory: " << inputImgFolder << std::endl;
            int i = 0;
            /* print all the files and directories within directory */
            auto startParallelExecutionTime_Horizontal = std::chrono::high_resolution_clock::now();
            while ((ent = readdir(dir)) != NULL) {
                if (ent->d_type == 8) { // 8 stands for image
                    std::string buf(inputImgFolder);
                    buf.append(ent->d_name);

                    // buf.c_str() contains the current file in the directory
                    try {

                        cv::Mat inputImage = imread(buf.c_str(), cv::IMREAD_GRAYSCALE);
                        ompSobel mySobel(inputImage, buf.c_str());
                        mySobel.setThreadsNum(THREADS);
                        mySobel.setChunksNum(CHUNKS);
                        mySobel.computeVertical();

                        //mySobel.displayOutputImg("Result of EdgeDetection");
                        //std::cout << "Input image: " << buf.c_str() << "\t-\tComputation time: " << mySobel.getComputationTime() << "[us]" << std::endl;
                        i++;

                    } catch (const cv::Exception &e) {
                        std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                    }

                }

            }
            auto endParallelExecutionTime_Horizontal = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    endParallelExecutionTime_Horizontal - startParallelExecutionTime_Horizontal);
            closedir(dir);

            // std::cout << "Total processed files: " << i << "\nTotal execution time: " << duration.count() << "[us] -> " << duration.count() / 1000 << "[ms] -> " << duration.count() / 1000000 << "[s]" << std::endl;

            executionTime += duration.count();
        } else {
            /* could not open directory */
            perror("");
        }
    }
    return executionTime / times; // average time
}

double parallelBlocksExecution(int times) {

    double executionTime = 0;

    for (int k = 0; k < times; k++) {

        DIR *dir;
        struct dirent *ent;

        if ((dir = opendir(inputImgFolder)) != NULL) {

            //std::cout << "Working directory: " << inputImgFolder << std::endl;
            int i = 0;
            /* print all the files and directories within directory */
            auto startParallelExecutionTime_Horizontal = std::chrono::high_resolution_clock::now();
            while ((ent = readdir(dir)) != NULL) {
                if (ent->d_type == 8) { // 8 stands for image
                    std::string buf(inputImgFolder);
                    buf.append(ent->d_name);

                    // buf.c_str() contains the current file in the directory
                    try {

                        cv::Mat inputImage = imread(buf.c_str(), cv::IMREAD_GRAYSCALE);
                        ompSobel mySobel(inputImage, buf.c_str());
                        mySobel.setThreadsNum(THREADS);
                        mySobel.setChunksNum(CHUNKS);
                        mySobel.computeBlocks(BLOCKS);

                        //mySobel.displayOutputImg("Result of EdgeDetection");
                        //std::cout << "Input image: " << buf.c_str() << "\t-\tComputation time: " << mySobel.getComputationTime() << "[us]" << std::endl;
                        i++;

                    } catch (const cv::Exception &e) {
                        std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                    }

                }

            }
            auto endParallelExecutionTime_Horizontal = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    endParallelExecutionTime_Horizontal - startParallelExecutionTime_Horizontal);
            closedir(dir);

            // std::cout << "Total processed files: " << i << "\nTotal execution time: " << duration.count() << "[us] -> " << duration.count() / 1000 << "[ms] -> " << duration.count() / 1000000 << "[s]" << std::endl;

            executionTime += duration.count();
        } else {
            /* could not open directory */
            perror("");
        }
    }
    return executionTime / times; // average time
}