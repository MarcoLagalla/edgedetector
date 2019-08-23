#include <iostream>
#include <dirent.h>
#include <ostream>
#include <chrono>
#include "sobel.h"
#include "ompSobel.h"
#include <omp.h>

#define REPETITIONS 1
#define MAX_FILES 300

#define CHUNKS 8
#define THREADS 16
#define BLOCKS 16

void readFolder(const char *inputImgFolder);
double serialExecution(int times,const char *inputImgFolder);
double parallelHorizontalExecution(int times,const char *inputImgFolder);
double parallelVerticalExecution(int times,const char *inputImgFolder);
double parallelBlocksExecution(int times, const char *inputImgFolder);



int main() {


    std::cout << "Welcome to Sobel edges detector :)" << std::endl;

    std::cout << "Today we will work with " << MAX_FILES << " images for class!" << std::endl;

    char *inputImgFolder = "/home/marco/Scrivania/dataset/"; // without the last slash will fail!

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (inputImgFolder)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            if (ent->d_type == 4) {
                if ( !strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..") ) {
                   // do nothing
                } else {
                    std::cout << "Processing folder: " << ent->d_name << std::endl;
                    std::string buf(inputImgFolder);
                    buf.append(ent->d_name);
                    buf.append("/"); // hard coded ? :)

                    readFolder(buf.c_str());

                }

            }
        }


    }
  //  readFolder(inputImgFolder);

    std::cout << "Done, bye :)";

    return 0;
}

void readFolder(const char *inputImgFolder) {
    std::cout << "\n\n[SERIAL]" << std::endl;
    double serial = serialExecution(REPETITIONS,inputImgFolder);
    std::cout << "Computation time: " << serial / 1000 << "[msec]" << std::endl;

    std::cout << "[PARALLEL - HORIZONTAl]" << std::endl;
    double horizontal = parallelHorizontalExecution(REPETITIONS,inputImgFolder);
    std::cout << "Computation time: " << horizontal / 1000 << "[msec]" << std::endl;

    std::cout << "[PARALLEL - VERTICAL]" << std::endl;
    double vertical = parallelVerticalExecution(REPETITIONS,inputImgFolder);
    std::cout << "Computation time: " << vertical / 1000 << "[msec]" << std::endl;

    std::cout << "[PARALLEL - BLOCKS] --> " << BLOCKS << " BLOCKS" << std::endl;
    double blocks = parallelBlocksExecution(REPETITIONS,inputImgFolder);
    std::cout << "Computation time: " << blocks / 1000 << "[msec]" << std::endl;

    // speed up
    double hSpeedUP = serial / horizontal;
    double vSpeedUP = serial / vertical;
    double bSpeedUP = serial / blocks;
    std::cout << "\n\n\t\t\t" << "H\t\tV\t\tB" << std::endl;
    std::cout << "SpeedUp:\t" <<  hSpeedUP << "\t" << vSpeedUP << "\t" << bSpeedUP << std::endl;
}

double serialExecution(int times,const char *inputImgFolder) {

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

                if (i >= MAX_FILES) {
                    break; // exit after MAX_FILES
                }


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

double parallelHorizontalExecution(int times,const char *inputImgFolder) {

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

                if (i >= MAX_FILES) {
                    break; // exit after MAX_FILES
                }

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

double parallelVerticalExecution(int times, const char *inputImgFolder) {

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

                if (i >= MAX_FILES) {
                    break; // exit after MAX_FILES
                }

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

double parallelBlocksExecution(int times, const char *inputImgFolder) {

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

                if (i >= MAX_FILES) {
                    break; // exit after MAX_FILES
                }

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