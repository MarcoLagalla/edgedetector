#include <iostream>
#include <dirent.h>
#include <ostream>
#include <chrono>
#include <fstream>
#include <omp.h>

#include "sobel.h"
#include "ompSobel.h"


#define REPETITIONS 10
#define MAX_FILES 10

#define CHUNKS 8
#define THREADS 8

void readFolder(const char *inputImgFolder);
double serialExecution(int times,const char *inputImgFolder);
double parallelHorizontalExecution(int times,const char *inputImgFolder);
double parallelVerticalExecution(int times,const char *inputImgFolder);
double parallelBlocksExecution(int times, int nBlocks, const char *inputImgFolder);



int main() {


    std::cout << "Welcome to Sobel edges detector :)" << std::endl;

    std::cout << "Today we will work with " << MAX_FILES << " images for class!" << std::endl;
    std::cout << "Threads number is set to: " << THREADS << std::endl;
    std::cout << "Chunks number for horizontal and vertical processing is set to: " << CHUNKS << std::endl;

    char *inputImgFolder = "./dataset/"; // without the last slash will fail!

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (inputImgFolder)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            if (ent->d_type == 4) {
                if ( !strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..") ) {
                   // do nothing
                } else {
                    std::cout << "Processing folder: " << ent->d_name << "\n\n";
                    std::string buf(inputImgFolder);
                    buf.append(ent->d_name);
                    buf.append("/"); // hard coded ? :)

                    readFolder(buf.c_str());

                }

            }
        }


    }

    std::cout << "Done, bye :)";

    return 0;
}

/***
 * scan all image files into the given folder and make computation for serial, horizontal and vertical
 * @param inputImgFolder
 */
void readFolder(const char *inputImgFolder) {

    std::cout << "\t[SERIAL] ";
    double serial = serialExecution(REPETITIONS,inputImgFolder);
    std::cout << "Computation time: " << serial / 1000 << "[msec]" << std::endl;

    std::cout << "\t[PARALLEL - VERTICAL] ";
    double vertical = parallelVerticalExecution(REPETITIONS,inputImgFolder);
    std::cout << "Computation time: " << vertical / 1000 << "[msec]" << std::endl;

    std::cout << "\t[PARALLEL - HORIZONTAl] ";
    double horizontal = parallelHorizontalExecution(REPETITIONS,inputImgFolder);
    std::cout << "Computation time: " << horizontal / 1000 << "[msec]" << std::endl;


    // try different block numbers --> 8 - 16 - 32 - 64
    for (int i = 3; i < 7; i++) {
        int nBlocks = pow(2,i);
        std::cout << "\t[PARALLEL - "<< nBlocks << " BLOCKS] ";
        double blocks = parallelBlocksExecution(REPETITIONS,nBlocks, inputImgFolder);
        std::cout << "Computation time: " << blocks / 1000 << "[msec]" << std::endl;
    }

    // speed up
    double hSpeedUP = serial / horizontal;
    double vSpeedUP = serial / vertical;
    std::cout << "\n\n\t\t\t\t" << "H\t\tV" << std::endl;
    std::cout << "\tSpeedUp:\t" <<  hSpeedUP << "\t" << vSpeedUP << "\n\n";

 }

 /***
  * performs a serial execution of sobel edge detector. Experiments are repeated <times> to
  * remove almost dependence from internal CPU state
  * @param times
  * @param inputImgFolder
  * @return average execution time over <times> repetitions in microseconds
  */
double serialExecution(int times,const char *inputImgFolder) {

    double executionTime = 0;

    for (int k = 0; k < times; k++) {

        // for each iteration over this folder

        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (inputImgFolder)) != NULL) {

            int i = 0;
            bool lock = 1;
            /* print all the files and directories within directory */

            auto startSerialExecutionTime = std::chrono::high_resolution_clock::now();
            while ((ent = readdir (dir)) != NULL && lock) {

                if (i >= MAX_FILES) {
                    lock = 0;
                    break; // exit after MAX_FILES
                }


                if (ent->d_type == 8) { // 8 stands for image
                    std::string buf(inputImgFolder);
                    buf.append(ent->d_name);

                    // buf.c_str() contains the current file in the directory
                    try {

                        cv::Mat inputImage = imread(buf.c_str() , cv::IMREAD_GRAYSCALE);
                        Sobel mySobel(inputImage,buf.c_str());

                    //    std::string s = "/home/marco/Scrivania/test1/serial/" + std::string(ent->d_name);
                   //     mySobel.writeToFile(s);
                        i++;

                    } catch (const cv::Exception& e) {
                        std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                    }

                }

            }

            auto endSerialExecutionTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endSerialExecutionTime - startSerialExecutionTime);
            closedir (dir);

            executionTime += duration.count();
        } else {
            /* could not open directory */
            perror ("");
        }
    }

    return executionTime / times; // average time over #times repetitions
}

/***
 * performs a parallel execution of sobel edge detector, images are scanned in horizontal
 * and are subdivided into <CHUNKS> rows. Experiments are repeated <times> to
 * remove almost dependence from internal CPU state
 * @param times
 * @param inputImgFolder
 * @return average computation time over <times> repetitions
 */
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

            bool lock = 1;
            //#pragma omp parallel private(ent) shared(i,lock) num_threads(THREADS)
            {
                while ((ent = readdir(dir)) != NULL && lock) {

                    if (i >= MAX_FILES) {
                        lock = 0;
                        break; // exit after MAX_FILES
                    }

                    if (ent->d_type == 8) { // 8 stands for image
                        std::string buf(inputImgFolder);
                        buf.append(ent->d_name);

                        // buf.c_str() contains the current file in the directory
                        try {

                            cv::Mat inputImage = imread(buf.c_str(), cv::IMREAD_GRAYSCALE);

                            ompSobel mySobel(inputImage, buf.c_str());
                            mySobel.setThreadsNum(THREADS);   // set threads number
                            mySobel.setChunksNum(CHUNKS);     // how many chunks for image subdivision
                            mySobel.computeHorizontal();      // start computation

                        //    std::string s = "/home/marco/Scrivania/test1/horizontal/" + std::string(ent->d_name);
                        //    mySobel.writeToFile(s);

                          //  #pragma omp atomic
                            i++;


                        } catch (const cv::Exception &e) {
                            std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                        }

                    }

                }
            }
            auto endParallelExecutionTime_Horizontal = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    endParallelExecutionTime_Horizontal - startParallelExecutionTime_Horizontal);
            closedir(dir);

            executionTime += duration.count();
        } else {
            /* could not open directory */
            perror("");
        }
    }

    return executionTime / times; // average time
}

/***
 * performs a parallel execution of sobel edge detector, images are scanned in vertical
 * and are subdivided into <CHUNKS> columns. Experiments are repeated <times> to
 * remove almost dependence from internal CPU state
 * @param times
 * @param inputImgFolder
 * @return average computation time over <times> repetitions
 */

double parallelVerticalExecution(int times, const char *inputImgFolder) {

    double executionTime = 0;

    for (int k = 0; k < times; k++) {

        DIR *dir;
        struct dirent *ent;

        if ((dir = opendir(inputImgFolder)) != NULL) {

            int i = 0;
            bool lock = 1;
            /* print all the files and directories within directory */
            auto startParallelExecutionTime_Horizontal = std::chrono::high_resolution_clock::now();
            //#pragma omp parallel private(ent) shared(i, lock) num_threads(THREADS)
            {
                while ((ent = readdir(dir)) != NULL && lock) {

                    if (i >= MAX_FILES) {
                        lock = 0;
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

                           // std::string s = "/home/marco/Scrivania/test1/vertical/" + std::string(ent->d_name);
                           // mySobel.writeToFile(s);

                          //  #pragma omp critical
                            {
                                i++;
                            };

                        } catch (const cv::Exception &e) {
                            std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                        }

                    }

                }
            }
            auto endParallelExecutionTime_Horizontal = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    endParallelExecutionTime_Horizontal - startParallelExecutionTime_Horizontal);
            closedir(dir);

            executionTime += duration.count();
        } else {
            /* could not open directory */
            perror("");
        }
    }
    return executionTime / times; // average time
}

/***
 * performs a parallel execution of sobel edge detector, images are subdivided into <BLOCKS> rectangular blocks.
 * Blocks are then scanned in horizontal.
 *  Experiments are repeated <times> to remove almost dependence from internal CPU state
 * @param times
 * @param inputImgFolder
 * @return average computation time over <times> repetitions
 */
double parallelBlocksExecution(int times, int nBlocks, const char *inputImgFolder) {

    double executionTime = 0;

    for (int k = 0; k < times; k++) {

        DIR *dir;
        struct dirent *ent;

        if ((dir = opendir(inputImgFolder)) != NULL) {

            int i = 0;

            /* print all the files and directories within directory */
            auto startParallelExecutionTime_Horizontal = std::chrono::high_resolution_clock::now();

          //  #pragma omp parallel private(ent) shared(i) num_threads(THREADS)  ---> DOES not WORKS
            {
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
                            mySobel.computeBlocks(nBlocks);

                         //   std::string s = "/home/marco/Scrivania/test1/blocks/" + std::to_string(nBlocks) + "/" + std::string(ent->d_name);
                         //   mySobel.writeToFile(s);
                            i++;

                        } catch (const cv::Exception &e) {
                            std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                        }

                    }

                }
            }
            auto endParallelExecutionTime_Horizontal = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    endParallelExecutionTime_Horizontal - startParallelExecutionTime_Horizontal);
            closedir(dir);

            executionTime += duration.count();
        } else {
            /* could not open directory */
            perror("");
        }
    }
    return executionTime / times; // average time
}