#include <iostream>
#include <dirent.h>
#include <ostream>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <vector>
#include <map>

#include "Sobel/Sobel.h"
#include "Sobel/ompSobel.h"

#include "Canny/Canny.h"
#include "Canny/ompCanny.h"

#define REPETITIONS 1
#define MAX_FILES 1

#define CANNY_FILTER_SIZE 5
#define CANNY_FILTER_SIGMA 2

void readFolder(const char *inputImgFolder, char name[]);
std::vector<double> serialExecution(int times,const char *inputImgFolder);
std::vector<double> parallelHorizontalExecution(int times,const char *inputImgFolder);
std::vector<double> parallelVerticalExecution(int times,const char *inputImgFolder);
std::vector<double> parallelBlocksExecution(int times, int nBlocks, const char *inputImgFolder);

int CHUNKS = 8;
int THREADS = 8;


std::ofstream myFile;
bool writePerformances = 0;

int main(int argc, char** argv) {

    if (argc >= 5) {
        // arguments mode
        // usage: edgedetector -t <threads> -c <chunks>
        THREADS = atoi(argv[2]);
        CHUNKS = atoi(argv[4]);
        if (argc == 6 && strcmp(argv[5],"-w")) {
            // write performance file
            writePerformances = 1;
        }
    }

    if (writePerformances) {

        std::string file = "result " + std::to_string(THREADS) + "T-" + std::to_string(CHUNKS) + "C.txt";
        myFile.open(file);
    }

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
                    if (writePerformances) myFile << "[" << ent->d_name << "]\n";
                    std::string buf(inputImgFolder);
                    buf.append(ent->d_name);
                    buf.append("/"); // hard coded ? :)

                    readFolder(buf.c_str(), ent->d_name);

                }

            }
        }


    }

    std::cout << "Done, bye :)";

    if (writePerformances) {
        myFile << "Terminated\n";
        myFile.close();
    }
    return 0;
}

/***
 * scan all image files into the given folder and make computation for serial, horizontal and vertical
 * @param inputImgFolder
 */
void readFolder(const char *inputImgFolder, char name[]) {

    std::cout << "\t[SERIAL] ";
    std::vector<double> serial = serialExecution(REPETITIONS,inputImgFolder);

    std::cout << "Computation time:\n\t\tSobel: " << serial[0] * 1000 << "[msec]\n\t\tCanny: " << serial[1] * 1000 << "[msec]" << std::endl;

    std::cout << "\t[PARALLEL - VERTICAL] ";
    std::vector<double> vertical = parallelVerticalExecution(REPETITIONS,inputImgFolder);
    std::cout << "Computation time:\n\t\tSobel: " << vertical[0] * 1000 << "[msec]\n\t\tCanny: " << vertical[1] * 1000 << "[msec]" << std::endl;

    std::cout << "\t[PARALLEL - HORIZONTAl] ";
    std::vector<double> horizontal = parallelHorizontalExecution(REPETITIONS,inputImgFolder);
    std::cout << "Computation time:\n\t\tSobel: " << horizontal[0] * 1000 << "[msec]\n\t\tCanny: " << horizontal[1] * 1000 << "[msec]" << std::endl;

    if (writePerformances) {

        myFile << "Serial:\t" << serial[0] * 1000 << "\t" << serial[1] * 1000 << "\n";
        myFile << "Vertical:\t" << vertical[0] * 1000 << "\t" << vertical[1] * 1000 << "\n";
        myFile << "Horizontal:\t" << horizontal[0] * 1000 << "\t" << horizontal[1] * 1000<< "\n";
    }

    // try different block numbers --> 8 - 16 - 32 - 64
    for (int i = 3; i < 7; i++) {
        int nBlocks = pow(2,i);
        std::cout << "\t[PARALLEL - "<< nBlocks << " BLOCKS] ";
        std::vector<double> blocks = parallelBlocksExecution(REPETITIONS,nBlocks, inputImgFolder);
        std::cout << "Computation time:\n\t\tSobel: " << blocks[0] * 1000 << "[msec]\n\t\tCanny: " << blocks[1] * 1000 << "[msec]" << std::endl;
        if (writePerformances) myFile << "Blocks - [" << nBlocks << "]\t" << blocks[0] * 1000<< "\t" << blocks[1] * 1000 << "\n";
    }

    // speed up
    double hSpeedUP_Sobel = serial[0] / horizontal[0];
    double hSpeedUP_Canny = serial[1] / horizontal[1];

    double vSpeedUP_Sobel = serial[0] / vertical[0];
    double vSpeedUP_Canny = serial[1] / vertical[1];


    std::cout << "\n\nSpeedUp:\t\t" << "H\t\t\tV" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Sobel\t\t\t" <<  hSpeedUP_Sobel << "\t\t" << vSpeedUP_Sobel << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Canny\t\t\t" <<  hSpeedUP_Canny << "\t\t" << vSpeedUP_Canny << std::endl;
    std::cout << "------------------------------------" << std::endl;


    if (writePerformances) {
        myFile << "\n\nSpeedUp:\t\t" << "H\t\t\tV\n";
        myFile << "------------------------------------\n";
        myFile << "Sobel\t\t\t" <<  hSpeedUP_Sobel << "\t\t" << vSpeedUP_Sobel << "\n";
        myFile << "------------------------------------\n";
        myFile << "Canny\t\t\t" <<  hSpeedUP_Canny << "\t\t" << vSpeedUP_Canny << "\n";
        myFile << "------------------------------------\n";
    }

 }

 /***
  * performs a serial execution of sobel edge detector. Experiments are repeated <times> to
  * remove almost dependence from internal CPU state
  * @param times
  * @param inputImgFolder
  * @return average execution time over <times> repetitions in microseconds
  */
std::vector<double> serialExecution(int times,const char *inputImgFolder) {

     std::vector<double> executionTime;
     double duration_Sobel, duration_Canny;

    for (int k = 0; k < times; k++) {

        // for each iteration over this folder

        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (inputImgFolder)) != NULL) {

            int i = 0;
            bool lock = 1;
            /* print all the files and directories within directory */


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

                        auto startSobel = std::chrono::high_resolution_clock::now();

                        Sobel mySobel(inputImage,buf.c_str());

                        std::chrono::duration<double> time_cpu_Sobel =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startSobel);
                        duration_Sobel += time_cpu_Sobel.count();


                        auto startCanny = std::chrono::high_resolution_clock::now();

                        Canny myCanny(inputImage, buf.c_str(), CANNY_FILTER_SIZE, CANNY_FILTER_SIGMA);
                        myCanny.computeCannyEdgeDetector();

                        std::chrono::duration<double> time_cpu_Canny = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startCanny);
                        duration_Canny += time_cpu_Sobel.count();

                        i++;

                    } catch (const cv::Exception& e) {
                        std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                    }

                }

            }

            closedir (dir);

        } else {
            /* could not open directory */
            perror ("");
        }
    }

    duration_Sobel = duration_Sobel / times;
    duration_Canny = duration_Canny / times;

    executionTime.push_back(duration_Sobel);
    executionTime.push_back(duration_Canny);

    return executionTime; // average time over #times repetitions
}

/***
 * performs a parallel execution of sobel edge detector, images are scanned in horizontal
 * and are subdivided into <CHUNKS> rows. Experiments are repeated <times> to
 * remove almost dependence from internal CPU state
 * @param times
 * @param inputImgFolder
 * @return average computation time over <times> repetitions
 */
std::vector<double> parallelHorizontalExecution(int times,const char *inputImgFolder) {

    std::vector<double> executionTime;
    double duration_Sobel = 0, duration_Canny = 0;

    for (int k = 0; k < times; k++) {

        DIR *dir;
        struct dirent *ent;

        if ((dir = opendir(inputImgFolder)) != NULL) {

            //std::cout << "Working directory: " << inputImgFolder << std::endl;
            int i = 0;
            /* print all the files and directories within directory */

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


                            auto startSobel = std::chrono::high_resolution_clock::now();

                            ompSobel mySobel(inputImage, buf.c_str());
                            mySobel.setThreadsNum(THREADS);   // set threads number
                            mySobel.setChunksNum(CHUNKS);     // how many chunks for image subdivision
                            mySobel.computeHorizontal();      // start computation

                            std::chrono::duration<double> time_cpu_Sobel =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startSobel);
                            duration_Sobel += time_cpu_Sobel.count();



                            auto startCanny = std::chrono::high_resolution_clock::now();

                            ompCanny myCanny(inputImage, buf.c_str(), CANNY_FILTER_SIZE, CANNY_FILTER_SIGMA);
                            myCanny.setThreadsNum(THREADS);
                            myCanny.setChunksNum(CHUNKS);
                            myCanny.computeCannyEdgeDetector_Horizontal();

                            std::chrono::duration<double> time_cpu_Canny = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startCanny);
                            duration_Canny += time_cpu_Sobel.count();

                            i++;


                        } catch (const cv::Exception &e) {
                            std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                        }

                    }

                }
            }
            closedir(dir);

        } else {
            /* could not open directory */
            perror("");
        }
    }


    duration_Sobel = duration_Sobel / times;
    duration_Canny = duration_Canny / times;

    executionTime.push_back(duration_Sobel);
    executionTime.push_back(duration_Canny);

    return executionTime; // average time
}

/***
 * performs a parallel execution of sobel edge detector, images are scanned in vertical
 * and are subdivided into <CHUNKS> columns. Experiments are repeated <times> to
 * remove almost dependence from internal CPU state
 * @param times
 * @param inputImgFolder
 * @return average computation time over <times> repetitions
 */

std::vector<double> parallelVerticalExecution(int times, const char *inputImgFolder) {

    std::vector<double> executionTime;
    double duration_Sobel = 0, duration_Canny = 0;

    for (int k = 0; k < times; k++) {

        DIR *dir;
        struct dirent *ent;

        if ((dir = opendir(inputImgFolder)) != NULL) {

            int i = 0;
            bool lock = 1;
            /* print all the files and directories within directory */

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

                            auto startSobel = std::chrono::high_resolution_clock::now();

                            ompSobel mySobel(inputImage, buf.c_str());
                            mySobel.setThreadsNum(THREADS);   // set threads number
                            mySobel.setChunksNum(CHUNKS);     // how many chunks for image subdivision
                            mySobel.computeHorizontal();      // start computation

                            std::chrono::duration<double> time_cpu_Sobel =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startSobel);
                            duration_Sobel += time_cpu_Sobel.count();


                            auto startCanny = std::chrono::high_resolution_clock::now();

                            ompCanny myCanny(inputImage, buf.c_str(), CANNY_FILTER_SIZE, CANNY_FILTER_SIGMA);
                            myCanny.setThreadsNum(THREADS);
                            myCanny.setChunksNum(CHUNKS);
                            myCanny.computeCannyEdgeDetector_Vertical();

                            std::chrono::duration<double> time_cpu_Canny = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startCanny);
                            duration_Canny += time_cpu_Sobel.count();


                            i++;

                        } catch (const cv::Exception &e) {
                            std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                        }

                    }

                }
            }
            closedir(dir);

        } else {
            /* could not open directory */
            perror("");
        }
    }
    duration_Sobel = duration_Sobel / times;
    duration_Canny = duration_Canny / times;

    executionTime.push_back(duration_Sobel);
    executionTime.push_back(duration_Canny);
    return executionTime; // average time
}

/***
 * performs a parallel execution of sobel edge detector, images are subdivided into <BLOCKS> rectangular blocks.
 * Blocks are then scanned in horizontal.
 *  Experiments are repeated <times> to remove almost dependence from internal CPU state
 * @param times
 * @param inputImgFolder
 * @return average computation time over <times> repetitions
 */
std::vector<double> parallelBlocksExecution(int times, int nBlocks, const char *inputImgFolder) {

    std::vector<double> executionTime;
    double duration_Sobel = 0, duration_Canny = 0;

    for (int k = 0; k < times; k++) {

        DIR *dir;
        struct dirent *ent;

        if ((dir = opendir(inputImgFolder)) != NULL) {

            int i = 0;

            /* print all the files and directories within directory */

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

                            auto startSobel = std::chrono::high_resolution_clock::now();

                            ompSobel mySobel(inputImage, buf.c_str());
                            mySobel.setThreadsNum(THREADS);
                            mySobel.computeBlocks(nBlocks);

                            std::chrono::duration<double> time_cpu_Sobel =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startSobel);
                            duration_Sobel += time_cpu_Sobel.count();


                            auto startCanny = std::chrono::high_resolution_clock::now();

                            ompCanny myCanny(inputImage, buf.c_str(), CANNY_FILTER_SIZE, CANNY_FILTER_SIGMA);
                            myCanny.setThreadsNum(THREADS);
                            myCanny.computeCannyEdgeDetector_Blocks(nBlocks);

                            std::chrono::duration<double> time_cpu_Canny = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startCanny);
                            duration_Canny += time_cpu_Sobel.count();

                            i++;

                        } catch (const cv::Exception &e) {
                            std::cerr << e.what() << " in file: " << buf.c_str() << std::endl;
                        }

                    }

                }
            }
            closedir(dir);

        } else {
            /* could not open directory */
            perror("");
        }
    }
    duration_Sobel = duration_Sobel / times;
    duration_Canny = duration_Canny / times;

    executionTime.push_back(duration_Sobel);
    executionTime.push_back(duration_Canny);
    return executionTime; // average time
}