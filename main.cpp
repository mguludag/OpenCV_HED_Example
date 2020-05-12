/****************************************************************************************************
 * 
 * This example shows dnn edge detection vs canny
 * dnn edge detection slower than canny but its more realistic on most cases
 * 
*****************************************************************************************************/
#include "mycroplayer.hpp"
#include <iostream>
#include <string>
#include <vector>

void hedEdgeDetectDNN(cv::Mat &image, std::string prototxt, std::string caffemodel, int size = 128)
{
    cv::dnn::Net net = cv::dnn::readNet(prototxt, caffemodel);

    cv::Size reso(size, size);
    cv::Mat theInput;

    cv::resize(image, theInput, reso);
    cv::Mat blob = cv::dnn::blobFromImage(theInput,
                                          1.0,
                                          reso,
                                          cv::Scalar(104.00698793, 116.66876762, 122.67891434),
                                          false,
                                          false);
    net.setInput(blob);
    cv::Mat out
        = net.forward(); // outputBlobs contains all output blobs for each layer specified in outBlobNames.

    std::vector<cv::Mat> vectorOfImagesFromBlob;
    cv::dnn::imagesFromBlob(out, vectorOfImagesFromBlob);
    cv::Mat tmpMat = vectorOfImagesFromBlob[0] * 255;
    cv::Mat tmpMatUchar;
    cv::cvtColor(tmpMat, tmpMatUchar, cv::COLOR_GRAY2BGR);
    cv::resize(tmpMatUchar, image, image.size());
}

int main(int argc, char *argv[])
{
    std::string image_name;
    if (argc > 1)
        image_name = argv[1];
    else {
        std::cout << "please enter image name with path and extension: ";
        std::cin >> image_name;
    }

    CV_DNN_REGISTER_LAYER_CLASS(Crop, MyCropLayer); //register reimplemented layer class to opencv

    cv::Mat image = cv::imread(image_name);
    if (!image.empty()) {
        cv::Mat cannyd, win_mat(cv::Size(image.cols, static_cast<int>(image.rows * 1.5)),
                                CV_8UC3);                              //show images side by side
        image.copyTo(win_mat(cv::Rect(0, 0, image.cols, image.rows))); //copy image to left
        cv::resize(image,
                   image,
                   cv::Size(image.cols / 2, image.rows / 2)); //resize image to its half size
        cv::Canny(image, cannyd, 90, 150);                    //canny edge detection
        cv::cvtColor(cannyd, cannyd, cv::COLOR_GRAY2BGR);
        cannyd.copyTo(win_mat(cv::Rect(0, image.rows * 2, image.cols, image.rows)));
        hedEdgeDetectDNN(image,
                         "c:/opencv/deploy.prototxt",
                         "c:/opencv/hed_pretrained_bsds.caffemodel",
                         1024);
        image.copyTo(win_mat(
            cv::Rect(image.cols, image.rows * 2, image.cols, image.rows))); //copy image to right
        cv::namedWindow("OpenCV hed example", cv::WINDOW_GUI_EXPANDED);
        cv::imshow("OpenCV hed example", win_mat);
        cv::waitKey(0);
    }
}
