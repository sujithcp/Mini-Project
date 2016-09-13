#include<iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include "FaceRecognizer.h"
using namespace std;
using  namespace cv;

int capture()
{
    VideoCapture cap;
    Mat frame;
    cap.open(0);
    if(!cap.isOpened())
        return -1;
    while(true)
    {
        cap>>frame;
        cvtColor(frame,frame,COLOR_BGR2GRAY);
        imshow("Webcam",frame);
        waitKey(5);

    }
    return 0;
}