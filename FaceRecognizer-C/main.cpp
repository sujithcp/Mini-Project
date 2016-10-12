#include <dlib/gui_widgets.h>
#include <dlib/opencv/cv_image.h>
#include "Video.h"
#include "Generals.h"

int main(int argc,char* argv[]) {
    //capture();
    //fooDriver();
    setParam(atof(argv[1]),atof(argv[2]),atof(argv[3]),atof(argv[4]),atof(argv[5]));
    //recognitionTest(' ');
    train();
    FrameCapture cap;
    cap.capture();
    /*
    dlib::image_window original,processed;
    Mat img;
    img = imread("/home/sujith/ClionProjects/Comma/res/images/OUT/AM02-15.jpg",0);
    original.set_title("Original");
    original.set_image(dlib::cv_image<uchar >(img));
    img = preProcess(img);
    processed.set_title("Processed");
    processed.set_image(dlib::cv_image<uchar >(img));
    waitKey(20000);

     */



}