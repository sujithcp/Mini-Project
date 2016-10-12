//
// Created by sujith on 09/10/16.
//
#ifndef COMMA_VIDEO_H
#define COMMA_VIDEO_H

#include <iostream>
#include <unistd.h>
using namespace std;
class FrameCapture
{
public:
    int src=0;
    string CURR_DIR = "/home/sujith/ClionProjects/Comma/";
    string strUrl="";
    FrameCapture();
    void capture();
};

#endif //COMMA_VIDEO_H
