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
    map<int,string> class_map;
    vector< string > classmap_list;
    int src=0;
    string strUrl="";
    FrameCapture();
    void capture();
};

#endif //COMMA_VIDEO_H
