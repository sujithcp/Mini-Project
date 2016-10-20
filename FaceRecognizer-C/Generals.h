//
// Created by sujith on 13/09/16.
//
#include <iostream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing.h>
/*
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
*/
#include <dlib/opencv/cv_image.h>
#ifndef COMMA_GENERALS_H
#define COMMA_GENERALS_H
static double ALPHA,BETA,GAMMA,B1,B2;
using namespace std;
using namespace cv;
const string CURR_DIR = "/home/sujith/ClionProjects/Comma/";
void readLines(const string&,vector<string>& );
void fooDriver();
void setParam(double,double, double, double, double);
void extractFaces(Mat,std::vector<dlib::full_object_detection> & ,dlib::array<dlib::array2d<dlib::rgb_pixel> > &);
bool extractFacesInit();
vector<Mat> getFacesFromImage(Mat);
void readLines(const string&,vector<string>&);
void prepareTrainData(string& , vector<Mat>& , vector<int>& ,char,bool=false);
void train(bool= false);
bool initEngine();
void recognitionTest(char);
int recognize(Mat);
int recognize(Mat,dlib::full_object_detection &);
Mat preProcess(Mat);
Mat preProcess(Mat,dlib::full_object_detection &);
#endif //COMMA_GENERALS_H
