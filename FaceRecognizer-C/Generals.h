//
// Created by sujith on 13/09/16.
//
#include <iostream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#ifndef COMMA_GENERALS_H
#define COMMA_GENERALS_H
static double ALPHA,BETA,GAMMA,B1,B2;
using namespace std;
using namespace cv;
void fooDriver();
void setParam(double,double, double, double, double);
vector<Mat> getFacesFromImage(Mat);
void readLines(const string&,vector<string>&);
void prepareTrainData(string& , vector<Mat>& , vector<int>& , char);
void train();
bool initEngine();
void recognitionTest(char);
int recognize(Mat);
Mat preProcess(Mat);
#endif //COMMA_GENERALS_H
