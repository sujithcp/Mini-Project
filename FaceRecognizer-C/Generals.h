//
// Created by sujith on 13/09/16.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#ifndef COMMA_GENERALS_H
#define COMMA_GENERALS_H
using namespace std;
using namespace cv;
void fooDriver();
vector<Mat> getFacesFromImage(Mat);
void readLines(const string&,vector<string>&);
void prepareTrainData(string& , vector<Mat>& , vector<int>& , char);
void train();
void recognitionTest(char);
Mat preProcess(Mat);
#endif //COMMA_GENERALS_H
