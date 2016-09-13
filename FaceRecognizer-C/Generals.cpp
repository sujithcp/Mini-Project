//
// Created by sujith on 13/09/16.
//
#include <iostream>
#include "Generals.h"
#include <fstream>
#include "opencv2/face.hpp"


using namespace cv;
using namespace cv::face;
using namespace std;
CascadeClassifier face_detect;
Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
vector<Mat> getFacesFromImage(Mat image)
{
    vector<Rect> faces;
    vector<Mat> face_imgs;
    cvtColor(image,image,COLOR_BGR2GRAY);
    face_detect.detectMultiScale(image,faces);
    for(size_t i=0;i<faces.size();i++)
    {
        int x=faces[i].x,y=faces[i].y,w=faces[i].width,h=faces[i].height;
        cout<<x<<" "<<y<<" "<<w<<" "<<h<<endl;
        imshow("FACE",Mat(image,Rect(x,y,w,h)));
        waitKey(1000);
        Mat tmpFace = Mat(image,Rect(x,y,w,h));
        if(tmpFace.rows<250 || tmpFace.cols<250)
            continue;
        face_imgs.push_back(tmpFace);
    }
    cout<<face_imgs.size()<<endl;
    return face_imgs;
}

void readLines(const string& filename,vector<string>& lines)
{
    cout<<filename.c_str()<<endl;
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line;
    while (getline(file, line)) {
        lines.push_back(line);
    }
}

void prepareTrainData(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ' ')
{
    vector<string> lines;
    readLines(filename,lines);
    string line,path,classlabel;
    for(size_t i =0;i<lines.size();i++) {
        line = lines[i];
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()) {
            images.push_back(imread("/home/sujith/ClionProjects/Comma/res/images/OUT/" + path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }

}

void train()
{

    vector<Mat> images;
    vector<int> labels;
    prepareTrainData("/home/sujith/ClionProjects/Comma/res/images/OUT/faces_list.txt",images,labels,' ');
    model->train(images,labels);
}

void recognitionTest(char separator = ' ')
{
    vector<string> lines;
    readLines("/home/sujith/ClionProjects/Comma/res/images/OUT/faces_list.txt",lines);
    vector<int> labels;
    vector<Mat> trainImages;
    vector<int> trainLabels;

    random_shuffle(lines.begin(),lines.end());
    int mid = (int) lines.size()/2;
    string line,path,classlabel;
    cout<<mid;
    for(size_t i =0;i<lines.size();i++)
    {
        line = lines[i];
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty())
        {
            if(i<mid)
            {
                trainImages.push_back(imread("/home/sujith/ClionProjects/Comma/res/images/OUT/" + path, 0));
                trainLabels.push_back(atoi(classlabel.c_str()));
                labels.push_back(trainLabels[i]);
            }
            else
                labels.push_back(atoi(classlabel.c_str()));
            //cout<<trainLabels[i]<<"--"<<endl;
        }
    }

    model->clear();
    model->train(trainImages,trainLabels);
    int t=0,f=0;
    for(int i = mid;i<lines.size();i++)
    {
        line = lines[i];
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty())
        {
            int res = model->predict(imread("/home/sujith/ClionProjects/Comma/res/images/OUT/" + path, 0));
            if (res == labels[i])
                t++;
            else
                f++;
            cout << res << " " << labels[i] << endl;
        }
    }
    cout<<(100*t/(t+f))<<"%\n";
}


void fooDriver()
{
    //face_detect.load("./res/haarcascade_profileface.xml");
    //Mat image = imread("./res/images/AF01/AF01AFS.JPG");
    //imshow("IMAGE",image);
    //waitKey(0);
    //getFacesFromImage(image);
    //train();
    recognitionTest();
}