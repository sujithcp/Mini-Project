//
// Created by sujith on 13/09/16.
//
#include <iostream>
#include <fstream>
#include <math.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/face.hpp>
#include "Generals.h"
using namespace cv::face;
CascadeClassifier face_detect;
CascadeClassifier eye_detect;
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
        Mat tmpFace = preProcess(Mat(image,Rect(x,y,w,h)));
        if(tmpFace.rows<100 || tmpFace.cols<100)
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
    model->save("/home/sujith/ClionProjects/Comma/MODEL.YAML");
}

int recognize(Mat image)
{
    cvtColor(image,image,COLOR_BGR2GRAY);
    int res=-1;
    double conf=-1;
    model->predict(image, res, conf);
    cout<<res<<" "<<conf<<endl;
    return res;
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
        if (!path.empty() && !classlabel.empty()) {
            int res = -1;
            double conf = -1;
            model->predict(imread("/home/sujith/ClionProjects/Comma/res/images/OUT/" + path, 0), res, conf);
            cout << res << " " << conf << " " << labels[i];
            if (res == labels[i])
            {
                t++;
                cout<<endl;
            }
            else
            {
                f++;
                cout<<" ** "<<endl;
            }

        }
    }
    cout<<(100*t/(t+f))<<"%\n";
}

Mat preProcess(Mat image)
{
    Point2d e1,e2;
    vector<Rect> eyes;
    Mat img = Mat::zeros(image.rows,image.cols,image.type());
    eye_detect.detectMultiScale(image,eyes);
    dlib::full_object_detection shape;
    dlib::shape_predictor sp;
    shape = sp(dlib::cv_image<uchar>(image),dlib::rectangle(0,0,image.rows,image.cols));
    cout<<shape.num_parts()<<endl;
    cout<<"Hello\n"<<eyes.size()<<endl;
    for(size_t i=0;i<2;i++)
    {
        cout<<eyes[i].x<<" "<<eyes[i].y<<" "<<eyes[i].width<<" "<<eyes[i].height<<endl;
        rectangle(image,CvPoint(eyes[i].x,eyes[i].y),cvPoint(eyes[i].x+eyes[i].width,eyes[i].y+eyes[i].height),Scalar(0,0,250),2);
    }
    imshow("Origiinal",image);
    waitKey(5000);
    /*if(eyes.size()!=2)
        return image;
        */
    e1.x = eyes[0].x+(eyes[0].width*0.5);
    e1.y = eyes[0].y+(eyes[0].height*0.5);
    e2.x = eyes[1].x+(eyes[1].width*0.5);
    e2.y = eyes[1].y+(eyes[1].height*0.5);

    if(e2.x<e1.x)
    {
        Point2d tmp = e2;
        e2 = e1;
        e1=tmp;
    }
    double angle = 90-atan2(e2.x-e1.x,e2.y-e1.y)*180/3.1415;
    cout<<"Angle "<<angle<<endl;
    warpAffine(image,img,getRotationMatrix2D(e1,angle,1),img.size());
    imshow("Rotated",img);
    waitKey(5000);
    return img;

}



void fooDriver()
{
    face_detect.load("/home/sujith/ClionProjects/Comma/res/haarcascade_profileface.xml");
    eye_detect.load("/home/sujith/ClionProjects/Comma/res/haarcascade_eye.xml");
    //Mat image = imread("./res/images/AF01/AF01AFS.JPG");
    //imshow("IMAGE",image);
    //waitKey(0);
    //getFacesFromImage(image);
    //recognitionTest();
    //train();
    //cout<<recognize(imread("/home/sujith/ClionProjects/Comma/res/images/OUT/AF01-4.jpg"))<<endl;
    preProcess(imread("/home/sujith/ClionProjects/Comma/res/images/OUT/BM01-6.jpg",0));
}
