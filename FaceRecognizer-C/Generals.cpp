//
// Created by sujith on 13/09/16.
//
#include <iostream>
#include <fstream>
#include <opencv2/face.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include "Generals.h"
using namespace cv::face;
CascadeClassifier face_detect;
Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
dlib::frontal_face_detector detector;
dlib::shape_predictor pose_model;
void setParam(double a,double b,double g,double b1,double b2)
{
    ALPHA = a;
    BETA = b;
    GAMMA = g;
    B1 = b1;
    B2 = b2;
    cout<<"GEN "<<ALPHA<<BETA<<GAMMA<<B1<<B2<<endl;
}

void extractFaces(Mat image,std::vector<dlib::full_object_detection> & shapes,dlib::array<dlib::array2d<dlib::rgb_pixel> > & face_chips)
{
    dlib::cv_image<dlib::bgr_pixel> cimg(image);
    // Detect face
    std::vector<dlib::rectangle> faces = detector(cimg,0);
    // Find the pose of each face.
    for (unsigned long i = 0; i < faces.size(); ++i)
        shapes.push_back(pose_model(cimg, faces[i]));
    extract_image_chips(cimg, get_face_chip_details(shapes), face_chips);
}
bool extractFacesInit()
{
    try
    {
        detector = dlib::get_frontal_face_detector();
        dlib::deserialize(CURR_DIR+"/res/shape_predictor_68_face_landmarks.dat") >> pose_model;
    }
    catch (Exception e)
    {
        cout<<e.code<<" "<<e.err<<"\n"<<e.msg<<"\n";
        return 1;
    }

    return true;
}


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

void prepareTrainData(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ' ',bool mask=false)
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
            cout<<"Reading "<<path<<"\n";
            if(mask)
            {
                std::vector<dlib::full_object_detection> shapes;
                dlib::array<dlib::array2d<dlib::rgb_pixel> > face_chips;
                extractFaces(imread(CURR_DIR+"/res/images/OUT/" + path),shapes,face_chips);
                if(face_chips.size()==1)
                {
                    images.push_back(preProcess(dlib::toMat(face_chips[0]),shapes[0]));
                }
                else
                    continue;
            }
            else
            {
                images.push_back(preProcess(imread(CURR_DIR+"/res/images/OUT/" + path)));
            }

            labels.push_back(atoi(classlabel.c_str()));
        }
    }

}

void train(bool mask)
{

    vector<Mat> images;
    vector<int> labels;
    if(mask)
        prepareTrainData(CURR_DIR+"res/images/OUT/faces_list.txt",images,labels,' ',true);
    else
        prepareTrainData(CURR_DIR+"res/images/OUT/faces_list.txt",images,labels,' ');
    cout<<"***\n";
    model->train(images,labels);
    model->save(CURR_DIR+"MODEL.YAML");
}

int recognize(Mat image)
{
    //cvtColor(image,image,COLOR_BGR2GRAY);
    image = preProcess(image);
    int res=-1;
    double conf=-1;
    model->predict(image, res, conf);
    //cout<<res<<" "<<conf<<endl;
    cout<<"With confidence "<<conf<<endl;
    return res;
}
int recognize(Mat image,dlib::full_object_detection &shape)
{
    //cvtColor(image,image,COLOR_BGR2GRAY);
    image = preProcess(image,shape);
    int res=-1;
    double conf=-1;
    model->predict(image, res, conf);
    //cout<<res<<" "<<conf<<endl;
    cout<<"With confidence "<<conf<<endl;
    return res;
}

void recognitionTest(char separator = ' ')
{
    vector<string> lines;
    readLines(CURR_DIR+"/res/images/OUT/faces_list.txt",lines);
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
                trainImages.push_back(preProcess(imread(CURR_DIR+"/res/images/OUT/" + path, 0)));
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
            model->predict(preProcess(imread(CURR_DIR+"/res/images/OUT/" + path, 0)), res, conf);
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
    if(image.channels()==3)
        cvtColor(image,image,COLOR_BGR2GRAY);
    resize(image,image,CvSize(300,300));
    equalizeHist(image,image);
    normalize(image,image,0,255,NORM_MINMAX,CV_8U);
    Mat blur;
    //GaussianBlur(image,blur,Size(21,21),0,0);
    //addWeighted(image,1,blur,-0.9,0,image);
    //cout<<"GEN "<<ALPHA<<BETA<<GAMMA<<B1<<B2<<endl;
    GaussianBlur(image,blur,Size((int)B1,(int)B2),0,0);
    addWeighted(image,ALPHA,blur,BETA,GAMMA,image);
    imshow("Hist",image);
    waitKey(10);


    return image;

}

Mat preProcess(Mat image,dlib::full_object_detection &shape)
{
    /*Point2d e1,e2;
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
    if(eyes.size()!=2)
    {
        equalizeHist(image,img);
        return img;
    }

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
     */
    if(image.channels()==3)
        cvtColor(image,image,COLOR_BGR2GRAY);
    Mat res;
    Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    vector< vector<Point> >  co_ordinates;
    co_ordinates.push_back(vector<Point>());
    int orgx = (int) shape.get_rect().tl_corner().x();
    int orgy = (int) shape.get_rect().tl_corner().y();
    co_ordinates[0].push_back(Point((int)shape.part(1).x()-orgx,(int)shape.part(20).y()-orgy+10));
    for(size_t i = 2;i<17;i++)
    {
        //cout<<shape.part(i)<<endl;
        co_ordinates[0].push_back(Point((int)shape.part(i).x()-orgx,(int)shape.part(i).y()-orgy));
    }
    co_ordinates[0].push_back(Point((int)shape.part(17).x()-orgx,(int)shape.part(25).y()-orgy+10));
    co_ordinates[0].push_back(Point((int)shape.part(0).x()-orgx,(int)shape.part(0).y()-orgy));
    drawContours( mask,co_ordinates,0, Scalar(255),CV_FILLED, 8 );
    image.copyTo(res,mask);
    imshow("mask",mask);
    waitKey(10);
    imshow("res",res);
    waitKey(10);


    return preProcess(res);

}

bool initEngine()
{
    try {
        model->load(CURR_DIR+"/MODEL.YAML");
        face_detect.load(CURR_DIR+"/res/haarcascade_profileface.xml");
    }
    catch (Exception e)
    {
        cout<<"Model Load failed\n";
        return false;
    }
    return true;
}

void fooDriver()
{
    face_detect.load(CURR_DIR+"/res/haarcascade_profileface.xml");
    //eye_detect.load(CURR_DIR+"/res/haarcascade_eye.xml");
    //Mat image = imread("./res/images/AF01/AF01AFS.JPG");
    //imshow("IMAGE",image);
    //waitKey(0);
    //getFacesFromImage(image);
    //recognitionTest();
    //cout<<"Training\n";
    //train();
    //cout<<"Training completed\n";
    cout<<recognize(imread(CURR_DIR+"/res/images/OUT/BM01-6.jpg",0))<<endl;
    preProcess(imread(CURR_DIR+"/res/images/OUT/BM01-6.jpg",0));
}
