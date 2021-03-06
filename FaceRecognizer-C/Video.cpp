//
// Created by sujith on 09/10/16.
//
#include "Generals.h"
#include "Video.h"
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
using namespace dlib;
FrameCapture::FrameCapture()
{

    readLines(CURR_DIR+"/res/images/OUT/class_map.txt",classmap_list);
    string tmp,sub;
    int sub_id;
    for(size_t i = 0;i<classmap_list.size();i++)
    {
        tmp = classmap_list[i];
        stringstream liness(tmp);
        getline(liness, sub, ' ');
        getline(liness, tmp);
        sub_id=stoi(tmp);
        class_map[sub_id]=sub;

    }

}

void FrameCapture::capture()
{
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return;
        }
        //cap.set(1, CV_CAP_PROP_FPS);
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize(CURR_DIR+"/res/shape_predictor_68_face_landmarks.dat") >> pose_model;
        image_window win,win_face;
        cv::Mat temp;
        int face_count = 0;
        if(!initEngine() || !extractFacesInit())
        {
            cout << "Model Load failed\n";
            return;
        }
        while(!win.is_closed())
        {
            // Grab a frame
            //const int frames = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);
            //Seek video to last frame
            //cap.set(CV_CAP_PROP_POS_FRAMES,frames-1);
            cap>>temp;
            flip(temp,temp,1);
            //cap.retrieve(temp,CAP_OPENNI_DEPTH_MAP);
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);
            // Detect face
            std::vector<full_object_detection> shapes;
            dlib::array<array2d<rgb_pixel> > face_chips;
            extractFaces(temp,shapes,face_chips);
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes));

            if(face_chips.size()>0)
            {
                /*
                string fname = CURR_DIR+"/sujith/"+to_string(face_count++)+".jpg";
                try
                {
                    imwrite(fname, toMat(face_chips[0]));
                }
                catch (Exception e)
                {
                    cout<<"Exception found"<<endl<<e.what()<<endl;
                }
                */

                cout<<shapes[0].get_rect().tl_corner()<<" "<<shapes[0].get_rect().br_corner()<<endl;
                cout<<shapes[0].part(0)<<endl;
                win_face.set_image(face_chips[0]);
                //int res = recognize(toMat(face_chips[0]),shapes[0]);
                int res = recognize(toMat(face_chips[0]));
                if(class_map.find(res)!=class_map.end())
                {
                    cout<<"Recognized "<<class_map[res]<<endl;
                    string audio = "espeak -p 60 "+class_map[res];
                    system(audio.c_str());
                }
            }
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}
