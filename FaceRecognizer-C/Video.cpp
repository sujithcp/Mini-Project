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
        if(!initEngine())
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
            //cap.retrieve(temp,CAP_OPENNI_DEPTH_MAP);
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);
            // Detect face
            std::vector<dlib::rectangle> faces = detector(cimg,0);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(cimg, faces[i]));
            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes));
            dlib::array<array2d<rgb_pixel> > face_chips;
            extract_image_chips(cimg, get_face_chip_details(shapes), face_chips);
            win_face.set_image(tile_images(face_chips));
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
                recognize(toMat(face_chips[0]));
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
