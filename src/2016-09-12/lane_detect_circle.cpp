#include<iostream>
#include<fstream>
#include<string>
#include <unistd.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

Mat src, src_gray, src_bak;
//Mat dst, detected_edges;

vector<Mat> img;
vector<string> list;
int edgeThresh = 1;
int lowThreshold=100;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
//char* window_name = "Edge Map";

Mat lane_perspective(Mat src);
Mat sobel(Mat src);
Mat set_ROI(Mat src);

int main( int argc, char** argv )
{
    string filename = "data/04/image_0/list.txt";
    ifstream ifile;
    ifile.open(filename.c_str());

    int count=0;

    string tmp;
    while( getline(ifile,tmp) )
    {
        list.push_back(tmp);
        cout << "Read from file: " << list[count] << endl;
        count++;
        tmp.clear();
    }
    ifile.close();

  int img_num=list.size()-4;
  cout<<"img num: "<<img_num<<endl;
  int img_begin=3;
  int img_end=list.size()-2;

  cout<<"img_end: "<<img_end<<endl;

  for(int i=img_begin;i<=img_end;i++)  //读图
  {
      char *addr_buf;
      int len = list[i].length();
      addr_buf =new char[len+1];
      strcpy(addr_buf,list[i].c_str());
      char img_buf[100];
      sprintf(img_buf, "./data/04/image_0/%s", addr_buf);
      string img_addr = img_buf;
      cout<<img_addr<<endl;
      Mat src=imread(img_addr,0);
      if( !src.data )
      { return -1; }
      Mat src_heq;
      equalizeHist( src, src_heq );
      img.push_back(src_heq);
//      img.push_back(src);
  }

  for(int i=0;i<img.size(); i++)
  {
     cout<<i<<endl;
     Mat dst, detected_edges;

      /// Create a matrix of the same type and size as src (for dst)
      dst.create( img[i].size(), img[i].type() );

      /// Create a window
//      namedWindow( "Edge Map", CV_WINDOW_AUTOSIZE );

      Mat rotated = lane_perspective(img[i]);

      Mat ROI_image = set_ROI(rotated);

//      medianBlur(rotated,detected_edges,3);

      Mat grad, abs_grad_x;
   //      medianBlur(rotated,grad,3);
         medianBlur(ROI_image,grad,3);
      abs_grad_x = sobel(grad);

//        Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
        Canny( abs_grad_x, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
        /// Using Canny's output as a mask, we display our result
        dst = Scalar::all(0);

//        rotated.copyTo( dst, detected_edges);
//        imshow( "Edge Map", dst );

        Mat hdst, roi_show;
        cvtColor(detected_edges, hdst, CV_GRAY2BGR);
        cvtColor(rotated, roi_show, CV_GRAY2BGR);
        vector<Vec4i> lines;
        HoughLinesP(detected_edges, lines, 1, CV_PI/180, 50, 1, 20 );
//        HoughLinesP(abs_grad_x, lines, 1, CV_PI/180, 50, 1, 10 );
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            float k = ((l[0]-l[2])*1.0)/(l[1]-l[3]);
            if(k<1 && k >-1)
            {
                line( hdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
                l[0]=l[0]+500;
                l[2]=l[2]+500;
                l[1]=l[1]+200;
                l[3]=l[3]+200;
                line( roi_show, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
            }
        }
        imshow("source", detected_edges);
        imshow("detected lines", hdst);
        imshow("detected lines on rotated", roi_show);
        char img_buf[100];
        sprintf(img_buf, "./output/04/image_0/%d.png", i);
        string img_addr = img_buf;
//        imwrite(img_addr, hdst);
        imwrite(img_addr, roi_show);
        waitKey(0);

  }


  return 0;
  }

Mat lane_perspective(Mat src)
{
    vector<Point> not_a_rect_shape;
    not_a_rect_shape.push_back(Point(400,233));
    not_a_rect_shape.push_back(Point(800,233));
    not_a_rect_shape.push_back(Point(894,369));
    not_a_rect_shape.push_back(Point(305,369));

   const Point* point = &not_a_rect_shape[0];
   int n = (int )not_a_rect_shape.size();
   Mat draw = src.clone();
   polylines(draw, &point, &n, 1, true, Scalar(0, 255, 0), 3, CV_AA);
   imshow( "draw", draw);

   //  topLeft, topRight, bottomRight, bottomLeft
  Point2f src_vertices[4];
  src_vertices[0] = not_a_rect_shape[0];
  src_vertices[1] = not_a_rect_shape[1];
  src_vertices[2] = not_a_rect_shape[2];
  src_vertices[3] = not_a_rect_shape[3];

  Point2f dst_vertices[4];
  dst_vertices[0] = Point(400,233);
  dst_vertices[1] = Point(800,233);
  dst_vertices[2] = Point(700,369);
  dst_vertices[3] = Point(500,369);

  Mat warpMatrix = getPerspectiveTransform(src_vertices, dst_vertices);
  Mat rotated;
  warpPerspective(src, rotated, warpMatrix, rotated.size(), INTER_LINEAR, BORDER_CONSTANT);

  // Display the image
   namedWindow( "warp perspective");
   imshow( "warp perspective",rotated);

   return rotated;
}

Mat sobel(Mat src)
{
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

  Sobel( src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
//   Sobel( detected_edges, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
//   convertScaleAbs( grad_y, abs_grad_y );

//   /// Total Gradient (approximate)
//   addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  return abs_grad_x;
}

Mat set_ROI(Mat src)
{
    Rect roi(500, 200, 200, 169);
    Mat roi_of_image = src(roi);
    return roi_of_image;
}
