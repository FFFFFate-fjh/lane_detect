#include<iostream>
#include<string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

Mat src, src_gray,src_heq;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";
int ROI_x_l=500;
int ROI_y_l=200;
int ROI_x_r=600;
int ROI_y_r=200;

void CannyThreshold(int, void*);
Mat lane_perspective(Mat src);
Mat sobel(Mat src);
Mat set_ROI(Mat src, int ROI_L, int ROI_R);


void CannyThreshold(int, void*)
{

   Mat rotated = lane_perspective(src_heq);

   /// Apply Histogram Equalization
//   equalizeHist( rotated, rotated );

  /// Reduce noise with a kernel 3x3
//  blur( src_gray, detected_edges, Size(3,3) );
//   medianBlur(rotated,detected_edges,3);
//   GaussianBlur(rotated,detected_edges,Size(3,3),0,0);

   Mat ROI_img_left = set_ROI(rotated,ROI_x_l,ROI_y_l);
//  imshow("ROI_image", ROI_image);
//    waitKey(0);

   Mat grad_l, abs_grad_x_l;
//      medianBlur(rotated,grad,3);
      medianBlur(ROI_img_left,grad_l,3);
   abs_grad_x_l = sobel(grad_l);

/// Canny detector
//  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
  Canny( abs_grad_x_l, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

//  src.copyTo( dst, detected_edges);
//  rotated.copyTo( dst, detected_edges);
//  imshow( window_name, dst );

  Mat hdst, roi_show;
  cvtColor(detected_edges, hdst, CV_GRAY2BGR);
  cvtColor(rotated, roi_show, CV_GRAY2BGR);
  vector<Vec4i> lines;

//  HoughLinesP(detected_edges, lines, 1, CV_PI/180, 100, 1, 10 );
    HoughLinesP(detected_edges, lines, 1, CV_PI/180, 50, 1, 15 );
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
 }


/** @function main */
int main( int argc, char** argv )
{
  /// Load an image
  src = imread( "./data/04/image_1/000000.png" );

//  分割
//  Mat tmp ;
////  cout<<src.rows<<endl;
//  src.rowRange(170,370).colRange(300,926).copyTo(tmp);
////  cout<<tmp.cols<<" "<<tmp.rows<<endl;
//  src.resize(200,626);
//  src=tmp.clone();

  if( !src.data )
  { return -1; }

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );

  equalizeHist( src_gray, src_heq );

//  erode(src_heq,src_heq,Mat(5,5,CV_8U),Point(-1,-1),3);

//  dilate(src_heq,src_heq,Mat(1,1,CV_8U),Point(-1,-1),2);


  /// Create a window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  /// Show the image
  CannyThreshold(0, 0);

  /// Wait until user exit program by pressing a key
  waitKey(0);

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

  Sobel( src, grad_x, ddepth, 1, 0, 3, 1, 0, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
//   Sobel( detected_edges, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
//   convertScaleAbs( grad_y, abs_grad_y );

//   /// Total Gradient (approximate)
//   addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  return abs_grad_x;
}

Mat set_ROI(Mat src, int ROI_L, int ROI_R)
{
//    Rect roi(500, 200, 200, 169);
//    Rect roi(500, 200, 200, 169);
    Rect roi(ROI_L, ROI_R, 200, 169);
    Mat roi_of_image = src(roi);
    return roi_of_image;
}
