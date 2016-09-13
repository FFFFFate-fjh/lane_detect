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

//ROI位置
int ROI_x_l=475;
int ROI_y_l=100;
int ROI_x_r=600;
int ROI_y_r=100;

Mat lane_perspective(Mat src);  //透射投影，不知道相机参数时，可以通过固定梯形上边改变下边到达平行效果
Mat sobel(Mat src);  //sobel计算
Mat set_ROI(Mat src, int ROI_L, int ROI_R);   //设置ROI大小
int autoCanny(Mat &srcmat, Mat &dstmat, float type);  //自动求取阈值参数

int main( int argc, char** argv )
{
    string filename = "data/21/image_0/list.txt"; //使用正则表达生存图片名字的list
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
      sprintf(img_buf, "./data/21/image_0/%s", addr_buf);
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
      equalizeHist( rotated, rotated );

      Mat ROI_img_left = set_ROI(rotated,ROI_x_l,ROI_y_l);
      Mat ROI_img_right = set_ROI(rotated,ROI_x_r,ROI_y_r);

      Mat grad_l, abs_grad_x_l,grad_r, abs_grad_x_r;
      medianBlur(ROI_img_left,grad_l,3);
      abs_grad_x_l = sobel(grad_l);
      medianBlur(ROI_img_right,grad_r,3);
      abs_grad_x_r = sobel(grad_r);

      Mat detected_edges_l, detected_edges_r;

      /// Canny detector
      int thres_l, thres_r,thres_max;
      thres_l=autoCanny(grad_l, detected_edges_l, 1);
      thres_r=autoCanny(grad_r, detected_edges_r, 1);
      thres_max= max(thres_l,thres_r);
      if(thres_max>200)
      {
          ;
      }
      else if(thres_max>100)
          thres_max=thres_max*2;
      else if(thres_max>50)
          thres_max=thres_max*4;
      else if(thres_max>25)
          thres_max=thres_max*8;
      cout<<"thres_max: "<<thres_max<<endl;
      Canny( abs_grad_x_l, detected_edges_l, thres_max, thres_max*3, kernel_size );
      Canny( abs_grad_x_r, detected_edges_r, thres_max, thres_max*3, kernel_size );
      //  Canny( abs_grad_x_l, detected_edges_l, lowThreshold, lowThreshold*ratio, kernel_size );
      //    Canny( abs_grad_x_r, detected_edges_r, lowThreshold, lowThreshold*ratio, kernel_size );

     Mat hdst_l, hdst_r,roi_show;
     cvtColor(detected_edges_l, hdst_l, CV_GRAY2BGR);
     cvtColor(detected_edges_r, hdst_r, CV_GRAY2BGR);
     cvtColor(rotated, roi_show, CV_GRAY2BGR);
     vector<Vec4i> lines_l,lines_r;

     HoughLinesP(detected_edges_l, lines_l, 1, CV_PI/180, 50, 10, 10 );
     HoughLinesP(detected_edges_r, lines_r, 1, CV_PI/180, 50, 10, 10 );
     for( size_t i = 0; i < lines_l.size(); i++ )
     {
       Vec4i l = lines_l[i];
       float k = ((l[0]-l[2])*1.0)/(l[1]-l[3]);
       if(k<0.2 && k >-0.2)
       {
           line( hdst_l, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
           l[0]=l[0]+ROI_x_l;
           l[2]=l[2]+ROI_x_l;
           l[1]=l[1]+ROI_y_l;
           l[3]=l[3]+ROI_y_l;
           line( roi_show, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
       }
     }
     for( size_t i = 0; i < lines_r.size(); i++ )
     {
       Vec4i l = lines_r[i];
       float k = ((l[0]-l[2])*1.0)/(l[1]-l[3]);
       if(k<0.2 && k >-0.2)
       {
           line( hdst_r, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
           l[0]=l[0]+ROI_x_r;
           l[2]=l[2]+ROI_x_r;
           l[1]=l[1]+ROI_y_r;
           l[3]=l[3]+ROI_y_r;
           line( roi_show, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
       }
     }

     imshow("source_l", detected_edges_l);
     imshow("detected lines left", hdst_l);
     imshow("source_r", detected_edges_r);
     imshow("detected lines right", hdst_r);
     imshow("detected lines on rotated", roi_show);
        char img_buf[100];
        sprintf(img_buf, "./output/21/image_0/%d.png", i);
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
  dst_vertices[3] = Point(515,369);

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

Mat set_ROI(Mat src, int ROI_L, int ROI_R)
{
//    Rect roi(500, 200, 200, 169);
//    Rect roi(500, 200, 200, 169);
    Rect roi(ROI_L, ROI_R, 100, 269);
    Mat roi_of_image = src(roi);
    return roi_of_image;
}

int autoCanny(Mat &srcmat,  Mat &dstmat, float type)
{
    // Check input Mat type is CV_8UC1
    if (srcmat.type() != CV_8UC1 || dstmat.type() != CV_8UC1)
        CV_Error(CV_StsUnsupportedFormat, "");

    // CHeck output Mat size equal to input Mat size or not
    if (!CV_ARE_SIZES_EQ(&srcmat, &dstmat)) {
        dstmat.create(srcmat.size(), srcmat.type());
    }

    int total_piexl = 0;
    int nr = srcmat.rows;
    int nc = srcmat.cols;

    // If the input and output mat is store continuous in memory, then loop
    // the Mat just in one rows will be much more quickly.
    if (srcmat.isContinuous() && dstmat.isContinuous()) {
        nr = 1;
        nc = nc * srcmat.rows;
    }

    // Means gray level in image.
    for (int i = 0; i < nr; i++) {
        const uchar *src_data = srcmat.ptr<uchar>(i);
        for (int j = 0; j < nc; j++) {
            total_piexl += *src_data;
        }
    }

    total_piexl /= nr * nc; // means gray level

    // threshold 1 is 0.66 * means gray level
    // threshold 2 is 1.33 * means gray level
    // the following threshould value is calc from experience.
    int thres1 = int(type * total_piexl);
    int thres2 = int(type * 3 * total_piexl);
    cout<<"thres1  "<<thres1<<"  "<<"thres2  "<<thres2<<endl;
//    Canny(cannymat, dstmat, thres1, thres2, kernel_size);
    return thres1;
}


