# 增加opencv的依赖
FIND_PACKAGE( OpenCV REQUIRED )

# 添加头文件和库文件

# INCLUDE_DIRECTORIES( ${PROJECT_SOURSE_DIR}/include )

ADD_EXECUTABLE( lane_detect lane_detect.cpp )
TARGET_LINK_LIBRARIES( lane_detect ${OpenCV_LIBS} 
 )

ADD_EXECUTABLE( lane_detect_circle lane_detect_circle.cpp )
TARGET_LINK_LIBRARIES( lane_detect_circle ${OpenCV_LIBS} 
 )

