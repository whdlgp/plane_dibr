cmake_minimum_required(VERSION 2.8)
project( plane_dibr.out )

set( SRCS )
aux_source_directory( "." SRCS)

set( ld_flag /home/choyg/workspace/opencv_ffmpeg/opencv-3.4.2/install/lib )
set( include_flag /home/choyg/workspace/opencv_ffmpeg/opencv-3.4.2/install/include )
set( libs opencv_img_hash opencv_world OpenMP::OpenMP_CXX )

find_package(OpenMP REQUIRED)

include_directories( ${include_flag} )
link_directories( ${ld_flag} )

add_executable( ${CMAKE_PROJECT_NAME} ${SRCS} )

target_link_libraries( ${CMAKE_PROJECT_NAME} ${libs} )

