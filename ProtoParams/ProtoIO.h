#ifndef ProtoIO_h__
#define ProtoIO_h__



#include "google/protobuf/message.h"

CODE_LIB_EXPORT bool ReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto);
CODE_LIB_EXPORT bool WriteProtoToTextFile(const google::protobuf::Message& proto, const char* filename);

CODE_LIB_EXPORT bool ReadProtoFromBinaryFile(const char* filename, google::protobuf::Message* proto);
CODE_LIB_EXPORT bool WriteProtoToBinaryFile(const google::protobuf::Message& proto, const char* filename);

CODE_LIB_EXPORT bool ReadProtoFromString(const std::string proto_string, google::protobuf::Message* proto);
CODE_LIB_EXPORT bool WriteProtoToString(const google::protobuf::Message& proto, std::string* proto_string);

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename, const int height, const int width, const bool is_color);
cv::Mat ReadImageToCVMat(const string& filename, const int height, const int width);
cv::Mat ReadImageToCVMat(const string& filename, const bool is_color);
cv::Mat ReadImageToCVMat(const string& filename);

static bool matchExt(const std::string & fn, std::string en);
#endif


#endif // ProtoIO_h__
