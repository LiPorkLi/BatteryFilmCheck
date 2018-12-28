#include "ProtoIO.h"

#if defined(_MSC_VER)
#include <io.h>
#endif

#include <fcntl.h>
#include <stdint.h>
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/descriptor.h>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdio.h>

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

using namespace std;
using std::fstream;
using std::ios;

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.


bool ReadProtoFromTextFile(const char* filename, Message* proto) {
	FILE* ff = fopen(filename, "r");
	if (ff == nullptr) return false;
	FileInputStream* input = new FileInputStream(ff->_file);
	bool success = google::protobuf::TextFormat::Parse(input, proto);
	delete input;
	fclose(ff);
	return success;
}

#ifdef _MSC_VER
bool CODE_LIB_EXPORT ReadProtoFromTextFile(const wchar_t* filename, google::protobuf::Message* proto)
{
	FILE* ff = _wfopen(filename, L"r");
	if (ff == nullptr) return false;
	FileInputStream* input = new FileInputStream(ff->_file);
	bool success = google::protobuf::TextFormat::Parse(input, proto);
	delete input;
	fclose(ff);
	return success;
}
#endif

bool WriteProtoToTextFile(const Message& proto, const char* filename) {
	FILE* ff = fopen(filename, "w");
	if (ff == nullptr) return false;
	FileOutputStream* output = new FileOutputStream(ff->_file);
	bool success = (google::protobuf::TextFormat::Print(proto, output));
	delete output;
	fclose(ff);
	return success;
}

#ifdef _MSC_VER
bool CODE_LIB_EXPORT WriteProtoToTextFile(const google::protobuf::Message& proto, const wchar_t* filename)
{
	FILE* ff = _wfopen(filename, L"w");
	if (ff == nullptr) return false;
	FileOutputStream* output = new FileOutputStream(ff->_file);
	bool success = (google::protobuf::TextFormat::Print(proto, output));
	delete output;
	fclose(ff);
	return success;
}
#endif

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
	FILE* ff = fopen(filename, "r");
	if (ff == nullptr) return false;
	ZeroCopyInputStream* raw_input = new FileInputStream(ff->_file);
	CodedInputStream* coded_input = new CodedInputStream(raw_input);
	coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 1073741823);

	bool success = proto->ParseFromCodedStream(coded_input);

	delete coded_input;
	delete raw_input;
	fclose(ff);
	return success;
}

#ifdef _MSC_VER
bool CODE_LIB_EXPORT ReadProtoFromBinaryFile(const wchar_t* filename, google::protobuf::Message* proto)
{
	FILE* ff = _wfopen(filename, L"w");
	if (ff == nullptr) return false;
	ZeroCopyInputStream* raw_input = new FileInputStream(ff->_file);
	CodedInputStream* coded_input = new CodedInputStream(raw_input);
	coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 1073741823);

	bool success = proto->ParseFromCodedStream(coded_input);

	delete coded_input;
	delete raw_input;
	fclose(ff);
	return success;
}
#endif

bool WriteProtoToBinaryFile(const Message& proto, const char* filename) {
	fstream output(filename, ios::out | ios::trunc | ios::binary);
	return (proto.SerializeToOstream(&output));
}

#ifdef _MSC_VER
bool CODE_LIB_EXPORT WriteProtoToBinaryFile(const google::protobuf::Message& proto, const wchar_t* filename)
{
	fstream output(filename, ios::out | ios::trunc | ios::binary);
	return (proto.SerializeToOstream(&output));
}
#endif

bool ReadProtoFromString(const std::string proto_string, Message* proto) {
	bool success = google::protobuf::TextFormat::ParseFromString(proto_string, proto);
	return success;
}

bool WriteProtoToString(const Message& proto, std::string* proto_string) {
	return (google::protobuf::TextFormat::PrintToString(proto, proto_string));
}

void ClearRepeatedFields(Message* proto)
{
	auto msgDesc = proto->GetDescriptor();
	auto reflection = proto->GetReflection();
	for (int iField = 0; iField < msgDesc->field_count(); ++iField)
	{
		if (msgDesc->field(iField)->label() == google::protobuf::FieldDescriptor::LABEL_REPEATED)
		{
			reflection->ClearField(proto, msgDesc->field(iField));
		}
	}
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
	const int height, const int width, const bool is_color) {
	cv::Mat cv_img;
	int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
		CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
	if (!cv_img_origin.data) {
		LOG(ERROR) << "Could not _open or find file " << filename;
		return cv_img_origin;
	}
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	}
	else {
		cv_img = cv_img_origin;
	}
	return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
	const int height, const int width) {
	return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
	const bool is_color) {
	return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
	return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
	std::string en) {
	size_t p = fn.rfind('.');
	std::string ext = p != fn.npos ? fn.substr(p) : fn;
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	std::transform(en.begin(), en.end(), en.begin(), ::tolower);
	if (ext == en)
		return true;
	if (en == "jpg" && ext == "jpeg")
		return true;
	return false;
}
#endif  // USE_OPENCV
