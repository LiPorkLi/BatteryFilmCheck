#pragma once
#include "proto/param.pb.h"
#include "ProtoIO.h"
#include <mutex>

class CODE_LIB_EXPORT ParamHelperTool
{
public:
	static std::mutex& getFieldMutex(MainParam::param* pMainParam, const std::string& fieldName);
	static std::mutex& getMutex();
private:
	static std::map<std::pair<MainParam::param*, std::string>, std::mutex> m_gloabDataMtx;
};

class CODE_LIB_EXPORT ParamIO
{
public:
	bool ReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto);
	bool WriteProtoToTextFile(const google::protobuf::Message& proto, const char* filename);

	bool ReadProtoFromBinaryFile(const char* filename, google::protobuf::Message* proto);
	bool WriteProtoToBinaryFile(const google::protobuf::Message& proto, const char* filename);

	bool ReadProtoFromString(const std::string proto_string, google::protobuf::Message* proto);
	bool WriteProtoToString(const google::protobuf::Message& proto, std::string* proto_string);
};

template <class T>
class ParamHelper
{
public:
	ParamHelper<T>(MainParam::param* pMainParam)
	{
		m_dataName = T::descriptor()->name();
		m_pMainParam = pMainParam;
		m_bChanged = false;
		update(m_bChanged);
		//printf("initia.. %s\n", m_dataName.c_str());
	}

	ParamHelper<T>(MainParam::param* pMainParam, T individData)
	{
		m_dataName = T::descriptor()->name();
		m_pMainParam = pMainParam;
		m_individData = individData;
		m_bChanged = true;
	}
	~ParamHelper<T>()
	{
		if (m_bChanged)
		{
			update(m_bChanged);
		}
	}

	T* getPtr()
	{
		if (m_lck.owns_lock() == false)
		{
			m_lck = std::unique_lock<std::mutex>(getFieldMutex());
		}
		m_bChanged = true;
		return &m_individData;
	}
	const T& getRef() const
	{
		return m_individData;
	}

	//T* operator ->()
	//{
	//	if (m_lck.owns_lock() == false)
	//	{
	//		m_lck = std::unique_lock<std::mutex>(getFieldMutex());
	//	}
	//	m_bChanged = true;
	//	return &m_individData;
	//}
	//const T* operator ->() const
	//{
	//	return &m_individData;
	//}
	//T& operator* ()
	//{
	//	if (m_lck.owns_lock() == false)
	//	{
	//		m_lck = std::unique_lock<std::mutex>(getFieldMutex());
	//	}
	//	m_bChanged = true;
	//	return m_individData;
	//}
	const T& operator* () const
	{
		return m_individData;
	}

	void operator= (T &data)
	{
		if (m_lck.owns_lock() == false)
		{
			m_lck = std::unique_lock<std::mutex>(getFieldMutex());
		}
		m_individData = data;
		m_bChanged = true;
	}

	ParamHelper<T>(const ParamHelper<T>&) = delete;
	ParamHelper<T>& operator= (const ParamHelper<T> &data) = delete;
private:
	std::unique_lock<std::mutex> m_lck;

	MainParam::param* m_pMainParam;
	std::string m_dataName;
	T m_individData;
	bool m_bChanged;

	std::mutex& getFieldMutex()
	{
		return ParamHelperTool::getFieldMutex(m_pMainParam, m_dataName);
	}

	void update(bool save = true)
	{
		if (save)
		{
			std::string* targetMsg;
			{
				std::unique_lock<std::mutex> lck(ParamHelperTool::getMutex());
				targetMsg = &((*m_pMainParam->mutable_param_list())[m_dataName]);
			}
			WriteProtoToString(m_individData, targetMsg);
			m_lck.unlock();
		}
		else
		{
			if (m_pMainParam->param_list().count(m_dataName) == 0)
			{
				if (m_lck.owns_lock() == false)
				{
					m_lck = std::unique_lock<std::mutex>(getFieldMutex());
				}
				m_bChanged = true;
				return;
			}
			std::unique_lock<std::mutex> lck(getFieldMutex());
			ReadProtoFromString(m_pMainParam->param_list().at(m_dataName), &m_individData);
		}
	}
};
