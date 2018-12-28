#ifndef __SAFE_QUEUE_H_  
#define __SAFE_QUEUE_H_  

#include <mutex>  
#include <memory>  
#include <functional>  
#include <condition_variable>  

template <typename T>
class safe_queue
{

public:
	safe_queue() : head(new node), tail(head.get()), iPopCount(0), iPushCount(0)
	{
		
	}
	safe_queue(const safe_queue&) = delete;
	safe_queue operator=(const safe_queue&) = delete;
	~safe_queue() = default;

	std::shared_ptr<T> try_pop()
	{
		std::unique_lock<std::mutex> ulkh(head_mut);
		if (head.get() == get_tail())
			return nullptr;
		auto old_head = std::move(head);
		head = std::move(old_head->next);
		iPopCount++;
		return old_head->data;
	}

	std::shared_ptr<T> wait_and_pop()
	{
		std::unique_lock<std::mutex> ulkh(head_mut);
		{
			std::unique_lock<std::mutex> ulkt(tail_mut);
			if (data_cond.wait_for(ulkt, std::chrono::seconds(1), [&]()
			{
				return head.get() != tail;
			}) == /*std::cv_status::timeout*/false)
			{
				return nullptr;
			}
// 			data_cond.wait(ulkt, [&]()
// 			{
// 				return head.get() != tail;
// 			});
		}
		auto old_head = std::move(head);
		head = std::move(old_head->next);
		iPopCount++;
		return old_head->data;
	}

	/*void push(const T& t)
	{
		std::shared_ptr<T> new_data(std::make_shared<T>(std::forward<T>(t)));
		std::unique_ptr<node> new_tail(new node);
		node *p = new_tail->get();
		{
			std::unique_lock<std::mutex> ulkt(tail_mut);
			tail->data = new_data;
			tail->next = std::move(new_tail);
			tail = p;
		}
		data_cond.notify_one();
	}*/

	void push(T &&t)
	{
		std::shared_ptr<T> new_data(std::make_shared<T>(std::forward<T>(t)));
		std::unique_ptr<node> new_tail(new node);
		node *p = new_tail.get();
		{
			std::unique_lock<std::mutex> ulkt(tail_mut);
			tail->data = new_data;
			tail->next = std::move(new_tail);
			tail = p;
		}
		iPushCount++;
		data_cond.notify_one();
	}

	int getPopCount()
	{
		std::unique_lock<std::mutex> ulkh(head_mut);
		return iPopCount;
	}
	int getPushCount()
	{
		std::unique_lock<std::mutex> ulkt(tail_mut);
		return iPushCount;
	}
private:

	struct node
	{
		std::shared_ptr<T> data;
		std::unique_ptr<node> next;
	};

	std::mutex head_mut;
	std::mutex tail_mut;
	std::unique_ptr<node> head;
	node *tail;
	std::condition_variable data_cond;
	int iPopCount, iPushCount;
private:

	node* get_tail()
	{
		std::unique_lock<std::mutex> ulkt(tail_mut);
		return tail;
	}

};
/*
template <typename T>
class threadsafe_queue2<T*>
{

public:

	threadsafe_queue2() : head(new node), tail(head)
	{

	}
	threadsafe_queue2(const threadsafe_queue2&) = delete;
	threadsafe_queue2 operator=(const threadsafe_queue2&) = delete;

	~threadsafe_queue2()
	{
		node *pre;
		for (; head != tail;)
		{
			pre = head;
			head = head->next;
			delete pre;
		}
		delete tail;
	}

	T* try_pop()
	{
		node *old_head = nullptr;
		{
			std::unique_lock<std::mutex> ulkh(head_mut);
			if (head == get_tail())
				return nullptr;
			old_head = head;
			head = head->next;
		}
		T *data = old_head->data;
		delete old_head;
		return data;
	}


	T* wait_and_pop()
	{
		node *old_head = nullptr;
		{
			std::unique_lock<std::mutex> ulkh(head_mut);
			{
				std::unique_lock<std::mutex> ulkt(tail_mut);
				data_cond.wait(ulkt, [&]()
				{
					return head != tail;
				});
			}
			old_head = head;
			head = head->next;
		}
		T *data = old_head->data;
		delete old_head;
		return data;
	}

	void push(T *t)
	{
		node *new_tail = new node;
		{
			std::unique_lock<std::mutex> ulkt(tail_mut);
			tail->data = t;
			tail->next = new_tail;
			tail = new_tail;
		}
		data_cond.notify_one();
	}

private:

	struct node
	{
		T *data;
		node *next;
	};

	std::mutex head_mut;
	std::mutex tail_mut;
	node *head;
	node *tail;
	std::condition_variable data_cond;

private:

	node* get_tail()
	{
		std::unique_lock<std::mutex> ulkt(tail_mut);
		return tail;
	}

};*/

#endif