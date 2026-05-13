#pragma once

#include "readerwriterqueue.h"
#include <stdexcept>
#include <cstddef>
#include <chrono>

/// FastQueue —— 基于 moodycamel::ReaderWriterQueue 的封装
///
/// 保留原始 FastQueue 的接口语义，底层委托给 ReaderWriterQueue。
///
/// 注意事项：
///   1. ReaderWriterQueue 是严格的 SPSC（单生产者-单消费者）队列，
///      不支持随机访问。因此 operator[]、at()、front()、back()
///      在本实现中 **不再提供**。如果你的业务逻辑依赖随机访问，
///      请继续使用原始的环形缓冲区实现。
///   2. size() 返回的是近似值（size_approx），在并发场景下这是合理的。
///   3. empty() 同样基于 size_approx，语义与原版一致。

template <typename T>
class FastQueue {
public:
    /// 构造函数
    /// @param max_queue_len  队列最大容量（与原版语义一致）
    explicit FastQueue(size_t max_queue_len)
        : queue_(max_queue_len)
    {
    }

    // ── 禁止拷贝 ──
    FastQueue(const FastQueue&) = delete;
    FastQueue& operator=(const FastQueue&) = delete;

    // ── 允许移动 ──
    FastQueue(FastQueue&& other) noexcept = default;
    FastQueue& operator=(FastQueue&& other) noexcept = default;

    /// 弹出队首元素到 dst，队列为空时返回 false
    bool pop(T& dst)
    {
        return queue_.try_dequeue(dst);
    }

    /// 阻塞弹出队首元素到 dst，队列为空时阻塞直到有元素可用
    void wait_pop(T& dst)
    {
        this->queue_.wait_dequeue(dst);
    }

    /// 阻塞弹出队首元素到 dst，超时返回 false
    template <typename Rep, typename Period>
    bool wait_pop(T& dst, const std::chrono::duration<Rep, Period>& timeout)
    {
        return this->queue_.wait_dequeue_timed(dst, timeout);
    }

    /// 弹出队首元素（不获取值），队列为空时返回 false
    bool pop()
    {
        return queue_.pop();
    }

    /// 压入元素（拷贝），队列满时返回 false（不会分配新内存）
    bool push(const T& src)
    {
        return queue_.try_enqueue(src);
    }

    /// 压入元素（移动），队列满时返回 false（不会分配新内存）
    bool push(T&& src)
    {
        return queue_.try_enqueue(std::move(src));
    }

    /// 队列是否为空
    bool empty() const
    {
        return queue_.size_approx() == 0;
    }

    /// 返回队列中元素的（近似）数量
    size_t size() const
    {
        return queue_.size_approx();
    }

    /// 查看队首元素（即下一个将被 pop 的元素）的指针。
    /// 队列为空时返回 nullptr。
    /// 仅消费者线程可调用。
    T* peek()
    {
        return queue_.peek();
    }

    const T* peek() const
    {
        return queue_.peek();
    }

    /// 返回队列在不触发新内存分配的前提下能容纳的最大元素数
    size_t max_capacity() const
    {
        return queue_.max_capacity();
    }


private:
    moodycamel::BlockingReaderWriterQueue<T> queue_;
};


template <typename T>
class FastQueueNoWait {
public:
    /// 构造函数
    /// @param max_queue_len  队列最大容量（与原版语义一致）
    explicit FastQueueNoWait(size_t max_queue_len)
        : queue_(max_queue_len)
    {
    }

    // ── 禁止拷贝 ──
    FastQueueNoWait(const FastQueueNoWait&) = delete;
    FastQueueNoWait& operator=(const FastQueueNoWait&) = delete;

    // ── 允许移动 ──
    FastQueueNoWait(FastQueueNoWait&& other) noexcept = default;
    FastQueueNoWait& operator=(FastQueueNoWait&& other) noexcept = default;

    /// 弹出队首元素到 dst，队列为空时返回 false
    bool pop(T& dst)
    {
        return queue_.try_dequeue(dst);
    }


    /// 弹出队首元素（不获取值），队列为空时返回 false
    bool pop()
    {
        return queue_.pop();
    }

    /// 压入元素（拷贝），队列满时返回 false（不会分配新内存）
    bool push(const T& src)
    {
        return queue_.try_enqueue(src);
    }

    /// 压入元素（移动），队列满时返回 false（不会分配新内存）
    bool push(T&& src)
    {
        return queue_.try_enqueue(std::move(src));
    }

    /// 队列是否为空
    bool empty() const
    {
        return queue_.size_approx() == 0;
    }

    /// 返回队列中元素的（近似）数量
    size_t size() const
    {
        return queue_.size_approx();
    }

    /// 查看队首元素（即下一个将被 pop 的元素）的指针。
    /// 队列为空时返回 nullptr。
    /// 仅消费者线程可调用。
    T* peek()
    {
        return queue_.peek();
    }

    const T* peek() const
    {
        return queue_.peek();
    }

    /// 返回队列在不触发新内存分配的前提下能容纳的最大元素数
    size_t max_capacity() const
    {
        return queue_.max_capacity();
    }
private:
    moodycamel::ReaderWriterQueue<T> queue_;    
};



