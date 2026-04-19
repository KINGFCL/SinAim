// FastQueue.hpp
//
// 基于 moodycamel::BlockingReaderWriterCircularBuffer 改写
// 原作者: Cameron Desrochers (©2013-2021)
// 原始许可: Simplified BSD License
// 原始仓库: https://github.com/cameron314/readerwriterqueue
//
// 修改内容:
//   - 类名改为 FastQueue
//   - 接口风格统一为 push/pop/front/back/at/operator[] 等
//   - 保留阻塞与超时入队/出队的扩展接口（wait_push / wait_pop 等）
//   - 新增 empty()、front()、back()、at()、operator[] 等访问接口
//
// 原始 Simplified BSD License 条款仍然适用，见文件末尾。

#pragma once

#include <utility>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <stdexcept>

// 来自 moodycamel 的轻量级信号量实现
#include "atomicops.h"

#ifndef MOODYCAMEL_CACHE_LINE_SIZE
#define MOODYCAMEL_CACHE_LINE_SIZE 64
#endif

template <typename T>
class FastQueue
{
public:
    typedef T value_type;

public:
    // 构造函数，capacity 为队列最大容量
    explicit FastQueue(std::size_t capacity)
        : maxcap(capacity), mask(), rawData(nullptr), data(nullptr),
          slots_(new moodycamel::spsc_sema::LightweightSemaphore(
              static_cast<moodycamel::spsc_sema::LightweightSemaphore::ssize_t>(capacity))),
          items(new moodycamel::spsc_sema::LightweightSemaphore(0)),
          nextSlot(0), nextItem(0)
    {
        // 将容量向上取整为 2 的幂，以便用位掩码代替取模
        // 算法来源: http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
        --capacity;
        capacity |= capacity >> 1;
        capacity |= capacity >> 2;
        capacity |= capacity >> 4;
        for (std::size_t i = 1; i < sizeof(std::size_t); i <<= 1)
            capacity |= capacity >> (i << 3);
        mask = capacity++;
        rawData = static_cast<char*>(std::malloc(capacity * sizeof(T) + std::alignment_of<T>::value - 1));
        data = align_for<T>(rawData);
    }

    // 移动构造
    FastQueue(FastQueue&& other)
        : maxcap(0), mask(0), rawData(nullptr), data(nullptr),
          slots_(new moodycamel::spsc_sema::LightweightSemaphore(0)),
          items(new moodycamel::spsc_sema::LightweightSemaphore(0)),
          nextSlot(), nextItem()
    {
        swap(other);
    }

    FastQueue(FastQueue const&) = delete;

    // 析构时需手动调用已入队元素的析构函数
    ~FastQueue()
    {
        for (std::size_t i = 0, n = items->availableApprox(); i != n; ++i)
            reinterpret_cast<T*>(data)[(nextItem + i) & mask].~T();
        std::free(rawData);
    }

    FastQueue& operator=(FastQueue&& other) noexcept
    {
        swap(other);
        return *this;
    }

    FastQueue& operator=(FastQueue const&) = delete;

    // 交换两个队列的内容（非线程安全）
    void swap(FastQueue& other) noexcept
    {
        std::swap(maxcap, other.maxcap);
        std::swap(mask, other.mask);
        std::swap(rawData, other.rawData);
        std::swap(data, other.data);
        std::swap(slots_, other.slots_);
        std::swap(items, other.items);
        std::swap(nextSlot, other.nextSlot);
        std::swap(nextItem, other.nextItem);
    }

    // ========================================================================
    //  基础接口（非阻塞），对应原 FastQueue 风格
    // ========================================================================

    // 入队（拷贝）。队列满时返回 false。
    // 生产者线程安全。
    bool push(T const& item)
    {
        if (!slots_->tryWait())
            return false;
        inner_enqueue(item);
        return true;
    }

    // 入队（移动）。队列满时返回 false。
    // 生产者线程安全。
    bool push(T&& item)
    {
        if (!slots_->tryWait())
            return false;
        inner_enqueue(std::move(item));
        return true;
    }

    // 出队并通过引用返回元素。队列空时返回 false。
    // 消费者线程安全。
    bool pop(T& dst)
    {
        if (!items->tryWait())
            return false;
        inner_dequeue(dst);
        return true;
    }

    // 仅弹出队首元素（不返回值）。队列空时返回 false。
    // 消费者线程安全。
    bool pop()
    {
        if (!items->tryWait())
            return false;
        inner_pop();
        return true;
    }

    // 队列是否为空（近似判断，线程安全）
    bool empty() const
    {
        return items->availableApprox() == 0;
    }

    // 返回当前队列中的元素数量（近似值，线程安全）
    std::size_t size() const
    {
        return items->availableApprox();
    }

    // 返回队列最大容量
    std::size_t max_capacity() const
    {
        return maxcap;
    }

    // 窥视队首（最早入队的元素），不弹出。
    // 队列为空时返回 nullptr。消费者线程安全。
    T* peek()
    {
        if (!items->availableApprox())
            return nullptr;
        return inner_peek();
    }

    // 返回队首元素的引用（最早入队的元素）。
    // 注意：调用前请确保队列非空，否则行为未定义。
    T& back()
    {
        return reinterpret_cast<T*>(data)[nextItem & mask];
    }

    // 返回队尾元素的引用（最近入队的元素）。
    // 注意：调用前请确保队列非空，否则行为未定义。
    T& front()
    {
        return reinterpret_cast<T*>(data)[(nextSlot - 1) & mask];
    }

    // 按索引访问（0 = 最早入队的元素）。不做边界检查。
    T& operator[](std::size_t index)
    {
        return reinterpret_cast<T*>(data)[(nextItem + index) & mask];
    }

    // 按索引访问（带边界检查）。越界时抛出 std::out_of_range。
    T& at(std::size_t index)
    {
        std::size_t current_size = items->availableApprox();
        if (index >= current_size)
            throw std::out_of_range("FastQueue::at - index out of range!");
        return reinterpret_cast<T*>(data)[(nextItem + index) & mask];
    }

    // ========================================================================
    //  阻塞接口（扩展），来自原 moodycamel 实现
    // ========================================================================

    // 阻塞入队（拷贝），直到有空位
    void wait_push(T const& item)
    {
        while (!slots_->wait());
        inner_enqueue(item);
    }

    // 阻塞入队（移动），直到有空位
    void wait_push(T&& item)
    {
        while (!slots_->wait());
        inner_enqueue(std::move(item));
    }

    // 带超时的入队（拷贝），超时返回 false
    bool wait_push_timed(T const& item, std::int64_t timeout_usecs)
    {
        if (!slots_->wait(timeout_usecs))
            return false;
        inner_enqueue(item);
        return true;
    }

    // 带超时的入队（移动），超时返回 false
    bool wait_push_timed(T&& item, std::int64_t timeout_usecs)
    {
        if (!slots_->wait(timeout_usecs))
            return false;
        inner_enqueue(std::move(item));
        return true;
    }

    // 带超时的入队（拷贝），接受 std::chrono::duration
    template <typename Rep, typename Period>
    bool wait_push_timed(T const& item, std::chrono::duration<Rep, Period> const& timeout)
    {
        return wait_push_timed(item,
            std::chrono::duration_cast<std::chrono::microseconds>(timeout).count());
    }

    // 带超时的入队（移动），接受 std::chrono::duration
    template <typename Rep, typename Period>
    bool wait_push_timed(T&& item, std::chrono::duration<Rep, Period> const& timeout)
    {
        return wait_push_timed(std::move(item),
            std::chrono::duration_cast<std::chrono::microseconds>(timeout).count());
    }

    // 阻塞出队，直到有元素可用
    void wait_pop(T& dst)
    {
        while (!items->wait());
        inner_dequeue(dst);
    }

    // 带超时的出队，超时返回 false
    bool wait_pop_timed(T& dst, std::int64_t timeout_usecs)
    {
        if (!items->wait(timeout_usecs))
            return false;
        inner_dequeue(dst);
        return true;
    }

    // 带超时的出队，接受 std::chrono::duration
    template <typename Rep, typename Period>
    bool wait_pop_timed(T& dst, std::chrono::duration<Rep, Period> const& timeout)
    {
        return wait_pop_timed(dst,
            std::chrono::duration_cast<std::chrono::microseconds>(timeout).count());
    }

private:
    template <typename U>
    void inner_enqueue(U&& item)
    {
        std::size_t i = nextSlot++;
        new (reinterpret_cast<T*>(data) + (i & mask)) T(std::forward<U>(item));
        items->signal();
    }

    template <typename U>
    void inner_dequeue(U& item)
    {
        std::size_t i = nextItem++;
        T& element = reinterpret_cast<T*>(data)[i & mask];
        item = std::move(element);
        element.~T();
        slots_->signal();
    }

    T* inner_peek()
    {
        return reinterpret_cast<T*>(data) + (nextItem & mask);
    }

    void inner_pop()
    {
        std::size_t i = nextItem++;
        reinterpret_cast<T*>(data)[i & mask].~T();
        slots_->signal();
    }

    template <typename U>
    static inline char* align_for(char* ptr)
    {
        const std::size_t alignment = std::alignment_of<U>::value;
        return ptr + (alignment - (reinterpret_cast<std::uintptr_t>(ptr) % alignment)) % alignment;
    }

private:
    std::size_t maxcap;    // 实际容量（非 2 的幂）
    std::size_t mask;      // 位掩码，用于快速取模
    char* rawData;         // 原始内存
    char* data;            // 对齐后的内存

    std::unique_ptr<moodycamel::spsc_sema::LightweightSemaphore> slots_;  // 空闲槽位信号量
    std::unique_ptr<moodycamel::spsc_sema::LightweightSemaphore> items;   // 已入队元素信号量

    // 缓存行填充，避免 false sharing
    char cachelineFiller0[MOODYCAMEL_CACHE_LINE_SIZE
        - sizeof(char*) * 2
        - sizeof(std::size_t) * 2
        - sizeof(std::unique_ptr<moodycamel::spsc_sema::LightweightSemaphore>) * 2];

    std::size_t nextSlot;  // 下一个入队位置（生产者使用）
    char cachelineFiller1[MOODYCAMEL_CACHE_LINE_SIZE - sizeof(std::size_t)];
    std::size_t nextItem;  // 下一个出队位置（消费者使用）
};



// ============================================================================
// 原始许可证 (Simplified BSD License):
//
// Copyright (c) 2013-2021, Cameron Desrochers
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// ============================================================================

