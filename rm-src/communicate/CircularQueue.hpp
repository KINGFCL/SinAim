#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

template <typename T>
class CircularQueue {
private:
    // 1. 使用 unique_ptr 管理纯粹的字节数组（生内存）
    // uint8_t[] 确保了它只是内存，不会触发 T 的构造函数
    std::unique_ptr<uint8_t[]> raw_memory;
    
    // 2. 指向这块内存的 T 类型指针，方便我们进行偏移计算
    T* data;

    size_t capacity;
    size_t head;
    size_t tail;
    size_t count; // 引入 count 让判断空/满以及最终清理变得非常简单

public:
    /**
     * 构造函数：预先分配整块内存
     */
    CircularQueue(int k) 
        : capacity(k), head(0), tail(0), count(0) {
        
        if (k <= 0) throw std::invalid_argument("Capacity must be > 0");

        // 一次性分配容纳 k 个 T 对象的纯字节内存
        // make_unique<uint8_t[]> 天然支持数组释放 (delete[])
        raw_memory = std::make_unique<uint8_t[]>(capacity * sizeof(T));
        
        // 将纯字节指针转换为 T* 指针
        data = reinterpret_cast<T*>(raw_memory.get());
    }
    
    CircularQueue(const CircularQueue&) = delete;
    CircularQueue& operator=(const CircularQueue&) = delete;

    CircularQueue& operator=(CircularQueue&& other) noexcept {
        if (this != &other) {
            // 1. 先清理自身现有的元素（调用析构）
            while (!isEmpty()) {
                deQueue();
            }
            
            // 2. 接管 other 的资源
            raw_memory = std::move(other.raw_memory);
            data = other.data;
            capacity = other.capacity;
            head = other.head;
            tail = other.tail;
            count = other.count;

            // 3. 将 other 掏空
            other.data = nullptr;
            other.capacity = 0;
            other.count = 0;
            other.head = 0;
            other.tail = 0;
        }
        return *this;
    }

    // 显式实现移动构造函数
    CircularQueue(CircularQueue&& other) noexcept 
        : raw_memory(std::move(other.raw_memory)), data(other.data),
        capacity(other.capacity), head(other.head), tail(other.tail), count(other.count) {
        // 关键：将 moved-from 对象的状态置空，防止它的析构函数作妖
        // 3. 将 other 掏空
        other.data = nullptr;
        other.capacity = 0;
        other.count = 0;
        other.head = 0;
        other.tail = 0;
    }

    /**
     * 析构函数：极为重要！
     * unique_ptr 会自动释放底层那块字节内存，
     * 但我们需要手动调用队列中**仍存活的对象**的析构函数！
     */
    ~CircularQueue() {
        while (!isEmpty()) {
            deQueue(); // 复用 deQueue 逻辑，里面包含了显式的析构调用
        }
        // 循环结束后，raw_memory 离开作用域，自动回收这块连续内存
    }

    /**
     * 入队：使用 Placement new
     */
    void enQueue(const T& value) {
        if (isFull()) this->deQueue();
        
        T* target = reinterpret_cast<T*>(&raw_memory[tail * sizeof(T)]);
        new (target) T(value);
        
        tail = (tail + 1) % capacity;
        count++;
    }

    void enQueue(T&& value) {
        if (isFull()) this->deQueue();
        
        // 统一使用严谨的生内存偏移计算
        T* target = reinterpret_cast<T*>(&raw_memory[tail * sizeof(T)]);
        new (target) T(std::move(value));
        
        tail = (tail + 1) % capacity;
        count++;
    }

    /**
     * 出队：手动调用析构函数
     */
    bool deQueue() {
        if (isEmpty()) return false;
        
        // 另一个核心：对象出队了，它的生命周期结束，必须手动调用析构函数清理其内部资源
        data[head].~T();
        
        head = (head + 1) % capacity;
        count--;
        return true;
    }

    const T& operator[](size_t index) const { 
    // 队列为空或索引越界时，明确抛出异常
        if (index >= static_cast<size_t>(count)) {
            throw std::out_of_range("Index out of bounds or queue is empty");
        }
        return data[(head + index) % capacity]; 
    }

    void clear() {
        while (count > 0) {
            data[head].~T();
            head = (head + 1) % capacity;
            count--;
        }
        this->head = 0;
        this->tail = 0;
        this->count = 0;
    }

    const T* Front() const {
        if (isEmpty()) return nullptr;
        return &data[head];
    }

    const T* Rear() const {
        if (isEmpty()) return nullptr;
        // tail 总是指向下一个要插入的位置，所以队尾元素在 tail 的前一个位置
        size_t rear_index = ( (tail + capacity) -1 ) % capacity;
        return &data[rear_index];
    }

    bool isEmpty() const {
        return count == 0;
    }

    bool isFull() const {
        return count == capacity;
    }
};