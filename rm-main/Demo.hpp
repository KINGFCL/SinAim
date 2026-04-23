#pragma once
#include "Function.hpp"
#include <opencv2/videoio.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

class DemoReader {
public:
    explicit DemoReader(const std::string& path) : cap_(path), done_(false) {}

    bool isOpened() const { return cap_.isOpened(); }
    bool isDone()   const { return done_.load(); }

    void feedQueue(FastQueue<FrameData>& frames) {
        auto base_time = std::chrono::steady_clock::now();
        double accumulated = 0.0;
        int n = 0;

        cv::Mat raw;
        while (cap_.read(raw)) {
            if (raw.total() * raw.elemSize() < sizeof(DoubleQuaternion)) continue;

            DoubleQuaternion dq;
            std::memcpy(&dq, raw.data, sizeof(dq));

            accumulated += dq.time;
            auto frame_time = base_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(accumulated));

            FrameData fd(raw.clone(), cv::Quatd(dq.w, dq.x, dq.y, dq.z), frame_time);
            while (!frames.push(fd))
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            ++n;
        }
        std::printf("[DEMO] finished, total=%d frames\n", n);
        done_.store(true);
    }

private:
    #pragma pack(push, 1)
    struct DoubleQuaternion { double time, w, x, y, z; };
    #pragma pack(pop)

    cv::VideoCapture cap_;
    std::atomic<bool> done_;
};
