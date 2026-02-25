#include "Function.hpp"

void rm::IMUAndImageMatchFunction(io::HikCamera &Hik, io::RTSerial<Packet> &ser, FastQueue<FrameData> &Frames)
{
    while (true) {

        // 读取相机数据
        io::HikCamera::ImageData HikData;

        Hik.read(HikData);
        if(HikData.image.empty()) continue;

        // 读取串口数据
        std::chrono::steady_clock::time_point time ;
        Packet IMU;
        while( true )
        {
            bool ret = ser.readPacket(IMU, time);
            if( !ret ) break;

            const auto& t = ((double)(HikData.time - time).count()) * 1e-6;

            //配对超时

            //串口数据比相机数据早8ms以上
            if( t > 8 ) continue;

            //串口数据比相机数据早5ms以下
            if( t < 5 ) break;

            //配对成功
            cv::Quatd quat( IMU.q3, IMU.q0, IMU.q1, IMU.q2 );
            FrameData frame(HikData.image, quat, HikData.time);

            Frames.push(frame);
            break;
        }

    }
}

void rm::SendMessageToRobot(io::RTSerial<Packet> &ser, float pitch, float yaw, bool fire)
{
    ShootPosi posimsg{pitch, yaw};
    ShootFire firemsg{fire};

    ser.writeBytes(&posimsg,sizeof(posimsg));
    ser.writeBytes(&firemsg,sizeof(firemsg));
}