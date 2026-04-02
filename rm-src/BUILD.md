# RoboMaster 自瞄系统 - 模块化构建

## 模块结构

```
rm-src/
├── CMakeLists.txt          # 主构建文件
├── communicate/            # 通信模块 (rm_communicate)
│   ├── CMakeLists.txt
│   └── logger.cpp
├── HikCamera/              # 相机模块 (rm_hikcamera)
│   ├── CMakeLists.txt
│   └── HikCamera.cpp
├── identify/               # 识别模块 (rm_identify)
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
├── aim/                    # 瞄准模块 (rm_aim)
│   ├── CMakeLists.txt
│   ├── include/
│   ├── src/
│   └── planner/            # 规划子模块 (rm_planner)
│       ├── CMakeLists.txt
│       └── tinympc/
└── example_main.cpp        # 示例程序
```

## 编译方法

### 1. 作为子模块使用
```cmake
add_subdirectory(rm-src)
target_link_libraries(your_target
    rm_aim
    rm_identify
    rm_communicate
    rm_hikcamera
)
```

### 2. 编译示例程序
```bash
cd rm-src
mkdir build && cd build
cmake .. -DBUILD_EXAMPLE=ON
make
```

## 静态库说明

- **rm_communicate**: 串口通信、日志
- **rm_hikcamera**: 海康相机驱动
- **rm_identify**: CV/YOLO 检测、数字分类
- **rm_aim**: PnP解算、追踪、卡尔曼滤波
- **rm_planner**: MPC 轨迹规划
- **tinympcstatic**: TinyMPC 求解器

## 依赖关系

```
rm_aim → rm_identify, rm_planner
rm_planner → tinympcstatic
rm_communicate → (头文件依赖 aim, identify)
```
