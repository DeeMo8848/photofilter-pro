# PhotoFilter Pro — 图片调色批量处理

本地图片滤镜批量调色工具，支持 GPU 加速（CuPy），基于 FastAPI + 原生 JS。

## 功能

- **实时预览** — 上传图片后即时预览调色效果
- **批量处理** — 多图批量应用同一滤镜配置，多进程并行
- **导出下载** — 处理后图片可单张下载或打包 ZIP 下载
- **GPU 加速** — 支持 NVIDIA GPU（CuPy），处理速度显著提升
- **滤镜参数** — 曝光、对比度、饱和度、色温、色调、锐化、暗角等完整调色控制
- **预设管理** — 保存/加载调色预设，快速复用

## 技术栈

- 后端：Python 3 + FastAPI + Pillow + NumPy + SciPy
- GPU 加速：CuPy（NVIDIA GPU，可选）
- 前端：原生 HTML/CSS/JS
- 队列：asyncio + ProcessPoolExecutor 多进程批量

## 快速开始

```bash
pip install -r requirements.txt
# GPU 加速（可选）
pip install cupy-cuda12x
python main.py
```

或双击 `start.bat`，自动安装依赖并打开 `http://localhost:8899`。

## 目录结构

```
├── main.py              # FastAPI 后端
├── processor.py         # 图像处理引擎（numpy/scipy/cupy）
├── templates/
│   └── index.html       # 前端界面
├── requirements.txt
├── start.bat            # Windows 一键启动
├── uploads/             # 上传缓存
├── outputs/             # 处理输出
└── presets/             # 调色预设
```
