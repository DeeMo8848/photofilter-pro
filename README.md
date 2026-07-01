# PhotoFilter Pro — 图片调色批量处理

ai跑的本地图片滤镜批量调色工具，支持 GPU 加速（CuPy），有待完善。

## 功能

- **实时预览** — 上传图片后即时预览调色效果
- **批量处理** — 多图批量应用同一滤镜配置，多进程并行
- **导出下载** — 处理后图片可单张下载或打包 ZIP 下载
- **GPU 加速** — 支持 NVIDIA GPU（CuPy），处理速度显著提升
- **滤镜参数** — 曝光、对比度、饱和度、色温、色调、锐化、暗角等完整调色控制
- **预设管理** — 保存/加载调色预设，快速复用

## 快速开始

```bash
pip install -r requirements.txt
# GPU 加速（可选）
pip install cupy-cuda12x
python main.py
```

或双击 `start.bat`，自动安装依赖并打开 `http://localhost:8899`。


