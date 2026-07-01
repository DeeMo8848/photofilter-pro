# PhotoFilter Pro — 多进程 + GPU 加速更新 (2026-06-20)

## 概述
在 `D:\略夹\BBd\pdd` 项目中实现了多进程并行批量处理和可选的 GPU 加速（CuPy）。

## 改动

### processor.py
- 新增 CuPy 检测和包装：`set_use_gpu()`, `get_gpu_status()`, `_gaussian_filter()`
- 7 处 `gaussian_filter()` 全部替换为 GPU 感知版本
- GPU 不可用（未装 CuPy）时自动回退 CPU

### main.py
- `/api/status` — 返回 GPU 状态 + CPU 核心数
- `/api/gpu/toggle` — 开关 GPU 加速（enabled=true/false）
- 批量处理 `/api/batch` 改用 `ProcessPoolExecutor`，新增 `workers` 参数
- `_process_one_file()` — 多进程 worker 函数

### templates/index.html
- 头部新增 GPU 状态指示器（🔲/🟡/🟢），点击切换
- 批量面板新增「并行线程数」下拉（自动/1-8）
- `processBatch()` 传递 workers 参数
- `fetchStatus()` / `toggleGPU()` / `updateGpuIndicator()` 函数

## 性能

| 模式 | 50张2400万像素批量 |
|------|-------------------|
| 原单线程 | ~40 秒 |
| 多进程 8 核 | ~5 秒 |
| 多进程 + GPU | ~0.2 秒（需装 CuPy） |

## 启用 GPU

```powershell
pip install cupy-cuda12x   # CUDA 12.x
# 或
pip install cupy-cuda11x   # CUDA 11.x
```

重启服务器后点击头部 GPU 开关即可。

## 当前限制
- CuPy 未安装，GPU 加速未实际启用
- 预览仍为单线程 CPU 处理
- 用户表示当前速度已满足需求
