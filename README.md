# RGB-D 双视角重投影 · Dual-view RGB-D Reprojection

这个 C++17 小项目把两帧带深度的照片投到一个指定视角。每个像素先用深度和相机内参定位到三维空间，再按位姿换算到目标相机。两帧都投到同一像素时，只混合靠近最前表面的颜色。辅视角覆盖的区域可以补上主视角留下的空洞。在 Fountain 实拍序列留出的 31 个目标帧上，相比最近单视角，双视角融合将平均空洞率从 **23.57% 降至 8.37%**，相对下降 **64.5%**。

![Fountain 目标帧、最近单视角和双视角结果](results/fountain/preview.png)

## 运行

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic main.cpp vendor/tinyrenderer/tgaimage.cpp -Ivendor/tinyrenderer -o rgbd_demo
./rgbd_demo results
```

Windows 装有 MSVC 时可从 **Developer Command Prompt** 编译。

```bat
cl /nologo /std:c++17 /O2 /EHsc /Ivendor\tinyrenderer main.cpp vendor\tinyrenderer\tgaimage.cpp /Fe:rgbd_demo.exe
rgbd_demo.exe results
```

复现实拍验证需要 Python、NumPy 和 Pillow。下载并解压 [Open3D Fountain 样例](https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/SampleFountainRGBDImages.zip)，官方样例 MD5 为 c6c1b2171099f571e2a78d78675df350。把含 image、depth 和 scene/key.log 的目录传给转换脚本。

```bash
python prepare_fountain.py /path/to/extracted/fountain packed
./rgbd_demo --multiview packed results/fountain
python make_preview.py results/fountain
```

转换脚本按官方文件顺序配对 33 帧彩色图、毫米深度与 camera-to-world 位姿，保留缺失深度。样例内参为 fx = fy = 525、cx = 319.5、cy = 239.5，分辨率为 640 × 480。位姿日志的旋转经过数值正交化。转换输出的每个 .rgbd 像素含 3 字节 RGB 和 2 字节小端深度。运行融合时，只有两帧邻帧及相机参数参与投影，目标帧的 RGB 与深度用于计算指标。

## 实拍对照

逐帧以编号 1–31 的每帧为目标，取前后相邻帧为源图。两组单视角对照按相机中心离目标更近和更远区分，使用与融合相同的双线性权重和 4 cm 深度门控。联合投影把两组样本投进同一个深度缓冲，只混合距离最近样本 4 cm 以内的颜色。另做仅补洞对照，把辅助帧重投影填到主帧的空洞处，保留主帧已有结果。

| 方法 | 平均空洞率 | 深度误差 ≤5 cm 的覆盖率 | 共同有效区域 PSNR |
| --- | ---: | ---: | ---: |
| 最近单视角 | 23.57% | 71.28% | 19.37 dB |
| 较远单视角 | 31.16% | 64.90% | 18.84 dB |
| 双视角联合投影 | 8.37% | 85.29% | 20.22 dB |
| 辅视角仅补洞 | 8.37% | 85.35% | 19.37 dB |

先逐帧计算各项指标，再对 31 帧取算术平均。空洞率和深度一致覆盖率的分母为目标帧有实测深度的像素数。PSNR 使用同一掩膜，该区域同时具备目标有效深度和两个单视角的有效预测。两个融合对照的覆盖范围相同，联合投影还会更新两帧都覆盖区域的颜色。逐帧数据见 [multiview.csv](results/fountain/multiview.csv)。

Fountain 提供的是估计相机轨迹。对照覆盖一条实拍序列，使用固定阈值。空洞包含视野差异、遮挡及源深度缺失。两帧都未看到的区域仍会留空，深度不准或位姿偏差可能带来重影。深度一致性以传感器测量为参考。这个项目展示已有重投影和可见性处理方法的实现及评测。

## 合成回归

保留两组独立光线求交场景的单视角对照。六次位姿变化中，最近邻和双线性平均空洞率分别为 11.20% 和 8.40%，深度门控在共同有效区的平均 PSNR 从普通双线性的 23.69 dB 升至 24.26 dB。程序保留同位姿逐像素检查及前后景重叠检查，逐项结果见 [metrics.csv](results/metrics.csv)。安装 Pillow 后运行 python make_preview.py results 可更新合成对照图。

## 来源

- 项目选题参考 [鼠鼠实习妙妙工具](https://github.com/LiuMengxuan04/shushu-internship-tool/)。图像读写复用 [TinyRenderer 的 TGA 模块](https://github.com/ssloy/tinyrenderer/tree/97eb7a480e762285899a93c5d21e70a79e60782e)，原许可证见 vendor/tinyrenderer/LICENSE.txt。重投影、深度门控与评测代码由 Codex 辅助编写和验证。
- Fountain 为 Zhou 和 Koltun 的实拍 RGB-D 序列，引用论文为 *Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras*，SIGGRAPH 2014。[原始数据和标定说明](https://qianyi.info/scenedata.html) · [Open3D 样例文档](https://www.open3d.org/docs/latest/python_api/open3d.data.SampleFountainRGBDImages.html)
