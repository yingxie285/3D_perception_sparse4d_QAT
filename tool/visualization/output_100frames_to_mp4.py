#!/usr/bin/env python3
"""Combine numbered Sparse4D frame folders into an MP4 video."""

import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="将 Sparse4D 逐帧图片合成为 MP4")
    parser.add_argument("input_dir", nargs="?", type=Path, default=Path("/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/output_100frames"), help="包含 0、1、2 等帧目录的 output_100frames")
    parser.add_argument("output_mp4", nargs="?", type=Path, default=Path("/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/output_100frames.mp4"), help="原有 mp4v 视频的输出路径")
    parser.add_argument("--image", default="four_cameras.jpg", help="每帧选用的图片文件名")
    parser.add_argument("--fps", type=float, default=10.0, help="视频帧率，默认 10")
    args = parser.parse_args()

    h264_mp4 = args.output_mp4.with_name(args.output_mp4.stem + "_h264.mp4")
    if args.output_mp4.resolve() == h264_mp4.resolve():
        parser.error("两个输出文件路径不能相同")

    if args.fps <= 0:
        parser.error("--fps 必须大于 0")
    if not args.input_dir.is_dir():
        parser.error(f"输入目录不存在：{args.input_dir}")

    frame_dirs = sorted(
        (p for p in args.input_dir.iterdir() if p.is_dir() and p.name.isdecimal()),
        key=lambda p: int(p.name),
    )
    if not frame_dirs:
        parser.error("未找到数字命名的帧目录")

    missing = [str(p / args.image) for p in frame_dirs if not (p / args.image).is_file()]
    if missing:
        parser.error("帧图片缺失：\n" + "\n".join(missing))

    try:
        import cv2
    except ImportError as exc:
        parser.error("缺少 OpenCV，请执行：python3 -m pip install opencv-python-headless")
        raise exc

    first = cv2.imread(str(frame_dirs[0] / args.image))
    if first is None:
        parser.error(f"图片无法读取：{frame_dirs[0] / args.image}")
    height, width = first.shape[:2]
    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        parser.error("MP4 编码器无法启动，请检查服务器 OpenCV 的 MP4 编码支持")

    count = 0
    try:
        for folder in frame_dirs:
            path = folder / args.image
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"图片无法读取：{path}")
            if image.shape[:2] != (height, width):
                raise RuntimeError(f"图片尺寸不一致：{path}，预期 {width}x{height}")
            writer.write(image)
            count += 1
    finally:
        writer.release()

    print(f"已生成 {args.output_mp4}，共 {count} 帧，{args.fps:g} FPS，时长 {count / args.fps:.2f} 秒")

    # 从原始逐帧图片直接编码另一份 H.264；不读取或改写上面的 mp4v 文件。
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("原有 MP4 已生成；生成 H.264 版本还需要服务器安装 ffmpeg")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "image2pipe", "-framerate", str(args.fps),
        "-vcodec", "mjpeg", "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23", "-movflags", "+faststart",
        str(h264_mp4),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for folder in frame_dirs:
            with (folder / args.image).open("rb") as image_file:
                shutil.copyfileobj(image_file, process.stdin)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg 生成 H.264 视频失败")
    except Exception:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    print(f"已生成 {h264_mp4}，共 {count} 帧，{args.fps:g} FPS")


if __name__ == "__main__":
    main()
