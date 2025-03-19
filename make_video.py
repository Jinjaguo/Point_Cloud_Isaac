import cv2
import os


def create_video_from_folder(folder_path, output_path, fps=30):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    image_files.sort()

    image_paths = [os.path.join(folder_path, f) for f in image_files]

    # 检查是否有图片
    if not image_paths:
        print("未找到图片文件。")
        return

    # 读取第一张图片获取尺寸
    frame = cv2.imread(image_paths[0])
    if frame is None:
        print("无法读取图片:", image_paths[0])
        return

    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for path in image_paths:
        frame = cv2.imread(path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"skip unreadable image: {path}")

    video.release()
    print("saved in", output_path)


folder_path = './data/experiments/allegro_screwdriver_diff_init_sdf_guided_1./csvgd/trial_1'
output_path = folder_path + '/diff_init_sdf_guided.mp4'
create_video_from_folder(folder_path, output_path)
