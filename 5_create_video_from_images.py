import cv2
import os

def create_video_from_images(image_folder, output_video_path, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # Ensure images are in order

    if len(images) == 0:
        print("No images found in the specified folder.")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved to {output_video_path}")

# Parameters
image_folder = "/home/ailab/git/AUE8088_PA2/jaehwan/AUE8088-PA2/runs/detect_test_1/test_results_240614_1pixel"
output_video_path = "/home/ailab/git/AUE8088_PA2/jaehwan/AUE8088-PA2/runs/detect_test_1/detection_results_video.mp4"
fps = 30  # Frames per second

# Create the video
create_video_from_images(image_folder, output_video_path, fps)
