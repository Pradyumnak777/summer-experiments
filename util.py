import cv2
import os

def convert_vid_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    current_frame = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break

        # Save the frame as an image file
        file_name = os.path.join(output_folder, f"frame_{current_frame:04d}.jpg")
        cv2.imwrite(file_name, frame)
        
        current_frame += 1
    
    video.release()
    print("Done!")
    
if __name__ == "__main__":
    convert_vid_to_frames("bleach_corrected_DPHM_Sox2_MCP_halo549_snap646-01_MIP_merged (1).avi", "output_frames")
    