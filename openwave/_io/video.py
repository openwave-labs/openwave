"""
Video generation engine for OPENWAVE using Taichi GGUI.
"""

from pathlib import Path

import taichi as ti

from openwave._io import render

# Get the video_export directory path
# Navigate from _io module to parent package, then to video_export
package_dir = Path(__file__).parent.parent
export_dir = package_dir / "video_export"

# Initialize VideoManager
manager = ti.tools.VideoManager(output_dir=export_dir, framerate=24, automatic_build=False)


def write_frame(image, current_frame, final_frame):
    """Write a frame to the video manager.
    To be called inside the main simulation loop.

    Args:
        image (numpy.ndarray): The image array to write as a frame.
        current_frame (int): The current frame number for logging.
        final_frame (int): The target frame number to stop recording and
            finalize the video export.
    """
    manager.write_frame(image)
    print(f"\nFrame {current_frame}/{final_frame} is recorded", end="")


def finalize_video():
    """Finalize and export the video files.
    To be called after the main simulation loop.
    """
    print("")
    print("\nExporting .mp4 and .gif videos...")
    manager.make_video(gif=False, mp4=True)
    print("\n================================================================")
    print("VIDEO MANAGER")
    print(f'MP4 video is saved to {manager.get_output_filename(".mp4")}')
    print("================================================================")
    print("")


def export(current_frame, final_frame):
    """Capture current frame and finalize video when reaching the final frame.

    Called during the simulation loop to record each frame. When the current
    frame reaches the final frame, automatically finalizes the video export
    (MP4 and GIF) and stops the simulation window.

    Args:
        current_frame (int): The current frame number being rendered.
        final_frame (int): The target frame number to stop recording and
            finalize the video export.
    """

    # Capture frame for video export
    image = render.window.get_image_buffer_as_numpy()
    write_frame(image, current_frame, final_frame)

    # Set desired frame to stop the simulation
    if current_frame >= final_frame:
        finalize_video()
        render.window.running = False
