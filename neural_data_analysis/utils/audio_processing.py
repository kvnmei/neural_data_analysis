from moviepy.editor import VideoFileClip
from pathlib import Path
from typing import Union


def extract_audio_from_video(
    video_path: Union[str, Path], audio_path: Union[str, Path]
) -> None:
    """
    Extract audio from video file and save to file.

    Parameters:
        video_path: path to video file
        audio_path: path to save audio file

    Returns:
        None
    """
    # VideoFileClip supports any format supported by ffmpeg: .mp4, .avi, .mov
    with VideoFileClip(video_path) as video_clip:
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)
        print(f"Audio clip saved successfully to {audio_path})")
