from utils import read_video,save_video
import os
import sys
sys.path.append(os.path.dirname(__file__))

from my_tracker import Trackers

def main():
    #read video
    video_frames=read_video('input_video/08fd33_4.mp4')

    # initialze tracker
    tracker= Trackers('Models/best.pt')

    track= tracker.get_object_track(video_frames,read_from_stub=True,
                                    stub_path='stubs/track_stubs.pk1')
    
    #outpur_video_frames

    output_video_frames=tracker.draw_annotations(video_frames,track)
    

    #save video
    save_video( output_video_frames,'output_videos/output_videos.avi')

if __name__== '__main__':
    main()