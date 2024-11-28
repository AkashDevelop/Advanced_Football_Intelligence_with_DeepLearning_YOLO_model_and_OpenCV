from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
sys.path.append('../')
from utils.bbox_utils import get_center_of_bbox, get_bbox_width


class Trackers:

    def __init__(self,model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()

    def detect_frame(self,frames):
        batch_size=20
        detection=[]

        for i in range(0,len(frames),batch_size):
            detection_batch=self.model.predict(frames[i:i+batch_size],conf=0.1)
            detection+=detection_batch
            
        return detection

    def get_object_track(self,frames,read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks=pickle.load(f)

            return tracks

        detection=self.detect_frame(frames)

        tracks={
            "player":[],
            "referee":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detection):
            class_name=detection.names
            class_name_inv={v:k for k,v in class_name.items()}

            #convert to supervison detection format

            detection_supervision= sv.Detections.from_ultralytics(detection)

            #convert goalkeeper to player

            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if class_name[class_id]=='goalkeeper':
                    detection_supervision.class_id[object_ind]=class_name_inv['player']

            #track objects

            detection_with_tracks=self.tracker.update_with_detections(detection_supervision)

            tracks['player'].append({})
            tracks['referee'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox=frame_detection[0].tolist()
                class_id=frame_detection[3]
                track_id=frame_detection[4]

                if class_id==class_name_inv['player']:
                    tracks['player'][frame_num][track_id]={'bbox':bbox}

                if class_id==class_name_inv['referee']:
                    tracks['referee'][frame_num][track_id]={'bbox':bbox}

            for frame_detection in detection_supervision:
                bbox=frame_detection[0].tolist()
                class_id=frame_detection[3]

                if class_id==class_name_inv['ball']:
                    tracks['ball'][frame_num][1]={'bbox':bbox}

        if stub_path is not None:
            with open (stub_path,'wb') as f:
                pickle.dump(tracks,f)
    

        return tracks
    
    def draw_elipse(self,frame,bbox,color,track_id=None):
        y2=int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width=get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width=40
        rectangle_height=20
        x1_rect= x_center - rectangle_width//2
        x2_rect= x_center + rectangle_width//2
        y1_rect= (y2- rectangle_height//2)+15
        y2_rect=(y2+ rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(frame,
                           (int(x1_rect),int(y1_rect)),
                           (int(x2_rect),int(y2_rect)),
                           color,
                           cv2.FILLED)

            x1_text= x1_rect+12
            if track_id> 99:
                x1_text -=10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2

            )

        return frame
    
    def draw_triangle(self,frame,bbox,color):
        y=int(bbox[1])
        x,_=get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])

        cv2.drawContours(frame, [triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0),2)

        return frame


    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []  # Initialize list to store annotated frames

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # Work with a copy of the frame
            player_dict = tracks['player'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referee'][frame_num]

        # Draw ellipse for each player
            for track_id, player in player_dict.items():
                if 'bbox' in player:  # Ensure bbox exists
                    frame = self.draw_elipse(frame, player['bbox'], (255, 255, 0), track_id)
                else:
                    print(f"Warning: 'bbox' not found for player with track_id {track_id} on frame {frame_num}")

              # Draw ellipse for each ball (using a unique color, e.g., green)
            for track_id, ball in ball_dict.items():
                if 'bbox' in ball:  # Ensure bbox exists
                    frame = self.draw_elipse(frame, ball['bbox'], (255, 6, 180), track_id)  # red for ball
                else:
                    print(f"Warning: 'bbox' not found for ball with track_id {track_id} on frame {frame_num}")

        # Draw ellipse for each referee (using black)
            for track_id, referee in referee_dict.items():
                if 'bbox' in referee:  # Ensure bbox exists
                    frame = self.draw_elipse(frame, referee['bbox'], (191, 137, 241), track_id)  # purple for referee
                else:
                    print(f"Warning: 'bbox' not found for referee with track_id {track_id} on frame {frame_num}")

        # Append the processed frame to the output list
            output_video_frames.append(frame)

        return output_video_frames  # Return the list of annotated frames

