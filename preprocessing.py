# =============================Importing the necessary libraries=============================
import json
import cv2
import os
import functools
import threading
import click

import pandas as pd
import numpy as np
import mediapipe as mp

from tqdm import tqdm
from typing import TypeVar


class MediapipeDataset():
    # =============================Configs=============================
    def __init__(
            self, 
            root: str, 
            max_gloss_numb: int, 
            n_threads: int
            ) -> None:
        self.MP_HOLISTIC = mp.solutions.holistic

        self.ROOT = root
        self.INDEX_FILE = f'{root}/WLASL_v0.3.json'
        self.MISSING_FILE = f'{root}/missing.txt'

        self.MAX_POSE_SIZE = 33
        self.MAX_HAND_SIZE = 21
        self.MAX_FACE_SIZE = 468
        self.MAX_GLOSS_NUMB = max_gloss_numb
        self.N_THREADS = n_threads
    
    # ==============Сreating videos information dataframe==============
    def __create_df(self) -> None:
        fh = open(self.MISSING_FILE, 'r')
        missing = fh.read()
        fh.close()
        missing = missing.split("\n")
        missing = [x.strip() for x in missing]
        # all the information needed to process the video in the videos folder.
        # The table contains:
            # gloss: Glossary
            # instances: Video Instances
        root = json.load(open(self.INDEX_FILE))
        rows = []
        gloss_unique = set()
        for ri in range(len(root)):
            item = root[ri]
            gloss = item['gloss'] # target word 
            gloss_unique.add(gloss)
            instances = item['instances'] # videos information
            for inst in instances:
                video_id = str(inst['video_id'])
                if str(video_id) in missing: # skip if video is missing
                    continue
                frame_start = inst['frame_start']
                frame_end = inst['frame_end']
                fps = inst['fps']
                bbox_0 = inst['bbox'][0]
                bbox_1 = inst['bbox'][1]
                bbox_2 = inst['bbox'][2]
                bbox_3 = inst['bbox'][3]
                signer_id = inst['signer_id'] + 1000000
                path = f'train_landmark_files/{signer_id}/{video_id}.parquet'
                split = inst['split']
                rw = {
                    'path': path, 
                    'participant_id': signer_id, 
                    'sequence_id': video_id, 
                    'sign': gloss, 
                    'video_id': str(video_id), 
                    'fps': fps, 
                    'frame_start': frame_start, 
                    'frame_end': frame_end, 
                    'bbox_0': bbox_0, 
                    'bbox_1': bbox_1, 
                    'bbox_2': bbox_2, 
                    'bbox_3': bbox_3, 
                    'split': split,
                    }
                rows.append(rw)
            
        self.video_df = pd.DataFrame(rows) # create dataframe 
        limited_gloss = set(
            self.video_df['sign']
            .value_counts().index
            .to_list()[:self.MAX_GLOSS_NUMB]
            )
        self.video_df = self.video_df[self.video_df['sign'].isin(limited_gloss)] # sort dataframe 
        self.video_df.to_csv(f'{self.ROOT}/video.csv', index=False)

    # =============================Structure add functions=============================
    def __not_zero_add(self, frame: int, a_type: str, landmark: TypeVar('T')) -> TypeVar('T'):
        return {'frame': frame, 
                'row_id': f'{frame}-{a_type}-{landmark[0]}', 
                'type': a_type, 
                'landmark_index': landmark[0], 
                'x': landmark[1].x, 'y': landmark[1].y, 'z': landmark[1].z}

    def __zero_add(self, frame, a_type, idx):
        return {'frame': frame, 
                'row_id': f'{frame}-{a_type}-{idx}', 
                'type': a_type, 
                'landmark_index': idx, 
                'x': np.NaN, 'y': np.NaN, 'z': np.NaN}

    # =============================Frame-by-frame video recording in parquet format=============================
    def __process_video(self, video_idx: int) -> None:
        video_id = self.video_df['video_id'].iloc[video_idx]
        signer_id = self.video_df['participant_id'].iloc[video_idx]
        bbox = [
            self.video_df['bbox_0'].iloc[video_idx], 
            self.video_df['bbox_1'].iloc[video_idx], 
            self.video_df['bbox_2'].iloc[video_idx], 
            self.video_df['bbox_3'].iloc[video_idx],
            ]
        video_file = f'{self.ROOT}/videos/{video_id}.mp4'
        
        cap = cv2.VideoCapture(video_file) # read video frame
        rows = []
        frame = 0
        with self.MP_HOLISTIC.Holistic(
            model_complexity=0, # complexity of the pose landmark model: 0, 1 or 2. 
            # minimum confidence value ([0.0, 1.0]) from the person-detection model 
                # for the detection to be considered successful
            min_detection_confidence=0.5,
        ) as holistic:
            while cap.isOpened():
                success, full_image = cap.read()
                if not success:
                    break

                image = full_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image.flags.writeable = True

                # FACE START
                atype = 'face'
                if results.face_landmarks:
                    face_rows = list(map(
                        functools.partial(self.__not_zero_add, frame, atype), 
                        enumerate(results.face_landmarks.landmark),
                    ))
                else:
                    face_rows = list(map(
                        functools.partial(self.__zero_add, frame, atype),
                        [*range(self.MAX_FACE_SIZE)],
                    ))
                rows.extend(face_rows)
                # FACE END
                
                # LEFT HAND START
                atype = 'left_hand'
                if results.left_hand_landmarks:
                    left_hand_rows = list(map(
                        functools.partial(self.__not_zero_add, frame, atype), 
                        enumerate(results.left_hand_landmarks.landmark),
                    ))
                else:
                    left_hand_rows = list(map(
                        functools.partial(self.__zero_add, frame, atype),
                        [*range(self.MAX_HAND_SIZE)],
                    ))
                rows.extend(left_hand_rows)
                # LEFT HAND END

                # POSE START
                atype = 'pose'
                if results.pose_landmarks:
                    pose_rows = list(map(
                        functools.partial(self.__not_zero_add, frame, atype), 
                        enumerate(results.pose_landmarks.landmark),
                    ))
                else:
                    pose_rows = list(map(
                        functools.partial(self.__zero_add, frame, atype),
                        [*range(self.MAX_POSE_SIZE)],
                    ))
                rows.extend(pose_rows)
                # POSE END

                # RIGHT HAND START
                atype = 'right_hand'
                if results.right_hand_landmarks:
                    right_hand_rows = list(map(
                        functools.partial(self.__not_zero_add, frame, atype), 
                        enumerate(results.right_hand_landmarks.landmark),
                    ))
                else:
                    right_hand_rows = list(map(
                        functools.partial(self.__zero_add, frame, atype),
                        [*range(self.MAX_HAND_SIZE)],
                    ))
                rows.extend(right_hand_rows)
                # RIGHT HAND END

                frame += 1
        cap.release()
        df = pd.DataFrame(rows) # create preprocess dataframe
        df.to_parquet(f'{self.ROOT}/train_landmark_files/{signer_id}/{video_id}.parquet')

    # =============================Creating files path for .parquet data=============================
    def __creating_path(self, video_idx: int) -> None:
        signer_id = self.video_df['participant_id'].iloc[video_idx]
        if not os.path.exists(f'{self.ROOT}/train_landmark_files/{signer_id}'):
            os.makedirs(f'{self.ROOT}/train_landmark_files/{signer_id}')

    # =============================Multithreaded preprocessing=============================
    def __threads_process(self, START_SPLIT: int, STOP_SPLIT: int) -> None:
        for _, video_idx in enumerate(tqdm(range(START_SPLIT, STOP_SPLIT))):
            self.__process_video(video_idx)

    def run(self) -> None:
        self.__create_df()
        for _, video_idx in enumerate(tqdm(range(len(self.video_df)))):
            self.__creating_path(video_idx)

        SPLIT = len(self.video_df) // self.N_THREADS

        threads = [
           threading.Thread(
            target=self.__threads_process, 
            args=(i * SPLIT, (i + 1) * SPLIT,)
            )
           for i in range(self.N_THREADS)
           ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()        


@click.command(name='preprocess')
@click.option('--root', default='C:\\Users\\trene\\Diploma')
@click.option('--max_gloss_numb', default=25)
@click.option('--n_threads', default=4)
def main(root: str, max_gloss_numb: int, n_threads: int):
    preprocess = MediapipeDataset(root, max_gloss_numb, n_threads)
    preprocess.run()

if __name__ == '__main__':
    main()