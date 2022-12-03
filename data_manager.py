import json
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as T
import os
import cv2
from log.log import logger
from mtcnn.mtcnn import MTCNN
import ffmpeg
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import shutil
import sys
import util
class VideoProcess:
    def __init__(self, config) -> None:
        self.video_path = config.video_path
        self.file_suffix = config.file_suffix
        self.video_list = self.get_video_list()
        self.segment_num = config.segment_num 
        self.sample_num = config.sample_num
        self.save_imgs_path = config.save_imgs_path
    
        self.detector = MTCNN()

    def get_video_list(self):
        video_list = []
        for _, _, names in os.walk(self.video_path):
            for name in names:
                ext = os.path.splitext(name)[1]
                if ext in self.file_suffix:
                    video_list.append(name)
        print("the total video nums is :" + str(len(video_list)))
        return sorted(video_list)

    def get_video_info(self, in_file):
        try:
            probe = ffmpeg.probe(in_file)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            return video_stream
        except ffmpeg.Error as err:
            print(str(err.stderr, encoding='utf8'))
            return None
    
    def video_to_img(self, vid_file):
        # video_info = self.get_video_info(os.path.join(self.video_path, vid_file))
            # total_frame_nums = int(video_info['nb_frames'])
            # video_input = ffmpeg.input(os.path.join(self.video_path, vid_file))

            # if total_frame_nums < self.segment_num * self.sample_num:
            #     logger.error(str(vid_file) + " frames is less than save_frame_nums")
            #     continue

            # timeF = total_frame_nums / self.segment_num

            vid_dir = os.path.join(self.save_imgs_path, vid_file.split('.')[0])
            print(vid_dir)
            if not os.path.exists(vid_dir):
                os.mkdir(vid_dir)
            else:
                return
            video_input = cv2.VideoCapture(os.path.join(self.video_path, vid_file))
            total_frame_nums = int(video_input.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info("vid file is " + vid_file + ", total frame is :" + str(total_frame_nums))

            flag = True
            # for segment in range(self.segment_num):
            for sample in range(self.sample_num):
                # img_bytes, err = video_input.filter('select', 'gte(n,{})'.format(self.sample_num)).output('pipe:', vframes=1, format='image2', vcodec='mjpeg').run(capture_stdout=True)
                # img_array = np.asarray(bytearray(img_bytes), dtype="uint8")
                # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                success, img = video_input.read()
                if success:
                    face_img = self.detect_face(img)
                    if face_img != None :
                        if len(face_img) > 1 :
                            logger.debug("vid_file " + str(vid_file) + ", sample_num:"+  str(sample) + " is not one face")
                            face_img = face_img[0]
                        else:
                            face_img = face_img[0]
                        file_name = str(sample).zfill(3) + ".png"
                        save_path = os.path.join(vid_dir, file_name)
                        cv2.imwrite(save_path, face_img)
                    else:
                        flag = False
                        logger.error("vid_file " + str(vid_file) + ", sample_num:"+  str(sample) + " is not detect face!!!")
                        break
                else:
                    flag = False
                    logger.error("vid_file " + str(vid_file) + ", sample_num:"+  str(sample) + " is not read!!!")
                    break
            if not flag:
                shutil.rmtree(vid_dir)
                logger.error(str(vid_file) + " is not process complete!!")
            video_input.release()
        
    def process(self):
        print("video is processing")
        pool = ThreadPoolExecutor(max_workers=6)
        for vid_file in self.video_list:
            pool.submit(self.video_to_img, vid_file)
        pool.shutdown()
        print("video is processed")
        
    def detect_face(self, img):
        with torch.no_grad():
            detect_result = self.detector.detect_faces(img)

        if len(detect_result) == 0:
            return None
        
        shapes = np.array(img).shape
        face_images = []
        height = shapes[0]
        weight = shapes[1]
        for item in detect_result:
            box = item['box']

            top = int(box[1] * 0.9)
            buttom = int((box[1] + box[3]) * 1.1)
            left = int(box[0] * 0.9)
            right = int((box[0] + box[2]) * 1.1)

            if top < 0:
                top = 0
            if left < 0:
                left = 0
            if buttom > height:
                buttom = height
            if right > weight:
                right = weight

            cropped = img[top:buttom, left:right]
            face_img = cv2.resize(cropped, (224, 224),interpolation=cv2.INTER_LINEAR)
            face_images.append(face_img)
        return face_images





class Meta_Train_Dataset(data.Dataset):
    def __init__(self, imgs_path, mode, vids_pkg_size=32):
        self.mode = mode
        self.imgs_path = imgs_path
        self.vids_pkg_size = vids_pkg_size

        self.vids_pkg = sorted(os.listdir(self.imgs_path))
        
        if self.mode == "train":
            self.vids_pkg = self.vids_pkg[ : 448]
        elif self.mode == "test":
            self.vids_pkg = self.vids_pkg[448:]
        # for dbtype in dbtype_list:
        #     if os.path.isfile(os.path.join(sql_dir_war, dbtype)):
        #         dbtype_list.remove(dbtype)

    def __getitem__(self, index):
        vid_pkg_name = self.vids_pkg[index]
        vid_pkg_path = os.path.join(self.imgs_path, vid_pkg_name)

        vid_imgs = sorted(os.listdir(vid_pkg_path))
        
        if len(vid_imgs) != self.vids_pkg_size:
            logger.error(vid_pkg_path + " is not have enghout image!!!")
            sys.exit()
        
        pkg = []
        for vid_img in vid_imgs:
            inst = util.img_read(os.path.join(vid_pkg_path, vid_img))
            if inst == None:
                sys.exit(1)
            pkg.append(inst)
        #the data form is T C H W
        pkg = torch.stack(pkg, dim=0)
        label = torch.zeros(1, dtype=torch.float32) if "raw" in self.imgs_path else torch.ones(1, dtype=torch.float32)
        return pkg, label

        

    def __len__(self):
        return len(self.vids_pkg)