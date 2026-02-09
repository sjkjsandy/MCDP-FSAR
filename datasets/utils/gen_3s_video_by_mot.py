import os
import shutil
import sys
import cv2
import numpy as np


parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

recods = []

def parse_mot_res(input):
    mot_res = []
    boxes, scores, ids = input[0]
    for box, score, i in zip(boxes[0], scores[0], ids[0]):
        xmin, ymin, w, h = box
        res = [i, 0, score, xmin, ymin, xmin + w, ymin + h]
        mot_res.append(res)
    return {'boxes': np.array(mot_res)}

def crop_images_with_mot(frame, mot_res, final_w=320, final_h = 240):
    #mot_res = [[id, class, score, xmin, ymin, xmax, ymax]]
    tager_x, tager_y, tager_w, tager_h = 0, 0, final_w, final_h
    tager_rate = tager_w/tager_h
    crop_images = []
    #获取图片的宽高
    src_w, src_h = frame.shape[1], frame.shape[0]
    for mot_res_temp in mot_res:
        #获取人物框的坐标和高宽
        id, label, score, x, y, x2, y2, = mot_res_temp
        x2 = x2 + 60
        y2 = y2 + 30
        w = x2 - x
        h = y2 - y
        #计算出人物框的中心点
        center_x = x + w / 2
        center_y = y + h / 2
        is_need_scale = True
        if tager_w == w and tager_h == h:
            is_need_scale = False
        elif (tager_w <= w and tager_h <= h) or (tager_w >= w and tager_h >= h):
            if w/h > tager_rate:
                tager_w = w
                tager_h = w/tager_rate
            else:
                tager_h = h
                tager_w = h*tager_rate
        elif tager_w < w and tager_h > h:
            tager_w = w
            tager_h = w/tager_rate
        elif tager_w > w and tager_h < h:
            tager_h = h
            tager_w = h*tager_rate
        #判断新计算的目标框是否超出图片范围，如果超出，则调整起始的x,y
        if center_x - tager_w/2 < 0:
            tager_x = 0
        elif center_x + tager_w/2 > src_w:
            tager_x = src_w - tager_w
        else:
            tager_x = center_x - tager_w/2
        if center_y - tager_h/2 < 0:
            tager_y = 0
        elif center_y + tager_h/2 > src_h:
            tager_y = src_h - tager_h
        else:
            tager_y = center_y - tager_h/2
        #从image中根据目标框copy目标图片
        tager_x, tager_y, tager_w, tager_h = int(tager_x), int(tager_y), int(tager_w), int(tager_h)
        frame_temp = frame[tager_y:tager_y+tager_h, tager_x:tager_x+tager_w].copy()

        #将图片frame裁剪成420*512大小的图片
        if is_need_scale:
            frame_temp = cv2.resize(frame_temp, (int(final_w), int(final_h)), interpolation=cv2.INTER_AREA)
        if not (frame_temp.shape[0] == final_h and frame_temp.shape[1] == final_w):
            raise ValueError("frame_temp shape error")
        # #保存frame到本地
        # savepath = 'app_action/demo/output/'
        # if not os.path.exists(savepath):
        #     os.makedirs(savepath)
        # cv2.imwrite(savepath + str(time.time()) +".jpg", frame_temp)
        crop_images.append({"id":id, "label":label, "score":score, "oribox":[x, y, x2, y2], "frame":frame_temp})
    return crop_images

def gen_3s_video_by_mot_and_reid_crop(video_path, out_path, save_tags=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))
    if save_tags:
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (320, 240))
    frame_id = 0
    manayperson =0 
    oneperson = 0
    noperson = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_id +=1
            results = model([frame.copy()])[0].boxes.cpu().numpy()
            clss = results.cls
            confs = results.conf
            datas = results.data
            boxess = []
            for i in range(len(clss)):
                if clss[i] == 0:
                    boxess.append([0,0,confs[i],datas[i][0],datas[i][1],datas[i][2],datas[i][3]])
            mot_results = {'boxes': boxess}
            # 如果没有检测到人，跳出不处理
            if len(mot_results['boxes']) == 0:
                noperson +=1
                print("reid_Predict  no person!")
                continue
            #获取被追踪目标，如果有多个人，需要追踪并重识别；
            elif len(mot_results['boxes']) > 1:
                manayperson +=1
                continue
                # mot_results['boxes'] = reid_Predict.get_tracked_mot_res(frame_id, frame, mot_results['boxes'])
            oneperson +=1
            #裁剪每一帧的人物
            # crop_images = [{"id":0, "label":0, "score":0.9, "oribox":[20,100,100,130], "frame":frame}]
            crop_images = crop_images_with_mot(frame.copy(), mot_results['boxes'])
            if len(crop_images) == 1 and save_tags:
                #将frame转换成rgb格式
                # frame = cv2.cvtColor(crop_images[0]['frame'], cv2.COLOR_BGR2RGB)
                writer.write(crop_images[0]['frame'])
        else:
            break
    cap.release()
    if save_tags:
        writer.release()
        if oneperson < 25:
            #删除指定的文件
            os.remove(out_path)
        else:
            recods.append(out_path)
        print("save video to %s" % out_path)

if __name__ == '__main__':

    from ultralytics import YOLO

    # Load a model
    model = YOLO("datasets/utils/yolo11x.pt")

    video_path = '/home/ps/workdata/andy/data/ucf101/videos/HandstandWalking/'
    out_path = '/home/ps/workdata/andy/data/ucf101_crop/videos/HandstandWalking/'  # 输出路径

    # 定义视频文件扩展名
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')
    # 获取视频文件列表
    video_files = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi', '.mov'))]

    # 遍历文件夹a及其子文件夹
    for root, dirs, files in os.walk(video_path):
        for dir_name in dirs:
            sub_folder_path = os.path.join(root, dir_name)
            sub_folder_path_save = os.path.join(out_path, dir_name)
            # 如果不存在输出文件夹，则创建
            if not os.path.exists(sub_folder_path_save):
                os.makedirs(sub_folder_path_save)
            video_file = None
            
            # 遍历子文件夹中的文件，找到第一个视频文件
            for file_name in os.listdir(sub_folder_path):
                if file_name.lower().endswith(video_extensions):
                    video_file = file_name
                    full_path = os.path.join(sub_folder_path, video_file)
                    # file_name = os.path.basename(video_file)
                    # video_name = file_name.split('.')[0] + '.mp4'
                    saved_path = os.path.join(sub_folder_path_save, video_file)
                    # print(f"正在处理视频文件: {full_path}")
                    gen_3s_video_by_mot_and_reid_crop(full_path, saved_path, model, None)
