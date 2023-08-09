import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size,check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, strip_optimizer, set_logging,increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from sort import *

area1_pointA = (0,370)
area1_pointB = (185,370)
area1_pointC = (0,390)
area1_pointD = (185,390)

area2_pointA = (200,400)
area2_pointB = (600,400)
area2_pointC = (200,420)
area2_pointD = (600,420)

area3_pointA = (640,400)
area3_pointB = (1000,400)
area3_pointC = (640,420)
area3_pointD = (1000,420)

#vehicles total counting variables
l_arr = []
l_total = 0
a1_array_ids = []
a1_counting = 0

m_arr = []
m_total = 0
a2_array_ids = []
a2_counting = 0

r_arr = []
r_total = 0
a3_array_ids = []
a3_counting = 0

def timetrans(s):
    s1 = s%60
    min = s//60
    return s1, min

def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        label = names[cat]
        if label=='vehicle':
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        elif label=='bus':
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,0), 1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        elif label=='lightbus':
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        elif label=='truck':
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        elif label=='locomotive':
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 1)
            cv2.putText(img, 'scooter', (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        elif label=='pedestrian':
            cv2.rectangle(img, (x1, y1), (x2, y2), (125,125,125), 1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        elif label=='cyclist':
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        midpoint_x = x1+((x2-x1)/2)
        midpoint_y = y1+((y2-y1)/2)
        center_point = (int(midpoint_x),int(midpoint_y))
        midpoint_color = (0,255,0)
        if label == 'vehicle' or label =='bus' or label =='lightbus' or label =='truck' or label=='locomotive':
            if (midpoint_x < area2_pointA[0]):
                if len(l_arr) > 0:
                    if id not in l_arr:
                        l_arr.append(id)
                else:
                    l_arr.append(id)
            elif (midpoint_x > area2_pointA[0]) and (midpoint_x < area2_pointB[0]):
                if len(m_arr) > 0:
                    if id not in m_arr:
                        m_arr.append(id)
                else:
                    m_arr.append(id)
            elif (midpoint_x > area2_pointB[0]):
                if len(r_arr) > 0:
                    if id not in r_arr:
                        r_arr.append(id)
                else:
                    r_arr.append(id)
            if (midpoint_x > area1_pointA[0]and midpoint_x < area1_pointD[0] ) and (midpoint_y > area1_pointA[1]and midpoint_y < area1_pointD[1] ):
                midpoint_color = (0,0,255)
                if len(a1_array_ids) > 0:
                    if id not in a1_array_ids:
                        a1_array_ids.append(id)
                else:
                    a1_array_ids.append(id)
            elif (midpoint_x > area2_pointA[0]and midpoint_x < area2_pointD[0] ) and (midpoint_y > area2_pointA[1]and midpoint_y < area2_pointD[1] ):
                midpoint_color = (0,0,255)
                if len(a2_array_ids) > 0:
                    if id not in a2_array_ids:
                        a2_array_ids.append(id)
                else:
                    a2_array_ids.append(id)
            elif (midpoint_x > area3_pointA[0]and midpoint_x < area3_pointD[0] ) and (midpoint_y > area3_pointA[1]and midpoint_y < area3_pointD[1] ):
                midpoint_color = (0,0,255)
                if len(a3_array_ids) > 0:
                    if id not in a3_array_ids:
                        a3_array_ids.append(id)
                else:
                    a3_array_ids.append(id)
        cv2.circle(img,center_point,radius=1,color=midpoint_color,thickness=2)
    return img

def get_per10_count(route,nf10,counting):
    path = 'per10.txt'
    s = nf10*10
    s1,min = timetrans(s)
    if route==1:
        with open(path, 'a') as f:
            f.write('1 ')
            f.write(str(min)+' '+str(s1)+' ')
            f.write(str(counting)+'\n')
    elif route==2:
        with open(path, 'a') as f:
            f.write('2 ')
            f.write(str(min)+' '+str(s1)+' ')
            f.write(str(counting)+'\n')
    elif route==3:
        with open(path, 'a') as f:
            f.write('3 ')
            f.write(str(min)+' '+str(s1)+' ')
            f.write(str(counting)+'\n')
    return counting

def get_per300_count(route,nf10,counting):
    path = 'per300.txt'
    s = nf10*10
    s1,min = timetrans(s)
    if route==1:
        with open(path, 'a') as f:
            f.write('1 ')
            f.write(str(min)+' '+str(s1)+' ')
            f.write(str(counting)+'\n')
    elif route==2:
        with open(path, 'a') as f:
            f.write('2 ')
            f.write(str(min)+' '+str(s1)+' ')
            f.write(str(counting)+'\n')
    elif route==3:
        with open(path, 'a') as f:
            f.write('3 ')
            f.write(str(min)+' '+str(s1)+' ')
            f.write(str(counting)+'\n')
    return counting
    

def detect(save_img=False):
    path = 'per300.txt'
    with open(path, 'w') as f:
        f.write('lane min sec num\n')
    path = 'per10.txt'
    with open(path, 'w') as f:
        f.write('lane min sec num\n')
    nf10=1 #判斷每十秒的初始值
    nf300=1
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) 
    #......................... 
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    start = time.time()
    end = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                
                print('Tracked Detections : '+str(len(tracked_dets)))
                
                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names)
                
                #........................................................
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            cv2.line(im0,area1_pointA,area1_pointB,(0,0,255),1)
            cv2.line(im0,area1_pointC,area1_pointD,(0,0,255),1)
            
            cv2.line(im0,area2_pointA,area2_pointB,(0,0,255),1)
            cv2.line(im0,area2_pointC,area2_pointD,(0,0,255),1)
            
            cv2.line(im0,area3_pointA,area3_pointB,(0,0,255),1)
            cv2.line(im0,area3_pointC,area3_pointD,(0,0,255),1)

            thickness = 2
            fontScale = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            a1_org = (0,30)
            a2_org = (400,30)
            a3_org = (820,30)
            
            a1_counting = len(a1_array_ids)
            a2_counting = len(a2_array_ids)
            a3_counting = len(a3_array_ids)
            
            l_total = len(l_arr)
            m_total = len(m_arr)
            r_total = len(r_arr)

            l_arr.clear()
            m_arr.clear()
            r_arr.clear()

            end = time.time()
            print(end-start)
            #vartime=time.gmtime(end-start)
            if int(end-start)>=(10*nf10-0.185*nf10):
                get_per10_count(1,nf10,l_total)
                get_per10_count(2,nf10,m_total)
                get_per10_count(3,nf10,r_total)
                nf10+=1
                #start1=time.time()
                
            if int(end-start)>=(300*nf300-12*nf300):
                get_per300_count(1,nf10-1,a1_counting)
                get_per300_count(2,nf10-1,a2_counting)
                get_per300_count(3,nf10-1,a3_counting)
                #start=time.time()
                a1_array_ids.clear()
                a2_array_ids.clear()
                a3_array_ids.clear()
                nf300+=1
                
            cv2.putText(im0,"left:"+str(a1_counting), a1_org, font, fontScale, (0,0,255), thickness, cv2.LINE_AA)
            cv2.putText(im0,"mid:"+str(a2_counting), a2_org, font, fontScale, (0,255,0), thickness, cv2.LINE_AA)
            cv2.putText(im0,"right:"+str(a3_counting), a3_org, font, fontScale, (255,0,0), thickness, cv2.LINE_AA)    
            
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.65, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()