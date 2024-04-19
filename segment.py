import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
global input_box
global temp_coodrs
global time
time =0
temp_coodrs = []
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255,255,255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
# mouse callback function
def draw_point(event,x,y,flags,param):
    global input_box
    global temp_coodrs
    global time
    if event == cv2.EVENT_LBUTTONDOWN:
        time+=1
        cv2.circle(img,(x,y),5,(255,0,0),-1)
        if time == 1:
            temp_coodrs=[x,y]
        elif time == 2:
            temp_coodrs.extend([x,y])
            input_box=np.array(temp_coodrs)
            print(input_box)
            time =0



# def save_figure(event,x,y,flags,param):
#     if event == cv2.EVENT_RBUTTONDOWN:
#         cv2.write()

# Create a black image, a window and bind the function to window


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
 
for i in range(1398,2311):
    print("process image:",i)
    img = cv2.imread('./real_imgs/train/observe_{}.jpg'.format(i))


    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_point)
    
    input_label = np.array([1])
    while True:
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow('image')
    # input_box=np.array([109,1113,153,155])
    predictor.set_image(img)
    masks, _, _ = predictor.predict(
    point_coords=None,


    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
    )

    mask_img=show_mask(masks[-1], plt.gca())
    #show_points(input_point, input_label, plt.gca())
    #plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    # plt.savefig("./img/test/result_{}.jpg".format(i))
    cv2.imwrite("./real_imgs/result_{}.jpg".format(i),mask_img)
    plt.close()


