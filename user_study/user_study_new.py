import copy
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

DELTA_T = 5
DELTA_R = 10

def transform_objpatch(image, mask, translate=(0,0), theta=0):
    mask = mask.astype(np.uint8)
    H, W, _ = image.shape
    tx, ty = translate
    if tx > 0:
        image = cv2.copyMakeBorder(image[:, :-tx],0,0,tx,0,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[:, :-tx],0,0,tx,0,cv2.BORDER_CONSTANT,None,value=0)
    elif tx < 0:
        image = cv2.copyMakeBorder(image[:, -tx:],0,0,0,-tx,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[:, -tx:],0,0,0,-tx,cv2.BORDER_CONSTANT,None,value=0)
        
    if ty > 0:
        image = cv2.copyMakeBorder(image[ty:, :],0,ty,0,0,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[ty:, :],0,ty,0,0,cv2.BORDER_CONSTANT,None,value=0)        
    elif ty < 0:
        image = cv2.copyMakeBorder(image[:ty, :],-ty,0,0,0,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[:ty, :],-ty,0,0,0,cv2.BORDER_CONSTANT,None,value=0)
    
    py, px = np.where(mask)
    cy = int(np.round(np.mean(py)))
    cx = int(np.round(np.mean(px)))
    M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    img_rotated = cv2.warpAffine(image, M, (W, H))
    seg_rotated = cv2.warpAffine(mask, M, (W, H))
    #plt.imshow(img_rotated[:,:,::-1]*seg_rotated[:,:,None])
    #plt.show()
    return img_rotated, seg_rotated

def get_transformed_image(image, segmasks, transforms):
    result = np.ones_like(image).astype(int) * 110
    for i in range(len(segmasks)):
        trans, theta = transforms[i]
        mask = segmasks[i]
        img_rotated, seg_rotated = transform_objpatch(image, mask, translate=trans, theta=theta)
        result[seg_rotated==1] = img_rotated[seg_rotated==1]
    return result

def show_scene(image1, image2):
    image = np.concatenate([image1, image2], 1)
    cv2.imshow('Image Viewer', image.astype(np.uint8))


def evaluate(data_folder, num_scenes):
    scenes = sorted([s.split('_')[0] for s in os.listdir(data_folder) if s.endswith('_img.png')])
    scenes = np.random.choice(scenes, num_scenes, False)

    log_transforms = []
    log_images = []
    for scene in scenes:
        img_path = os.path.join(data_folder, '%s_img.png'%scene)
        seg_path = os.path.join(data_folder, '%s_seg.png'%scene)

        # Load a Scene
        image = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        segment_colors = np.vstack({tuple(r) for r in seg.reshape(-1,3)})
        segmasks = []
        for sc in segment_colors:
            mask = np.all(seg==sc, axis=2)
            if mask.sum() > 100000:
                continue
            segmasks.append(mask)

        # Copy the Scene
        current_image = np.ones_like(image).astype(int) * 110
        transforms = [[[0, 0], 0] for _ in range(len(segmasks))]
        current_image = get_transformed_image(image, segmasks, transforms)
        show_scene(image, current_image)

        # Move Each Object
        for i, mask in enumerate(segmasks):
            print("Object:", i)
            trans = [0, 0]
            theta = 0
            while True:
                key = cv2.waitKey(0)
                #print(key)
                if key==2:
                    trans[0] -= DELTA_T
                    if trans[0] < 0:
                        trans[0] = 0
                elif key==3:
                    trans[0] += DELTA_T
                    if trans[0] > 479:
                        trans[0] = 479
                elif key==0:
                    trans[1] += DELTA_T
                elif key==1:
                    trans[1] -= DELTA_T
                elif key==113:
                    theta += DELTA_R
                elif key==119:
                    theta -= DELTA_R
                elif key==13:
                    transforms[i] = [trans, theta]
                    break
                elif key==27:
                    cv2.destroyAllWindows()
                    exit()
                copy_trans = copy.deepcopy(transforms)
                copy_trans[i] = [trans, theta]
                current_image = get_transformed_image(image, segmasks, copy_trans)
                show_scene(image, current_image)
        current_image = get_transformed_image(image, segmasks, transforms)
        show_scene(image, current_image)
        log_transforms.append(transforms)
        log_images.append(current_image)
        print(log_transforms)
        print(log_images)
    return log_transforms, log_images



# left: 2, 97
# right: 3, 100
# up: 0, 119
# down: 1, 115
# enter: 13
# backspace: 127
# n: 110
# p: 112
# r: 114
# q: 113
# w: 119
if __name__=='__main__':
    folder_path = 'selected_study2' # '/ssd/disk/PreferenceDiffusion/selected/'
    evaluate(folder_path, 20)
