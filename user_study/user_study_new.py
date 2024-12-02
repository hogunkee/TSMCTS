import copy
import os
import numpy as np
import cv2
import platform
from matplotlib import pyplot as plt

DELTA_T = 5
DELTA_R = 10
IMAGE_H = 360
IMAGE_W = 480

pf = platform.platform()
if pf.startswith('mac'):
    # For IOS
    KEY_UP = 0
    KEY_DOWN = 1
    KEY_LEFT = 2
    KEY_RIGHT = 3
    KEY_Q = 113
    KEY_W = 119
    KEY_S = 115
    KEY_Y = 121
    KEY_ENTER = 13
    KEY_BACKSPACE = 127
    KEY_ESC = 27
else:
    # For Ubuntu
    KEY_UP = 82
    KEY_DOWN = 84
    KEY_LEFT = 81
    KEY_RIGHT = 83
    KEY_Q = 113
    KEY_W = 119
    KEY_S = 115
    KEY_Y = 121
    KEY_ENTER = 13
    KEY_BACKSPACE = 8
    KEY_ESC = 27

def transform_objpatch(image, mask, translate=(0,0), theta=0):
    mask = mask.astype(np.uint8)
    H, W, _ = image.shape
    py, px = np.where(mask)
    if len(py)==0:
        return np.zeros_like(image), np.zeros_like(mask)
    cy = int(np.round(np.mean(py)))
    cx = int(np.round(np.mean(px)))

    # Move the object patch to the center.
    tx1 = int(W/2 - cx)
    ty1 = -int(H/2 - cy)
    if tx1 > 0:
        image = cv2.copyMakeBorder(image[:, :-tx1],0,0,tx1,0,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[:, :-tx1],0,0,tx1,0,cv2.BORDER_CONSTANT,None,value=0)
    elif tx1 < 0:
        image = cv2.copyMakeBorder(image[:, -tx1:],0,0,0,-tx1,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[:, -tx1:],0,0,0,-tx1,cv2.BORDER_CONSTANT,None,value=0)
    if ty1 > 0:
        image = cv2.copyMakeBorder(image[ty1:, :],0,ty1,0,0,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[ty1:, :],0,ty1,0,0,cv2.BORDER_CONSTANT,None,value=0)        
    elif ty1 < 0:
        image = cv2.copyMakeBorder(image[:ty1, :],-ty1,0,0,0,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[:ty1, :],-ty1,0,0,0,cv2.BORDER_CONSTANT,None,value=0)

    # Get rotated images.
    M = cv2.getRotationMatrix2D((W/2, H/2), theta, 1.0)
    image = cv2.warpAffine(image, M, (W, H))
    mask = cv2.warpAffine(mask, M, (W, H))

    # Move the object patch to the transformed position.
    tx2 = translate[0] - tx1
    ty2 = translate[1] - ty1
    if tx2 > 0:
        image = cv2.copyMakeBorder(image[:, :-tx2],0,0,tx2,0,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[:, :-tx2],0,0,tx2,0,cv2.BORDER_CONSTANT,None,value=0)
    elif tx2 < 0:
        image = cv2.copyMakeBorder(image[:, -tx2:],0,0,0,-tx2,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[:, -tx2:],0,0,0,-tx2,cv2.BORDER_CONSTANT,None,value=0)
    if ty2 > 0:
        image = cv2.copyMakeBorder(image[ty2:, :],0,ty2,0,0,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[ty2:, :],0,ty2,0,0,cv2.BORDER_CONSTANT,None,value=0)        
    elif ty2 < 0:
        image = cv2.copyMakeBorder(image[:ty2, :],-ty2,0,0,0,cv2.BORDER_CONSTANT,None,value=0)
        mask = cv2.copyMakeBorder(mask[:ty2, :],-ty2,0,0,0,cv2.BORDER_CONSTANT,None,value=0)
    return image, mask

def get_transformed_image(image, segmasks, transforms, bg_image, current_sidx=-1):
    #result = np.ones_like(image).astype(int) * 110
    result = copy.deepcopy(bg_image)
    for i in range(len(segmasks)):
        trans, theta = transforms[i]
        mask = segmasks[i]
        img_rotated, seg_rotated = transform_objpatch(image, mask, translate=trans, theta=theta)
        if i==current_sidx:
            kernel = np.ones((3,3), np.uint8)
            seg_rotated_dilated = cv2.dilate(seg_rotated, kernel, iterations=1)
            result[seg_rotated_dilated==1] = np.array([0, 100, 255])
            result[seg_rotated==1] = img_rotated[seg_rotated==1]
        else:
            result[seg_rotated==1] = img_rotated[seg_rotated==1]
    return result

def show_scene(image1, image2):
    image = np.concatenate([image1, image2], 1)
    cv2.imshow('Image Viewer', image.astype(np.uint8))


def evaluate(data_folder, output_path, num_scenes, name):
    scenes = sorted([s.split('_')[0] for s in os.listdir(data_folder) if s.endswith('_img.png')])
    scenes = np.random.choice(scenes, num_scenes, False)

    bg_path = ('nv_background.png')
    bg_image = cv2.imread(bg_path)

    log_file = os.path.join(output_path, 'log_%s.txt'%name)
    with open(log_file, 'w') as file:
        file.write("Num scenes: %d\n" %num_scenes)
    log_transforms = []
    log_images = []
    for sidx, scene in enumerate(scenes):
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
        #removebg_image = np.ones_like(image).astype(int) * 0
        current_image = copy.deepcopy(bg_image)
        transforms = [[[0, 0], 0] for _ in range(len(segmasks))]
        current_image = get_transformed_image(image, segmasks, transforms, bg_image, 0)
        show_scene(image, current_image)

        # Move Each Object
        count_control = 0
        i = 0
        flag_save = False
        while not flag_save:
            i = i%len(segmasks)
            mask = segmasks[i]

            trans, theta = transforms[i]
            while True:
                key = cv2.waitKey(0)
                #print(key)
                if key==KEY_LEFT:
                    trans[0] -= DELTA_T
                    count_control += 1
                elif key==KEY_RIGHT:
                    trans[0] += DELTA_T
                    count_control += 1
                elif key==KEY_UP:
                    trans[1] += DELTA_T
                    count_control += 1
                elif key==KEY_DOWN:
                    trans[1] -= DELTA_T
                    count_control += 1
                elif key==KEY_Q:
                    theta += DELTA_R
                    count_control += 1
                elif key==KEY_W:
                    theta -= DELTA_R
                    count_control += 1
                elif key==KEY_ENTER:
                    transforms[i] = [trans, theta]
                    i += 1
                    break
                elif key==KEY_ESC:
                    cv2.destroyAllWindows()
                    return log_transforms, log_images
                elif key==KEY_BACKSPACE:
                    transforms[i] = [trans, theta]
                    i -= 1
                    i %= len(segmasks)
                    break
                copy_trans = copy.deepcopy(transforms)
                copy_trans[i] = [trans, theta]
                current_image = get_transformed_image(image, segmasks, copy_trans, bg_image, i)
                show_scene(image, current_image)

            current_image = get_transformed_image(image, segmasks, transforms, bg_image, i)
            show_scene(image, current_image)
            if i==len(segmasks):
                key = cv2.waitKey(0)
                if key==KEY_ENTER or key==KEY_S:
                    flag_save = True
                elif key==KEY_BACKSPACE:
                    i %= len(segmasks)
                    current_image = get_transformed_image(image, segmasks, transforms, bg_image, i)
                    show_scene(image, current_image)
        current_image = get_transformed_image(image, segmasks, transforms, bg_image)
        removebg_image = get_transformed_image(image, segmasks, transforms, np.zeros_like(bg_image))
        show_scene(image, current_image)

        # Save Results
        result_image = np.concatenate([image, current_image], 1).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, 's%d-%s.png'%(sidx, scene)), result_image)
        with open(log_file, 'a') as file:
            file.write("Scene %d: %s / %s / %d\n" %(sidx, scene, transforms, count_control))
        removebg_image = removebg_image.astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, 's%d-%s-nobg.png'%(sidx, scene)), removebg_image)

        log_transforms.append(transforms)
        log_images.append(current_image)
    return log_transforms, log_images


if __name__=='__main__':
    folder_path = 'selected_study2' # '/ssd/disk/PreferenceDiffusion/selected/'
    while True:
        name = input("Name: ").replace(' ', '')
        output_path = 'logs/study2/%s' %name
        if os.path.isdir(output_path):
            print("Same Name Already Exists!! Use another name.")
        else:
            os.makedirs(output_path)
            break
    log_transforms, log_images = evaluate(folder_path, output_path, 20, name)
