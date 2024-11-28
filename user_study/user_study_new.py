import copy
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

DELTA_T = 5
DELTA_R = 10
IMAGE_H = 360
IMAGE_W = 480

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
    if len(py)==0:
        return np.zeros_like(image), np.zeros_like(mask)
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


def evaluate(data_folder, output_path, num_scenes):
    scenes = sorted([s.split('_')[0] for s in os.listdir(data_folder) if s.endswith('_img.png')])
    scenes = np.random.choice(scenes, num_scenes, False)

    log_file = os.path.join(output_path, 'log.txt')
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
        current_image = np.ones_like(image).astype(int) * 110
        transforms = [[[0, 0], 0] for _ in range(len(segmasks))]
        current_image = get_transformed_image(image, segmasks, transforms)
        show_scene(image, current_image)

        # Move Each Object
        #for i, mask in enumerate(segmasks):
        i = 0
        flag_save = False
        while not flag_save: #Truei<len(segmasks):
            i = i%len(segmasks)
            print("Object:", i)
            mask = segmasks[i]

            trans, theta = transforms[i]
            #trans = [0, 0]
            #theta = 0
            while True:
                key = cv2.waitKey(0)
                #print(key)
                if key==KEY_LEFT:
                    trans[0] -= DELTA_T
                    #if trans[0] < 0:
                    #    trans[0] = 0
                elif key==KEY_RIGHT:
                    trans[0] += DELTA_T
                    #if trans[0] > 479:
                    #    trans[0] = 479
                elif key==KEY_UP:
                    trans[1] += DELTA_T
                elif key==KEY_DOWN:
                    trans[1] -= DELTA_T
                elif key==KEY_Q:
                    theta += DELTA_R
                elif key==KEY_W:
                    theta -= DELTA_R
                elif key==KEY_ENTER:
                    transforms[i] = [trans, theta]
                    break
                elif key==KEY_S:
                    flag_save = True
                    break
                elif key==KEY_ESC:
                    cv2.destroyAllWindows()
                    return log_transforms, log_images
                elif key==KEY_BACKSPACE:
                    transforms[i] = [trans, theta]
                    i -= 2
                    break

                copy_trans = copy.deepcopy(transforms)
                copy_trans[i] = [trans, theta]
                current_image = get_transformed_image(image, segmasks, copy_trans)
                show_scene(image, current_image)
            i += 1
        current_image = get_transformed_image(image, segmasks, transforms)
        show_scene(image, current_image)

        # Save Results
        result_image = np.concatenate([image, current_image], 1).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, 's%d-%s.png'%(sidx, scene)), result_image)
        with open(log_file, 'a') as file:
            file.write("Scene %d: %s / %s\n" %(sidx, scene, transforms))

        log_transforms.append(transforms)
        log_images.append(current_image)
        #print(log_transforms)
        #print(log_images)
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
    log_transforms, log_images = evaluate(folder_path, output_path, 20)
