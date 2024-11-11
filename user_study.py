import os
import cv2
import numpy as np

def browse_images(folder_path, num_eval=10):
    scenes = np.random.choice(np.arange(len(os.listdir(folder_path))), num_eval, replace=False)
    print("Press ENTER or →  : move to the next image.")
    print("Press ←  : move back to the previous image.")
    print("Press 1 : select a Well-Organized scene.")
    print("Press X : you cannot satisfy with the scenes.")
    print()

    name = input("Name: ")
    log_file = 'logs/%s.txt'%name.replace(' ', '')
    with open(log_file, 'w') as file:
        file.write("Selected Images:\n")

    scores = []

    for scene in scenes:
        scene_name = 'scene%02d'%scene
        with open(log_file, 'a') as file:
            file.write("%s: "%scene_name)
        scene_path = os.path.join(folder_path, scene_name)
        images = [img for img in os.listdir(scene_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        images.sort()

        #for image in images:
        index = 0
        while True: #index < len(images):
            img_path = os.path.join(scene_path, images[index])
            img = cv2.imread(img_path)
            cv2.imshow('Image Viewer', img)
            
            key = cv2.waitKey(0)
            #print(key)
            if key == 13 or key == 83: #2555904:
                if index < len(images)-1:
                    index += 1
                    cv2.destroyAllWindows()
            elif key == ord('1'):
                with open(log_file, 'a') as file:
                    file.write("%s\n"%images[index])
                cv2.destroyAllWindows()
                score = int(images[index].split('.png')[0].split('_')[-1])
                scores.append(score)
                break
            elif key == 81: #2424832:
                if index > 0:
                    index -= 1
            elif key == 27 or key == ord('x'):
                with open(log_file, 'a') as file:
                    file.write("X\n")
                cv2.destroyAllWindows()
                break

        #cv2.destroyAllWindows()
    print('Average tidiness score: %.3f'%(np.mean(scores)/1000))

folder_path = '/ssd/disk/PreferenceDiffusion/selected/'
browse_images(folder_path)
