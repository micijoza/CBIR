import numpy as np
from os import listdir
from paths import kth_dir, kth_path
from descriptors import bitdesc
import cv2

def main():
    listOflists = list()
    print('Extracting features ....')
    counter = 0
    for kth_class in kth_dir:
        print(f'Current folder: {kth_class}')
        for filename in listdir(kth_path + kth_class + "/"):
            counter += 1
            img_name = f'{kth_path}{kth_class}/{filename}'
            img = cv2.imread(img_name, 0)
            features = bitdesc(img) + [kth_class] + [img_name]
            print(f' Image count:{counter}')
            listOflists.append(features)
    final_array = np.array(listOflists)
    np.save('signatures.npy', final_array)
    print('Extraction conclued successufuly')

if __name__ == "__main__":
    main()