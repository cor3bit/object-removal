from collections import defaultdict
import os
import time

import cv2 as cv

from algorithm import remove_object

EXTENSIONS = {'bmp', 'jpeg', 'jpg', 'png'}

OPTIMAL_PARAMETERS = defaultdict(
    lambda: (21, 0.8),
    {
        'input_1.jpg': (17, 0.8),
        'input_2.jpg': (33, 0.5),
        'input_3.jpg': (21, 0.8),
    }
)


def get_images(input_dir):
    image_files = [f for f in os.listdir(input_dir) if f.split('.')[1] in EXTENSIONS]
    masks = sorted([f for f in image_files if '_mask.' in f])
    images = sorted([f for f in image_files if '_mask.' not in f])
    return images, masks


if __name__ == '__main__':
    # timing
    start_t = time.time()

    # main directories
    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, 'data', 'input')
    output_dir = os.path.join(base_dir, 'data', 'output')

    # get images & masks (sorted)
    image_names, mask_names = get_images(input_dir)

    # run algorithm
    for i, (image_name, mask_name) in enumerate(zip(image_names, mask_names), 1):
        try:
            # read original image and mask
            print(f'Reading image set: {image_name}<->{mask_name}.')
            image = cv.imread(os.path.join(input_dir, image_name))
            mask = cv.imread(os.path.join(input_dir, mask_name), 0)

            # run algorithm
            print('Running object removing algorithm.')
            patch_size, alpha = OPTIMAL_PARAMETERS[image_name]
            new_image = remove_object(image, mask, patch_size=patch_size, alpha=alpha)

            # save new artifacts
            res_name = f'result_{i}.jpg'
            print(f'Saving new image: {res_name}. Input image reference: {image_name}.\n')
            cv.imwrite(os.path.join(output_dir, res_name), new_image)
        except Exception as e:
            print(f'Pipeline for {image_name} failed with exception: {e}.')

    dt = (time.time() - start_t) / 60.
    print(f'The program took {dt:.1f} minutes to run.')
    print('Done!')
