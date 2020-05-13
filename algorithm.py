import numpy as np
import cv2 as cv
from numba import jit

DEBUG = False


def remove_object(image, mask, patch_size=21, alpha=0.8):
    # time
    t = 0

    # mask post-processing
    mask = cv.medianBlur(mask, 5)
    mask_indx = np.where(mask > 245)
    mask_boolean = np.ones_like(mask)
    mask_boolean[mask_indx] = 0

    if DEBUG:
        cv.imshow(f'mask t={t}', mask)
        cv.waitKey()

    frame_shape = mask.shape

    # source (phi)
    source = np.copy(image)
    source[mask_indx] = 255.0  # to speed up best-matching

    # target (omega)
    target = np.copy(source)
    target[mask_indx] = 0.0

    if DEBUG:
        cv.imshow(f'target t={t}', target)
        cv.waitKey()

    # Confidence
    C = np.ones_like(mask, dtype=np.float)
    C[mask_indx] = 0.0

    # Data
    # TODO improve data algorithm D(p)
    D = _find_data_property(image, mask_indx, t)

    # Main Loop
    print('Started main loop.')
    not_filled = True
    while not_filled:
        # find front contour
        front = _find_border(mask, t)

        # TODO check that always holds
        if np.max(front) == 0:
            break

        # find priority
        front_c = cv.blur(C, (patch_size, patch_size))
        front_d = cv.blur(D, (patch_size, patch_size))
        # front_priority = front * front_c * front_d
        front_priority = front * (alpha * front_c + (1 - alpha) * front_d)

        # find max priority index
        prio_indx = np.unravel_index(np.argmax(front_priority, axis=None), front_priority.shape)
        assert front_priority[prio_indx] > 0

        # get patch
        top, bottom, left, right = _get_patch_coords(prio_indx, patch_size, frame_shape)
        patch = target[top:bottom + 1, left:right + 1, :]

        if DEBUG:
            h, w, _ = patch.shape
            pt = (left, top)
            temp = np.copy(target)
            cv.imshow(f'patch t={t}', cv.rectangle(temp, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2))
            cv.waitKey()

        # find best template
        patch_c = C[top:bottom + 1, left:right + 1]
        best_match = _find_best_match(source, patch, patch_c, mask_boolean, t)

        # increment time
        t += 1

        # update target (Raw)
        target[top:bottom + 1, left:right + 1, :] = best_match

        # TODO update target (Seamless)
        # hp, wp, _ = best_match.shape
        # center = (left + wp // 2, top + hp // 2)
        # bm_mask = np.full_like(best_match, fill_value=255, dtype=np.float)
        # target = cv.seamlessClone(best_match, target, bm_mask, center, cv.NORMAL_CLONE)

        if DEBUG:
            cv.imshow(f'target t={t}', target)
            cv.waitKey()

        # update mask
        mask[top:bottom + 1, left:right + 1] = 0
        if DEBUG:
            cv.imshow(f'mask t={t}', mask)
            cv.waitKey()

        # break if filled
        if np.sum(mask) == 0:
            not_filled = False

        # update confidence
        prev_c = C[top:bottom + 1, left:right + 1]
        not_sure_ind = np.where(prev_c < 1)
        new_c = front_c[top:bottom + 1, left:right + 1]
        prev_c[not_sure_ind] = new_c[not_sure_ind]
        C[top:bottom + 1, left:right + 1] = prev_c

        # logging
        if t % 20 == 0:
            _log_mask_percentage(mask)

    return target


def _find_border(mask, t):
    edges = cv.Canny(mask, 100, 200)

    if DEBUG:
        cv.imshow(f'contour t={t}', edges)
        cv.waitKey()

    return edges / 255.


def _find_data_property(image, mask_indx, t):
    image_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(image_bw, 20, 100)
    edges[mask_indx] = 0

    if DEBUG:
        cv.imshow(f'data t={t}', edges)
        cv.waitKey()

    return edges / 255.


def _get_patch_coords(prio_indx, patch_size, frame_shape):
    i_center, j_center = prio_indx
    h, w = frame_shape
    half_patch = (patch_size - 1) // 2

    top = np.maximum(i_center - half_patch, 0)
    bottom = np.minimum(i_center + half_patch, h - 1)

    left = np.maximum(j_center - half_patch, 0)
    right = np.minimum(j_center + half_patch, w - 1)

    return top, bottom, left, right


def _find_best_match(source, patch, confidence, mask_boolean, t):
    h, w, _ = source.shape
    hp, wp, _ = patch.shape

    confidence3d = np.repeat(confidence[:, :, np.newaxis], 3, axis=2)

    source_float = source.astype(np.float)

    # Main loop
    best_left, best_top = _find_best_match_loop(confidence3d, h, hp, mask_boolean, patch, source_float, w, wp)

    best_bottom, best_right = best_top + hp, best_left + wp

    if DEBUG:
        pt = (best_left, best_top)
        temp = np.copy(source)
        cv.imshow(f'best_match t={t}', cv.rectangle(temp, pt, (best_right, best_bottom), (0, 0, 255), 2))
        cv.waitKey()

    return source[best_top:best_bottom, best_left:best_right, :]


@jit(nopython=True)
def _find_best_match_loop(confidence3d, h, hp, mask_boolean, patch, source, w, wp):
    min_distance = np.inf
    best_top = None
    best_left = None
    for top in range(h - hp + 1):
        for left in range(w - wp + 1):
            bottom, right = top + hp, left + wp
            if np.prod(mask_boolean[top:bottom, left:right]) == 0.:  # touches target region
                continue

            s = source[top:bottom, left:right]

            d = np.sum(np.power(s - patch, 2) * confidence3d)

            if d < min_distance:
                min_distance = d
                best_top = top
                best_left = left

    return best_left, best_top


def _log_mask_percentage(mask):
    mask_indx = np.where(mask > 245)
    mask_boolean = np.ones_like(mask)
    mask_boolean[mask_indx] = 0

    n = np.count_nonzero(mask_boolean)
    perc = n / mask_boolean.size * 100
    print(f'Covered {perc:.2f}% of the image.')
