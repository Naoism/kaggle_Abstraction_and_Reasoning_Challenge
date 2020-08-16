import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
from xgboost import XGBClassifier
import pdb
import itertools
import random
import copy
from numpy.lib.stride_tricks import as_strided
from tqdm.notebook import tqdm
import inspect
from skimage import measure


"""
First Solution
This can solve one task.
"""

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))
evaluation_tasks = sorted(os.listdir(evaluation_path))
test_tasks = sorted(os.listdir(test_path))

def get_data(task_filename):
    with open(task_filename, 'r') as f:
        task = json.load(f)
    return task

num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]
color2num = {c: n for n, c in enumerate(num2color)}


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


def one_right(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] > 0 and j < input_shape[1]-1 and x[i][j+1] == 0:
                x[i][j+1] = vinput[i][j]
    return x

def one_left(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] > 0 and j > 0 and x[i][j-1] == 0:
                x[i][j-1] = vinput[i][j]
    return x

def one_above(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] > 0 and i > 0 and x[i-1][j] == 0:
                x[i-1][j] = vinput[i][j]
    return x

def one_bottom(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] > 0 and i < input_shape[0]-1 and x[i+1][j] == 0:
                x[i+1][j] = vinput[i][j]
    return x



def plus_shape(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] > 0 and i > 0 and x[i-1][j] == 0:
                x[i-1][j] = vinput[i][j]
            if vinput[i][j] > 0 and j > 0 and x[i][j-1] == 0:
                x[i][j-1] = vinput[i][j]
            if vinput[i][j] > 0 and i < input_shape[0]-1 and x[i+1][j] == 0:
                x[i+1][j] = vinput[i][j]
            if vinput[i][j] > 0 and j < input_shape[1]-1 and x[i][j+1] == 0:
                x[i][j+1] = vinput[i][j]
    return x

def batsu_shape(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] > 0 and i > 0 and j > 0 and x[i-1][j-1] == 0:
                x[i-1][j-1] = vinput[i][j]
            if vinput[i][j] > 0 and i < input_shape[0]-1 and j > 0 and x[i+1][j-1] == 0:
                x[i+1][j-1] = vinput[i][j]
            if vinput[i][j] > 0 and i > 0 and j < input_shape[1]-1 and x[i-1][j+1] == 0:
                x[i-1][j+1] = vinput[i][j]
            if vinput[i][j] > 0 and i < input_shape[0]-1 and j < input_shape[1]-1 and x[i+1][j+1] == 0:
                x[i+1][j+1] = vinput[i][j]
    return x


def cropToContent_right_mirror(vinput):
    x = vinput.copy()
    """ Crop an image to fit exactly the non 0 pixels """
    # Op argwhere will give us the coordinates of every non-zero point
    true_points = np.argwhere(vinput)
    if len(true_points) == 0:
        return vinput
    # Take the smallest points and use them as the top left of our crop
    top_left = true_points.min(axis=0)
    # Take the largest points and use them as the bottom right of our crop
    bottom_right = true_points.max(axis=0)
    # Crop inside the defined rectangle
    pic = vinput[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    pic_shape = pic.shape
    if pic_shape[1] % 2 == 0:
        mid = pic_shape[1]//2
        pic_mir = pic[:, 0:mid]
        pic_mir = pic_mir[:, ::-1]
        x[top_left[0]:bottom_right[0]+1, top_left[1] + mid:bottom_right[1]+1] = pic_mir
        return x
    else:
        mid = pic_shape[1]//2
        pic_mir = pic[:, 0:mid]
        pic_mir = pic_mir[:, ::-1]
        x[top_left[0]:bottom_right[0]+1, top_left[1] + mid + 1:bottom_right[1]+1] = pic_mir
        return x


def cropToContent_left_mirror(vinput):
    x = vinput.copy()
    x_rot180 = np.rot90(x, 2)
    x_rot180 = cropToContent_right_mirror(x_rot180)
    x = np.rot90(x_rot180, 2)
    return x

def cropToContent_bottom_mirror(vinput):
    x = vinput.copy()
    x_rot90 = np.rot90(x, 1)
    x_rot90 = cropToContent_right_mirror(x_rot90)
    x = np.rot90(x_rot90, 3)
    return x

def cropToContent_above_mirror(vinput):
    x = vinput.copy()
    x_rot270 = np.rot90(x, 3)
    x_rot270 = cropToContent_right_mirror(x_rot270)
    x = np.rot90(x_rot270, 1)
    return x


def cropToContent_rot90(vinput):
    x = vinput.copy()
    """ Crop an image to fit exactly the non 0 pixels """
    # Op argwhere will give us the coordinates of every non-zero point
    true_points = np.argwhere(vinput)
    if len(true_points) == 0:
        return vinput
    # Take the smallest points and use them as the top left of our crop
    top_left = true_points.min(axis=0)
    # Take the largest points and use them as the bottom right of our crop
    bottom_right = true_points.max(axis=0)
    # Crop inside the defined rectangle
    pic = vinput[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    if pic.shape[0] == pic.shape[1]:
        pic_rot90 = np.rot90(pic, 1)
        x[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = pic_rot90
        return x
    return x

def cropToContent_rot180(vinput):
    x = vinput.copy()
    """ Crop an image to fit exactly the non 0 pixels """
    # Op argwhere will give us the coordinates of every non-zero point
    true_points = np.argwhere(vinput)
    if len(true_points) == 0:
        return vinput
    # Take the smallest points and use them as the top left of our crop
    top_left = true_points.min(axis=0)
    # Take the largest points and use them as the bottom right of our crop
    bottom_right = true_points.max(axis=0)
    # Crop inside the defined rectangle
    pic = vinput[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    if pic.shape[0] == pic.shape[1]:
        pic_rot90 = np.rot90(pic, 2)
        x[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = pic_rot90
        return x
    return x

def cropToContent_rot270(vinput):
    x = vinput.copy()
    """ Crop an image to fit exactly the non 0 pixels """
    # Op argwhere will give us the coordinates of every non-zero point
    true_points = np.argwhere(vinput)
    if len(true_points) == 0:
        return vinput
    # Take the smallest points and use them as the top left of our crop
    top_left = true_points.min(axis=0)
    # Take the largest points and use them as the bottom right of our crop
    bottom_right = true_points.max(axis=0)
    # Crop inside the defined rectangle
    pic = vinput[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    if pic.shape[0] == pic.shape[1]:
        pic_rot90 = np.rot90(pic, 3)
        x[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = pic_rot90
        return x
    return x


def coloring_leftaround(vinput):
    x = vinput.copy()
    x_shape = x.shape
    left_unique = np.unique(x[:, :x_shape[1]//2])
    if len(left_unique) == 2:
        left_unique = sorted(left_unique)
        color = left_unique[-1]
        x[0, :x_shape[1]//2] = color
        x[x_shape[0]-1, :x_shape[1]//2] = color
        for i in range(1, x_shape[0]-1):
            x[i, 0] = color
    return x

def coloring_rightaround(vinput):
    x = vinput.copy()
    x_rot180 = np.rot90(x, 2)
    x_rot180 = coloring_leftaround(x_rot180)
    x = np.rot90(x_rot180, 2)
    return x

def coloring_abovearound(vinput):
    x = vinput.copy()
    x_rot90 = np.rot90(x, 1)
    x_rot90 = coloring_leftaround(x_rot90)
    x = np.rot90(x_rot90, 3)
    return x

def coloring_bottomaround(vinput):
    x = vinput.copy()
    x_rot270 = np.rot90(x, 3)
    x_rot270 = coloring_leftaround(x_rot270)
    x = np.rot90(x_rot270, 1)
    return x


def bottom_gravity(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    for i in range(input_shape[0]-1, -1, -1):
        for j in range(input_shape[1]):
            if vinput[i][j] > 0:
                color = vinput[i][j]
                count = 1
                while True:
                    if i + count < input_shape[0]:
                        if x[i+count][j] > 0:
                            x[i][j] = 0
                            x[i+count-1][j] = color
                            break
                        else:
                            count += 1
                    else:
                        x[i][j] = 0
                        x[i+count-1][j] = color
                        break
    return x

def above_gravity(vinput):
    x = vinput.copy()
    x_rot180 = np.rot90(x, 2)
    x_rot180 = bottom_gravity(x_rot180)
    x = np.rot90(x_rot180, 2)
    return x

def right_gravity(vinput):
    x = vinput.copy()
    x_rot270 = np.rot90(x, 3)
    x_rot270 = bottom_gravity(x_rot270)
    x = np.rot90(x_rot270, 1)
    return x

def left_gravity(vinput):
    x = vinput.copy()
    x_rot90 = np.rot90(x, 1)
    x_rot90 = bottom_gravity(x_rot90)
    x = np.rot90(x_rot90, 3)
    return x


def bottom_gravity_overlap(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    for i in range(input_shape[0]-1, -1, -1):
        for j in range(input_shape[1]):
            if vinput[i][j] > 0:
                color = vinput[i][j]
                count = 1
                while True:
                    if i + count < input_shape[0]:
                        if x[i+count][j] == color:
                            x[i][j] = 0
                            x[i+count-1][j] = color
                            break
                        else:
                            count += 1
                    else:
                        x[i][j] = 0
                        x[i+count-1][j] = color
                        break
    return x

def above_gravity_overlap(vinput):
    x = vinput.copy()
    x_rot180 = np.rot90(x, 2)
    x_rot180 = bottom_gravity_overlap(x_rot180)
    x = np.rot90(x_rot180, 2)
    return x

def right_gravity_overlap(vinput):
    x = vinput.copy()
    x_rot270 = np.rot90(x, 3)
    x_rot270 = bottom_gravity_overlap(x_rot270)
    x = np.rot90(x_rot270, 1)
    return x

def left_gravity_overlap(vinput):
    x = vinput.copy()
    x_rot90 = np.rot90(x, 1)
    x_rot90 = bottom_gravity_overlap(x_rot90)
    x = np.rot90(x_rot90, 3)
    return x

def drow_hline_samecolor(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    for i in range(input_shape[0]):
        row = vinput[i, :]
        uni_list = np.unique(row)
        if len(uni_list) == 2:
            for color in uni_list:
                if color > 0 and sum(row==color) >= 2:
                    for j in range(input_shape[1]):
                        if row[j] == color:
                            pos1 = j
                            break
                    for j in range(input_shape[1]-1, -1, -1):
                        if row[j] == color:
                            pos2 = j
                            break
                    x[i, pos1:pos2+1] = color
    return x

def drow_vline_samecolor(vinput):
    x = vinput.copy()
    x_rot90 = np.rot90(x, 1)
    x_rot90 = drow_hline_samecolor(x_rot90)
    x = np.rot90(x_rot90, 3)
    return x





def backgroundblack_change_blue(vinput):
    x = vinput.copy()
    color = color2num["blue"]
    x[x == 0] = color
    return x

def backgroundblack_change_red(vinput):
    x = vinput.copy()
    color = color2num["red"]
    x[x == 0] = color
    return x

def backgroundblack_change_green(vinput):
    x = vinput.copy()
    color = color2num["green"]
    x[x == 0] = color
    return x

def backgroundblack_change_yellow(vinput):
    x = vinput.copy()
    color = color2num["yellow"]
    x[x == 0] = color
    return x

def backgroundblack_change_gray(vinput):
    x = vinput.copy()
    color = color2num["gray"]
    x[x == 0] = color
    return x

def backgroundblack_change_magenta(vinput):
    x = vinput.copy()
    color = color2num["magenta"]
    x[x == 0] = color
    return x


def backgroundblack_change_orange(vinput):
    x = vinput.copy()
    color = color2num["orange"]
    x[x == 0] = color
    return x


def backgroundblack_change_sky(vinput):
    x = vinput.copy()
    color = color2num["sky"]
    x[x == 0] = color
    return x

def backgroundblack_change_brown(vinput):
    x = vinput.copy()
    color = color2num["brown"]
    x[x == 0] = color
    return x

def backgroundgray_change_black(vinput):
    x = vinput.copy()
    color = color2num["black"]
    x[x == color2num["gray"]] = color
    return x



def reverse(vinput):

    def get_1stmost_color(x):
        color_1dim = x.flatten()
        count = np.bincount(color_1dim)
        mode1st = np.argsort(count)[-1]
        if mode1st == 0:
            mode1st = np.argsort(count)[-2]
        return mode1st

    x = vinput.copy()
    color_1dim = x.flatten()
    if len(np.unique(color_1dim))==1:
        return x
    mode1st = get_1stmost_color(vinput)
    x[vinput == 0] = mode1st
    x[vinput > 0] = 0
    return x



def above_mirror_in_sameshape(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    if input_shape[0] % 2 == 1:
        x_bottom = x[(input_shape[0]//2)+1:, :]
        x_mir = x_bottom[::-1, :]
        x[:input_shape[0]//2, :] = x_mir
    else:
        x_bottom = x[input_shape[0]//2:, :]
        x_mir = x_bottom[::-1, :]
        x[:input_shape[0]//2, :] = x_mir
    return x

def bottom_mirror_in_sameshape(vinput):
    x = vinput.copy()
    x_rot180 = np.rot90(x, 2)
    x_rot180 = above_mirror_in_sameshape(x_rot180)
    x = np.rot90(x_rot180, 2)
    return x

def left_mirror_in_sameshape(vinput):
    x = vinput.copy()
    x_rot270 = np.rot90(x, 3)
    x_rot270 = above_mirror_in_sameshape(x_rot270)
    x = np.rot90(x_rot270, 1)
    return x

def right_mirror_in_sameshape(vinput):
    x = vinput.copy()
    x_rot90 = np.rot90(x, 1)
    x_rot90 = above_mirror_in_sameshape(x_rot90)
    x = np.rot90(x_rot90, 3)
    return x


def drow_hline_and_vline_samecolor(vinput):
    x_h = vinput.copy()
    x_v = vinput.copy()
    vinput_rot90 = np.rot90(vinput, 1)
    input_shape = vinput.shape
    vinput_rot90_shape = vinput_rot90.shape
    for i in range(input_shape[0]):
        row = vinput[i, :]
        uni_list = np.unique(row)
        if len(uni_list) == 2:
            for color in uni_list:
                if color > 0 and sum(row==color) >= 2:
                    for j in range(input_shape[1]):
                        if row[j] == color:
                            pos1_h = j
                            break
                    for j in range(input_shape[1]-1, -1, -1):
                        if row[j] == color:
                            pos2_h = j
                            break
                    x_h[i, pos1_h:pos2_h+1] = color

    x_v = np.rot90(x_v, 1)
    for i in range(vinput_rot90_shape[0]):
        row = vinput_rot90[i, :]
        uni_list = np.unique(row)
        if len(uni_list) == 2:
            for color in uni_list:
                if color > 0 and sum(row==color) >= 2:
                    for j in range(vinput_rot90_shape[1]):
                        if row[j] == color:
                            pos1_v = j
                            break
                    for j in range(vinput_rot90_shape[1]-1, -1, -1):
                        if row[j] == color:
                            pos2_v = j
                            break
                    x_v[i, pos1_v:pos2_v+1] = color
    x_v = np.rot90(x_v, 3)

    x = x_h.copy()
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if x_h[i][j] == 0 and x_v[i][j] > 0:
                x[i][j] = x_v[i][j]    
    return x


def one_color_down_overlap(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    for i in range(1, input_shape[0]):
        for j in range(input_shape[1]):
            if j == 0:
                if vinput[i][j] > 0 and vinput[i][j+1] == 0 and vinput[i-1][j] == 0:
                    color = vinput[i][j]
                    count = 0
                    while i+count < input_shape[0]-1:
                        count += 1
                        if vinput[i+count][j] > 0:
                            x[i+count][j] = color
                            break
            elif j == input_shape[1]-1:
                if vinput[i][j] > 0 and vinput[i][j-1] == 0 and vinput[i-1][j] == 0:
                    color = vinput[i][j]
                    count = 0
                    while i+count < input_shape[0]-1:
                        count += 1
                        if vinput[i+count][j] > 0:
                            x[i+count][j] = color
                            break
            else:
                if vinput[i][j] > 0 and vinput[i][j-1] == 0 and vinput[i][j+1] == 0 and vinput[i-1][j] == 0:
                    color = vinput[i][j]
                    count = 0
                    while i+count < input_shape[0]-1:
                        count += 1
                        if vinput[i+count][j] > 0:
                            x[i+count][j] = color
                            break
    return x


def one_color_up_overlap(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    x_rot180 = np.rot90(x, 2)
    x_rot180 = one_color_down_overlap(x_rot180)
    x = np.rot90(x_rot180, 2)
    return x

def one_color_right_overlap(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    x_rot270 = np.rot90(x, 3)
    x_rot270 = one_color_down_overlap(x_rot270)
    x = np.rot90(x_rot270, 1)
    return x

def one_color_left_overlap(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    x_rot90 = np.rot90(x, 1)
    x_rot90 = one_color_down_overlap(x_rot90)
    x = np.rot90(x_rot90, 3)
    return x



def one_color_down(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    for i in range(1, input_shape[0]):
        for j in range(input_shape[1]):
            if j == 0:
                if vinput[i][j] > 0 and vinput[i][j+1] == 0 and vinput[i-1][j] == 0:
                    color = vinput[i][j]
                    count = 0
                    while i+count < input_shape[0]-1:
                        count += 1
                        if vinput[i+count][j] > 0:
                            x[i+count-1][j] = color
                            break
            elif j == input_shape[1]-1:
                if vinput[i][j] > 0 and vinput[i][j-1] == 0 and vinput[i-1][j] == 0:
                    color = vinput[i][j]
                    count = 0
                    while i+count < input_shape[0]-1:
                        count += 1
                        if vinput[i+count][j] > 0:
                            x[i+count-1][j] = color
                            break
            else:
                if vinput[i][j] > 0 and vinput[i][j-1] == 0 and vinput[i][j+1] == 0 and vinput[i-1][j] == 0:
                    color = vinput[i][j]
                    count = 0
                    while i+count < input_shape[0]-1:
                        count += 1
                        if vinput[i+count][j] > 0:
                            x[i+count-1][j] = color
                            break
    return x


def one_color_up(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    x_rot180 = np.rot90(x, 2)
    x_rot180 = one_color_down(x_rot180)
    x = np.rot90(x_rot180, 2)
    return x

def one_color_right(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    x_rot270 = np.rot90(x, 3)
    x_rot270 = one_color_down(x_rot270)
    x = np.rot90(x_rot270, 1)
    return x

def one_color_left(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    x_rot90 = np.rot90(x, 1)
    x_rot90 = one_color_down(x_rot90)
    x = np.rot90(x_rot90, 3)
    return x


def get_closed_area(arr, frame_color):
    # depth first search
    H, W = arr.shape
    Dy = [0, -1, 0, 1]
    Dx = [1, 0, -1, 0]
    arr_padded = np.pad(arr, ((1,1),(1,1)), "constant", constant_values=0)
    searched = np.zeros(arr_padded.shape, dtype=bool)
    searched[0, 0] = True
    q = [(0, 0)]
    while q:
        y, x = q.pop()
        for dy, dx in zip(Dy, Dx):
            y_, x_ = y+dy, x+dx
            if not 0 <= y_ < H+2 or not 0 <= x_ < W+2:
                continue
            if not searched[y_][x_] and arr_padded[y_][x_]==0:
                q.append((y_, x_))
                searched[y_, x_] = True
    res = searched[1:-1, 1:-1]
    res |= arr==frame_color
    return ~res


def color_black_closed_red_ares(vinput):

    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["red"], color2num["black"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x


def color_red_closed_red_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["red"], color2num["red"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x


def color_blue_closed_red_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["red"], color2num["blue"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_green_closed_red_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["red"], color2num["green"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_yellow_closed_red_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["red"], color2num["yellow"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_gray_closed_red_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["red"], color2num["gray"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_magenta_closed_red_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["red"], color2num["magenta"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_orange_closed_red_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["red"], color2num["orange"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_sky_closed_red_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["red"], color2num["sky"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_brown_closed_red_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["red"], color2num["brown"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_black_closed_blue_ares(vinput):

    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["blue"], color2num["black"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_red_closed_blue_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["blue"], color2num["red"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_blue_closed_blue_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["blue"], color2num["blue"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x


def color_green_closed_blue_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["blue"], color2num["green"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_yellow_closed_blue_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["blue"], color2num["yellow"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_gray_closed_blue_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["blue"], color2num["gray"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_magenta_closed_blue_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["blue"], color2num["magenta"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_orange_closed_blue_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["blue"], color2num["orange"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_sky_closed_blue_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["blue"], color2num["sky"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_brown_closed_blue_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["blue"], color2num["brown"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_black_closed_green_ares(vinput):

    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["green"], color2num["black"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_red_closed_green_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["green"], color2num["red"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x


def color_blue_closed_green_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["green"], color2num["blue"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_green_closed_green_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["green"], color2num["green"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_yellow_closed_green_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["green"], color2num["yellow"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_gray_closed_green_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["green"], color2num["gray"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_magenta_closed_green_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["green"], color2num["magenta"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_orange_closed_green_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["green"], color2num["orange"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_sky_closed_green_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["green"], color2num["sky"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_brown_closed_green_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["green"], color2num["brown"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_black_closed_yellow_ares(vinput):

    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["yellow"], color2num["black"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_red_closed_yellow_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["yellow"], color2num["red"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x


def color_blue_closed_yellow_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["yellow"], color2num["blue"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_green_closed_yellow_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["yellow"], color2num["green"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_yellow_closed_yellow_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["yellow"], color2num["yellow"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_gray_closed_yellow_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["yellow"], color2num["gray"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_magenta_closed_yellow_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["yellow"], color2num["magenta"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_orange_closed_yellow_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["yellow"], color2num["orange"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_sky_closed_yellow_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["yellow"], color2num["sky"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_brown_closed_yellow_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["yellow"], color2num["brown"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_black_closed_gray_ares(vinput):

    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["gray"], color2num["black"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_red_closed_gray_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["gray"], color2num["red"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x


def color_blue_closed_gray_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["gray"], color2num["blue"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_green_closed_gray_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["gray"], color2num["green"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_yellow_closed_gray_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["gray"], color2num["yellow"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_gray_closed_gray_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["gray"], color2num["gray"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_magenta_closed_gray_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["gray"], color2num["magenta"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_orange_closed_gray_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["gray"], color2num["orange"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_sky_closed_gray_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["gray"], color2num["sky"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_brown_closed_gray_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["gray"], color2num["brown"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_black_closed_magenta_ares(vinput):

    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["magenta"], color2num["black"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_red_closed_magenta_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["magenta"], color2num["red"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x


def color_blue_closed_magenta_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["magenta"], color2num["blue"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_green_closed_magenta_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["magenta"], color2num["green"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_yellow_closed_magenta_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["magenta"], color2num["yellow"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_gray_closed_magenta_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["magenta"], color2num["gray"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_magenta_closed_magenta_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["magenta"], color2num["magenta"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_orange_closed_magenta_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["magenta"], color2num["orange"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_sky_closed_magenta_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["magenta"], color2num["sky"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_brown_closed_magenta_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["magenta"], color2num["brown"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_black_closed_orange_ares(vinput):

    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["orange"], color2num["black"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_red_closed_orange_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["orange"], color2num["red"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x


def color_blue_closed_orange_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["orange"], color2num["blue"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_green_closed_orange_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["orange"], color2num["green"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_yellow_closed_orange_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["orange"], color2num["yellow"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_gray_closed_orange_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["orange"], color2num["gray"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_magenta_closed_orange_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["orange"], color2num["magenta"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_orange_closed_orange_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["orange"], color2num["orange"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_sky_closed_orange_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["orange"], color2num["sky"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_brown_closed_orange_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["orange"], color2num["brown"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_black_closed_sky_ares(vinput):

    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["sky"], color2num["black"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_red_closed_sky_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["sky"], color2num["red"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x


def color_blue_closed_sky_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["sky"], color2num["blue"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_green_closed_sky_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["sky"], color2num["green"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_yellow_closed_sky_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["sky"], color2num["yellow"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_gray_closed_sky_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["sky"], color2num["gray"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_magenta_closed_sky_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["sky"], color2num["magenta"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_orange_closed_sky_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["sky"], color2num["orange"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_sky_closed_sky_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["sky"], color2num["sky"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_brown_closed_sky_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["sky"], color2num["brown"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_black_closed_brown_ares(vinput):

    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["brown"], color2num["black"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_red_closed_brown_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["brown"], color2num["red"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x


def color_blue_closed_brown_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["brown"], color2num["blue"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_green_closed_brown_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["brown"], color2num["green"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_yellow_closed_brown_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["brown"], color2num["yellow"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_gray_closed_brown_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["brown"], color2num["gray"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_magenta_closed_brown_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["brown"], color2num["magenta"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_orange_closed_brown_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["brown"], color2num["orange"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_sky_closed_brown_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["brown"], color2num["sky"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x

def color_brown_closed_brown_ares(vinput):
    x = vinput.copy()
    color_1dim = vinput.flatten()
    frame_color, inside_color = color2num["brown"], color2num["brown"]
    if frame_color in color_1dim and color2num["black"] in color_1dim:
        x[get_closed_area(vinput, frame_color)] = inside_color
    return x


def drow_hline_dot_to_rightside(vinput):
    try:
        x = vinput.copy()
        input_shape = vinput.shape
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                if i == 0:
                    if j == 0:
                        if vinput[i][j] > 0 and vinput[i][j+1] == 0 and vinput[i+1][j] == 0:
                            color = vinput[i][j]
                            for k in range(input_shape[1]):
                                if k <= j:
                                    continue
                                x[i][k] = color
                    elif j == input_shape[1]-1:
                        if vinput[i][j] > 0 and vinput[i][j-1] == 0 and vinput[i+1][j] == 0:
                            color = vinput[i][j]
                            for k in range(input_shape[1]):
                                if k <= j:
                                    continue
                                x[i][k] = color
                    else:
                        if vinput[i][j] > 0 and vinput[i][j-1] == 0 and vinput[i][j+1] == 0 and vinput[i+1][j] == 0:
                            color = vinput[i][j]
                            for k in range(input_shape[1]):
                                if k <= j:
                                    continue
                                x[i][k] = color
                elif i == input_shape[0]-1:
                    if j == 0:
                        if vinput[i][j] > 0 and vinput[i][j+1] == 0 and vinput[i-1][j] == 0:
                            color = vinput[i][j]
                            for k in range(input_shape[1]):
                                if k <= j:
                                    continue
                                x[i][k] = color
                    elif j == input_shape[1]-1:
                        if vinput[i][j] > 0 and vinput[i][j-1] == 0 and vinput[i-1][j] == 0:
                            color = vinput[i][j]
                            for k in range(input_shape[1]):
                                if k <= j:
                                    continue
                                x[i][k] = color
                    else:
                        if vinput[i][j] > 0 and vinput[i][j-1] == 0 and vinput[i][j+1] == 0 and vinput[i-1][j] == 0:
                            color = vinput[i][j]
                            for k in range(input_shape[1]):
                                if k <= j:
                                    continue
                                x[i][k] = color
                else:
                    if j == 0:
                        if vinput[i][j] > 0 and vinput[i][j+1] == 0 and vinput[i-1][j] == 0 and vinput[i+1][j] == 0:
                            color = vinput[i][j]
                            for k in range(input_shape[1]):
                                if k <= j:
                                    continue
                                x[i][k] = color
                    elif j == input_shape[1]-1:
                        if vinput[i][j] > 0 and vinput[i][j-1] == 0 and vinput[i-1][j] == 0 and vinput[i+1][j] == 0:
                            color = vinput[i][j]
                            for k in range(input_shape[1]):
                                if k <= j:
                                    continue
                                x[i][k] = color
                    else:
                        if vinput[i][j] > 0 and vinput[i][j-1] == 0 and vinput[i][j+1] == 0 and vinput[i-1][j] == 0 and vinput[i+1][j] == 0:
                            color = vinput[i][j]
                            for k in range(input_shape[1]):
                                if k <= j:
                                    continue
                                x[i][k] = color
        return x
    except:
        return x


def drow_hline_dot_to_leftside(vinput):
    x = vinput.copy()
    x_rot180 = np.rot90(x, 2)
    x_rot180 = drow_hline_dot_to_rightside(x_rot180)
    x = np.rot90(x_rot180, 2)
    return x

def drow_vline_dot_to_ceilside(vinput):
    x = vinput.copy()
    x_rot270 = np.rot90(x, 3)
    x_rot270 = drow_hline_dot_to_rightside(x_rot270)
    x = np.rot90(x_rot270, 1)
    return x

def drow_vline_dot_to_floorside(vinput):
    x = vinput.copy()
    x_rot90 = np.rot90(x, 1)
    x_rot90 = drow_hline_dot_to_rightside(x_rot90)
    x = np.rot90(x_rot90, 3)
    return x




def coloring_plus_shape(vinput, center_color, plus_color):
    x = vinput.copy()
    input_shape = vinput.shape
    try:
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                if i == 0:
                    if j == 0:
                        if vinput[i][j] == center_color:
                            x[i+1][j] = plus_color
                            x[i][j+1] = plus_color
                    elif j == input_shape[1]-1:
                        if vinput[i][j] == center_color:
                            x[i+1][j] = plus_color
                            x[i][j-1] = plus_color
                    else:
                        if vinput[i][j] == center_color:
                            x[i+1][j] = plus_color
                            x[i][j-1] = plus_color
                            x[i][j+1] = plus_color
                elif i == input_shape[0]-1:
                    if j == 0:
                        if vinput[i][j] == center_color:
                            x[i-1][j] = plus_color
                            x[i][j+1] = plus_color
                    elif j == input_shape[1]-1:
                        if vinput[i][j] == center_color:
                            x[i-1][j] = plus_color
                            x[i][j-1] = plus_color
                    else:
                        if vinput[i][j] == center_color:
                            x[i-1][j] = plus_color
                            x[i][j-1] = plus_color
                            x[i][j+1] = plus_color
                else:
                    if j == 0:
                        if vinput[i][j] == center_color:
                            x[i+1][j] = plus_color
                            x[i][j+1] = plus_color
                            x[i-1][j] = plus_color
                    elif j == input_shape[1]-1:
                        if vinput[i][j] == center_color:
                            x[i+1][j] = plus_color
                            x[i][j-1] = plus_color
                            x[i-1][j] = plus_color
                    else:
                        if vinput[i][j] == center_color:
                            x[i+1][j] = plus_color
                            x[i][j-1] = plus_color
                            x[i][j+1] = plus_color
                            x[i-1][j] = plus_color
        return x
    except:
        return x

def coloring_plus_shape_blue_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["red"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_blue_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["green"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_blue_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_blue_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["gray"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_blue_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["magenta"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    

def coloring_plus_shape_blue_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["orange"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_blue_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["sky"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_blue_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["brown"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_red_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["blue"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_red_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["green"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_red_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_red_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["gray"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_red_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["magenta"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    

def coloring_plus_shape_red_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["orange"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_red_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["sky"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_red_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["brown"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x

def coloring_plus_shape_green_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["blue"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_green_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["green"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_green_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_green_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["gray"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_green_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["magenta"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    

def coloring_plus_shape_green_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["orange"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_green_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["sky"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_green_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["brown"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x

def coloring_plus_shape_yellow_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["blue"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_yellow_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_yellow_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_yellow_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["gray"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_yellow_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["magenta"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    

def coloring_plus_shape_yellow_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["orange"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_yellow_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["sky"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_yellow_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["brown"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x

def coloring_plus_shape_gray_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["blue"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_gray_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_gray_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["green"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_gray_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["gray"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_gray_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["magenta"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    

def coloring_plus_shape_gray_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["orange"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_gray_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["sky"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_gray_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["brown"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x

def coloring_plus_shape_magenta_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["blue"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_magenta_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_magenta_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["green"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_magenta_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_magenta_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["gray"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    

def coloring_plus_shape_magenta_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["orange"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_magenta_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["sky"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_magenta_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["brown"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x


def coloring_plus_shape_orange_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["blue"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_orange_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_orange_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["green"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_orange_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_orange_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["gray"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    

def coloring_plus_shape_orange_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["magenta"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_orange_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["sky"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_orange_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["brown"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x

def coloring_plus_shape_sky_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["blue"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_sky_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_sky_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["green"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_sky_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_sky_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["gray"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    

def coloring_plus_shape_sky_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["magenta"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_sky_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["orange"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_sky_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["brown"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x

def coloring_plus_shape_brown_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["blue"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_brown_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_brown_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["green"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_brown_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["yellow"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_brown_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["gray"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    

def coloring_plus_shape_brown_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["magenta"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_brown_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["orange"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x
    
def coloring_plus_shape_brown_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["sky"]
    x = coloring_plus_shape(vinput, center_color, plus_color)
    return x





def coloring_batsu_shape(vinput, center_color, plus_color):
    x = vinput.copy()
    input_shape = vinput.shape
    try:
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                if i == 0:
                    if j == 0:
                        if vinput[i][j] == center_color:
                            x[i+1][j+1] = plus_color
                    elif j == input_shape[1]-1:
                        if vinput[i][j] == center_color:
                            x[i+1][j-1] = plus_color
                    else:
                        if vinput[i][j] == center_color:
                            x[i+1][j-1] = plus_color
                            x[i+1][j+1] = plus_color
                elif i == input_shape[0]-1:
                    if j == 0:
                        if vinput[i][j] == center_color:
                            x[i-1][j+1] = plus_color
                    elif j == input_shape[1]-1:
                        if vinput[i][j] == center_color:
                            x[i-1][j-1] = plus_color
                    else:
                        if vinput[i][j] == center_color:
                            x[i-1][j-1] = plus_color
                            x[i-1][j+1] = plus_color
                else:
                    if j == 0:
                        if vinput[i][j] == center_color:
                            x[i+1][j+1] = plus_color
                            x[i-1][j+1] = plus_color
                    elif j == input_shape[1]-1:
                        if vinput[i][j] == center_color:
                            x[i+1][j-1] = plus_color
                            x[i-1][j-1] = plus_color
                    else:
                        if vinput[i][j] == center_color:
                            x[i+1][j+1] = plus_color
                            x[i-1][j+1] = plus_color
                            x[i+1][j-1] = plus_color
                            x[i-1][j+1] = plus_color
        return x
    except:
        return x


def coloring_batsu_shape_blue_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["red"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_blue_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["green"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_blue_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_blue_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["gray"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_blue_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["magenta"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    

def coloring_batsu_shape_blue_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["orange"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_blue_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["sky"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_blue_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["blue"]
    plus_color = color2num["brown"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_red_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["blue"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_red_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["green"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_red_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_red_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["gray"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_red_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["magenta"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    

def coloring_batsu_shape_red_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["orange"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_red_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["sky"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_red_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["red"]
    plus_color = color2num["brown"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x

def coloring_batsu_shape_green_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["blue"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_green_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["green"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_green_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_green_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["gray"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_green_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["magenta"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    

def coloring_batsu_shape_green_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["orange"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_green_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["sky"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_green_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["green"]
    plus_color = color2num["brown"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x

def coloring_batsu_shape_yellow_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["blue"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_yellow_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_yellow_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_yellow_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["gray"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_yellow_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["magenta"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    

def coloring_batsu_shape_yellow_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["orange"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_yellow_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["sky"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_yellow_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["yellow"]
    plus_color = color2num["brown"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x

def coloring_batsu_shape_gray_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["blue"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_gray_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_gray_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["green"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_gray_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["gray"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_gray_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["magenta"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    

def coloring_batsu_shape_gray_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["orange"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_gray_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["sky"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_gray_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["gray"]
    plus_color = color2num["brown"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x

def coloring_batsu_shape_magenta_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["blue"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_magenta_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_magenta_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["green"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_magenta_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_magenta_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["gray"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    

def coloring_batsu_shape_magenta_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["orange"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_magenta_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["sky"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_magenta_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["magenta"]
    plus_color = color2num["brown"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x


def coloring_batsu_shape_orange_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["blue"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_orange_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_orange_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["green"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_orange_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_orange_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["gray"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    

def coloring_batsu_shape_orange_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["magenta"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_orange_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["sky"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_orange_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["orange"]
    plus_color = color2num["brown"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x

def coloring_batsu_shape_sky_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["blue"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_sky_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_sky_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["green"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_sky_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_sky_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["gray"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    

def coloring_batsu_shape_sky_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["magenta"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_sky_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["orange"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_sky_to_brown(vinput):
    x = vinput.copy()
    center_color = color2num["sky"]
    plus_color = color2num["brown"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x

def coloring_batsu_shape_brown_to_blue(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["blue"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_brown_to_red(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_brown_to_green(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["green"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_brown_to_yellow(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["yellow"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_brown_to_gray(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["gray"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    

def coloring_batsu_shape_brown_to_magenta(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["magenta"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_brown_to_orange(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["orange"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x
    
def coloring_batsu_shape_brown_to_sky(vinput):
    x = vinput.copy()
    center_color = color2num["brown"]
    plus_color = color2num["sky"]
    x = coloring_batsu_shape(vinput, center_color, plus_color)
    return x


def coloring_blue_to_red(vinput):
    x = vinput.copy()
    color_org = color2num["blue"]
    color_changed = color2num["red"]
    x[vinput == color_org] == color_changed
    return x


def coloring_blue_to_green(vinput):
    x = vinput.copy()
    color_org = color2num["blue"]
    color_changed = color2num["green"]
    x[vinput == color_org] == color_changed
    return x

def coloring_blue_to_yellow(vinput):
    x = vinput.copy()
    color_org = color2num["blue"]
    color_changed = color2num["yellow"]
    x[vinput == color_org] == color_changed
    return x

def coloring_blue_to_gray(vinput):
    x = vinput.copy()
    color_org = color2num["blue"]
    color_changed = color2num["gray"]
    x[vinput == color_org] == color_changed
    return x

def coloring_blue_to_magenta(vinput):
    x = vinput.copy()
    color_org = color2num["blue"]
    color_changed = color2num["magenta"]
    x[vinput == color_org] == color_changed
    return x

def coloring_blue_to_orange(vinput):
    x = vinput.copy()
    color_org = color2num["blue"]
    color_changed = color2num["orange"]
    x[vinput == color_org] == color_changed
    return x

def coloring_blue_to_sky(vinput):
    x = vinput.copy()
    color_org = color2num["blue"]
    color_changed = color2num["sky"]
    x[vinput == color_org] == color_changed
    return x

def coloring_blue_to_brown(vinput):
    x = vinput.copy()
    color_org = color2num["blue"]
    color_changed = color2num["brown"]
    x[vinput == color_org] == color_changed
    return x

def coloring_red_to_blue(vinput):
    x = vinput.copy()
    color_org = color2num["red"]
    color_changed = color2num["blue"]
    x[vinput == color_org] == color_changed
    return x


def coloring_red_to_green(vinput):
    x = vinput.copy()
    color_org = color2num["red"]
    color_changed = color2num["green"]
    x[vinput == color_org] == color_changed
    return x

def coloring_red_to_yellow(vinput):
    x = vinput.copy()
    color_org = color2num["red"]
    color_changed = color2num["yellow"]
    x[vinput == color_org] == color_changed
    return x

def coloring_red_to_gray(vinput):
    x = vinput.copy()
    color_org = color2num["red"]
    color_changed = color2num["gray"]
    x[vinput == color_org] == color_changed
    return x

def coloring_red_to_magenta(vinput):
    x = vinput.copy()
    color_org = color2num["red"]
    color_changed = color2num["magenta"]
    x[vinput == color_org] == color_changed
    return x

def coloring_red_to_orange(vinput):
    x = vinput.copy()
    color_org = color2num["red"]
    color_changed = color2num["orange"]
    x[vinput == color_org] == color_changed
    return x

def coloring_red_to_sky(vinput):
    x = vinput.copy()
    color_org = color2num["red"]
    color_changed = color2num["sky"]
    x[vinput == color_org] == color_changed
    return x

def coloring_red_to_brown(vinput):
    x = vinput.copy()
    color_org = color2num["red"]
    color_changed = color2num["brown"]
    x[vinput == color_org] == color_changed
    return x


def coloring_green_to_blue(vinput):
    x = vinput.copy()
    color_org = color2num["green"]
    color_changed = color2num["blue"]
    x[vinput == color_org] == color_changed
    return x


def coloring_green_to_red(vinput):
    x = vinput.copy()
    color_org = color2num["green"]
    color_changed = color2num["red"]
    x[vinput == color_org] == color_changed
    return x

def coloring_green_to_yellow(vinput):
    x = vinput.copy()
    color_org = color2num["green"]
    color_changed = color2num["yellow"]
    x[vinput == color_org] == color_changed
    return x

def coloring_green_to_gray(vinput):
    x = vinput.copy()
    color_org = color2num["green"]
    color_changed = color2num["gray"]
    x[vinput == color_org] == color_changed
    return x

def coloring_green_to_magenta(vinput):
    x = vinput.copy()
    color_org = color2num["green"]
    color_changed = color2num["magenta"]
    x[vinput == color_org] == color_changed
    return x

def coloring_green_to_orange(vinput):
    x = vinput.copy()
    color_org = color2num["green"]
    color_changed = color2num["orange"]
    x[vinput == color_org] == color_changed
    return x

def coloring_green_to_sky(vinput):
    x = vinput.copy()
    color_org = color2num["green"]
    color_changed = color2num["sky"]
    x[vinput == color_org] == color_changed
    return x

def coloring_green_to_brown(vinput):
    x = vinput.copy()
    color_org = color2num["green"]
    color_changed = color2num["brown"]
    x[vinput == color_org] == color_changed
    return x

def coloring_yellow_to_blue(vinput):
    x = vinput.copy()
    color_org = color2num["yellow"]
    color_changed = color2num["blue"]
    x[vinput == color_org] == color_changed
    return x


def coloring_yellow_to_red(vinput):
    x = vinput.copy()
    color_org = color2num["yellow"]
    color_changed = color2num["red"]
    x[vinput == color_org] == color_changed
    return x

def coloring_yellow_to_green(vinput):
    x = vinput.copy()
    color_org = color2num["yellow"]
    color_changed = color2num["green"]
    x[vinput == color_org] == color_changed
    return x

def coloring_yellow_to_gray(vinput):
    x = vinput.copy()
    color_org = color2num["yellow"]
    color_changed = color2num["gray"]
    x[vinput == color_org] == color_changed
    return x

def coloring_yellow_to_magenta(vinput):
    x = vinput.copy()
    color_org = color2num["yellow"]
    color_changed = color2num["magenta"]
    x[vinput == color_org] == color_changed
    return x

def coloring_yellow_to_orange(vinput):
    x = vinput.copy()
    color_org = color2num["yellow"]
    color_changed = color2num["orange"]
    x[vinput == color_org] == color_changed
    return x

def coloring_yellow_to_sky(vinput):
    x = vinput.copy()
    color_org = color2num["yellow"]
    color_changed = color2num["sky"]
    x[vinput == color_org] == color_changed
    return x

def coloring_yellow_to_brown(vinput):
    x = vinput.copy()
    color_org = color2num["yellow"]
    color_changed = color2num["brown"]
    x[vinput == color_org] == color_changed
    return x


def coloring_gray_to_blue(vinput):
    x = vinput.copy()
    color_org = color2num["gray"]
    color_changed = color2num["blue"]
    x[vinput == color_org] == color_changed
    return x


def coloring_gray_to_red(vinput):
    x = vinput.copy()
    color_org = color2num["gray"]
    color_changed = color2num["red"]
    x[vinput == color_org] == color_changed
    return x

def coloring_gray_to_green(vinput):
    x = vinput.copy()
    color_org = color2num["gray"]
    color_changed = color2num["green"]
    x[vinput == color_org] == color_changed
    return x

def coloring_gray_to_yellow(vinput):
    x = vinput.copy()
    color_org = color2num["gray"]
    color_changed = color2num["yellow"]
    x[vinput == color_org] == color_changed
    return x

def coloring_gray_to_magenta(vinput):
    x = vinput.copy()
    color_org = color2num["gray"]
    color_changed = color2num["magenta"]
    x[vinput == color_org] == color_changed
    return x

def coloring_gray_to_orange(vinput):
    x = vinput.copy()
    color_org = color2num["gray"]
    color_changed = color2num["orange"]
    x[vinput == color_org] == color_changed
    return x

def coloring_gray_to_sky(vinput):
    x = vinput.copy()
    color_org = color2num["gray"]
    color_changed = color2num["sky"]
    x[vinput == color_org] == color_changed
    return x

def coloring_gray_to_brown(vinput):
    x = vinput.copy()
    color_org = color2num["gray"]
    color_changed = color2num["brown"]
    x[vinput == color_org] == color_changed
    return x

def coloring_magenta_to_blue(vinput):
    x = vinput.copy()
    color_org = color2num["magenta"]
    color_changed = color2num["blue"]
    x[vinput == color_org] == color_changed
    return x


def coloring_magenta_to_red(vinput):
    x = vinput.copy()
    color_org = color2num["magenta"]
    color_changed = color2num["red"]
    x[vinput == color_org] == color_changed
    return x

def coloring_magenta_to_green(vinput):
    x = vinput.copy()
    color_org = color2num["magenta"]
    color_changed = color2num["green"]
    x[vinput == color_org] == color_changed
    return x

def coloring_magenta_to_yellow(vinput):
    x = vinput.copy()
    color_org = color2num["magenta"]
    color_changed = color2num["yellow"]
    x[vinput == color_org] == color_changed
    return x

def coloring_magenta_to_gray(vinput):
    x = vinput.copy()
    color_org = color2num["magenta"]
    color_changed = color2num["gray"]
    x[vinput == color_org] == color_changed
    return x

def coloring_magenta_to_orange(vinput):
    x = vinput.copy()
    color_org = color2num["magenta"]
    color_changed = color2num["orange"]
    x[vinput == color_org] == color_changed
    return x

def coloring_magenta_to_sky(vinput):
    x = vinput.copy()
    color_org = color2num["magenta"]
    color_changed = color2num["sky"]
    x[vinput == color_org] == color_changed
    return x

def coloring_magenta_to_brown(vinput):
    x = vinput.copy()
    color_org = color2num["magenta"]
    color_changed = color2num["brown"]
    x[vinput == color_org] == color_changed
    return x


def coloring_orange_to_blue(vinput):
    x = vinput.copy()
    color_org = color2num["orange"]
    color_changed = color2num["blue"]
    x[vinput == color_org] == color_changed
    return x


def coloring_orange_to_red(vinput):
    x = vinput.copy()
    color_org = color2num["orange"]
    color_changed = color2num["red"]
    x[vinput == color_org] == color_changed
    return x

def coloring_orange_to_green(vinput):
    x = vinput.copy()
    color_org = color2num["orange"]
    color_changed = color2num["green"]
    x[vinput == color_org] == color_changed
    return x

def coloring_orange_to_yellow(vinput):
    x = vinput.copy()
    color_org = color2num["orange"]
    color_changed = color2num["yellow"]
    x[vinput == color_org] == color_changed
    return x

def coloring_orange_to_gray(vinput):
    x = vinput.copy()
    color_org = color2num["orange"]
    color_changed = color2num["gray"]
    x[vinput == color_org] == color_changed
    return x

def coloring_orange_to_magenta(vinput):
    x = vinput.copy()
    color_org = color2num["orange"]
    color_changed = color2num["magenta"]
    x[vinput == color_org] == color_changed
    return x

def coloring_orange_to_sky(vinput):
    x = vinput.copy()
    color_org = color2num["orange"]
    color_changed = color2num["sky"]
    x[vinput == color_org] == color_changed
    return x

def coloring_orange_to_brown(vinput):
    x = vinput.copy()
    color_org = color2num["orange"]
    color_changed = color2num["brown"]
    x[vinput == color_org] == color_changed
    return x


def coloring_sky_to_blue(vinput):
    x = vinput.copy()
    color_org = color2num["sky"]
    color_changed = color2num["blue"]
    x[vinput == color_org] == color_changed
    return x


def coloring_sky_to_red(vinput):
    x = vinput.copy()
    color_org = color2num["sky"]
    color_changed = color2num["red"]
    x[vinput == color_org] == color_changed
    return x

def coloring_sky_to_green(vinput):
    x = vinput.copy()
    color_org = color2num["sky"]
    color_changed = color2num["green"]
    x[vinput == color_org] == color_changed
    return x

def coloring_sky_to_yellow(vinput):
    x = vinput.copy()
    color_org = color2num["sky"]
    color_changed = color2num["yellow"]
    x[vinput == color_org] == color_changed
    return x

def coloring_sky_to_gray(vinput):
    x = vinput.copy()
    color_org = color2num["sky"]
    color_changed = color2num["gray"]
    x[vinput == color_org] == color_changed
    return x

def coloring_sky_to_magenta(vinput):
    x = vinput.copy()
    color_org = color2num["sky"]
    color_changed = color2num["magenta"]
    x[vinput == color_org] == color_changed
    return x

def coloring_sky_to_orange(vinput):
    x = vinput.copy()
    color_org = color2num["sky"]
    color_changed = color2num["orange"]
    x[vinput == color_org] == color_changed
    return x

def coloring_sky_to_brown(vinput):
    x = vinput.copy()
    color_org = color2num["sky"]
    color_changed = color2num["brown"]
    x[vinput == color_org] == color_changed
    return x

def coloring_brown_to_blue(vinput):
    x = vinput.copy()
    color_org = color2num["brown"]
    color_changed = color2num["blue"]
    x[vinput == color_org] == color_changed
    return x


def coloring_brown_to_red(vinput):
    x = vinput.copy()
    color_org = color2num["brown"]
    color_changed = color2num["red"]
    x[vinput == color_org] == color_changed
    return x

def coloring_brown_to_green(vinput):
    x = vinput.copy()
    color_org = color2num["brown"]
    color_changed = color2num["green"]
    x[vinput == color_org] == color_changed
    return x

def coloring_brown_to_yellow(vinput):
    x = vinput.copy()
    color_org = color2num["brown"]
    color_changed = color2num["yellow"]
    x[vinput == color_org] == color_changed
    return x

def coloring_brown_to_gray(vinput):
    x = vinput.copy()
    color_org = color2num["brown"]
    color_changed = color2num["gray"]
    x[vinput == color_org] == color_changed
    return x

def coloring_brown_to_magenta(vinput):
    x = vinput.copy()
    color_org = color2num["brown"]
    color_changed = color2num["magenta"]
    x[vinput == color_org] == color_changed
    return x

def coloring_brown_to_orange(vinput):
    x = vinput.copy()
    color_org = color2num["brown"]
    color_changed = color2num["orange"]
    x[vinput == color_org] == color_changed
    return x

def coloring_brown_to_sky(vinput):
    x = vinput.copy()
    color_org = color2num["brown"]
    color_changed = color2num["sky"]
    x[vinput == color_org] == color_changed
    return x


def drow_hline_blue(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["blue"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[i, :] = color
    return x

def drow_hline_red(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["red"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[i, :] = color
    return x

def drow_hline_green(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["green"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[i, :] = color
    return x


def drow_hline_yellow(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["yellow"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[i, :] = color
    return x
    
def drow_hline_gray(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["gray"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[i, :] = color
    return x

def drow_hline_magenta(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["magenta"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[i, :] = color
    return x

def drow_hline_orange(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["orange"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[i, :] = color
    return x


def drow_hline_sky(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["sky"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[i, :] = color
    return x

def drow_hline_brown(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["brown"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[i, :] = color
    return x

def drow_vline_blue(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["blue"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[:, j] = color
    return x

def drow_vline_red(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["red"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[:, j] = color
    return x

def drow_vline_green(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["green"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[:, j] = color
    return x


def drow_vline_yellow(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["yellow"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[:, j] = color
    return x
    
def drow_vline_gray(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["gray"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[:, j] = color
    return x

def drow_vline_magenta(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["magenta"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[:, j] = color
    return x

def drow_vline_orange(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["orange"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[:, j] = color
    return x


def drow_vline_sky(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["sky"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[:, j] = color
    return x

def drow_vline_brown(vinput):
    x = vinput.copy()
    input_shape = vinput.shape
    color = color2num["brown"]
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if vinput[i][j] == color:
                x[:, j] = color
    return x


# ShapeSize Change
def extract_maxcolorshape(vinput):
    x = vinput.copy()

    def get_1stmost_color(x):
        color_1dim = x.flatten()
        if len(np.unique(color_1dim))==1:
            return color_1dim[0]
        count = np.bincount(color_1dim)
        mode1st = np.argsort(count)[-1]
        return mode1st

    color = get_1stmost_color(x)
    x = vinput[vinput == color]
    return x


def extract_mincolorshape(vinput):
    x = vinput.copy()

    def get_1stleast_color(x):
        color_1dim = x.flatten()
        if len(np.unique(color_1dim))==1:
            return color_1dim[0]
        count = np.bincount(color_1dim)
        count = np.where(count == 0, 100, count)
        least = np.argsort(count)[0]
        return least

    color = get_1stleast_color(x)
    x = vinput[vinput == color]
    return x


def cropToContent(vinput):
    """ Crop an image to fit exactly the non 0 pixels """
    # Op argwhere will give us the coordinates of every non-zero point
    true_points = np.argwhere(vinput)
    if len(true_points) == 0:
        return vinput
    # Take the smallest points and use them as the top left of our crop
    top_left = true_points.min(axis=0)
    # Take the largest points and use them as the bottom right of our crop
    bottom_right = true_points.max(axis=0)
    # Crop inside the defined rectangle
    x = vinput[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    return x

# ========================
#    Shape big to Small 
# ========================

def check_symmetric(a):
    try:
        sym = 1
        if np.array_equal(a, a.T):
            sym *= 2 #Check main diagonal symmetric (top left to bottom right)
        if np.array_equal(a, np.flip(a).T):
            sym *= 3 #Check antidiagonal symmetric (top right to bottom left)
        if np.array_equal(a, np.flipud(a)):
            sym *= 5 # Check horizontal symmetric of array
        if np.array_equal(a, np.fliplr(a)):
            sym *= 7 # Check vertical symmetric of array
        return sym
    except:
        return 0


def bbox(a):
    try:
        # t[t<0]があるか否か
        r = np.any(a, axis=1)
        c = np.any(a, axis=0)
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    except:
        return 0,a.shape[0],0,a.shape[1]

def cmask(t_in):
    cmin = 999
    cm = 0
    for c in range(10):
        t = t_in.copy().astype('int8')
        t[t==c],t[t>0],t[t<0]=-1,0,1
        b = bbox(t)
        a = (b[1]-b[0])*(b[3]-b[2])
        s = (t[b[0]:b[1],b[2]:b[3]]).sum()
        if a>2 and a<cmin and s==a:
            cmin=a
            cm=c
    return cm


def in_out_diff(t_in, t_out):
    x_in, y_in = t_in.shape
    x_out, y_out = t_out.shape
    # (2, 4) -> [[0000], [0000]]
    diff = np.zeros((max(x_in, x_out), max(y_in, y_out)))
    # [[-t_in+t_out, -t_in+t_out, -t_in+t_out, -t_in+t_out], 
    # [-t_in+t_out, -t_in+t_out, -t_in+t_out, -t_in+t_out]]
    diff[:x_in, :y_in] -= t_in
    diff[:x_out, :y_out] += t_out
    return diff


def call_pred_train(t_in, t_out, pred_func):
    feat = {}
    feat['s_out'] = t_out.shape
    if t_out.shape==t_in.shape:
        diff = in_out_diff(t_in,t_out)
        feat['diff'] = diff
        feat['cm'] = t_in[diff!=0].max()
    else:
        # (2, 4) -> (3, 7)  ==> (-1, -3)
        feat['diff'] = (t_in.shape[0]-t_out.shape[0],t_in.shape[1]-t_out.shape[1])
        feat['cm'] = cmask(t_in)
    feat['sym'] = check_symmetric(t_out)
    # 引数に与えたファンクションの引数情報を、
    # tuple(args, varargs, keywords,　defaults)という形式で返す。argsに引数のリストが入っている。
    args = inspect.getargspec(pred_func).args
    if len(args)==1:
        return pred_func(t_in)
    elif len(args)==2:
        t_pred = pred_func(t_in,feat[args[1]])    
    elif len(args)==3:
        t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
    feat['sizeok'] = len(t_out)==len(t_pred)
    t_pred = np.resize(t_pred,t_out.shape)
    acc = (t_pred==t_out).sum()/t_out.size
    return t_pred, feat, acc


def mask_rect(a):
    r,c = a.shape
    m = a.copy().astype('uint8')
    for i in range(r-1):
        for j in range(c-1):
            if m[i,j]==m[i+1,j]==m[i,j+1]==m[i+1,j+1]>=1:m[i,j]=2
            if m[i,j]==m[i+1,j]==1 and m[i,j-1]==2:m[i,j]=2
            if m[i,j]==m[i,j+1]==1 and m[i-1,j]==2:m[i,j]=2
            if m[i,j]==1 and m[i-1,j]==m[i,j-1]==2:m[i,j]=2
    m[m==1]=0
    return (m==2)

def crop_min(t_in):
    try:
        b = np.bincount(t_in.flatten(),minlength=10)
        c = int(np.where(b==np.min(b[np.nonzero(b)]))[0])
        coords = np.argwhere(t_in==c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return t_in[x_min:x_max+1, y_min:y_max+1]
    except:
        return t_in



def get_tile(img ,mask):
    try:
        m,n = img.shape
        a = img.copy().astype('int8')
        a[mask] = -1
        r=c=0
        for x in range(n):
            if np.count_nonzero(a[0:m,x]<0):continue
            for r in range(2,m):
                if 2*r<m and (a[0:r,x]==a[r:2*r,x]).all():break
            if r<m:break
            else: r=0
        for y in range(m):
            if np.count_nonzero(a[y,0:n]<0):continue
            for c in range(2,n):
                if 2*c<n and (a[y,0:c]==a[y,c:2*c]).all():break
            if c<n:break
            else: c=0
        if c>0:
            for x in range(n-c):
                if np.count_nonzero(a[:,x]<0)==0:
                    a[:,x+c]=a[:,x]
                elif np.count_nonzero(a[:,x+c]<0)==0:
                    a[:,x]=a[:,x+c]
        if r>0:
            for y in range(m-r):
                if np.count_nonzero(a[y,:]<0)==0:
                    a[y+r,:]=a[y,:]
                elif np.count_nonzero(a[y+r,:]<0)==0:
                    a[y,:]=a[y+r,:]
        return a[r:2*r,c:2*c]
    except:
        return a[0:1,0:1]


def patch_image(t_in,s_out,cm=0):
    try:
        t = t_in.copy()
        ty,tx=t.shape
        if cm>0:
            m = mask_rect(t==cm)
        else:
            m = (t==cm)   
        tile = get_tile(t ,m)
        if tile.size>2 and s_out==t.shape:
            rt = np.tile(tile,(1+ty//tile.shape[0],1+tx//tile.shape[1]))[0:ty,0:tx]
            if (rt[~m]==t[~m]).all():
                return rt
        for i in range(6):
            m = (t==cm)
            t -= cm
            if tx==ty:
                a = np.maximum(t,t.T)
                if (a[~m]==t[~m]).all():t=a.copy()
                a = np.maximum(t,np.flip(t).T)
                if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.flipud(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.fliplr(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            t += cm
            m = (t==cm)
            lms = measure.label(m.astype('uint8'))
            for l in range(1,lms.max()+1):
                lm = np.argwhere(lms==l)
                lm = np.argwhere(lms==l)
                x_min = max(0,lm[:,1].min()-1)
                x_max = min(lm[:,1].max()+2,t.shape[0])
                y_min = max(0,lm[:,0].min()-1)
                y_max = min(lm[:,0].max()+2,t.shape[1])
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                if i==1:
                    sy//=2
                    y_max=y_min+sx
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                allst = as_strided(t, shape=(ty,tx,sy,sx),strides=2*t.strides)    
                allst = allst.reshape(-1,sy,sx)
                allst = np.array([a for a in allst if np.count_nonzero(a==cm)==0])
                gm = (gap!=cm)
                for a in allst:
                    if sx==sy:
                        fpd = a.T
                        fad = np.flip(a).T
                        if i==1:gm[sy-1,0]=gm[0,sx-1]=False
                        if (fpd[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fpd)
                            t[y_min:y_max,x_min:x_max] = gap
                            break
                        if i==1:gm[0,0]=gm[sy-1,sx-1]=False
                        if (fad[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fad)
                            t[y_min:y_max,x_min:x_max] = gap
                            break 
                    fud = np.flipud(a)
                    flr = np.fliplr(a)
                    if i==1:gm[sy-1,0]=gm[0,sx-1]=gm[0,0]=gm[sy-1,sx-1]=False
                    if (a[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,a)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (fud[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,fud)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (flr[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,flr)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
        if s_out==t.shape:
            return t
        else:
            m = (t_in==cm)
            return np.resize(t[m],crop_min(m).shape)
    except:
        return np.resize(t_in, s_out)


def call_pred_test(t_in, pred_func, feat):
    args = inspect.getargspec(pred_func).args
    if len(args)==1:
        return pred_func(t_in)
    elif len(args)==2:
        t_pred = pred_func(t_in,feat[args[1]]) 
    elif len(args)==3:
        t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
    return t_pred







def resize_c(a):
    c = np.count_nonzero(np.bincount(a.flatten(),minlength=10)[1:])
    return np.repeat(np.repeat(a, c, axis=0), c, axis=1)

def resize_2(a):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

def resize_o(a,s):
    try:
        nx,ny = s[1]//a.shape[1],s[0]//a.shape[0]
        return np.repeat(np.repeat(a, nx, axis=0), ny, axis=1)
    except:
        return a

def repeat_1(a,s):
    try:
        si = a.shape
        nx,ny = s[1]//si[1],s[0]//si[0]
        return np.tile(a,(nx,ny))
    except:
        return a

def repeat_2(a):
    return np.tile(a,a.shape)


def crop_max(a):
    try:
        b = np.bincount(a.flatten(),minlength=10)
        b[0] = 255
        c = np.argsort(b)[-2]
        coords = np.argwhere(a==c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return a[x_min:x_max+1, y_min:y_max+1]
    except:
        return a

def crop_min(a):
    try:
        b = np.bincount(a.flatten(),minlength=10)
        c = int(np.where(b==np.min(b[np.nonzero(b)]))[0])
        coords = np.argwhere(a==c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return a[x_min:x_max+1, y_min:y_max+1]
    except:
        return a 


def upsampled_and_tiled(x):
    x_upsampled = x.repeat(3, axis=0).repeat(3, axis=1)
    x_tiled = np.tile(x, (3, 3))
    y = x_upsampled & x_tiled
    return y
    
    

def Compare(voutput, vpred):

    if voutput.shape != vpred.shape:
        return 0

    voutput_shape = voutput.shape
    acc = 0
    for i in range(voutput_shape[0]):
        for j in range(voutput_shape[1]):
            if voutput[i][j] == vpred[i][j]:
                acc += 1
    acc = acc / (voutput_shape[0]*voutput_shape[1])
    return acc




pred_function_list = [resize_c, resize_2, resize_o, repeat_1, repeat_2, crop_max, crop_min,
                     upsampled_and_tiled]

answer_list_count = 0
count = 0
solved_count = 0
answer_list = []
random.seed(42)
for i in range(len(test_tasks)):
    task = get_data(str(test_path / test_tasks[i]))
#     print(test_tasks[i])

    count = 0
    solve_flag = False
    best_function_list = []
    tmp_acc_list = [0] * (len(task["train"]))
    max_acc_list = [0] * (len(task["train"]))
    max_pred_list = [0] * (len(task["train"]))
    
    input_shape = np.array(task["train"][0]["input"]).shape
    output_shape = np.array(task["train"][0]["output"]).shape
    if input_shape != output_shape:
        for pred_func_pos, pred_func in enumerate(pred_function_list):
#             print("pred_func :", pred_func)
            shape_ok = 0
            identify = 0
            for task_i, t in enumerate(task["train"]):
                t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
                sz_out = t_out.shape
                if len(inspect.getargspec(pred_func).args)==1:
                    t_pred = pred_func(t_in)
                else:
                    t_pred = pred_func(t_in,sz_out)
                max_pred_list[task_i] = t_pred
                if t_out.shape == t_pred.shape:
                    shape_ok += 1
                if t_out.shape == t_pred.shape:
                    if (t_pred==t_out).all():
                        identify += 1
            if identify == len(task["train"]):
                print("")
                print("SOLVED!!")
                print("IDENTIFY!!")
                print("")
                print("pred_func :", pred_func.__name__)
                for task_i, t in enumerate(task["test"]):
                    print(test_tasks[i])
                    t_in = np.array(t["input"]).astype('uint8')
                    if len(inspect.getargspec(pred_func).args)==1:
                        pred_flatten = pred_func(t_in)
                        pred_flatten = flattener(pred_flatten.tolist())
                        answer_list.append(pred_flatten)
                        print("vpred", pred_func(t_in))
                        solved_count += 1
                    else:
                        pred_flatten = pred_func(t_in, sz_out)
                        pred_flatten = flattener(pred_flatten.tolist())
                        answer_list.append(pred_flatten)
                        print("vpred", pred_func(t_in, sz_out))
                        solved_count += 1
                print("Next Task")
                break
                    
            if shape_ok == len(task["train"]):
                for task_i, t in enumerate(task["train"]):
                    t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
                    sz_out = t_out.shape
                    if len(inspect.getargspec(pred_func).args)==1:
                        t_pred = pred_func(t_in)
                    else:
                        t_pred = pred_func(t_in,sz_out)
                    acc = Compare(t_out, t_pred)
                    max_acc_list[task_i] = acc
                print("shape_ok")
#                 print("pred_func :", pred_func.__name__)
#                 print("max_pred_list", max_pred_list)
                function_list = [one_right, one_left, one_above, one_bottom, 
                                plus_shape, batsu_shape,
                                cropToContent_right_mirror, cropToContent_left_mirror, cropToContent_above_mirror, cropToContent_bottom_mirror,
                                cropToContent_rot90, cropToContent_rot180, cropToContent_rot270,
                                coloring_leftaround, coloring_rightaround, coloring_abovearound, coloring_bottomaround,
                                bottom_gravity, above_gravity, right_gravity, left_gravity,
                                bottom_gravity_overlap, above_gravity_overlap, right_gravity_overlap, left_gravity_overlap,
                                drow_hline_samecolor, drow_vline_samecolor,
                                backgroundblack_change_blue, backgroundblack_change_red, backgroundblack_change_green,
                                backgroundblack_change_yellow, backgroundblack_change_gray, backgroundblack_change_magenta,
                                backgroundblack_change_orange, backgroundblack_change_sky, backgroundblack_change_brown,
                                backgroundgray_change_black, 
                                reverse,
                                above_mirror_in_sameshape, bottom_mirror_in_sameshape, left_mirror_in_sameshape, right_mirror_in_sameshape,
                                drow_hline_and_vline_samecolor,
                                one_color_down, one_color_up, one_color_right, one_color_left,
                                one_color_down_overlap, one_color_up_overlap, one_color_right_overlap, one_color_left_overlap,
                                color_black_closed_red_ares, color_red_closed_red_ares, color_blue_closed_red_ares, 
                                color_green_closed_red_ares, color_yellow_closed_red_ares, color_gray_closed_red_ares,
                                color_magenta_closed_red_ares, color_orange_closed_red_ares, color_sky_closed_red_ares, color_brown_closed_red_ares,
                                color_black_closed_blue_ares, color_red_closed_blue_ares, color_blue_closed_blue_ares, 
                                color_green_closed_blue_ares, color_yellow_closed_blue_ares, color_gray_closed_blue_ares,
                                color_magenta_closed_blue_ares, color_orange_closed_blue_ares, color_sky_closed_blue_ares, color_brown_closed_blue_ares,
                                color_black_closed_green_ares, color_red_closed_green_ares, color_blue_closed_green_ares, 
                                color_green_closed_green_ares, color_yellow_closed_green_ares, color_gray_closed_green_ares,
                                color_magenta_closed_green_ares, color_orange_closed_green_ares, color_sky_closed_green_ares, color_brown_closed_green_ares,
                                color_black_closed_yellow_ares, color_red_closed_yellow_ares, color_blue_closed_yellow_ares, 
                                color_green_closed_yellow_ares, color_yellow_closed_yellow_ares, color_gray_closed_yellow_ares,
                                color_magenta_closed_yellow_ares, color_orange_closed_yellow_ares, color_sky_closed_yellow_ares, color_brown_closed_yellow_ares,
                                color_black_closed_gray_ares, color_red_closed_gray_ares, color_blue_closed_gray_ares, 
                                color_green_closed_gray_ares, color_yellow_closed_gray_ares, color_gray_closed_gray_ares,
                                color_magenta_closed_gray_ares, color_orange_closed_gray_ares, color_sky_closed_gray_ares, color_brown_closed_gray_ares,
                                color_black_closed_magenta_ares, color_red_closed_magenta_ares, color_blue_closed_magenta_ares, 
                                color_green_closed_magenta_ares, color_yellow_closed_magenta_ares, color_gray_closed_magenta_ares,
                                color_magenta_closed_magenta_ares, color_orange_closed_magenta_ares, color_sky_closed_magenta_ares, color_brown_closed_magenta_ares,
                                color_black_closed_orange_ares, color_red_closed_orange_ares, color_blue_closed_orange_ares, 
                                color_green_closed_orange_ares, color_yellow_closed_orange_ares, color_gray_closed_orange_ares,
                                color_magenta_closed_orange_ares, color_orange_closed_orange_ares, color_sky_closed_orange_ares, color_brown_closed_orange_ares,
                                color_black_closed_sky_ares, color_red_closed_sky_ares, color_blue_closed_sky_ares, 
                                color_green_closed_sky_ares, color_yellow_closed_sky_ares, color_gray_closed_sky_ares,
                                color_magenta_closed_sky_ares, color_orange_closed_sky_ares, color_sky_closed_sky_ares, color_brown_closed_sky_ares, 
                                color_black_closed_brown_ares, color_red_closed_brown_ares, color_blue_closed_brown_ares, 
                                color_green_closed_brown_ares, color_yellow_closed_brown_ares, color_gray_closed_brown_ares,
                                color_magenta_closed_brown_ares, color_orange_closed_brown_ares, color_sky_closed_brown_ares, color_brown_closed_brown_ares,
                                drow_hline_dot_to_rightside, drow_hline_dot_to_leftside, drow_vline_dot_to_ceilside, drow_vline_dot_to_floorside,
                                coloring_plus_shape_blue_to_red, coloring_plus_shape_blue_to_green, coloring_plus_shape_blue_to_yellow,
                                coloring_plus_shape_blue_to_gray, coloring_plus_shape_blue_to_magenta, coloring_plus_shape_blue_to_orange,
                                coloring_plus_shape_blue_to_sky, coloring_plus_shape_blue_to_brown,
                                coloring_plus_shape_red_to_blue, coloring_plus_shape_red_to_green, coloring_plus_shape_red_to_yellow,
                                coloring_plus_shape_red_to_gray, coloring_plus_shape_red_to_magenta, coloring_plus_shape_red_to_orange,
                                coloring_plus_shape_red_to_sky, coloring_plus_shape_red_to_brown,
                                coloring_plus_shape_green_to_blue, coloring_plus_shape_green_to_red, coloring_plus_shape_green_to_yellow,
                                coloring_plus_shape_green_to_gray, coloring_plus_shape_green_to_magenta, coloring_plus_shape_green_to_orange,
                                coloring_plus_shape_green_to_sky, coloring_plus_shape_green_to_brown,
                                coloring_plus_shape_yellow_to_blue, coloring_plus_shape_yellow_to_red, coloring_plus_shape_yellow_to_green,
                                coloring_plus_shape_yellow_to_gray, coloring_plus_shape_yellow_to_magenta, coloring_plus_shape_yellow_to_orange,
                                coloring_plus_shape_yellow_to_sky, coloring_plus_shape_yellow_to_brown,
                                coloring_plus_shape_gray_to_blue, coloring_plus_shape_gray_to_red, coloring_plus_shape_gray_to_green,
                                coloring_plus_shape_gray_to_yellow, coloring_plus_shape_gray_to_magenta, coloring_plus_shape_gray_to_orange,
                                coloring_plus_shape_gray_to_sky, coloring_plus_shape_gray_to_brown,
                                coloring_plus_shape_magenta_to_blue, coloring_plus_shape_magenta_to_red, coloring_plus_shape_magenta_to_green,
                                coloring_plus_shape_magenta_to_yellow, coloring_plus_shape_magenta_to_gray, coloring_plus_shape_magenta_to_orange,
                                coloring_plus_shape_magenta_to_sky, coloring_plus_shape_magenta_to_brown,
                                coloring_plus_shape_orange_to_blue, coloring_plus_shape_orange_to_red, coloring_plus_shape_orange_to_green,
                                coloring_plus_shape_orange_to_yellow, coloring_plus_shape_orange_to_gray, coloring_plus_shape_orange_to_magenta,
                                coloring_plus_shape_orange_to_sky, coloring_plus_shape_orange_to_brown,
                                coloring_plus_shape_sky_to_blue, coloring_plus_shape_sky_to_red, coloring_plus_shape_sky_to_green,
                                coloring_plus_shape_sky_to_yellow, coloring_plus_shape_sky_to_gray, coloring_plus_shape_sky_to_orange,
                                coloring_plus_shape_sky_to_orange, coloring_plus_shape_sky_to_brown,
                                coloring_plus_shape_brown_to_blue, coloring_plus_shape_brown_to_red, coloring_plus_shape_brown_to_green,
                                coloring_plus_shape_brown_to_yellow, coloring_plus_shape_brown_to_gray, coloring_plus_shape_brown_to_orange,
                                coloring_plus_shape_brown_to_sky, coloring_plus_shape_brown_to_sky,
                                coloring_batsu_shape_blue_to_red, coloring_batsu_shape_blue_to_green, coloring_batsu_shape_blue_to_yellow,
                                coloring_batsu_shape_blue_to_gray, coloring_batsu_shape_blue_to_magenta, coloring_batsu_shape_blue_to_orange,
                                coloring_batsu_shape_blue_to_sky, coloring_batsu_shape_blue_to_brown,
                                coloring_batsu_shape_red_to_blue, coloring_batsu_shape_red_to_green, coloring_batsu_shape_red_to_yellow,
                                coloring_batsu_shape_red_to_gray, coloring_batsu_shape_red_to_magenta, coloring_batsu_shape_red_to_orange,
                                coloring_batsu_shape_red_to_sky, coloring_batsu_shape_red_to_brown,
                                coloring_batsu_shape_green_to_blue, coloring_batsu_shape_green_to_red, coloring_batsu_shape_green_to_yellow,
                                coloring_batsu_shape_green_to_gray, coloring_batsu_shape_green_to_magenta, coloring_batsu_shape_green_to_orange,
                                coloring_batsu_shape_green_to_sky, coloring_batsu_shape_green_to_brown,
                                coloring_batsu_shape_yellow_to_blue, coloring_batsu_shape_yellow_to_red, coloring_batsu_shape_yellow_to_green,
                                coloring_batsu_shape_yellow_to_gray, coloring_batsu_shape_yellow_to_magenta, coloring_batsu_shape_yellow_to_orange,
                                coloring_batsu_shape_yellow_to_sky, coloring_batsu_shape_yellow_to_brown,
                                coloring_batsu_shape_gray_to_blue, coloring_batsu_shape_gray_to_red, coloring_batsu_shape_gray_to_green,
                                coloring_batsu_shape_gray_to_yellow, coloring_batsu_shape_gray_to_magenta, coloring_batsu_shape_gray_to_orange,
                                coloring_batsu_shape_gray_to_sky, coloring_batsu_shape_gray_to_brown,
                                coloring_batsu_shape_magenta_to_blue, coloring_batsu_shape_magenta_to_red, coloring_batsu_shape_magenta_to_green,
                                coloring_batsu_shape_magenta_to_yellow, coloring_batsu_shape_magenta_to_gray, coloring_batsu_shape_magenta_to_orange,
                                coloring_batsu_shape_magenta_to_sky, coloring_batsu_shape_magenta_to_brown,
                                coloring_batsu_shape_orange_to_blue, coloring_batsu_shape_orange_to_red, coloring_batsu_shape_orange_to_green,
                                coloring_batsu_shape_orange_to_yellow, coloring_batsu_shape_orange_to_gray, coloring_batsu_shape_orange_to_magenta,
                                coloring_batsu_shape_orange_to_sky, coloring_batsu_shape_orange_to_brown,
                                coloring_batsu_shape_sky_to_blue, coloring_batsu_shape_sky_to_red, coloring_batsu_shape_sky_to_green,
                                coloring_batsu_shape_sky_to_yellow, coloring_batsu_shape_sky_to_gray, coloring_batsu_shape_sky_to_orange,
                                coloring_batsu_shape_sky_to_orange, coloring_batsu_shape_sky_to_brown,
                                coloring_batsu_shape_brown_to_blue, coloring_batsu_shape_brown_to_red, coloring_batsu_shape_brown_to_green,
                                coloring_batsu_shape_brown_to_yellow, coloring_batsu_shape_brown_to_gray, coloring_batsu_shape_brown_to_orange,
                                coloring_batsu_shape_brown_to_sky, coloring_batsu_shape_brown_to_sky,
                                coloring_blue_to_red, coloring_blue_to_green, coloring_blue_to_yellow,
                                coloring_blue_to_gray, coloring_blue_to_magenta, coloring_blue_to_orange,
                                coloring_blue_to_sky, coloring_blue_to_brown,
                                coloring_red_to_blue, coloring_red_to_green, coloring_red_to_yellow,
                                coloring_red_to_gray, coloring_red_to_magenta, coloring_red_to_orange,
                                coloring_red_to_sky, coloring_red_to_brown,
                                coloring_green_to_blue, coloring_green_to_red, coloring_green_to_yellow,
                                coloring_green_to_gray, coloring_green_to_magenta, coloring_green_to_orange,
                                coloring_green_to_sky, coloring_green_to_brown,
                                coloring_yellow_to_blue, coloring_yellow_to_red, coloring_yellow_to_green,
                                coloring_yellow_to_gray, coloring_yellow_to_magenta, coloring_yellow_to_orange,
                                coloring_yellow_to_sky, coloring_yellow_to_brown,
                                coloring_gray_to_blue, coloring_gray_to_red, coloring_gray_to_green,
                                coloring_gray_to_yellow, coloring_gray_to_magenta, coloring_gray_to_orange,
                                coloring_gray_to_sky, coloring_gray_to_brown,
                                coloring_magenta_to_blue, coloring_magenta_to_red, coloring_magenta_to_green,
                                coloring_magenta_to_yellow, coloring_magenta_to_gray, coloring_magenta_to_orange,
                                coloring_magenta_to_sky, coloring_magenta_to_brown,
                                coloring_orange_to_blue, coloring_orange_to_red, coloring_orange_to_green,
                                coloring_orange_to_yellow, coloring_orange_to_gray, coloring_orange_to_magenta,
                                coloring_orange_to_sky, coloring_orange_to_brown,
                                coloring_sky_to_blue, coloring_sky_to_red, coloring_sky_to_green,
                                coloring_sky_to_yellow, coloring_sky_to_gray, coloring_sky_to_orange,
                                coloring_sky_to_orange, coloring_sky_to_brown,
                                coloring_brown_to_blue, coloring_brown_to_red, coloring_brown_to_green,
                                coloring_brown_to_yellow, coloring_brown_to_gray, coloring_brown_to_orange,
                                coloring_brown_to_sky, coloring_brown_to_sky,
                                drow_hline_blue, drow_hline_red, drow_hline_green, drow_hline_brown, drow_hline_gray,
                                drow_hline_magenta, drow_hline_orange, drow_hline_sky, drow_hline_yellow,
                                drow_vline_blue, drow_vline_red, drow_vline_green, drow_vline_brown, drow_vline_gray,
                                drow_vline_magenta, drow_vline_orange, drow_vline_sky, drow_vline_yellow,
                                ]
                best_function_list = []
                count = 0
                solve_flag = False
                while True:
                    count += 1
                    tmp_acc_list_matome = [[] for j in range(len(function_list))]
                    tmp_pred_list_matome = [[] for j in range(len(function_list))]
                    if count > 10:
                        print("")
                        print("best_function_list", best_function_list)
                        print("Not suitable")
                        print("")
                        break
                    for j, func in enumerate(function_list):
        #                 print(func.__name__)
                        tmp_pred_list = [0] * (len(task["train"]))
                        for task_num in range(len(task["train"])):
                            vinput = np.array(task["train"][task_num]["input"])
                            voutput = np.array(task["train"][task_num]["output"])
                            vpred = max_pred_list[task_num]
                            vpred = func(vpred)
                            tmp_pred_list[task_num] = vpred
                            acc = Compare(voutput, vpred)
                            tmp_acc_list[task_num] = acc
        #                 print(tmp_acc_list)
                        tmp_acc_list_matome[j] = copy.deepcopy(tmp_acc_list)
#                         print(tmp_acc_list_matome)
                        tmp_pred_list_matome[j] = copy.deepcopy(tmp_pred_list)
                    max_sum_acc = sum(max_acc_list)
                    func_pos = -1
                    for j in range(len(function_list)):
                        sum_ = sum(tmp_acc_list_matome[j])
                        if sum_ > max_sum_acc:
                            max_sum_acc = sum_
                            func_pos = j
                            
                    if func_pos == -1:
                        break
                    best_function_list.append(function_list[func_pos])
                    for task_num in range(len(task["train"])):
                        max_acc_list = copy.deepcopy(tmp_acc_list_matome[func_pos])
                        max_pred_list = copy.deepcopy(tmp_pred_list_matome[func_pos])
                    if sum(max_acc_list) == len(task["train"]):
                        solve_flag = True
                        print("")
                        print("SOLVED!!")
                        print("")
                        break

        
                if solve_flag:
                    print("best_function_list", best_function_list)
                    for te in range(len(task["test"])):
                        print(test_tasks[i])
                        vinput = np.array(task["test"][te]["input"])
                        vpred = vinput.copy()
                        if len(inspect.getargspec(pred_func).args)==1:
                            vpred = pred_func(vinput)
                        else:
                            vpred = pred_func(vinput, sz_out)
                        for func in best_function_list:
                            vpred = func(vpred)
                        vpred_flatten = flattener(vpred.tolist())
                        answer_list.append(vpred_flatten)
                        solved_count += 1
                        print("vpred", vpred_flatten)
                        print("")
                    break
                else:
                    if pred_func_pos == len(pred_function_list)-1:
                        print("NOT SOLVED")
                        for te in range(len(task["test"])):
                            print(test_tasks[i])
                            answer_list.append("|00|00|")
                        break
            else:
                if pred_func_pos == len(pred_function_list)-1:
                    print("NOT SOLVED")
                    for te in range(len(task["test"])):
                        print(test_tasks[i])
                        answer_list.append("|00|00|")
                    break

    else:
        print("Shape Not Match")
        
        for te in range(len(task["test"])):
            print(test_tasks[i])
            answer_list.append("|00|00|")


            
print("SOLVED COUNT :", solved_count)

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
sample_sub3 = pd.read_csv(data_path/'sample_submission.csv')
sample_sub3["output"] = answer_list
# sub.to_csv("submission.csv", index=False)



"""
Second Solution
This can solve one task.
"""
# I used this code <https://www.kaggle.com/meaninglesslives/using-decision-trees-for-arc> as reference
# Thank you for sharing the code in kaggle notebook.

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

def plot_result(test_input, test_prediction,
                input_shape):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    fig, axs = plt.subplots(1, 2, figsize=(15,15))
    test_input = test_input.reshape(input_shape[0],input_shape[1])
    axs[0].imshow(test_input, cmap=cmap, norm=norm)
    axs[0].axis('off')
    axs[0].set_title('Actual Target')
    test_prediction = test_prediction.reshape(input_shape[0],input_shape[1])
    axs[1].imshow(test_prediction, cmap=cmap, norm=norm)
    axs[1].axis('off')
    axs[1].set_title('Model Prediction')
    plt.tight_layout()
    plt.show()
    
def plot_test(test_prediction, task_name):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    fig, axs = plt.subplots(1, 1, figsize=(15,15))
    axs.imshow(test_prediction, cmap=cmap, norm=norm)
    axs.axis('off')
    axs.set_title(f'Test Prediction {task_name}')
    plt.tight_layout()
    plt.show()
    
# https://www.kaggle.com/inversion/abstraction-and-reasoning-starter-notebook
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred



sample_sub1 = pd.read_csv(data_path/'sample_submission.csv')
sample_sub1 = sample_sub1.set_index('output_id')
sample_sub1.head()

def get_moore_neighbours(color, cur_row, cur_col, nrows, ncols):

    if cur_row<=0: top = -1
    else: top = color[cur_row-1][cur_col]
        
    if cur_row>=nrows-1: bottom = -1
    else: bottom = color[cur_row+1][cur_col]
        
    if cur_col<=0: left = -1
    else: left = color[cur_row][cur_col-1]
        
    if cur_col>=ncols-1: right = -1
    else: right = color[cur_row][cur_col+1]
        
    return top, bottom, left, right

def get_tl_tr(color, cur_row, cur_col, nrows, ncols):
        
    if cur_row==0:
        top_left = -1
        top_right = -1
    else:
        if cur_col==0: top_left=-1
        else: top_left = color[cur_row-1][cur_col-1]
        if cur_col==ncols-1: top_right=-1
        else: top_right = color[cur_row-1][cur_col+1]   
        
    return top_left, top_right

def make_features(input_color, nfeat):
    nrows, ncols = input_color.shape
    feat = np.zeros((nrows*ncols,nfeat))
    cur_idx = 0
    for i in range(nrows):
        for j in range(ncols):
            feat[cur_idx,0] = i
            feat[cur_idx,1] = j
            feat[cur_idx,2] = input_color[i][j]
            feat[cur_idx,3:7] = get_moore_neighbours(input_color, i, j, nrows, ncols)
            feat[cur_idx,7:9] = get_tl_tr(input_color, i, j, nrows, ncols)
            feat[cur_idx,9] = len(np.unique(input_color[i,:]))
            feat[cur_idx,10] = len(np.unique(input_color[:,j]))
            feat[cur_idx,11] = (i+j)
            feat[cur_idx,12] = len(np.unique(input_color[i-local_neighb:i+local_neighb,
                                                         j-local_neighb:j+local_neighb]))

            cur_idx += 1
        
    return feat

def features(task, mode='train'):
    num_train_pairs = len(task[mode])
    feat, target = [], []
    
    global local_neighb
    for task_num in range(num_train_pairs):
        input_color = np.array(task[mode][task_num]['input'])
        #print(input_color)
        target_color = task[mode][task_num]['output']
        #print(target_color)
        nrows, ncols = len(task[mode][task_num]['input']), len(task[mode][task_num]['input'][0])

        target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])
        
        if (target_rows!=nrows) or (target_cols!=ncols):
            print('Number of input rows:',nrows,'cols:',ncols)
            print('Number of target rows:',target_rows,'cols:',target_cols)
            not_valid=1
            return None, None, 1

        imsize = nrows*ncols
        #offset = imsize*task_num*3 #since we are using three types of aug
        feat.extend(make_features(input_color, nfeat))
        target.extend(np.array(target_color).reshape(-1,))
            
    return np.array(feat), np.array(target), 0

# mode = 'eval'
mode = 'test'
if mode=='eval':
    task_path = evaluation_path
elif mode=='train':
    task_path = training_path
elif mode=='test':
    task_path = test_path

all_task_ids = sorted(os.listdir(task_path))

nfeat = 13
local_neighb = 5
valid_scores = {}

model_accuracies = {'ens': []}
pred_taskids = []

for task_id in all_task_ids:

    task_file = str(task_path / task_id)
    with open(task_file, 'r') as f:
        task = json.load(f)

    feat, target, not_valid = features(task)
    if not_valid:
        print('ignoring task', task_file)
        print()
        not_valid = 0
        continue

    xgb =  XGBClassifier(n_estimators=10, n_jobs=-1)
    xgb.fit(feat, target, verbose=-1)


#     training on input pairs is done.
#     test predictions begins here

    num_test_pairs = len(task['test'])
    for task_num in range(num_test_pairs):
        cur_idx = 0
        input_color = np.array(task['test'][task_num]['input'])
        nrows, ncols = len(task['test'][task_num]['input']), len(
            task['test'][task_num]['input'][0])
        feat = make_features(input_color, nfeat)

        print('Made predictions for ', task_id[:-5])

        preds = xgb.predict(feat).reshape(nrows,ncols)
        
        if (mode=='train') or (mode=='eval'):
            ens_acc = (np.array(task['test'][task_num]['output'])==preds).sum()/(nrows*ncols)

            model_accuracies['ens'].append(ens_acc)

            pred_taskids.append(f'{task_id[:-5]}_{task_num}')

#             print('ensemble accuracy',(np.array(task['test'][task_num]['output'])==preds).sum()/(nrows*ncols))
#             print()

        preds = preds.astype(int).tolist()
#         plot_test(preds, task_id)
        sample_sub1.loc[f'{task_id[:-5]}_{task_num}',
                       'output'] = flattener(preds)
        


if (mode=='train') or (mode=='eval'):
    df = pd.DataFrame(model_accuracies, index=pred_taskids)
    print(df.head(10))

    print(df.describe())
    for c in df.columns:
        print(f'for {c} no. of complete tasks is', (df.loc[:, c]==1).sum())

    df.to_csv('ens_acc.csv')



"""
Third Solution
This can solve one task.
"""
# I used this code <https://www.kaggle.com/szabo7zoltan/colorandcountingmoduloq> as reference
# Thank you for sharing the code in kaggle notebook.

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))
eval_tasks = sorted(os.listdir(evaluation_path))


T = training_tasks
Trains = []
for i in range(400):
    task_file = str(training_path / T[i])
    task = json.load(open(task_file, 'r'))
    Trains.append(task)
    
E = eval_tasks
Evals= []
for i in range(400):
    task_file = str(evaluation_path / E[i])
    task = json.load(open(task_file, 'r'))
    Evals.append(task)
    
    
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
plt.figure(figsize=(5, 2), dpi=200)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()

def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    
    plt.tight_layout()
    plt.show()
    

def plot_picture(x):
    plt.imshow(np.array(x), cmap = cmap, norm = norm)
    plt.show()
    
    
def Defensive_Copy(A): 
    n = len(A)
    k = len(A[0])
    L = np.zeros((n,k), dtype = int)
    for i in range(n):
        for j in range(k):
            L[i,j] = 0 + A[i][j]
    return L.tolist()


def Create(task, task_id = 0):
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output


def Recolor(task):
    Input = task[0]
    Output = task[1]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    N = len(Input)
    
    for x, y in zip(Input, Output):
        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1
        
    Best_Dict = -1
    Best_Q1 = -1
    Best_Q2 = -1
    Best_v = -1
    # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners
    Pairs = []
    for t in range(15):
        for Q1 in range(1,8):
            for Q2 in range(1,8):
                if Q1+Q2 == t:
                    Pairs.append((Q1,Q2))
                    
    for Q1, Q2 in Pairs:
        for v in range(4):
    
  
            if Best_Dict != -1:
                continue
            possible = True
            Dict = {}
                      
            for x, y in zip(Input, Output):
                n = len(x)
                k = len(x[0])
                for i in range(n):
                    for j in range(k):
                        if v == 0 or v ==2:
                            p1 = i%Q1
                        else:
                            p1 = (n-1-i)%Q1
                        if v == 0 or v ==3:
                            p2 = j%Q2
                        else :
                            p2 = (k-1-j)%Q2
                        color1 = x[i][j]
                        color2 = y[i][j]
                        if color1 != color2:
                            rule = (p1, p2, color1)
                            if rule not in Dict:
                                Dict[rule] = color2
                            elif Dict[rule] != color2:
                                possible = False
            if possible:
                
                # Let's see if we actually solve the problem
                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v ==2:
                                p1 = i%Q1
                            else:
                                p1 = (n-1-i)%Q1
                            if v == 0 or v ==3:
                                p2 = j%Q2
                            else :
                                p2 = (k-1-j)%Q2
                           
                            color1 = x[i][j]
                            rule = (p1,p2,color1)
                            
                            if rule in Dict:
                                color2 = 0 + Dict[rule]
                            else:
                                color2 = 0 + y[i][j]
                            if color2 != y[i][j]:
                                possible = False 
                if possible:
                    Best_Dict = Dict
                    Best_Q1 = Q1
                    Best_Q2 = Q2
                    Best_v = v
                
                
    if Best_Dict == -1:
        return -1 #meaning that we didn't find a rule that works for the traning cases
    
    #Otherwise there is a rule: so let's use it:
    n = len(Test_Picture)
    k = len(Test_Picture[0])
    
    answer = np.zeros((n,k), dtype = int)
   
    for i in range(n):
        for j in range(k):
            if Best_v == 0 or Best_v ==2:
                p1 = i%Best_Q1
            else:
                p1 = (n-1-i)%Best_Q1
            if Best_v == 0 or Best_v ==3:
                p2 = j%Best_Q2
            else :
                p2 = (k-1-j)%Best_Q2
           
            color1 = Test_Picture[i][j]
            rule = (p1, p2, color1)
            if (p1, p2, color1) in Best_Dict:
                answer[i][j] = 0 + Best_Dict[rule]
            else:
                answer[i][j] = 0 + color1
                                    
           
            
    return answer.tolist()


Function = Recolor

training_examples = []
for i in range(400):
    task = Trains[i]
    basic_task = Create(task,0)
    a = Function(basic_task)
  
    if  a != -1 and task['test'][0]['output'] == a:
        plot_picture(a)
        plot_task(task)
        print(i)
        training_examples.append(i)
        
        
evaluation_examples = []


for i in range(400):
    task = Evals[i]
    basic_task = Create(task,0)
    a = Function(basic_task)
    
    if a != -1 and task['test'][0]['output'] == a:
       
        plot_picture(a)
        plot_task(task)
        print(i)
        evaluation_examples.append(i)
        
        
sample_sub2 = pd.read_csv(data_path/ 'sample_submission.csv')
sample_sub2.head()


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


example_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
display(example_grid)
print(flattener(example_grid))

Solved = []
Problems = sample_sub2['output_id'].values
Proposed_Answers = []
for i in  range(len(Problems)):
    output_id = Problems[i]
    task_id = output_id.split('_')[0]
    pair_id = int(output_id.split('_')[1])
    f = str(test_path / str(task_id + '.json'))
   
    with open(f, 'r') as read_file:
        task = json.load(read_file)
    
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][j]['input']) for j in range(n)]
    Output = [Defensive_Copy(task['train'][j]['output']) for j in range(n)]
    Input.append(Defensive_Copy(task['test'][pair_id]['input']))
    
    solution = Recolor([Input, Output])
   
    
    pred = ''
        
    if solution != -1:
        Solved.append(i)
        pred1 = flattener(solution)
        pred = pred+pred1+' '
        
    if pred == '':
        pred = flattener(example_grid)
        
    Proposed_Answers.append(pred)
    
sample_sub2['output'] = Proposed_Answers



"""
Ensemble
"""
sample_sub1 = sample_sub1.reset_index()
sample_sub1 = sample_sub1.sort_values(by="output_id")

sample_sub2 = sample_sub2.sort_values(by="output_id")

sample_sub3 = sample_sub3.sort_values(by="output_id")
sample_sub3 = sample_sub3.reset_index(drop=True)
out1 = sample_sub1["output"].astype(str).values
out2 = sample_sub2["output"].astype(str).values
out3 = sample_sub3["output"].astype(str).values

merge_output = []
for o1, o2, o3 in zip(out1, out2, out3):
    o = o1.strip().split(" ")[:1] + o2.strip().split(" ")[:2] + o3.strip().split(" ")[:1]
    o = " ".join(o)
    merge_output.append(o)
    
sample_sub1["output"] = merge_output
sample_sub1["output"] = sample_sub1["output"].astype(str)
sample_sub1.to_csv("submission.csv", index=False)