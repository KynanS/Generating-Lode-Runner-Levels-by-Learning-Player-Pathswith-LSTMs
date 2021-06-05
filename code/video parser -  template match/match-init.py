from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import glob
from scipy.spatial import distance as dist

def back_track_dig(y,x,positions):
    print("bt from "+str(y)+","+str(x))
    for i in reversed(xrange(len(positions))):
        print("i "+str(i) + " "+str(positions[i][0])+","+str(positions[i][1]))
        if positions[i][0] == y - 1:
            if positions[i][1] == x - 1:
                #player at top left of the hole
                return i,'D'
            if positions[i][1] == x + 1:
                #player at top right of the hole
                return i,'d'
    return -1, -1

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--level', type=int, help='level for the current play', default=1)

args = parser.parse_args()

capture = cv.VideoCapture("..\\videos\\"+args.input)
if not capture.isOpened:
    print('Unable to open: ' + "..\\videos\\" + args.input)
    exit(0)


player_refs = []
for filename in glob.glob(".\\player_ref_diff\\ladder\\*.png"):
    splits = filename.split("\\")
    spritename = splits[-1]
    player_refs.append([spritename,cv.imread(filename)])

for filename in glob.glob(".\\player_ref_diff\\others\\*.png"):
    splits = filename.split("\\")
    spritename = splits[-1]
    player_refs.append([spritename,cv.imread(filename)])

player_half_ref = []
for filename in glob.glob(".\\player_ref_diff\\upper_half\\*.png"):
    splits = filename.split("\\")
    spritename = splits[-1]
    player_half_ref.append([spritename,cv.imread(filename),1])

for filename in glob.glob(".\\player_ref_diff\\lower_half\\*.png"):
    splits = filename.split("\\")
    spritename = splits[-1]
    player_half_ref.append([spritename,cv.imread(filename),-1])

cur_level = args.level
augmented_level = []
with open("..\\processed\\Level "+str(cur_level)+".txt",'r') as fin:
    rin = fin.readlines()
    for l in rin:
        augmented_level.append(l[:-1])

#removables that will be treated as empty space
removables = ['M','E','G']

map = {}
player_loc = []
for i in xrange(len(augmented_level)):
    map[i] = ""
    for c in xrange(len(augmented_level[i])):
        if augmented_level[i][c] in removables:
            if augmented_level[i][c] is 'M':
                player_loc.append([i,c])
            map[i] += '.'
        else:
            map[i] += augmented_level[i][c]


enemy_color = [120,40,110]
color_range = 40

upper_range = np.array([0, 0, 0])
lower_range = np.array([0, 0, 0])
for i in xrange(3):
    upper_range[i] = min(255,enemy_color[i]+color_range)
    lower_range[i] = max(0,enemy_color[i]-color_range)

search_size = 1.5

width = 32
height = 22

erode_kernel = np.ones((5,5), np.uint8)
dilate_kernel = np.ones((7,7), np.uint8)

frame_num = 1

frame_h = 0
frame_w = 0

window_h = 0
window_w = 0

#last_mask = []

#record into video
v_out = None

print("player starts at " + str(player_loc[-1]))

first_frame = None

while True:
    print("frame_num: "+str(frame_num))
    ret, frame = capture.read()
    if frame is None:
        break


    print("frame size: " + str(frame.shape))


    if  frame_num == 1:

        first_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        first_frame = cv.GaussianBlur(first_frame, (11, 11), 0)

        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        window_h = float(frame_h) / float(height)
        window_w = float(frame_w) / float(width)

        frame_num += 1
        continue

    print("each block of size y "+str(window_h)+" x "+str(window_w))


    enemy_mask = cv.inRange(frame,lower_range,upper_range)
    #cv.imshow("enemy mask",enemy_mask)

    print("frame number " + str(frame_num))

    best_score = 0.0
    best_loc = [0,0]
    best_sprite = "-"
    best_shape = [0,0]

    search_y = player_loc[-1][0] * window_h
    search_x = player_loc[-1][1] * window_w

    print("search around y "+str(max(0, int(search_y - search_size * window_h))) + " to " + str(min(frame.shape[0]-1,int(search_y + (search_size+1) * window_h))) +
          ", x " + str(max(0, int(search_x - search_size * window_w))) + " to " + str(min(frame.shape[1]-1,int(search_x + (search_size+1) * window_w))))

    new_frame = frame.copy()

    cv.rectangle(frame, (max(0, int(search_x - search_size * window_w)), max(0, int(search_y - search_size * window_h))),
                 (min(frame.shape[1]-1,int(search_x + (search_size+1) * window_w)), min(frame.shape[0]-1,int(search_y + (search_size+1) * window_h))),
                 (0, 255, 0), 2)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (11, 11), 0)
    frameDelta = cv.absdiff(first_frame, gray)

    thresh = cv.threshold(frameDelta, 50, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations=4)

    #new_frame = frame.copy()
    new_frame[thresh==0] = [0,0,0]
    new_frame[enemy_mask!=0] = [0,0,0]
    cp_frame = new_frame

    #template match
    for img in player_refs:
        result = cv.matchTemplate(cp_frame[max(0, int(search_y - search_size * window_h)):min(cp_frame.shape[0]-1,int(search_y + (search_size+1) * window_h)),
                                  max(0, int(search_x - search_size * window_w)):min(cp_frame.shape[1]-1,int(search_x + (search_size+1) * window_w))], img[1], cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_sprite = img[0]
            best_shape = img[1].shape

    if best_score > 0.7:
        best_loc = [best_loc[0] + max(0, int(search_x - search_size * window_w)),
                   best_loc[1] + max(0, int(search_y - search_size * window_h))]
        player_loc.append([int((best_loc[1] + best_shape[0]/2) / window_h), int((best_loc[0] + best_shape[1]/2) / window_w)])
        print("player located at " + str(player_loc[-1][0]) + ", " + str(player_loc[-1][1]) + " with confidence " + str(best_score))

        cv.rectangle(frame, (best_loc[0], best_loc[1]), (int(best_loc[0] + best_shape[1]), int(best_loc[1] + best_shape[0])), (255, 0, 0), 2)

    else:
        #nothing matches, check for half body in bricks

        best_score = 0
        player_pos = [0,0]

        for img in player_half_ref:
            result = cv.matchTemplate(cp_frame[max(0, int(search_y - search_size * window_h)):min(cp_frame.shape[0] - 1,
                                                                                                  int(search_y + (
                                                                                                              search_size + 1) * window_h)),
                                      max(0, int(search_x - search_size * window_w)):min(cp_frame.shape[1] - 1, int(
                                          search_x + (search_size + 1) * window_w))], img[1], cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

            if max_val > best_score:
                cur_loc = [max_loc[0] + max(0, int(search_x - search_size * window_w)), max_loc[1] + max(0, int(search_y - search_size * window_h))]
                cur_pos = [int((cur_loc[1] + img[1].shape[0] / 2) / window_h), int((cur_loc[0] + img[1].shape[1] / 2) / window_w)]
                cur_pos[0] += img[2]
                if cur_pos[0] in map and map[cur_pos[0]][cur_pos[1]] is 'b':
                    best_score = max_val
                    best_loc = cur_loc
                    best_sprite = img[0]
                    best_shape = img[1].shape
                    player_pos = cur_pos

        if best_score > 0.7:
            player_loc.append(player_pos)
            print("half player located at " + str(player_loc[-1][0]) + ", " + str(
                player_loc[-1][1]) + " with confidence " + str(best_score))

            cv.rectangle(frame, (best_loc[0], best_loc[1]),
                         (int(best_loc[0] + best_shape[1]), int(best_loc[1] + best_shape[0])), (255, 0, 0), 2)

    cv.imshow("frame",frame)
    cv.imshow("diff",cp_frame)
    #cv.waitKey(0)

    frame_num += 1

player_moves = []
player_true_loc = [player_loc[0]]
dig_track = {}
print("init loc " + str(player_loc[0]))
cur_ind = 0
for pos in player_loc[1:]:
    cur_ind += 1
    diff_y = pos[0] - player_true_loc[-1][0]
    diff_x = pos[1] - player_true_loc[-1][1]

    print("loc " + str(pos))

    if abs(diff_y) + abs(diff_x) > 1:
        #check with second last move in case last move was noise
        if cur_ind <= 1:
            print("too far move")
            continue
        diff_y = pos[0] - player_true_loc[-2][0]
        diff_x = pos[1] - player_true_loc[-2][1]
        if abs(diff_y) + abs(diff_x) > 1:
            print("too far move")
            continue
        else:
            del player_moves[-1]
            del player_true_loc[-1]

    if abs(diff_y) + abs(diff_x) is 0:
        continue
    elif diff_x is 1:
        player_moves.append('r')
        player_true_loc.append(pos)
    elif diff_x is -1:
        player_moves.append('l')
        player_true_loc.append(pos)
    elif diff_y is -1:
        if map[player_true_loc[-1][0]][player_true_loc[-1][1]] is not '#':
            print("cannot move up without ladder!")
            continue
        player_moves.append('u')
        player_true_loc.append(pos)
    elif diff_y is 1:
        if map[pos[0]][pos[1]] is '#':
            player_moves.append('c')
        else:
            player_moves.append('f')
        player_true_loc.append(pos)
    else:
        print("not sure about this move")
        continue

    if map[pos[0]][pos[1]] is 'b':
        #check for dig
        p,m = back_track_dig(pos[0], pos[1], player_true_loc)
        if p is not -1 and m is not -1:
            dig_track[p] = m


with open(".\\results-no-dig\\level "+str(cur_level)+".txt",'w') as fout:
    for m in player_moves:
        fout.write(m)

player_moves_with_dig = []
for i in xrange(len(player_moves)):
    if i in dig_track:
        player_moves_with_dig.append(dig_track[i])
    player_moves_with_dig.append(player_moves[i])


with open(".\\results-dig\\level "+str(cur_level)+".txt",'w') as fout:
    for m in player_moves_with_dig:
        fout.write(m)

