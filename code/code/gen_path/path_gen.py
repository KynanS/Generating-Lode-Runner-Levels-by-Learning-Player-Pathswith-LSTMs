from __future__ import print_function
from builtins import range
import argparse
import pickle
from random import seed
from random import randint
import numpy as np

seed(1)
np.random.seed(1)

def add(p_list,m):
    if m not in p_list:
        p_list.append(m)

parser = argparse.ArgumentParser(description='This program generates a map from given path')
parser.add_argument('--input', type=str, help='file containing path information', default='path1.txt')

args = parser.parse_args()

path = []
with open("path_data\\"+args.input,'r') as fin:
    while True:
        c = fin.read(1)
        if not c:
            print("End of file")
            break
        path.append(c)

print(str(path))

# player locations at furthest of 4 directions
l_most = 0
r_most = 0
t_most = 0
b_most = 0

# keep track of player's move locations
player_loc = [[0,0]]

# keep track of locations where it is empty so we can place enemies and gold
empty_loc = []

# keeps track of each tile's functionalities in the path
puzzle_move = {}

for m in path:

    last_loc = player_loc[-1]

    # update puzzle move
    if last_loc[0] not in puzzle_move:
        puzzle_move[last_loc[0]] = {}
    if last_loc[1] not in puzzle_move[last_loc[0]]:
        puzzle_move[last_loc[0]][last_loc[1]] = []
    if m not in puzzle_move[last_loc[0]][last_loc[1]]:
        puzzle_move[last_loc[0]][last_loc[1]].append(m)

    cur_loc = []

    if m == 'u': #player moving up
        cur_loc = [last_loc[0] - 1, last_loc[1]]
        if cur_loc[0] < t_most:
            t_most = cur_loc[0]
    elif m == 'c' or m == 'f': #player moving down
        cur_loc = [last_loc[0] + 1, last_loc[1]]
        if cur_loc[0] > b_most:
            b_most = cur_loc[0]
    elif m == 'l': #player moving left
        cur_loc = [last_loc[0], last_loc[1] - 1]
        if cur_loc[1] < l_most:
            l_most = cur_loc[1]
    elif m == 'r': #player moving right
        cur_loc = [last_loc[0], last_loc[1] + 1]
        if cur_loc[1] > r_most:
            r_most = cur_loc[1]

    if cur_loc != []:
        player_loc.append(cur_loc)
        if cur_loc not in empty_loc:
            empty_loc.append(cur_loc)

print("loc limits: top " + str(t_most) + ", bottom " + str(b_most) + ", left " + str(l_most) + ", right " + str(r_most))

map_width = r_most - l_most + 1
map_height = b_most - t_most + 1

for i in range(len(empty_loc)):
    empty_loc[i][0] = empty_loc[i][0] - t_most
    empty_loc[i][1] = empty_loc[i][1] - l_most

map = [[[] for i in range(map_width)] for j in range(map_height)]
for y in reversed(range(map_height)):
    for x in range(map_width):
        puzzle_y = y + t_most
        puzzle_x = x + l_most


        if puzzle_y in puzzle_move and puzzle_x in puzzle_move[puzzle_y]:
            cur_moves = puzzle_move[puzzle_y][puzzle_x]


            # ladders are guaranteed when climbing up or down
            if 'u' in cur_moves:
                add(map[y][x],'#')
                continue

            if 'c' in cur_moves:
                add(map[y+1][x],'#')
                add(map[y][x],'.')
                add(map[y][x],'-')


            # check for locations where the player can move left or right
            if 'l' in cur_moves or 'r' in cur_moves:
                #check what is underneath
                if puzzle_y+1 in puzzle_move and puzzle_x in puzzle_move[puzzle_y]:
                    add(map[y][x],'-')
                    add(map[y][x],'.')

                    if y + 1 < map_height:
                        add(map[y+1][x],'b')
                #bottom does not exit or no moves made
                else:
                    add(map[y][x], '.')
                    if y+1 < map_height:
                        add(map[y+1][x], 'b')
                    continue

            if 'f' in cur_moves:
                add(map[y][x], '.')
                if  ((puzzle_y-1 in puzzle_move and puzzle_x-1 in puzzle_move[puzzle_y-1]) or (puzzle_y-1 in puzzle_move and puzzle_x+1 in puzzle_move[puzzle_y-1])):
                    add(map[y][x], 'b')

print("map:\n")
for y in range(map_height):
    line = ""
    for x in range(map_width):
        line += str(map[y][x])
    print(line)

#all possible types here, b . - #

final_map = [['?' for i in range(map_width)] for j in range(map_height)]
for y in reversed(range(map_height)):
    for x in range(map_width):
        if map[y][x] == []:
            final_map[y][x] = '?'
        elif '#' in map[y][x]:
            final_map[y][x] = '#'
        elif len(map[y][x]) == 1:
            final_map[y][x] = map[y][x][0]
            if map[y][x][0] == 'b' and y > 0 and '-' in map[y-1][x]:
                map[y-1][x].remove('-')
        else:
            # if there is a brick underneath, then current is not '-'
            final_map[y][x] = map[y][x]

print('t_most:'+str(t_most)+' ,l_most:'+str(l_most))
print("puzzle_move:\n"+str(puzzle_move))

print("final_map:\n")
for y in range(map_height):
    line = ""
    for x in range(map_width):
        if type(final_map[y][x]) is str:
            line += final_map[y][x] + "  "
        else:
            for i in final_map[y][x]:
                line += i
            for i in range(3-len(final_map[y][x])):
                line += " "
    print(line)

with open("generated_path\\" + args.input,'w') as fout:
    for y in range(map_height):
        line = ""
        for x in range(map_width):
            if type(final_map[y][x]) is str:
                line += final_map[y][x] + "  "
            else:
                for i in final_map[y][x]:
                    line += i
                for i in range(3 - len(final_map[y][x])):
                    line += " "
        fout.write(line+"\n")

#get enemy and gold distributions
enemy_mean = 0
enemy_std = 0
gold_mean = 0
gold_std = 0
with open('stat_eg.pkl','rb') as fin:
    enemy_mean = pickle.load(fin)
    enemy_std = pickle.load(fin)
    gold_mean = pickle.load(fin)
    gold_std = pickle.load(fin)

#sample to find the stat for current generated map
total_enemies = int(np.random.normal(enemy_mean,enemy_std,1)[0] * len(path))
total_gold = int(np.random.normal(gold_mean,gold_std,1)[0] * len(path))

print("total of " + str(total_enemies) + " enemies and " + str(total_gold) + " gold in generated map")

markov_stat = []
with open("markov_stat.pkl",'rb') as fin:
    markov_stat = pickle.load(fin)

half_markov_stat = []
with open("half_markov_stat.pkl",'rb') as fin:
    half_markov_stat = pickle.load(fin)

markov_map = [['?' for i in range(map_width)] for j in range(map_height)]
for y in reversed(range(map_height)):
    line = ""
    for x in range(map_width):
        left = ''
        if x-1 < 0:
            left = 'v'
        else:
            left = markov_map[y][x-1]

        bottom = ''
        if y+1 >= map_height:
            bottom = 'v'
        else:
            bottom = markov_map[y+1][x]

        right = ''
        if x+1 >= map_width:
            right = 'v'
        else:
            if y + t_most in puzzle_move and x + l_most + 1 in puzzle_move[y + t_most]:
                r_list = puzzle_move[y + t_most][x + l_most + 1]
                for m in r_list:
                    right += m

        top = ''
        if y - 1 < 0:
            top = 'v'
        else:
            if y + t_most - 1 in puzzle_move and x + l_most in puzzle_move[y + t_most - 1]:
                t_list = puzzle_move[y + t_most - 1][x + l_most]
                for m in t_list:
                    top += m

        center = ''
        if y + t_most in puzzle_move and x + l_most in puzzle_move[y + t_most]:
            c_list = puzzle_move[y + t_most][x + l_most]
            for m in c_list:
                center += m

        full_key = left + ',' + bottom + ',' + right + ',' + top + ',' + center
        half_key = left + ',' + bottom

        if final_map[y][x] is str and final_map[y][x] != '?':
            markov_map[y][x] = final_map[y][x]
        elif y+1 < map_height and markov_map[y+1][x] == '.' and '-' in final_map[y][x]:
            markov_map[y][x] = '-'
        else:
            #now find the random chosen cell type from possible options given by final_map
            if full_key in markov_stat:
                prob = markov_stat[full_key]
                prob_map = {}
                sum = 0
                #current cell is on the path
                if final_map[y][x] != '?':
                    for type in final_map[y][x]:
                        if type not in prob:
                            prob[type] = 0
                        prob_map[type] = prob[type]
                        sum += prob[type]
                #current cell is off the path
                else:
                    for type in prob:
                        prob_map[type] = prob[type]
                        sum += prob[type]
                #none of the pattern had seen before. just simply choose first possible type
                if sum == 0:
                    markov_map[y][x] = final_map[y][x][0]
                else:
                    rand = randint(1,sum)
                    aggre = 0
                    for type in prob_map:
                        aggre += prob_map[type]
                        if rand <= aggre:
                            markov_map[y][x] = type
                            break
            #this key combination does not exist, try half key
            else:
                prob = half_markov_stat[half_key]
                prob_map = {}
                sum = 0
                # current cell is on the path
                if final_map[y][x] != '?':
                    for type in final_map[y][x]:
                        if type not in prob:
                            prob[type] = 0
                        prob_map[type] = prob[type]
                        sum += prob[type]
                else:
                    for type in prob:
                        prob_map[type] = prob[type]
                        sum += prob[type]
                rand = randint(1, sum)
                aggre = 0
                for type in prob_map:
                    aggre += prob_map[type]
                    if rand <= aggre:
                        markov_map[y][x] = type
                        break

#add player initial placement
markov_map[-t_most][-l_most] = 'M'

placeble_loc = [p for p in empty_loc if markov_map[p[0]][p[1]]=='.']
chosen_inx = []
while len(chosen_inx) < total_enemies + total_gold:
    r = randint(1,len(placeble_loc)) - 1
    if r not in chosen_inx:
        chosen_inx.append(r)

#now place the enemies
for i in range(total_enemies):
    markov_map[placeble_loc[chosen_inx[i]][0]][placeble_loc[chosen_inx[i]][1]] = 'E'

#now place the gold
for i in range(total_gold):
    markov_map[placeble_loc[chosen_inx[i+total_enemies]][0]][placeble_loc[chosen_inx[i+total_enemies]][1]] = 'G'

print("generated_map:")
for y in range(map_height):
    line = ""
    for x in range(map_width):
        line += markov_map[y][x]
    print(line)

with open("filled_path\\filled_map_" + args.input,'w') as fout:
    for y in range(map_height):
        line = ""
        for x in range(map_width):
            line += markov_map[y][x]
        fout.write(line+"\n")
