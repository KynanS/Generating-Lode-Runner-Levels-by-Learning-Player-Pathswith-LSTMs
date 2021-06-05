from __future__ import print_function
from builtins import range
import argparse
import pickle
import numpy as np


parser = argparse.ArgumentParser(description='This program generates markov stats')
parser.add_argument('--level', type=int, help='maximum level that has full path', default='1')

args = parser.parse_args()
max_level = args.level

markov_stat = {}
half_markov_stat = {}

cur_level = 0

map_width = 32
map_height = 22

enemies = []
gold = []

while cur_level < max_level:

    cur_level += 1
    print('process level ' + str(cur_level))
    #read the annotated map data
    cur_map = []
    with open("..\\Processed\\Level " + str(cur_level) + ".txt", 'r') as fin:
        rin = fin.readlines()
        for l in rin:
            cur_map.append(l[:-1])

    #read the player movement
    moves = []
    try:
        with open("..\\video parser -  template match\\results-no-dig\\level " + str(cur_level) + ".txt", 'r') as fin:
            while True:
                c = fin.read(1)
                if not c:
                    break
                moves.append(c)
    except IOError:
        print("file level " + str(cur_level) + ".txt does not exist")
        continue

    cur_enemy = 0.0
    cur_gold = 0.0

    #now label the locations the player has moved
    labeled_map = [[[] for x in range(map_width)] for y in range(map_height)]

    #find player's init position
    player_x = 0
    player_y = 0

    for y in range(map_height):
        for x in range(map_width):
            if cur_map[y][x] == 'M':
                player_y = y
                player_x = x

    #now track the player's move and label corresponding cell
    for m in moves:
        if m == 'u':
            if 'u' not in labeled_map[player_y][player_x]:
                labeled_map[player_y][player_x].append('u')
            player_y -= 1
        elif m == 'c':
            if 'c' not in labeled_map[player_y][player_x]:
                labeled_map[player_y][player_x].append('c')
            player_y += 1
        elif m == 'f':
            if 'f' not in labeled_map[player_y][player_x]:
                labeled_map[player_y][player_x].append('f')
            player_y += 1
        elif m == 'l':
            if 'l' not in labeled_map[player_y][player_x]:
                labeled_map[player_y][player_x].append('l')
            player_x -= 1
        elif m == 'r':
            if 'r' not in labeled_map[player_y][player_x]:
                labeled_map[player_y][player_x].append('r')
            player_x += 1

    #now build markov unnormalized model
    for y in range(map_height):
        for x in range(map_width):
            #the bot and left are calculated from real cells
            #check left
            left = '?'
            #check for void
            if x-1 < 0:
                left = 'v'
            elif cur_map[y][x-1] == 'M' or cur_map[y][x-1] == 'E' or cur_map[y][x-1] == 'G':
                left = '.'
                if cur_map[y][x-1] == 'E':
                    cur_enemy += 1
                elif cur_map[y][x-1] == 'G':
                    cur_gold += 1
            else:
                left = cur_map[y][x-1]

            # check bottom
            bottom = '?'
            # check for void
            if y + 1 >= map_height:
                bottom = 'v'
            elif cur_map[y+1][x] == 'M' or cur_map[y+1][x] == 'E' or cur_map[y+1][x] == 'G':
                bottom = '.'
            else:
                bottom = cur_map[y + 1][x]

            #the right, top and center are calculated from path info
            #check right
            right = []
            if x + 1 >= map_width:
                right = ['v']
            else:
                if 'u' in labeled_map[y][x]:
                    right.append('u')
                if 'c' in labeled_map[y][x]:
                    right.append('c')
                if 'f' in labeled_map[y][x]:
                    right.append('f')
                if 'l' in labeled_map[y][x]:
                    right.append('l')
                if 'r' in labeled_map[y][x]:
                    right.append('r')
            right_line = ''
            for i in right:
                right_line += i

            # the right, top and center are calculated from path info
            # check top
            top = []
            if y - 1 < 0:
                top = ['v']
            else:
                if 'u' in labeled_map[y][x]:
                    top.append('u')
                if 'c' in labeled_map[y][x]:
                    top.append('c')
                if 'f' in labeled_map[y][x]:
                    top.append('f')
                if 'l' in labeled_map[y][x]:
                    top.append('l')
                if 'r' in labeled_map[y][x]:
                    top.append('r')
            top_line = ''
            for i in top:
                top_line += i


            # check center
            cen = []
            if 'u' in labeled_map[y][x]:
                cen.append('u')
            if 'c' in labeled_map[y][x]:
                cen.append('c')
            if 'f' in labeled_map[y][x]:
                cen.append('f')
            if 'l' in labeled_map[y][x]:
                cen.append('l')
            if 'r' in labeled_map[y][x]:
                cen.append('r')
            cen_line = ''
            for i in cen:
                cen_line += i


            #concatenate the directions to for the key
            key = left + ',' + bottom + ',' + right_line + ',' + top_line + ',' + cen_line
            half_key = left + ',' + bottom
            if key not in markov_stat:
                markov_stat[key] = {}
            if half_key not in half_markov_stat:
                half_markov_stat[half_key] = {}
            center_char = cur_map[y][x]
            if center_char == 'M' or center_char == 'E' or center_char == 'G':
                center_char = '.'
            if center_char not in markov_stat[key]:
                markov_stat[key][center_char] = 1
            else:
                markov_stat[key][center_char] += 1
            if center_char not in half_markov_stat[half_key]:
                half_markov_stat[half_key][center_char] = 1
            else:
                half_markov_stat[half_key][center_char] += 1

    if len(moves)!=0:
        enemies.append(cur_enemy/len(moves))
        gold.append(cur_gold/len(moves))



print("markov stat:\n")
for k in markov_stat:
    print(k + ": "+str(markov_stat[k]))
print("half_markov stat:\n")
for k in half_markov_stat:
    print(k + ": "+str(half_markov_stat[k]))
with open("markov_stat.pkl",'wb') as fout:
    pickle.dump(markov_stat,fout)
with open("half_markov_stat.pkl",'wb') as fout:
    pickle.dump(half_markov_stat,fout)

#find the parameter for emeny and gold distribution(assume gaussian)
enemy_np = np.array(enemies)
enemy_std = np.std(enemy_np)
enemy_mean = np.mean(enemy_np)

gold_np = np.array(gold)
gold_std = np.std(gold_np)
gold_mean = np.mean(gold_np)

with open('stat_eg.pkl','wb') as fout:
    pickle.dump(enemy_mean, fout)
    pickle.dump(enemy_std, fout)
    pickle.dump(gold_mean,fout)
    pickle.dump(gold_std, fout)


print("enemy distribution: mean " + str(enemy_mean) + " std " + str(enemy_std))
print("gold distribution: mean " + str(gold_mean) + " std " + str(gold_std))
