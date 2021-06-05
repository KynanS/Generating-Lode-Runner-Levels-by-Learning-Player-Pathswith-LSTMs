from __future__ import print_function
from builtins import range
import sys
import os
import glob
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='This program generates visual map for given level')
parser.add_argument('--input', type=str, help='name of file containing mao information', default='filled_map_path1.txt')
args = parser.parse_args()

#Load sprites
sprites = {}
for filename in glob.glob(".\\sprites_LR\\*.png"):
    im = Image.open(filename)
    #splits = filename.split("/")
    splits = filename.split("\\")
    name = splits[-1][:-4]
    sprites[name] = im

visualization = {}
visualization["#"] = "#ladder"
visualization["-"] = "-bar"
visualization["B"] = "brick_plain"
visualization["b"] = "brick"
visualization["E"] = "Enemy"
visualization["G"] = "Gold"
visualization["M"] = "Me"

#Visualize Output Level
maxY = 0
maxX = 0

level = {}
with open(args.input,'r') as fp:
    for line in fp:
        maxX = len(line)-1
        level[maxY] = line[:maxX]
        maxY+=1

image = Image.new("RGB", (maxX*8, maxY*8), color=(0, 0, 0))
pixels = image.load()



for y in range(0, maxY):
    for x in range(0, maxX):
        imageToUse = None
        if level[y][x] in visualization.keys():
            imageToUse = sprites[visualization[level[y][x]]]
        if not imageToUse == None:
            pixelsToUse = imageToUse.load()
            for x2 in range(0, 8):
                for y2 in range(0, 8):
                    pixels[x*8+x2,y*8+y2] = pixelsToUse[x2,y2]

image.save("visualized_level.png", "PNG")
