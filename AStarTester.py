# Kynan's attempt to write a testing script for the lode runner levels
# March 13th, 2021
# IMPORTANT:
# In order to use this pathfinder, changes must be made to main.  I would make changes to track specific metrics depending on what I was looking for from levels.
# I also would change the way I iterate through the level names in order to account for the way a group of levels change in their names level to level.  
# I have not included all variations here so make sure that you alter everything in main and in other places in the code that reference metric outputs to the csv file that this program creates
import csv



class Node():

    def __init__(self, parent=None, position=None, blockType=None):
        self.parent = parent
        self.position = position
        self.blockType = blockType
        
        self.g = 0
        self.h = 0
        self.f = 0

def astar(mapstr, start, gold, block, infoFile, levelNum1, levelNum2):

    # indicate the starting position and end position
    start_node = start
    start_node.g = start_node.h = start_node.f = 0
    end_node = gold
    end_node.g = end_node.h = end_node.f = 0

    #initialize lists
    openList = []
    closedList = []

    openList.append(start_node)

    tries = 0
    # loop until you find the end
    while len(openList) > 0 and tries < 28000:

        # get the current node
        tries += 1
        current_node = openList[0]
        current_index = 0
        for index, item in enumerate(openList):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        
        # pop current off open list, add it to closed list
        openList.pop(current_index)
        closedList.append(current_node)
        
        

        # Found the goal
        if current_node == end_node:
            return len(closedList)

        # Generate Children
        children = []

        # get the nodes all around the current node for checking purposes
        nodeAbove = None
        nodeBelow = None
        nodeRight = None
        nodeLeft = None
        
        for thing in block:
            
            # node above
            
            if thing.position == (current_node.position[0] - 1, current_node.position[1]):
                nodeAbove = thing
                
                
            # node below
            elif thing.position == (current_node.position[0] + 1, current_node.position[1]):
                nodeBelow = thing
                
                
            # node left
            elif thing.position == (current_node.position[0], current_node.position[1] - 1):
                nodeLeft = thing
                
                
            # node right
            elif thing.position == (current_node.position[0], current_node.position[1] + 1):
                nodeRight = thing
                
               
            
            
            
            

        # for each block type, there are only so many directions we can go, the following is for all scenarios
        # for when the node 
       
            
        # if there is no ground or rope, or we aren't at the bottom of the map, make the below node the only node to possibly go to
        if nodeBelow != None and current_node.blockType != "-" and current_node.blockType != "#" and (nodeBelow.blockType == "." or nodeBelow.blockType == "-" or nodeBelow.blockType == "G" or nodeBelow.blockType == "E"):
            nodeBelow.parent = current_node
            children.append(nodeBelow)
        else:
            # check to make sure if each node exists and is possible to move through.  If it is, add it to children
            # node below
            if nodeBelow != None and (nodeBelow.blockType == "#" or nodeBelow.blockType == "-" or nodeBelow.blockType == "b"):
                nodeBelow.parent = current_node
                children.append(nodeBelow)       
            elif nodeBelow != None and current_node.blockType == "-" and (nodeBelow.blockType == "#" or nodeBelow.blockType == "-" or nodeBelow.blockType == "b" or nodeBelow.blockType == "." or nodeBelow.blockType == "G" or nodeBelow.blockType == "E"):   
                nodeBelow.parent = current_node
                children.append(nodeBelow)
            elif nodeBelow != None and current_node.blockType == "b" and (nodeBelow.blockType == "#" or nodeBelow.blockType == "-" or nodeBelow.blockType == "b" or nodeBelow.blockType == "." or nodeBelow.blockType == "G" or nodeBelow.blockType == "E"):   
                nodeBelow.parent = current_node
                children.append(nodeBelow)  
            elif nodeBelow != None and current_node.blockType == "#" and (nodeBelow.blockType == "#" or nodeBelow.blockType == "-" or nodeBelow.blockType == "b" or nodeBelow.blockType == "." or nodeBelow.blockType == "G" or nodeBelow.blockType == "E"):   
                nodeBelow.parent = current_node
                children.append(nodeBelow)
            # node to the left     
            if nodeLeft != None and (nodeLeft.blockType == "." or nodeLeft.blockType == "-" or nodeLeft.blockType == "#" or nodeLeft.blockType == "G" or nodeLeft.blockType == "E"):
                nodeLeft.parent = current_node
                children.append(nodeLeft)
            elif nodeLeft != None and current_node.blockType == "-" and (nodeLeft.blockType == "." or nodeLeft.blockType == "-" or nodeLeft.blockType == "#" or nodeLeft.blockType == "G" or nodeLeft.blockType == "E"):
                nodeLeft.parent = current_node
                children.append(nodeLeft)
            # node to the right
            if nodeRight != None and (nodeRight.blockType == "." or nodeRight.blockType == "-" or nodeRight.blockType == "#" or nodeRight.blockType == "G" or nodeRight.blockType == "E"):
                nodeRight.parent = current_node
                children.append(nodeRight)
            elif nodeRight != None and current_node.blockType == "-" and (nodeRight.blockType == "." or nodeRight.blockType == "-" or nodeRight.blockType == "#" or nodeRight.blockType == "G" or nodeRight.blockType == "E"):
                nodeRight.parent = current_node
                children.append(nodeRight)
            # node above
            if nodeAbove != None and (nodeAbove.blockType == "-" or nodeAbove.blockType == "#"):
                nodeAbove.parent = current_node
                children.append(nodeAbove)
            elif nodeAbove != None and current_node.blockType == "#" and (nodeAbove.blockType == "." or nodeAbove.blockType == "-" or nodeAbove.blockType == "G" or nodeAbove.blockType == "E" or nodeAbove.blockType == "#"):
                nodeAbove.parent = current_node
                children.append(nodeAbove)


        
        for child in children:
            addChild = True
            # check if any of the children are in the closed list
            for closedChild in closedList:
                if child.position == closedChild.position:                    
                    addChild = False
            # create f, h, and g scores
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0])**2) + ((child.position[1] - end_node.position[1])**2)
            child.f = child.g + child.h

            for openNode in openList:
                if child.position == openNode.position and child.g > openNode.g:                    
                    addChild = False

            
            # add the child to the open list
            if addChild == True:
                openList.append(child)
    return 0


def testingPlayability(mapName, infoFile, levelNum1, levelNum2):
    levelFileName = mapName
    level = []
    nodesExplored = 0
    with open(levelFileName) as (level_file):
        for line in level_file:
            level.append(line.rstrip())
        

    gold = []
    block = []
    for i in range(len(level)):
        for j in range(len(level[0])):
            spot = Node(None, (i, j), level[i][j])     
            block.append(spot)    
            if level[i][j] == "G":                
                gold.append(spot)
            if level[i][j] == "M":
                gold.insert(0,spot)
            
    
    # find a path to each gold, should be able to get to each one from the starting position
    successes = 0
    for i in range(1, len(gold)):
        goldtravel = astar(level, gold[0], gold[i], block, infoFile, levelNum1, levelNum2)
        
    
        # keep track of successful aquisitions
        # success
        if goldtravel > 0:
            
            successes += 1
            nodesExplored += goldtravel
    
    infoFile.writerow({'Level' : str(levelNum1) + '.' + str(levelNum2), 'Gold Total' : str(len(gold) - 1), 'Gold Collected' : str(successes), '% Collected' : str(successes/(len(gold) - 1)), 'Nodes Explored' : nodesExplored})
                   
            
if __name__ == '__main__':
    import sys
    with open('Results.csv', 'w', newline='') as csvFile:
        fieldNames = ['Level', 'Gold Total', 'Gold Collected', '% Collected', 'Nodes Explored']
        infoFile = csv.DictWriter(csvFile, fieldnames = fieldNames)
        
        for i in range(1,39):
            for j in range(1,4):
                lvlText = "Loderunner_Maps/filled_map_path" + str(i) + "." + str(j)+ ".txt"
                testingPlayability(lvlText, infoFile, i, j)
