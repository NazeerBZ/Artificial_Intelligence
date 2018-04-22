#Breadth First Search

#G = {
#     1: [1,2,3],
#     2: [],
#     3: [3]
#     }
#
#output = [];
#queue = [];
#currentNode = 1;
#queue.append(currentNode);
#output.append(currentNode);
#
#while queue:
#    for adjacentNode in G[currentNode]:
#        if adjacentNode not in output:
#            queue.append(adjacentNode);
#            output.append(adjacentNode);
#
#    queue.remove(currentNode)
#    if queue == []: break;
#    item = queue[0:1];
#    currentNode = item[0];
#
#print(queue);
#print(output);

##############################################################################

## Depth First Search
#
##G = {
##     1: [5,2,4],
##     2: [4,3],
##     3: [],
##     4: [1,5,3],
##     5: []
##     }
#
##G = {
##     1: [1,2,3],
##     2: [5],
##     3: [],
##     4: [],
##     5: [3,4]
##     }
#
#G = {
#     1: [2,5],
#     2: [3,4],
#     3: [],
#     4: [],
#     5: []
#     }
#
#
#currentNode = 1;
#top = -1;
#stack = [];
#visited = [];
#adjacentNodes= [];
#
#
#def isVisited(currentNode):
#    if currentNode in visited: return True;
#    else: return False; 
#
#def pushInVisited(currentNode):
#    if currentNode not in visited:
#        visited.append(currentNode);
#        getAdjacentNodes(currentNode);
#
#def getAdjacentNodes(currentNode):
#        global adjacentNodes, top;    
#        adjacentNodes = G[currentNode];
#        
#        if adjacentNodes: # Nodes that have their adjacent nodes
#            stack.remove(currentNode);
#            top = top - 1;
#            
#            for node in adjacentNodes:
#                if not isVisited(node):
#                    pushInStack(node);
#        else: # Nodes that dont have adjacent Nodes
#            stack.remove(currentNode);
#            top = top - 1;
#
#def pushInStack(currentNode):    
#    stack.append(currentNode);
#    topOfStack(); # it will make currentNode
#
#def topOfStack():
#    global currentNode, top;
#    top = top + 1;
#    currentNode = stack[top];   
#    pushInVisited(currentNode);     
#
#pushInStack(currentNode);
#
#print(visited)
#print(stack)


























