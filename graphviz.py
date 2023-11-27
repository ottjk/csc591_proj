# Paste output into https://dreampuf.github.io/GraphvizOnline/
#
# 'circo' or 'dot' tend to look best

COLORS = ['#1E88E5', '#D81B60', '#FFC107', '#004D40']
USE_EDGE_COLORS = False

graph = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(2,3),(2,4),(3,4)]
coloring = '0011011011'

node_count = len(coloring) // 2
node_colors = [COLORS[int(coloring[i*2:i*2+2], 2)] for i in range(node_count)]

print('graph G {')
print('    node [style="filled" label="" color="white" shape="circle"]')
for i in range(len(coloring) // 2):
    print(f'    {i} [fillcolor="{node_colors[i]}"];')
for a, b in graph:
    if USE_EDGE_COLORS:
        print(f'    {a} -- {b} [color="{node_colors[a]};0.5:{node_colors[b]}"];')
    else:
        print(f'    {a} -- {b};')
print('}')
