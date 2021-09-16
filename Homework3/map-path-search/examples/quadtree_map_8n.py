import sys
sys.path.append('..')
from tkinter import *
from quadtreemap import QuadTreeMap
from dijkstra_quadtree import dijkstra_quadtree
# from a_star_quadtree import a_star_quadtree


canvas_width = 425
canvas_height = 420
start_node = (360.0, 330.0)
goal_node = (285.0, 86.0)
filename = 'maps/example_map_binary.png'
# filename = 'maps/example_map_occupancy.png'
qtmap = QuadTreeMap.from_png(filename, cell_size=1, occupancy_threshold=0.6, tile_capacity=100)
click = 0

def plan(event):
    print(event.x, event.y)
    global start_node, goal_node, click
    global qtmap
    if click == 0:
        start_node = (event.x, canvas_height-event.y)
        cv.create_oval( event.x-5, event.y-5, event.x+5, event.y+5, fill = "green" )
        click += 1
        return
    else:
        goal_node = (event.x, canvas_height-event.y)
        cv.create_oval( event.x-5, event.y-5, event.x+5, event.y+5, fill = "red" )

        # uncomment below for 'a_star_quadtree'
        # path, path_px = a_star_quadtree(start_node, goal_node, qtmap, movement='8N')
        # if path_px:
            ## plot resulting path in pixels over the map
            ## plot_on_canvas(path_px, cv, color="yellow", width=4)
            # draw_listOfTiles_on_canvas(path_px, cv, color="yellow", width=3)
        # else:
            # print('Goal is not reachable')

        # uncomment below for 'dijkstra_quadtree'
        path, path_px = dijkstra_quadtree(start_node, goal_node, qtmap, movement='8N')
        if path_px:
            # plot resulting path in pixels over the map
            # plot_on_canvas(path_px, cv, color="purple", width=2)
            draw_listOfTiles_on_canvas(path_px, cv, color="purple", width=3)
        else:
            print('Goal is not reachable')


def plot_on_canvas(path, cv, color="yellow", width=3):
    # plot path
    cv.create_line( path[0][0], canvas_height-path[0][1], path[1][0], canvas_height-path[1][1], fill=color, width=width)
    for i in range(1, len(path)):
        cv.create_line( path[i-1][0], canvas_height-path[i-1][1], path[i][0], canvas_height-path[i][1], fill=color, width=width)

def draw_listOfTiles_on_canvas(path_idx, cv, color="blue", width=3):
    for til in path_idx:
        cv.create_rectangle(til.boundary.x0, canvas_height - til.boundary.y0,
                            til.boundary.x0 + til.boundary.width,
                            canvas_height - til.boundary.y0 - til.boundary.height,
                            outline=None, fill=color, width=width)

if __name__ == '__main__':
    
    window = Tk()

    window.title("occupance map")

    img = PhotoImage(file="maps/example_map_occupancy.png")
    cv = Canvas(window, width=canvas_width, height=canvas_height)
    cv.pack(side='top', fill='both', expand='yes')
    cv.create_image(0, 0, image=img, anchor='nw')
    qtmap.drawQuadTreeMapByCanvas(cv, canvas_height, color="gray", width=2)
    cv.bind( "<Button-1>", plan )

    window.mainloop()