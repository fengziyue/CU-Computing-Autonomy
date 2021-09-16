import sys
sys.path.append('..')
from tkinter import *
import numpy
from gridmap import OccupancyGridMap
import matplotlib.pyplot as plt
from a_star_occupancy import a_star_occupancy
# from dijkstra_occupancy import dijkstra_occupancy
from utils import plot_path
import copy

canvas_width = 425
canvas_height = 420
start_node = (360.0, 330.0)
goal_node = (285.0, 86.0)
gmap = OccupancyGridMap.from_png('maps/example_map_occupancy.png', 1)
click = 0

def plan(event):
    print(event.x, event.y)
    global start_node, goal_node, click
    global gmap
    if click == 0:
        start_node = (event.x, canvas_height-event.y)
        cv.create_oval( event.x-5, event.y-5, event.x+5, event.y+5, fill = "green" )
        click += 1
        return
    else:
        goal_node = (event.x, canvas_height-event.y)
        cv.create_oval( event.x-5, event.y-5, event.x+5, event.y+5, fill = "red" )

        # uncomment below for a_star_occupancy
        path, path_px = a_star_occupancy(start_node, goal_node, copy.deepcopy(gmap), movement='8N')
        if path_px:
            ## plot resulting path in pixels over the map
            plot_on_canvas(path_px, cv, color="yellow", width=4)
        else:
            print('Goal is not reachable')

        # uncomment below for dijkstra_occupancy
        # path, path_px = dijkstra_occupancy(start_node, goal_node, gmap, movement='8N')
        # if path_px:
            ## plot resulting path in pixels over the map
            # plot_on_canvas(path_px, cv, color="purple", width=2)
        # else:
            # print('Goal is not reachable')


def plot_on_canvas(path, cv, color="yellow", width=3):
    # plot path
    cv.create_line( path[0][0], canvas_height-path[0][1], path[1][0], canvas_height-path[1][1], fill=color, width=width)
    for i in range(1, len(path)):
        cv.create_line( path[i-1][0], canvas_height-path[i-1][1], path[i][0], canvas_height-path[i][1], fill=color, width=width)



if __name__ == '__main__':
    
    window = Tk()

    window.title("occupance map")

    img = PhotoImage(file="maps/example_map_occupancy.png")
    cv = Canvas(window, width=canvas_width, height=canvas_height)
    cv.pack(side='top', fill='both', expand='yes')
    cv.create_image(0, 0, image=img, anchor='nw')

    cv.bind( "<Button-1>", plan )

    window.mainloop()