import sys
sys.path.append('..')
from tkinter import *
import numpy
from gridmap import OccupancyGridMap
import matplotlib.pyplot as plt
from a_star import a_star
from utils import plot_path

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
        path, path_px = a_star(start_node, goal_node, gmap, movement='8N')
        if path:
            # plot resulting path in pixels over the map
            plot_on_canvas(path_px, cv)
        else:
            print('Goal is not reachable')

def plot_on_canvas(path, cv):
    # plot path
    cv.create_line( path[0][0], canvas_height-path[0][1], path[1][0], canvas_height-path[1][1], fill="yellow", width=3)
    for i in range(1, len(path)):
        cv.create_line( path[i-1][0], canvas_height-path[i-1][1], path[i][0], canvas_height-path[i][1], fill="yellow", width=3)



if __name__ == '__main__':
    
    window = Tk()

    window.title("occupance map")

    img = PhotoImage(file="maps/example_map_occupancy.png")
    cv = Canvas(window, width=canvas_width, height=canvas_height)
    cv.pack(side='top', fill='both', expand='yes')
    cv.create_image(0, 0, image=img, anchor='nw')

    cv.bind( "<Button-1>", plan )

    window.mainloop()