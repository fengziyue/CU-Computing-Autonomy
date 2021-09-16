import queue
import quadtreemap

def _get_movements_4n(qtm, tile):
    neighborList = []
    neighborList.append(qtm.quadtree.tileIntersect(quadtreemap.BoundingBox(tile.boundary.x0-1, tile.boundary.y0,
                                                                    tile.boundary.width+2, tile.boundary.height)))
    neighborList.append(qtm.quadtree.tileIntersect(quadtreemap.BoundingBox(tile.boundary.x0, tile.boundary.y0-1,
                                                                    tile.boundary.width, tile.boundary.height+2)))
    movements = [(til, quadtreemap.Point.disOf2Points(tile.getCenter(), til.getCenter())) for til in neighborList]
    return movements

def _get_movements_8n(qtm: quadtreemap.QuadTreeMap , tile: quadtreemap.Tile):
    neighborList = qtm.quadtree.tileIntersect(quadtreemap.BoundingBox(tile.boundary.x0-1, tile.boundary.y0-1,
                                            tile.boundary.width+2, tile.boundary.height+2))
    movements = [(til, quadtreemap.Point.disOf2Points(tile.getCenter(), til.getCenter())) for til in neighborList]
    return movements

def dijkstra_quadtree(start_m, goal_m, qtm, movement='8n', occupancy_cost_factor=3):
    path_record = {}
    candidates = queue.PriorityQueue()

    # get array indices of start and goal
    start = qtm.quadtree.searchTileByIdx(quadtreemap.Point(start_m[0], start_m[1]))
    goal = qtm.quadtree.searchTileByIdx(quadtreemap.Point(goal_m[0], goal_m[1]))

    # check if start and goal nodes correspond to free spaces
    if not start or start.tile_points:
        raise Exception('Start node is not traversable')
    if not goal or goal.tile_points:
        raise Exception('Goal node is not traversable')

    candidates.put((0, None, start))   # store (distance, previous-tile, current-tile)
    while candidates:
        dis, prev_node, curr_node = candidates.get()
        # print(curr_node, "\t", goal)
        if curr_node == goal:
            # print(True)
            path_record[curr_node] = prev_node
            break
        if curr_node in path_record:
            continue
        path_record[curr_node] = prev_node

        # get possible movements
        if movement == '4N':
            movements = _get_movements_4n(qtm, curr_node)
        elif movement == '8N':
            movements = _get_movements_8n(qtm, curr_node)
        else:
            raise ValueError('Unknown movement')

        # check all neighbors
        for til, deltacost in movements:
            # check whether new position is inside the map or is an obstacle
            # if not, skip node
            if til.tile_points:
                continue
            if til not in path_record:
                candidates.put((dis + deltacost, curr_node, til))
    # reconstruct path backwards (only if we reached the goal)
    path = []
    path_idx = []
    # print(len(path_record))
    # print(path_record)
    if goal in path_record:
        node = goal
        while node:
            path_idx.append(node)
            # transform array indices to meters
            # node_m_x, node_m_y = gmap.get_coordinates_from_index(node[0], node[1])
            # path.append((node_m_x, node_m_y))
            node = path_record[node]
        # reverse so that path is from start to goal.
        path.reverse()
        path_idx.reverse()

    # print("path_idx len: ", len(path_idx))
    # print("path_idx:\n", path_idx)
    return path, path_idx