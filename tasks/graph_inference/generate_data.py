# coding; utf-8
import tensorflow as tf
import numpy as np
import getopt
import sys
import os
import json
import random


def bfs_shortest_path(graph, start, goal):
    # keep track of explored nodes
    explored = []
    # keep track of all the paths to be checked
    queue = [[start]]

    # return path if start is goal
    if start == goal:
        return "That was easy! Start = goal"

    # keeps looping until all possible paths have been checked
    while queue:
        # pop the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        if node not in explored:
            neighbours = graph[str(node)]
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # return path if neighbour is goal
                if neighbour == goal:
                    return new_path

            # mark node as explored
            explored.append(node)

    # in case there's no path between the 2 nodes
    return "So sorry, but a connecting path doesn't exist :("


def search_edge(from_id, to_id, edges):
    for _edge in edges:
        if from_id in _edge and to_id in _edge:
            return _edge
    return None


def generate_data(batch_size, edges, graph):
    '''
    - batch_size : int
    - seq_length : int
    - edges : np.array
    - graph : dict
    '''
    def convert_one_hot(_arr):
        out_list = []
        for idx, i in enumerate(_arr):
            str_i = "{0:03d}".format(i)
            for char_i in str_i:
                one_hot_char_i = np.eye(10)[[int(char_i)]].tolist()[0]
                out_list.extend(one_hot_char_i)
            # if idx < 2:
            #     out_list.append(0)
        return out_list
    input_vecs = []
    out_vecs = []
    for _b in range(batch_size):
        input_vec = []
        out_vec = []
        np.random.shuffle(edges)
        for _edge in edges.tolist():
            input_vec.append(convert_one_hot(_edge))
            out_vec.append(convert_one_hot([0, 0, 0]))
        start_id, goal_id = random.sample(range(140), 2)
        shortest_path = bfs_shortest_path(graph, start_id, goal_id)
        input_vec.append(convert_one_hot([start_id, goal_id, 0]))
        out_vec.append(convert_one_hot([0, 0, 0]))
        for j in range(len(shortest_path)-1):
            input_vec.append(convert_one_hot([0, 0, 0]))
            out_vec.append(convert_one_hot(search_edge(shortest_path[j], shortest_path[j+1], edges)))
        input_vecs.append(input_vec)
        out_vecs.append(out_vec)
    return (input_vecs, out_vecs)


if __name__ == '__main__':
    with open("./json/metro_training_data.json", "r") as f:
        data_dict = json.load(f)
    edges = data_dict["edge"]
    graph = data_dict["graph"]
    print(generate_data(1, np.array(edges), graph)[0])