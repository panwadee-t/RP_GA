# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:26:35 2020

@author: pguillerm
"""

import pandas as pd
import math
import numpy as np
import time
import random

#Constant values
v_fly = 50 #in km/h
bcr = 0.67 # battery consumption rate in %/minute
s_need = 5 # takeoff and landing time in minute
b_min = 5 # minimum reamaining battery level in %
R = 60 #Battery recharging time to full in minute
T = 1440 #Available operating time in minute

def prepare_data(file_instance):
    # Read data from text file
    with open(file_instance) as f:
        line_req_taxi = f.readline().split()
        nb_req = int(line_req_taxi[0])
        nb_taxi = int(line_req_taxi[1])
        line_center = f.readline().split()
        center_val = [int(elem) for elem in line_center]
        matrix = []
        for line in f:
            line1 = line.split()
            value = [int(elem) for elem in line1]
            matrix.append(value)
            
    data = pd.DataFrame(matrix, columns=['req_id','ori_x','ori_y','des_x','des_y','pick_t'])
    
    #Compute duration time from origin to destination + takeoff and landing time
    data['dist'] = data.apply(lambda x: round(math.sqrt((x['ori_x']-x['des_x'])**2 + (x['ori_y']-x['des_y'])**2),2), axis=1)
    data['dur_t'] = data.apply(lambda x: round(x['dist']/(v_fly*1000/60)+2*s_need,2), axis=1)
    # print(data)
    return nb_req, nb_taxi, data, center_val

# First objective function (To maximize the number of demanded which are serviced)
# def cal_obj1(matrix_task_chrom):
#     nb_assigned_tasks = 0
#     for j in range(nb_taxi):
#         nb_assigned_tasks += len(matrix_task_chrom["taxi"+str(j+1)]) - matrix_task_chrom["taxi"+str(j+1)].count("b")
#     return nb_assigned_tasks

# Second objective function (To maximize the demand service time)
def cal_obj2(matrix_task_chrom, matrix_start_time_chrom, matrix_fini_time_chrom):
    accumulate_service_time = 0
    for j in range(nb_taxi):
        for i in range(len(matrix_task_chrom["taxi"+str(j+1)])):
            if matrix_task_chrom["taxi"+str(j+1)][i] != "b":
                accumulate_service_time += matrix_fini_time_chrom["taxi"+str(j+1)][i] - matrix_start_time_chrom["taxi"+str(j+1)][i]
    return accumulate_service_time

def cal_dur_move_time(next_ori, prev_des=-1):
    x_prev_des = center_val[0]
    y_prev_des = center_val[1]
    if prev_des != -1:
        x_prev_des = data['des_x'][prev_des]
        y_prev_des = data['des_y'][prev_des]
    dist_move = round(math.sqrt((data['ori_x'][next_ori]-x_prev_des)**2 + (data['ori_y'][next_ori]-y_prev_des)**2),2)
    move_t = round(dist_move/(v_fly*1000/60),2) + 2*s_need
    return move_t

def cal_dur_back_to_center(req_test):
    dist_back = round(math.sqrt((data['des_x'][req_test]-center_val[0])**2 + (data['des_y'][req_test]-center_val[1])**2),2)
    back_t = round(dist_back/(v_fly*1000/60),2) + 2*s_need
    return back_t

def decode_GA(chromo, detail = None):
    chrom_priority = chromo.copy()
    
    order_assign = sorted(range(len(chrom_priority)), key=lambda k: chrom_priority[k], reverse=True)
    # print(order_assign)
    
    # Define decision variables
    # Demand i = [1, nb_req] or b (battery recharging)
    matrix_task = {}
    for x in range(1, nb_taxi+1):
        matrix_task["taxi{0}".format(x)]=[]
    
    matrix_start_time = {}
    for x in range(1, nb_taxi+1):
        matrix_start_time["taxi{0}".format(x)]=[]
        
    matrix_fini_time = {}
    for x in range(1, nb_taxi+1):
        matrix_fini_time["taxi{0}".format(x)]=[]
    
    #The battery level of each taxi at time 0
    battery_level = [100]*nb_taxi
    
    for i_gene in order_assign:
        # print("order_assign = ", order_assign)
        # print("i_gene = ", i_gene)
        i_req = (i_gene//nb_taxi)+1
        j_taxi = (i_gene%nb_taxi)+1
        dur_prep = cal_dur_move_time(i_req - 1)
        prev_fini_time = 0 
        prev_req = 'c'
        if (len(matrix_task["taxi"+str(j_taxi)]) != 0):
            prev_req = matrix_task["taxi"+str(j_taxi)][-1]
            if (prev_req != "b"):
                dur_prep = cal_dur_move_time(i_req - 1, prev_req - 1)
            prev_fini_time = matrix_fini_time["taxi"+str(j_taxi)][-1]
        
        #Check enough time to pick up  
        enough_time_to_pick = False
        if prev_fini_time + dur_prep <= data["pick_t"][i_req-1]:
            enough_time_to_pick = True
        
        dur_t_req = data["dur_t"][i_req-1]
        dur_back_center = cal_dur_back_to_center(i_req-1)
        #Check enough battery to service and go back to center if needed
        enough_battery = False
        if battery_level[j_taxi-1] - bcr*(dur_prep+dur_t_req+dur_back_center) >= b_min:
            enough_battery = True
        
        # Possible to response the demand?
        if enough_time_to_pick:
            if enough_battery:
                if data["pick_t"][i_req-1]+dur_t_req <= T:
                    matrix_task["taxi"+str(j_taxi)].append(i_req)
                    matrix_start_time["taxi"+str(j_taxi)].append(data["pick_t"][i_req-1])
                    matrix_fini_time["taxi"+str(j_taxi)].append(round((data["pick_t"][i_req-1]+dur_t_req), 2))
                    battery_level[j_taxi-1] = round(battery_level[j_taxi-1] - bcr*(dur_prep+dur_t_req), 2)
                    # print("battery level after assign1 = ", battery_level)
                    #Remove the assigned demand from the chromosome
                    for j_count in range(nb_taxi):
                        elem_remove = nb_taxi*(i_req-1) + j_count
                        if (i_gene != elem_remove):
                            order_assign.remove(elem_remove)
            else:
                matrix_task["taxi"+str(j_taxi)].append("b")
                dur_prev_to_center = cal_dur_back_to_center(prev_req - 1)
                matrix_start_time["taxi"+str(j_taxi)].append(prev_fini_time + dur_prev_to_center)
                matrix_fini_time["taxi"+str(j_taxi)].append(prev_fini_time + dur_prev_to_center + R)
                battery_level[j_taxi-1] = 100
                # print("battery level after recharging = ", battery_level)
                dur_prep1 = cal_dur_move_time(i_req - 1)
                prev_fini_time1 = matrix_fini_time["taxi"+str(j_taxi)][-1]
                if ((prev_fini_time1 + dur_prep1 <= data["pick_t"][i_req-1]) & (battery_level[j_taxi-1] - bcr*(dur_prep1+dur_t_req+dur_back_center) >= b_min)):
                    if data["pick_t"][i_req-1]+dur_t_req <= T:
                        matrix_task["taxi"+str(j_taxi)].append(i_req)
                        matrix_start_time["taxi"+str(j_taxi)].append(data["pick_t"][i_req-1])
                        matrix_fini_time["taxi"+str(j_taxi)].append(round((data["pick_t"][i_req-1]+dur_t_req), 2))
                        battery_level[j_taxi-1] = round(battery_level[j_taxi-1] - bcr*(dur_prep1+dur_t_req), 2)
                        # print("battery level after assign2 = ", battery_level)
                        #Remove the assigned demand from the chromosome
                        for j_count in range(nb_taxi):
                            elem_remove = nb_taxi*(i_req-1) + j_count
                            if (i_gene != elem_remove):
                                order_assign.remove(elem_remove)
            
    # print("matrix_task = ", matrix_task)
    # print("matrix_start_time = ", matrix_start_time)
    # print("matrix_fini_time = ", matrix_fini_time)

    obj_val_chrom = cal_obj2(matrix_task, matrix_start_time, matrix_fini_time)    
    if detail != None:
        return matrix_task, matrix_start_time, matrix_fini_time, obj_val_chrom
    else:
        return obj_val_chrom
        
def sorting_population(arr_pop, arr_res):
    n = len(arr_pop)
    while n > 1:
        for i in range (n-1):
            if (arr_res[i] < arr_res[i+1]):
                arr_res[i], arr_res[i+1] = arr_res[i+1], arr_res[i]
                arr_pop[i], arr_pop[i+1] = arr_pop[i+1], arr_pop[i]
        n = n - 1


#GA
def genetic_algorithm(i_run):
    chrom_size = nb_req*nb_taxi
    nb_chrom = 2*chrom_size
    population = np.random.rand(nb_chrom, chrom_size)

    result_GA = []
    for i in range(len(population)):
        chromosome = population[i].copy()
        result_GA_elem = decode_GA(chromosome)
        result_GA.append(result_GA_elem)
    
    sorting_population(population, result_GA)
    best_res = result_GA[0]
    
    exec_st_time = time.time()
    stop_value = 30
    nb_select = int(0.1*nb_chrom)
    nb_crossover = int(0.7*nb_chrom)
    nb_mutation = nb_chrom - nb_select - nb_crossover
    n_not_improve = 0
    n_iteration = 0
    
    while n_not_improve < stop_value:
        new_pop = []
        for i in range (nb_select):
            new_pop.append(population[i])
        #new_pop = population[:nb_select]
        for i in range (nb_crossover):
            chrom_p1 = population[random.randint(0,nb_select-1)]
            chrom_p2 = population[random.randint(nb_select,nb_chrom-1)]
            crossover_chrom = []
            for j in range (chrom_size):
                if random.randint(1,10) <= 7:
                    crossover_chrom.append(chrom_p1[j])
                else:
                    crossover_chrom.append(chrom_p2[j])
            new_pop.append(crossover_chrom)
        for i in range (nb_mutation):
            new_pop.append(np.random.rand(chrom_size))
    
        population = new_pop
    
        result_GA = []
        for i in range(nb_chrom):
            chromosome = population[i].copy()
            result_GA_elem = decode_GA(chromosome)
            result_GA.append(result_GA_elem)
    
        sorting_population(population, result_GA)
        best_res_iter = result_GA[0]
        if (best_res_iter > best_res):
            best_res = best_res_iter
            n_not_improve = 0
        else:
            n_not_improve = n_not_improve + 1
        n_iteration = n_iteration + 1
        # print('best_res = ', best_res)
        # print('n_not_improve = ', n_not_improve)

    final_chrom = population[0]
    final_task, final_st_time, final_fi_time, final_obj = decode_GA(final_chrom, True)
    comp_time = time.time() - exec_st_time
    # print("--- %s seconds ---" % (comp_time))
    # print("final_task = ", final_task)
    # print("final_st_time = ", final_st_time)
    # print("final_fi_time = ", final_fi_time)
    # print("final_obj = ", final_obj)
    
    #Keep solution detail for the first run of each instance
    if (i_run == 1):
        file2.write("--- %s seconds ---\n" % (comp_time))
        file2.write("final_task = "+ str(final_task)+"\n")
        file2.write("final_st_time = "+ str(final_st_time)+"\n")
        file2.write("final_fi_time = "+ str(final_fi_time)+"\n")
        file2.write("final_obj = "+ str(final_obj)+"\n")
    return final_obj, comp_time

file1 = open("result_repair.txt","w")
file2 = open("result_example_repair.txt","w")
for inst in ["instance10_2.txt", "instance20_2.txt", "instance20_3.txt", "instance30_2.txt",
                "instance30_3.txt", "instance30_4.txt", "instance50_2.txt", "instance50_3.txt",
                "instance50_5.txt", "instance100_3.txt", "instance100_5.txt"]:
    print(inst)
    file1.write(str(inst)+"\n")
    file2.write(str(inst)+"\n")
    nb_req, nb_taxi, data, center_val = prepare_data(inst)
    for i_run in range(1,2):
        obj_v, c_time = genetic_algorithm(i_run)
        print("run no.", i_run, " obj = ", round(obj_v, 2), " c_time = ", round(c_time, 2))
        file1.write("run no."+ str(i_run)+ " obj= " + str(round(obj_v, 2))+ " c_time= "+ str(round(c_time, 2))+"\n")
file1.close()
file2.close()

