import csv
import numpy as np
import matplotlib.pyplot as plt

with open('data.csv', newline='') as file:
    reader = csv.reader(file, delimiter=',')
    keys = next(reader)
    data = np.loadtxt(file, delimiter=',')
print(data.shape)

# for i in data:
#     print(i.shape)

def gender_box_plot(set_index, set_name):
    split_sets = [[], [], []]
    for i in data:
        gender = int(i[37])
        if gender > 0 and gender <= 3:
            if set_index == 36:
                if i[set_index] < 150:
                    split_sets[gender-1].append(i[set_index])
            else:
                split_sets[gender-1].append(i[set_index])
    print(split_sets)
    fig, ax = plt.subplots()
    ax.set_title(set_name)
    ax.boxplot(split_sets)
    plt.show()

def by_age(set_index, set_name):
    fig, ax = plt.subplots()
    ax.set_title(set_name)
    filtered_data = [[],[]]
    for i in data:
        age = i[36]
        if age < 150:
            filtered_data[0].append(age)
            filtered_data[1].append(i[set_index])
    print(filtered_data)
    ax.plot(filtered_data[0], filtered_data[1], 'ro')
    plt.show()

def age_bplot():
    fdata = []
    for i in data:
        age = i[36]
        if age < 150:
            fdata.append(age)
    fig, ax = plt.subplots()
    ax.set_title('Age distribution')
    ax.boxplot(fdata)
    plt.show()

age_bplot()