import csv
import numpy as np
import matplotlib.pyplot as plt

with open('data.csv', newline='') as file:
    reader = csv.reader(file, delimiter=',')
    keys = next(reader)
    data = np.loadtxt(file, delimiter=',')

def gender_box_plot(set_index, set_name):
    split_sets = [[], [], []]
    genders = ['male', 'female', 'other']

    for i in data:
        gender = int(i[37])
        if gender > 0 and gender <= 3:
            if set_index == 36:
                if i[set_index] < 150:
                    split_sets[gender-1].append(i[set_index])
            else:
                split_sets[gender-1].append(i[set_index])
    fig, ax = plt.subplots()
    ax.set_title(set_name)
    ax.set_xticklabels(genders)
    ax.grid(axis='y', linestyle = '--')
    ax.boxplot(split_sets, showmeans=True,whis=4)
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
    ax.grid(axis='y', linestyle = '--')
    ax.grid(axis='x', linestyle = '--')
    ax.plot(filtered_data[0], filtered_data[1], 'ro')
    plt.show()

def age_bplot():
    fdata = []
    for i in data:
        age = i[36]
        if age < 124:
            fdata.append(age)
    fig, ax = plt.subplots()
    ax.set_title('Age Boxplot')
    ax.boxplot(fdata, showmeans=True, whis=3)
    ax.grid(axis='y', linestyle = '--')
    plt.show()

gender_box_plot(36, 'Age distribution by gender')

count = 0
total = 0
valid_ages = []
for i in data:
    age = i[36]
    if age < 124:
        total += age
        count += 1
        valid_ages.append(age)
print(f'avg {total/count}')
print(np.median(valid_ages))
print(np.quantile(valid_ages, 0.75))

for i in range(32, 36):
    by_age(i, f'{keys[i]} score by age')