import pandas as pd
import math
train_df = pd.read_csv('Training.csv')
test_df = pd.read_csv('Testing.csv')
max_x = train_df['X'].max()
try:
    n = int(input("Enter the number of columns you want: "))
    if n <= 0:
        raise ValueError
except ValueError:
    print("Please enter a positive integer for the number of columns.")
try:
    k = int(input("Enter k: "))
    if k <= 0 or k > len(train_df):
        raise ValueError
except ValueError:
    print(f"Please enter a positive integer for k that is less than or equal to {len(train_df)}.")

if set(train_df.columns) != set(test_df.columns):
    print("The training and testing datasets have different column names.")

width = max_x / n

# this algorithm here makes the partition
def partitions(n, max_x, width):
    min_x = (0,0)
    p = [min_x]
    r = []
    for i in range(n):
        min_x = tuple(x + y for x, y in zip(min_x, (width, 0)))
        p.append(min_x)
    for points in range(len(p)):
        rnge =  tuple(x + y for x, y in zip(p[points], (width, max_x)))
        r.append(rnge)
    return p,r 

p, r = partitions(n, max_x, width)

del p[len(p)-1]
del r[len(r)-1]


# returns the partition number
def get_partition(point, p, r):
    x,y = point
    for i in range(len(p)):
        if (x >= p[i][0] and x <= r[i][0]):
            return i+1  # partition number is index+1
    return 0  # point does not fall into any partition



# Create an empty dictionary to store the points for each partition
partition_points = {i: {} for i in range(1, n+1)}

# Partition the user-provided points and store them in the dictionary
for index, row in train_df.iterrows():
    point = (row['X'], row['Y'])
    outcome = row['Outcome']
    partition = get_partition((point[0], 0), p, r)
    partition_points[partition][point] = outcome


# Print out the points in each partition
for partition, points in partition_points.items():
    print(f"Points in partition {partition}:")
    for point, outcome in points.items():
        print(f"{point}: {outcome}")

# Lists Points in partition, given the partition number
# updated get_points function
def get_points(num):
    points_list = []
    for point, outcome in partition_points[num].items():
        points_list.append((point, outcome))
    return points_list

# updated knn function
def knn(train_df, k, test_df):
    correct_predictions = 0
    total_predictions = 0
    for index, row in test_df.iterrows():
        point = (row['X'], row['Y'])
        datas = []
        distances = []
        actual_outcome = row['Outcome']
        partition = get_partition((point[0], 0), p, r)

        # get the points in the current partition and the adjacent partitions
        if partition == 1:
            for i in range(1, 3):
                datas.extend(get_points(i))
        elif partition == n:
            for i in range(n-1, n+1):
                datas.extend(get_points(i))
        else:
            for i in range(partition-1, partition+2):
                datas.extend(get_points(i))

        # calculate the distances and store them in a list
        for data in datas:
            distance = math.sqrt((point[0]-data[0][0])**2 + (point[1]-data[0][1])**2)
            distances.append((distance, data[1]))

        # sort the distances in ascending order and get the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]

        # get the outcome for the majority of the nearest neighbors
        votes = [neighbor[1] for neighbor in neighbors]
        predicted_outcome = max(set(votes), key=votes.count)

        # check if the predicted outcome is correct
        if predicted_outcome == actual_outcome:
            correct_predictions += 1
        total_predictions += 1

    accuracy = (correct_predictions / total_predictions)*100
    return accuracy


accuracy = knn(train_df, k, test_df)
print(f"Accuracy: {accuracy}")

