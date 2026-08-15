import csv
import numpy as np
import random
import os


dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"


def extract_block(filename, left, upper, right, lower, delimiter=";", encoding="cp1252"):
    num_lines = sum(1 for _ in open(filename, encoding=encoding))
    with open(filename, newline="", encoding=encoding) as file:
        reader = csv.reader(file, delimiter=delimiter)
        contents = []
        for i, row in enumerate(reader):
            if i < upper or((lower is not None) and  i >= (lower % num_lines)):
                continue
            else:
                contents.append(row[left:right])

    contents = list(filter(lambda line: line != [], contents))
    return contents

def merge_cleanup(file_1, box_1, file_2, box_2, dest):
    contents_1 = extract_block(file_1, *box_1)
    contents_2 = extract_block(file_2, *box_2)
    print(contents_2)

    contents = contents_1 + contents_2

    with open(dest, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(contents)

def to_table_2d_in_3d(filename, index_1, index_2, names_1, names_2):
    with open(filename, newline="") as file:
        reader = csv.reader(file)
        contents = []
        for row in reader:
            contents.append(row)

    # extract field names
    for row in contents:
        if not row[0] in names_1:
            print(f"warning: name {row[0]} unknown")
        if not row[1] in names_2:
            print(f"warning: name {row[1]} unknown")

    # generate map to indices
    names_1_map = {name:index for index, name in enumerate(names_1)}
    names_2_map = {name:index for index, name in enumerate(names_2)}

    data_shape = [1, 1, 1]
    data_shape[index_1] = len(names_1)
    data_shape[index_2] = len(names_2)

    data = np.zeros(data_shape)
    for line in contents:
        if "Nicht klassifizierbar <0>" in line:
            continue

        name_1_id = names_1_map[line[0]]
        name_2_id = names_2_map[line[1]]

        if line[2] == "-":
            value = 0
        else:
            value = int(line[2])

        index_3d = [0, 0, 0]
        index_3d[index_1] = name_1_id
        index_3d[index_2] = name_2_id

        # write to table
        data[*index_3d] = value

    return data

def extract_names(filename, left, upper, right, lower):
    names = extract_block(filename, left, upper, right, lower)
    names = [element[0] for element in names]
    return names

names_age = extract_names(dir_path + "names_age.csv", 1, 7, 2, -8)
names_places = extract_names(dir_path + "names_places.csv", 2, 10, 3, -26)

origin_dest = to_table_2d_in_3d(dir_path + "origin_destination.csv", 0, 1, names_places, names_places)
dest_age = to_table_2d_in_3d(dir_path + "age_dest.csv", 2, 1, names_age, names_places)
origin_age = to_table_2d_in_3d(dir_path + "age_origin.csv", 2, 0, names_age, names_places)

def random_names(n, names):
    names_map = {name:index for index, name in enumerate(names)}
    rand_names = random.sample(names, n)
    return sorted(rand_names, key=lambda name: names_map[name])

def extract_subtable(filename, names_1, index_1, names_2, index_2, names_3, index_3, value_index, left, upper, right, lower, remove_nonclassifiable=False, delimiter=";", encoding="cp1252"):
    extracted = extract_block(filename, left, upper, right, lower, delimiter=delimiter, encoding=encoding)

    names_1_map = {name:index for index, name in enumerate(names_1)}
    names_2_map = {name:index for index, name in enumerate(names_2)}
    names_3_map = {name:index for index, name in enumerate(names_3)}

    data = np.zeros((len(names_1), len(names_2), len(names_3)))
    data[:,:,:] = np.nan
    for line in extracted:
        if "Nicht klassifizierbar <0>" in line and remove_nonclassifiable:
            continue

        data_index = [None, None, None]
        data_index[0] = names_1_map[line[index_1]]
        data_index[1] = names_2_map[line[index_2]]
        data_index[2] = names_3_map[line[index_3]]
        if line[value_index] =="-":
            value = 0
        else:
            value = int(line[value_index])

        data[*data_index] = value

    return data

def extract_table(*args, **kwargs):
    table = extract_subtable(*args, **kwargs)

    print(table)
    assert not np.any(np.isnan(table))

    return table

names_places_federal = extract_names(dir_path + "names_places_federal.csv", 1, 7, 2, -10)
federal_values = extract_table(dir_path + "full_federal.csv", names_places_federal, 2, names_places_federal, 4, names_age, 0, value_index=5,  left=0, upper=7, right=-1, lower=-9, remove_nonclassifiable=True)
federal_cuts = (np.cumsum([9, 10, 25, 18, 6, 21, 9, 4, 23][:-1]), np.cumsum([9, 10, 25, 18, 6, 21, 9, 4, 23][:-1]), np.arange(1, len(names_age)))


num_districts = 23
names_places_vienna = names_places[-23:]
names_age_5_group = extract_names(dir_path + "names_age_5_group.csv", left=0, upper=7, right=-1, lower=-8)
vienna_cuts = (np.arange(1, num_districts), np.arange(1, num_districts), np.arange(5, 20 * 5, 5))
vienna_values = extract_table(dir_path + "vienna.csv", names_places_vienna, 2, names_places_vienna, 3, names_age_5_group, 0, value_index=4, left=0, upper=1, right=None, lower=None, remove_nonclassifiable=True, delimiter=",", encoding="utf8")
vienna_subtable = (slice(-23, None), slice(-23, None), slice(None))


if __name__=="__main__":
    # merge_cleanup("age_destination_1.csv", (2, 10, -1, -3),"age_destination_2.csv", (2, 10, -1, -3), "age_dest.csv")
    # merge_cleanup("age_origin_1.csv", (2, 10, -1, -3),"age_origin_2.csv", (2, 10, -1, -3), "age_origin.csv")
    # merge_cleanup("origin_destination_1.csv", (2, 10, -1, -3),"origin_destination_2.csv", (2, 10, -1, -3), "origin_destination.csv")
    #
    # merge_cleanup("vienna_1.csv", (1, 9, -1, -9), "vienna_2.csv", (1, 10, -1, -1), "vienna.csv")


    assert np.allclose(np.sum(origin_dest, axis=(1, 2)), np.sum(origin_age, axis=(1, 2)))
    assert np.allclose(np.sum(origin_dest, axis=(0, 2)), np.sum(dest_age, axis=(0, 2)))
    assert np.allclose(np.sum(dest_age, axis=(0, 1)), np.sum(origin_age, axis=(0, 1)))

    print(random_names(7, names_places))
    print(random_names(7, names_age))

    print(extract_subtable("sub_table_sample_1.csv", names_places, 3, names_places, 2, names_age, 0, value_index=4,  left=1, upper=10, right=-1, lower=-9))
