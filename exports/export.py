import sys
import numpy as np
import pathlib

sys.path += [str(pathlib.Path(__file__).parent / ".." / "data"), str(pathlib.Path(__file__).parent / ".." / "calculations")]

import ipf_2025_full
import ipf_2025_full_tiled
import ipf_2025_full_tiled_with_vienna
import data_preprocessing as data_pre


def export_migration_table(filename, table, names_places, names_age):
    "export table as filename, a csv file; dimensions of table have to be origin, destination, age"

    first_line = ["origin", "destination", "age", "value"]
    lines = [first_line]

    it = np.nditer(table, flags=["multi_index"])
    for _ in it:
        index = it.multi_index
        line = [names_places[index[0]], names_places[index[1]], names_age[index[2]],
                str(table[index])
               ]
        lines.append(line)

    for i in range(len(lines)):
        lines[i] = ",".join(lines[i])

    text = "\n".join(lines)

    with open(filename, "w") as file:
        file.write(text)

if __name__=="__main__":
    # runtime very roughly one and a half minutes
    print("exporting table 1 of 3")
    table_full = ipf_2025_full.ipf_estimate()
    export_migration_table("2025_full.csv", table_full, data_pre.names_places, data_pre.names_age)

    print("exporting table 2 of 3")
    table_full_tiled = ipf_2025_full_tiled.ipf_estimate()
    export_migration_table("2025_full_tiled.csv", table_full_tiled, data_pre.names_places, data_pre.names_age)

    print("exporting table 3 of 3")
    table_full_tiled_with_vienna = ipf_2025_full_tiled_with_vienna.ipf_estimate()
    export_migration_table("2025_full_tiled_with_vienna.csv", table_full_tiled_with_vienna, data_pre.names_places, data_pre.names_age)
