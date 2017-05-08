import json


def write_to_file(file_name, data_list):
    file = open(file_name, "w")
    file.write(json.dumps(data_list, indent=4, separators=(',', ': ')))
    file.flush()
    file.close()