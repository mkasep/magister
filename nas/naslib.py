import base64


def water_meter_data_parser(nas_data_str):
    btea_orig = bytearray(nas_data_str, 'UTF-8')
    btea = base64.b64decode(btea_orig)
    return int.from_bytes(btea[:4], byteorder='little', signed=False)



def map_nas_data(div):
    def map_fn(row):
        data_int = float(water_meter_data_parser(row['data']))
        data = data_int / float(div)  # to l
        return {
            'date': row['date'],
            'data': data
        }
    return map_fn