from mainlib import *


water_meter_data = [
    {'id': None, 'file_name': 'data/a.json', 'div': '100'},
    {'id': None, 'file_name': 'data/b.json', 'div': '1'},
    {'id': None, 'file_name': 'data/c.json', 'div': '1'}
]
for wm in water_meter_data:
    init_data(water_meter_id=wm['id'], file_name=wm['file_name'], div=wm['div'])

    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['hour'] = data.index.hour
    data['minute'] = data.index.minute
    data['weekday_nb'] = data.index.dayofweek
    data['weekday'] = data.index.weekday_name