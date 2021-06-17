import json
import shutil

# data_json1 = json.load(open("val.json", 'r'))
# data_json2 = json.load(open("train.json", 'r'))

# data_json = {}
# for key in data_json1.keys():
    # data_json[key] = data_json1[key]
    
# for key in data_json2.keys():
    # data_json[key] = data_json2[key]

# json.dump(data_json, open('trainval.json', 'w'), indent=4)


######################
# data_json = json.load(open("train.json", 'r'))
# new_data = {}
# for key in data_json.keys():
   # # file_path = key + '.csv'
   # # shutil.copyfile(file_path, 'features/' + file_path.split('/')[-1])
    
    # new_data[key.split('/')[-1]] = data_json[key]

# json.dump(new_data, open('train_d1.json', 'w'), indent=4)


# res_dict = {}
# data_json = json.load(open("val_d1.json", 'r'))
# for key in data_json.keys():
    # data_json[key]['duration'] = data_json[key]['duration_second']
    # data_json[key]['subset']   = "validation"
    # data_json[key]['resolution'] = "1920x1080"

    # res_dict[key] = data_json[key]

# data_json = json.load(open("train_d1.json", 'r'))
# for key in data_json.keys():
    # data_json[key]['duration'] = data_json[key]['duration_second']
    # data_json[key]['subset']   = "training"
    # data_json[key]['resolution'] = "1920x1080"

    # res_dict[key] = data_json[key]

# json.dump(res_dict, open('trainval_ASFD.json', 'w'), indent=4)

res_dict = {}
data_json = json.load(open("val_d1.json", 'r'))
for key in data_json.keys():
    data_json[key]['duration'] = data_json[key]['duration_second']
    data_json[key]['subset']   = "validation"
    data_json[key]['resolution'] = "1920x1080"

    res_dict[key] = data_json[key]

json.dump({"database": res_dict}, open('val_ASFD.json', 'w'), indent=4)
    


