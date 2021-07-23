import os
import cv2
import json
import numpy as np 
from pathlib import Path
import csv 

MASK_WIDTH = 256	# Dimensions should match those of ground truth image
MASK_HEIGHT = 256

img_dir='images/sample'

def process_filename(filename):
    f = filename.split('/')[-1]
    bhoomi = ['Bhoomi_data', 'bhoomi']
    if any(x in filename for x in bhoomi):
        return img_dir + '/Bhoomi_data/' + f
    if 'penn_in_hand' in filename:
        return img_dir+'/pen_in_hand/'+f
    if 'penn-in-hand' in filename:
        return img_dir+'/pen-in-hand/'+f
    if ('jain-mscripts' in filename):
        return img_dir+'/jain-mscripts/'+f
    if ('Jain_manuscripts' in filename):
        return img_dir+'/Jain_mansucsripts/'+f
    if 'ASR_Images' in filename:
        return img_dir+'/ASR_Images/'+f
    if 'pdf' in filename:
        return img_dir+"/pdf_images"+filename.split('pdf_images')[1]
    if 'google' in filename:
        return img_dir+'/google_scraped/'+f     

    if 'Stacked' in filename:
        arr = filename.split('/')
        return '/'.join(arr[1:])+'.png'

def get_metadata(path_to_json_file):
    f = open(path_to_json_file,'r')
    j = f.read()
    f.close()
    jj1 = json.loads(j)
    metadata= jj1.get('_via_img_metadata')
    return metadata

def get_file_list(files):
    file_list = []
    for csv_file in files:
        with open(csv_file,'r') as myfile:
            csv_reader = csv.reader(myfile, delimiter=',')
            for row in csv_reader:
                file_list += row
    
    return file_list

def get_img_shape(filename):
    path = filename
    img = cv2.imread(path)
    return img.shape

def generate_mask(regions,shape):
    MASK_WIDTH,MASK_HEIGHT,_ = shape
    total = []
    for r in regions:
        try:
            x_cord = r['shape_attributes']['all_points_x']
            y_cord = r['shape_attributes']['all_points_y']
        except:
            continue  
        points = []
        for i, x in enumerate(x_cord):
            points.append([int(x), int(y_cord[i])])
        points = np.array(points)
        total.append(points)
    mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
    cv2.fillPoly(mask,total,color = (255,255,255))
    return mask

def get_ground_truth(metadata):
    img_urls=metadata.keys()
    for url in img_urls:
        filename = process_filename(metadata[url]['filename'])
        # print("\n",filename)
        try:
            shape = get_img_shape(filename)
        except:
            continue
        all_regions=metadata[url]['regions']
        mask = generate_mask(all_regions,shape)
        print("Saving mask")
        outfile = 'ground_truth/'+ filename
        directory = os.path.dirname(outfile)
        Path(directory).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(outfile,mask)

data=get_metadata('doc_pb/val/temporary.json')
get_ground_truth(data)

url = 'http://bhoomi.csa.iisc.ernet.in:8080/ihg/manuscript_editor/images/INGYA%20RATNAM/GOML/1456/1.jpg'
# print(process_filename(url))