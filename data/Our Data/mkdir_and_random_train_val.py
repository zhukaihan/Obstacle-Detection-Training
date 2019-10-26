import os
import json
import random

if __name__ == '__main__':
  
  dataturks_json = 'Obstacle Detection Dataset.json'
  data_dir = os.path.join(os.getcwd(), 'formatted data')
  
  img_dir = os.path.join(data_dir, 'imgs')
  annotated_dir = os.path.join(data_dir, 'annotated')
  
  os.mkdir(data_dir)
  os.mkdir(img_dir)
  os.mkdir(annotated_dir)
  
  with open(dataturks_json, 'r') as f:
    lines = f.readlines()

  image_names = []
  for line in lines:
    data = json.loads(line)
    if (data['annotation'] == None):
      continue
    if len(data['annotation']) == 0:
      continue

    image_url = data['content']

    image_names.append(image_url.split("/")[-1])

  random.shuffle(image_names)
  train_imgs = image_names[:int(len(image_names) * 0.75)]
  val_imgs = image_names[int(len(image_names) * 0.75):]
  
  train_txt = ' 1\n'.join(train_imgs) + ' 1'
  val_txt = ' 1\n'.join(val_imgs) + ' 1'
  
  with open('train.txt', 'w') as f:
    f.write(train_txt)
  with open('val.txt', 'w') as f:
    f.write(val_txt)


