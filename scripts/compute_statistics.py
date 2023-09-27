import argparse
import io
import json
import tarfile
from PIL import Image
import numpy as np
from p_tqdm import p_map
import os
from tqdm import tqdm
from collections import defaultdict
import logging
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process tar files in a directory')
    parser.add_argument('--max_files', type=int, required=True, help='Max number of tar files to process (integer)')
    parser.add_argument('--log_level', type=str, default='INFO', help='Log level (string)')
    parser.add_argument('--total_workers', type=int, required=True, help='Total number of workers (integer)')
    parser.add_argument('--path', type=str, default='/p/fastdata/mmlaion/ocr/RenderedText/', help='path')
    parser.add_argument('--logdir', type=str, default='./log/', help='output path for logs and results')
    parser.add_argument('--type', type=str, default="tar", help='tar or parquet')
    return parser.parse_args()

def bbox2str(bbox):
    res = '[(%.2f, %.2f)'%(bbox[0][0], bbox[0][1])
    for coord in bbox[1:]:
        res += ', (%.2f, %.2f)'%(coord[0], coord[1])
    res += ']'
    return res

def ocr_caption(json_content, ocr_key):
    if 'caption' not in json_content:
        ocr_caption = '<ocr>\n'
    else:
        ocr_caption = json_content['caption'] + '\n<ocr>\n'
    for bbox, text in zip(json_content[ocr_key]['bb_relative'], json_content[ocr_key]['text']):
        ocr_caption += bbox2str(bbox) + ': ' + text + '\n' 
    ocr_caption += '</ocr>' 
    return ocr_caption
            
def work_parquet(parquet_name):
    # do the same thing as work_tar but for parquet files
    statistics = defaultdict(list)
    # open parquet file
    df = pd.read_parquet(parquet_name)
    if 'ocr_annotation' in df.columns:
        ocr_key = 'ocr_annotation'
    elif 'ocr_result' in df.columns:
        ocr_key = 'ocr_result'
    else:
        ocr_key = None

    # iterate over rows
    for idx, row in tqdm(df.iterrows()):
        statistics['height'] += [row['height']]
        statistics['width'] += [row['width']]
        # check if ocr_annotation is present
        if ocr_key is not None:
            statistics['num_ocr_lines'] += [len(row[ocr_key]['text'])]
            statistics['ocr_lines'] += [row[ocr_key]['text'].tolist()]
            statistics['ocr_caption'] += [ocr_caption(row, ocr_key)]
            statistics['len_ocr_caption'] += [len(statistics['ocr_caption'][-1])]
        # check if caption is present
        if 'caption' in row:
            statistics['len_caption'] += [len(row['caption'])]
    return statistics

def work_tar(tar_name):
    statistics = defaultdict(list)
    with tarfile.open(tar_name, mode='r') as input_tar:
        
        images = {}
        json_files = {}
        
        for tarinfo in input_tar:
            file_data = input_tar.extractfile(tarinfo)
            if file_data:
                if tarinfo.name.lower().endswith('.json'):
                    json_files[tarinfo.name] = json.load(file_data)
                elif tarinfo.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images[tarinfo.name] = file_data.read()
        
        
        for idx, (image_name, image_data) in tqdm(enumerate(images.items())):
            img_ending = image_name.split('.')[-1]
            json_name = image_name.replace(img_ending, 'json')
            logging.debug('Processing image: ' + image_name)
            logging.debug('Processing json: ' + json_name)
            if json_name in json_files:
                json_content = json_files[json_name]
                if 'ocr_annotation' in json_content:
                    ocr_key = 'ocr_annotation'
                elif 'ocr_result' in json_content:
                    ocr_key = 'ocr_result'
                else:
                    ocr_key = None

                try:
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    width, height = image.size
                    assert width == json_content['width'] and height == json_content['height'], 'json image dimensions missmatch with the loaded image'
                    statistics['height'] += [json_content['height']]
                    statistics['width'] += [json_content['width']]
                    if ocr_key is not None:
                        statistics['num_ocr_lines'] += [len(json_content[ocr_key]['text'])]
                        statistics['ocr_lines'] += [json_content[ocr_key]['text']]
                        statistics['ocr_caption'] += [ocr_caption(json_content, ocr_key)]
                        statistics['len_ocr_caption'] += [len(statistics['ocr_caption'][-1])]
                    if 'caption' in json_content:
                        statistics['len_caption'] += [len(json_content['caption'])]
                except Exception as e:
                    logging.error(f"Error processing image: {image_name} with error: {e}")
                
        return statistics

if __name__ == '__main__':
    args = parse_arguments()
    logging.basicConfig(level=args.log_level)

    logging.info('Computing statistics over %s'%args.path)
    # create logdir if it does not exist already
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    # create a list of tar files contained in args.path
    tar_files = []
    for root, dirs, files in os.walk(args.path):
        for file in files:
            if len(tar_files) >= args.max_files:
                break
            if file.endswith('.'+args.type):
                tar_files += [os.path.join(root, file)]
            

    logging.info(f"Processing tar files: {len(tar_files)}")

    n = len(tar_files)
    if args.type == 'tar':
        results = p_map(work_tar, tar_files, num_cpus=args.total_workers)
    elif args.type == 'parquet':
        results = p_map(work_parquet, tar_files, num_cpus=args.total_workers)

    all_results = defaultdict(list)
    for result in results:
        for key, value in result.items():
            all_results[key] += value
        all_results['num_imgs_per_tar'] += [len(result['height'])]

    print(all_results['ocr_caption'][0])
   
    for key in all_results.keys():
        vals = all_results[key]
        if not (isinstance(vals[0], int) or isinstance(vals[0], float)):
            continue
        fig = plt.figure()
        plt.boxplot(all_results[key])
        plt.title(f"Boxplot for {key}")
        plt.savefig(os.path.join(args.logdir, f'boxplot_{key}.png'))
        plt.close(fig)
        logging.info(f"Mean of {key}: {np.mean(all_results[key])}")
        logging.info(f"Std of {key}: {np.std(all_results[key])}")
        logging.info(f"Min of {key}: {np.min(all_results[key])}")
        logging.info(f"Max of {key}: {np.max(all_results[key])}")

    # make word cloud aggregating the words in  all_results['ocr_lines']
    #print(all_results['ocr_lines'])
    all_ocr_lines = []
    for ocr_lines in all_results['ocr_lines']:
        all_ocr_lines += ocr_lines
    all_strings = ' '.join(all_ocr_lines)
    all_words = all_strings.split(' ')
    #remove leading and trailing spaces
    all_words = [word.strip() for word in all_words]
    #filter out every word that is of length 1
    all_words_filtered = [word for word in all_words if len(word) > 1]
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(all_words_filtered))
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(os.path.join(args.logdir, 'wordcloud.png'))
    plt.close()
    # save results to json
    with open(os.path.join(args.logdir, 'results.json'), 'w') as f:
        json.dump(all_results, f)
    