import json
import pandas as pd
import io
import requests
import os
from adbutils import adb
import scrcpy.core as scrcpy
from difflib import get_close_matches
import numpy as np
import cv2
from PIL import Image
import re
import yaml
from datetime import datetime
import pickle
import shutil
from roi_ui import RoiSelector
import tkinter as tk

def load_pokemon_names():
    # Load the JSON files
    with open('json_files/pk.json', 'r') as file:
        return json.load(file)

def load_pokemon_details():
    # Load the JSON files
    with open('json_files/pokemon.json', 'r') as file:
        return json.load(file)
    
def load_moves_info():
    with open('json_files/moves.json', 'r') as file:
        return json.load(file)

def load_alignment_df(counts=4):
    alignment_info = """,1,2,3,4,5
    1,,"1,2","2,3","3,4","4,5"
    2,,,"1,3","1,2","2,5"
    3,,"1,2",,"1,4","3,5"
    4,,,"2,3",,"1,5"
    5,,"1,2","1,4","3,4","""

    df = pd.read_csv(io.StringIO(alignment_info), index_col=0)

    for index, row in df.iterrows():
        for col in df.columns:
            result = find_correct_alignment(df, index, col, counts)
            df.at[index, col] = result
    return df

def load_phone_data(device_name):
    with open("phone_roi.yaml", 'r') as file:
        data = yaml.safe_load(file)
    if data is None:
        return None
    if device_name in data:
        return data[device_name]
    else:
        return None

def get_phone_data(client):
    phone_data = load_phone_data(client.device_name)

    if phone_data is None:
        app = RoiSelector(client)
        app.update_ui(client)
        app.mainloop()
        phone_data = load_phone_data(client.device_name)

    if phone_data:
        roi_dict = {roi_key: phone_data.get(roi_key) for roi_key in 
                    ['my_roi', 'opp_roi', 'msgs_roi', 'my_pokeballs_roi', 
                     'opp_pokeballs_roi', 'my_typing_roi', 'opp_typing_roi']}
    else:
        print("Failed to retrieve phone data")
        return None

    return roi_dict

def find_correct_alignment(df, row, col, counts):
    if pd.isna(df.at[row, col]):
        return None
    move_counts = [int(count) for count in df.at[row, col].split(',')]
    if len(move_counts) < 2:
        return None
    first_count, step = move_counts[:2]
    return [first_count + step * i for i in range(counts)]

import shutil

def update_json_files():
    try:
        repo_owner = 'pvpoke'
        repo_name = 'pvpoke'
        folder_path = 'src/data/rankings/all/overall/'
        destination_directory = 'json_files/rankings'

        # Delete all files in the destination directory
        if os.path.isdir(destination_directory):
            shutil.rmtree(destination_directory)
        os.makedirs(destination_directory)

        headers = {'Accept': 'application/vnd.github+json'}
        url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}'
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            files = response.json()
            for file in files:
                if file['type'] == 'file' and file['name'].endswith('.json'):
                    download_url = file['download_url']
                    file_name = file['name']

                    if file_name == 'rankings-1500.json':
                        new_file_name = 'Great League.json'
                    elif file_name == 'rankings-2500.json':
                        new_file_name = 'Ultra League.json'
                    elif file_name == 'rankings-10000.json':
                        new_file_name = 'Master League.json'
                    elif file_name == 'rankings-500.json':
                        new_file_name = 'Little Cup.json'
                    else:
                        new_file_name = file_name

                    local_path = os.path.join(destination_directory, new_file_name)

                    file_content = requests.get(download_url).content
                    with open(local_path, 'wb') as f:
                        f.write(file_content)
                        print(f"Downloaded {local_path}")
        else:
            print(f"Failed to get folder content, status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def update_leagues_and_cups(update):
    cup_names_combo_box = ['Great League', 'Ultra League', 'Master League']
    save_cup_names = []
    try:
        if update:
            update_json_files()
            avail_cups = download_current_cups()
            for cup in avail_cups:
                cup_names_combo_box.append(cup['title'])
                save_cup_names.append(cup['title'])
            for format in avail_cups:
                title = format['title']
                cup = format['cup']
                category = 'overall'
                league = format['cp']
                download_ranking_data(cup, category, league, title)
            with open('json_files/saved_cup_names.pkl', 'wb') as f:
                pickle.dump(save_cup_names, f)
        else:
            with open('json_files/saved_cup_names.pkl', 'rb') as f:
                avail_cups = pickle.load(f)
            cup_names_combo_box.extend(avail_cups)
        return cup_names_combo_box
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return cup_names_combo_box


def download_ranking_data(cup, category, league,title):
    key = f"{cup}{category}{league}"
    object_rankings = {}
    repo_owner = 'pvpoke'
    repo_name = 'pvpoke'
    headers = {'Accept': 'application/vnd.github+json'}
    if key not in object_rankings:
        file_path = f"src/data/rankings/{cup}/{category}/rankings-{league}.json"
        url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}'

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            file_info = response.json()
            download_url = file_info['download_url']
            file_content = requests.get(download_url).content
            
            local_path = os.path.join('json_files/rankings/', f'{title}.json')

            with open(local_path, 'wb') as f:
                f.write(file_content)
                print(f"Downloaded {local_path}")
        else:
            print(f"Failed to get file content")

def update_format_select(formats):
    visible_formats = [format for format in formats if format['showFormat'] and not format.get('hideRankings', False) and 'Silph' not in format['title'] and format['title'] != 'Custom']
    return visible_formats

def update_pk_info():
    repo_owner = 'pvpoke'
    repo_name = 'pvpoke'
    file_path = 'src/data/gamemaster/pokemon.json'
    destination_directory = 'json_files/'

    headers = {'Accept': 'application/vnd.github+json'}
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}'
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_info = response.json()
        download_url = file_info['download_url']
        file_name = file_info['name']

        local_path = os.path.join(destination_directory, file_name)

        file_content = requests.get(download_url).content
        with open(local_path, 'wb') as f:
            f.write(file_content)
            print(f"Downloaded {local_path}")
    else:
        print('failed')

def update_move_info():
    repo_owner = 'pvpoke'
    repo_name = 'pvpoke'
    file_path = 'src/data/gamemaster/moves.json'
    destination_directory = 'json_files/'

    headers = {'Accept': 'application/vnd.github+json'}
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}'
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_info = response.json()
        download_url = file_info['download_url']
        file_name = file_info['name']

        local_path = os.path.join(destination_directory, file_name)

        file_content = requests.get(download_url).content
        with open(local_path, 'wb') as f:
            f.write(file_content)
            print(f"Downloaded {local_path}")
    else:
        print('failed')
           
def download_current_cups():
    repo_owner = 'pvpoke'
    repo_name = 'pvpoke'
    file_path = 'src/data/gamemaster.json'
    destination_directory = 'json_files/rankings'

    headers = {'Accept': 'application/vnd.github+json'}
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}'
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_info = response.json()
        download_url = file_info['download_url']
        file_name = file_info['name']

        local_path = os.path.join(destination_directory, file_name)

        file_content = requests.get(download_url).content
        with open(local_path, 'wb') as f:
            f.write(file_content)
            print(f"Downloaded {local_path}")
    else:
        return None
    
    with open(local_path) as f:
        data = json.load(f)

    # Extract formats and call update_format_select
    formats = data['formats']

    return update_format_select(formats)

def connect_to_device(ip):
    adb.connect("127.0.0.1:5037")
    try:
        client = scrcpy.Client(device=adb.device_list()[0])
    except IndexError:
        raise Exception("No devices connected.")

    client.start(threaded=True)
    print(f'Connected to: {client.device_name}')
    return client


def get_roi_images(frame,roi_dict):
    roi_images = {}
    for roi_name,roi in roi_dict.items():
        roi_images[roi_name] = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    return roi_images


def draw_display_frames(frame, roi_dict, feed_res, roi_color=(0, 0, 0), roi_thick=12):
    for i,roi in enumerate(roi_dict.values()):
        if i == 0:
            frame_with_rois = frame.copy()
        frame_with_rois = cv2.rectangle(frame_with_rois, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), roi_color, roi_thick)
    
    resized_image = cv2.resize(frame_with_rois, feed_res, interpolation=cv2.INTER_AREA)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(resized_image)
    return pil_img


# Function to find the closest Pokémon name
def closest_name(name, names_list):
    closest_name = get_close_matches(name, names_list, n=1, cutoff=0.6)
    if closest_name:
        return closest_name[0]
    return None

def process_image(img):
    prev_img = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    thresh_img = Image.fromarray(thresh_img)
    return prev_img, thresh_img

def mse(image1, image2):
    if image1.size == 0 or image2.size == 0:
        return float("inf")
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    return error


def get_next_filename(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.mp4')]
    
    if not files:
        return os.path.join(directory, '1.mp4')
    else:
        nums = sorted([int(f.split('.')[0]) for f in files])
        next_num = nums[-1] + 1
        return os.path.join(directory, f'{next_num}.mp4')
    
class LeagueDetector:
    def __init__(self):
        self.league = None
        self.league_pok = None
        
    @staticmethod
    def extract_cp(info):
        cp = re.search(r'\bCP\s+(\d+)\b', info)
        return int(cp.group(1)) if cp else None

    def set_league_based_on_cp(self, cp):
        if cp <= 500:
            self.league = "Little Cup"
        elif cp <= 1500:
            self.league = "Great League"
        elif cp <= 2500:
            self.league = "Ultra League"
        else:
            self.league = "Master League"

    def load_league_json(self):
        if self.league:
            self.league_pok = f"json_files/rankings/{self.league}.json"
            try:
                with open(self.league_pok, 'r') as file:
                    self.league_pok = json.load(file)
                    print(f"Loaded {self.league} JSON data")
            except FileNotFoundError:
                print(f"Failed to load {self.league} JSON data")

    def detect_league(self, my_info, opp_info):
        my_cp = self.extract_cp(my_info)
        opp_cp = self.extract_cp(opp_info)
        # print(f"My Pokémon CP: {my_cp}")
        # print(f"Opponent Pokémon CP: {opp_cp}")

        if my_cp and opp_cp:
            higher_cp = max(my_cp, opp_cp)
            self.set_league_based_on_cp(higher_cp)
            self.load_league_json()
        # else:
        #     print("Could not determine league")
        
        return self.league, self.league_pok


def count_pokeballs(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 50, 50])

    mask = cv2.inRange(image_rgb, lower_red, upper_red)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb[::-1]

def detect_emblems(image, color_range=30, save_images=False):
    hex_colors = {
        'normal': '#a0a29f',
        'fire': '#fba64c',
        'water': '#539ddf',
        'electric': '#f2d94e',
        'grass': '#60bd58',
        'ice': '#76d1c1',
        'fighting': '#d3425f',
        'poison': '#b763cf',
        'ground': '#da7c4d',
        'flying': '#a1bbec',
        'psychic': '#fa8582',
        'bug': '#92bd2d',
        'rock': '#c9bc8a',
        'ghost': '#5f6dbc',
        'dragon': '#0c6ac8',
        'dark': '#595761',
        'steel': '#5795a3',
        'fairy': '#ef90e6',
    }

    color_ranges = {pokemon_type: (list(map(lambda x: max(0, x-color_range), hex_to_bgr(color))), list(map(lambda x: min(255, x+color_range), hex_to_bgr(color)))) for pokemon_type, color in hex_colors.items()}

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.Canny(gray, 10, 80)

    gray = cv2.dilate(gray, None, iterations=2)
    gray = cv2.erode(gray, None, iterations=1)

    if save_images:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f'debug/gray_{timestamp}.png'
        filename1 = f'debug/gray_{timestamp}_1.png'
        try:
            cv2.imwrite(filename, gray)
            cv2.imwrite(filename1,image)
        except Exception as e:
            pass

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=30, param2=13, minRadius=30, maxRadius=40)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Sort the circles by their radius, from largest to smallest
        sorted_circles = sorted(circles[0], key=lambda x: -float(x[2]))
        top_circles = sorted_circles[:2]
        number_of_emblems = len(top_circles)
    else:
        return []

    # Detect each type of emblem
    type_counts = {}
    for i in circles[0, :]:
        # Create an empty mask
        mask = np.zeros_like(image)
        mask = cv2.circle(mask, (i[0], i[1]), i[2], (255,255,255), -1)
        masked_image = cv2.bitwise_and(image, mask)
        # if save_images:
        #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        #     filename = f'debug/detected_circle_{timestamp}.png'
        #     try:
        #         cv2.imwrite(filename, masked_image)
        #     except Exception as e:
        #         pass

        for pokemon_type, (lower, upper) in color_ranges.items():
            temp_mask = cv2.inRange(masked_image, np.array(lower), np.array(upper))
            pixel_count = np.count_nonzero(temp_mask)
            type_counts[pokemon_type] = pixel_count + type_counts.get(pokemon_type, 0)

            # if save_images:
            #     result = cv2.bitwise_and(image, image, mask=temp_mask)
            #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            #     cv2.imwrite(f'debug/{pokemon_type}_detection_{timestamp}.png', result)

    sorted_types = [pokemon_type for pokemon_type, pixel_count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:number_of_emblems]]
    
    return sorted(sorted_types)


class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)

    def flush(self):
        pass