import json
import pandas as pd
import io
import requests
import os
from adbutils import adb
import scrcpy.core as scrcpy
from difflib import get_close_matches
import math
import numpy as np

def load_pokemon_names():
    # Load the JSON files
    with open('json_files/pk.json', 'r') as file:
        return json.load(file)

def load_moves_info():
    with open('json_files/moves.json', 'r') as file:
        return json.load(file)

def load_alignment_df():
    alingment_info = """,1,2,3,4,5
    1,,"1,2","2,3","3,4","4,5"
    2,,,"1,3","1,2","2,5"
    3,,"1,2",,"1,4","3,5"
    4,,,"2,3",,"1,5"
    5,,"1,2","1,4","3,4","""

    df = pd.read_csv(io.StringIO(alingment_info), index_col=0)
    def find_correct_first_three_counts(row, col):
        if pd.isna(df.at[row, col]):
            return None
        move_counts = [int(count) for count in df.at[row, col].split(',')]
        first_count, step = move_counts
        return [first_count + step * i for i in range(3)]

    for index, row in df.iterrows():
        for col in df.columns:
            result = find_correct_first_three_counts(index, col)
            df.at[index, col] = result
    return df

def update_json_files():
    repo_owner = 'pvpoke'
    repo_name = 'pvpoke'
    folder_path = 'src/data/rankings/all/overall/'
    destination_directory = 'json_files'

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
                    new_file_name = 'great-league.json'
                elif file_name == 'rankings-2500.json':
                    new_file_name = 'ultra-league.json'
                elif file_name == 'rankings-10000.json':
                    new_file_name = 'master-league.json'
                else:
                    new_file_name = file_name

                local_path = os.path.join(destination_directory, new_file_name)

                file_content = requests.get(download_url).content
                with open(local_path, 'wb') as f:
                    f.write(file_content)
                    print(f"Downloaded {local_path}")
    else:
        print(f"Failed to get folder content")

def connect_to_device(ip):
    adb.connect("127.0.0.1:5037")
    client = scrcpy.Client(device=adb.device_list()[0])
    client.start(threaded=True)
    print(f'Connected to: {client.device_name}')
    return client


# Function to find the closest Pokémon name
def closest_pokemon_name(name, names_list):
    closest_name = get_close_matches(name, names_list, n=1, cutoff=0.6)
    if closest_name:
        return closest_name[0]
    return None

def calculate_move_counts(fast_move, charged_move):
    counts = []
    counts.append(math.ceil((charged_move['energy'] * 1) / fast_move['energyGain']))
    counts.append(math.ceil((charged_move['energy'] * 2) / fast_move['energyGain']) - counts[0])
    counts.append(math.ceil((charged_move['energy'] * 3) / fast_move['energyGain']) - counts[0] - counts[1])

    return counts

def get_moveset_and_counts(pokemon_name, pokemon_data, move_data):
    moveset = None
    for pokemon in pokemon_data:
        if pokemon_name.lower() == pokemon['speciesName'].lower():
            moveset = pokemon['moveset']
            break

    if moveset is None:
        return None , 0

    fast_move = moveset[0]
    charged1 = moveset[1]
    charged2 = moveset[2]

    fast_move_data = None
    for move in move_data:
        if fast_move.lower() == move['moveId'].lower():
            fast_move_data = move
            break

    if fast_move_data is None:
        return None , 0

    fast_count = round(fast_move_data['cooldown'] / 500)
    move_counts = {}
    move_counts[fast_move] = fast_count
    for charged_move_name in [charged1, charged2]:
        for move in move_data:
            if charged_move_name.lower() == move['moveId'].lower():
                move_counts[charged_move_name] = calculate_move_counts(fast_move_data, move)
                break

    return move_counts , fast_count


def get_moveset_and_counts_udpated(pokemon_name, pokemon_data, move_data):
    moveset = None
    for pokemon in pokemon_data:
        if pokemon_name.lower() == pokemon['speciesName'].lower():
            moveset = pokemon['moveset']
            break

    if moveset is None:
        return None , 0

    fast_move = moveset[0]
    charged1 = moveset[1]
    charged2 = moveset[2]

    fast_move_data = None
    for move in move_data:
        if fast_move.lower() == move['moveId'].lower():
            fast_move_data = move
            break

    if fast_move_data is None:
        return None , 0

    fast_count = round(fast_move_data['cooldown'] / 500)
    move_counts = {
        'fast_move': [fast_move, fast_count]
    }
    for charged_move_name, key in zip([charged1, charged2], ['charge_move1', 'charge_move2']):
        for move in move_data:
            if charged_move_name.lower() == move['moveId'].lower():
                move_counts[key] = [charged_move_name, calculate_move_counts(fast_move_data, move)]
                break

    return move_counts , fast_count


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