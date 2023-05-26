#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import time
import json
import re
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
import utils
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import threading
import pickle
import math


# In[ ]:


# parameters
record_video = False
print_out = False
display_img = True
img_scale = 0.1
update_timer = 1
alignment_count_display = 5
update_json_files = False
update_pokemon = False
phones = ['Pixel 3 XL', 'Pixel 7 Pro','Nexus 6P']


# In[ ]:


# Load the JSON files
pokemon_names = utils.load_pokemon_names()
pokemon_details = utils.load_pokemon_details()
moves = utils.load_moves_info()

# load alignment info
alignment_df = utils.load_alignment_df(alignment_count_display)
# update json files if prompted:
if update_json_files:
    utils.update_json_files()
    avail_cups = utils.download_current_cups()
if update_pokemon:
    utils.update_pk_info()
    utils.update_move_info()
# connect to phone
client = utils.connect_to_device("127.0.0.1:5037")
phone_t = phones.index(client.device_name)


# In[ ]:


roi_adjust =[[50,370,860],[50,350,860],[50,370,860]]
roi_adjust = roi_adjust[phone_t]
roi_dim = [530,50]
my_roi = (roi_adjust[0], roi_adjust[1], roi_dim[0], roi_dim[1])
opp_roi = (roi_adjust[2], roi_adjust[1], roi_dim[0], roi_dim[1])
center = client.resolution[0]/2
msg_width = 850
msgs_roi = (int(center-msg_width/2),995,msg_width,150)
# Load the template images in color format
my_pokemon_template_color = cv2.imread('templates/my-temp2.png')
opp_pokemon_template_color = cv2.imread('templates/opp-temp2.png')
my_pokemon_template_color = cv2.resize(my_pokemon_template_color, (269, 77))
opp_pokemon_template_color = cv2.resize(opp_pokemon_template_color, (269, 77))
# Convert the template images to grayscale
my_pokemon_template = cv2.cvtColor(my_pokemon_template_color, cv2.COLOR_BGR2GRAY)
opp_pokemon_template = cv2.cvtColor(opp_pokemon_template_color, cv2.COLOR_BGR2GRAY)
feed_res = (int(client.resolution[0]*img_scale), int(client.resolution[1]*img_scale))
cup_names_combo_box = ['Great League', 'Ultra League', 'Master League']
save_cup_names = []
if update_json_files:
    for cup in avail_cups:
        cup_names_combo_box.append(cup['title'])
        save_cup_names.append(cup['title'])
    for format in avail_cups:
        title = format['title']
        cup = format['cup']
        category = 'overall'
        league = format['cp']
        utils.download_ranking_data(cup, category, league,title)
    with open('json_files/saved_cup_names.pkl', 'wb') as f:
        pickle.dump(save_cup_names, f)
else:
    with open('json_files/saved_cup_names.pkl', 'rb') as f:
        avail_cups = pickle.load(f)
    cup_names_combo_box.extend(avail_cups)


# In[ ]:


class Move:
    def __init__(self, name,move_id, move_type, category, energy):
        self.name = name
        self.move_id = move_id
        self.move_type = move_type  # move's type (Ghost, Steel, Grass, etc.)
        self.category = category  # Fast or Charge
        self.energy = energy  # For fast moves, it's energy gain. For charge moves, it's energy cost.

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, move_type={self.move_type}, category={self.category}, energy={self.energy})"


class FastMove(Move):
    def __init__(self, name, move_id, move_type, energy_gain, cooldown):
        super().__init__(name, move_id, move_type, "fast", energy_gain)
        self.cooldown = cooldown
        self.move_turns = int(self.cooldown/500)
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, move_type={self.move_type}, category={self.category}, energy={self.energy}, cooldown={self.cooldown}, move_turns={self.move_turns})"
    def move_count_str(self):
        return f'{self.name} - {self.move_turns}'

class ChargeMove(Move):
    def __init__(self, name, move_id, move_type, energy_cost, counts):
        super().__init__(name, move_id, move_type, "charge", energy_cost)
        self.counts = counts
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, move_type={self.move_type}, category={self.category}, energy={self.energy}, counts={self.counts})"
    def move_count_str(self,fast_move):
        return f'{self.name} - {self.counts[fast_move]}'


class Pokemon:
    def __init__(self, name,id):
        self.species_name = name
        self.species_id = id
        self.fast_moves = {} 
        self.charge_moves = {}  
        self.energy = 0 # accumulated energy
        self.time_on_field = 0
        self.last_update_time = time.time()  # to calculate the energy accumulation
        self.recommended_moveset = None  # recommended Fast and Charge Moves based on league
        self.ui_chosen_moveset = None
        self.rating = None

    def __repr__(self):
        return f"{self.__class__.__name__}(species_name={self.species_name}, energy={self.energy}, time_on_field={self.time_on_field}, recommended_moveset={self.recommended_moveset})"
    
    def add_fast_move(self, name,id, move_type, energy_gain, cooldown):
        fast_move = FastMove(name, id, move_type, energy_gain, cooldown)
        self.fast_moves[id] = fast_move 

    def add_charge_move(self, name,id, move_type, energy_cost, counts):
        charge_move = ChargeMove(name, id, move_type, energy_cost, counts)
        self.charge_moves[id] = charge_move 

    def set_recommended_moveset(self, league_details):
        for league_pokemon in league_details:
            if self.species_name.lower() == league_pokemon['speciesName'].lower():
                self.recommended_moveset = league_pokemon['moveset']
                self.rating = league_pokemon['rating']
                break
        self.ui_chosen_moveset = self.recommended_moveset

    def load_fast_move(self, move_details,fast_move_name):
        for move in move_details:
            if move['moveId'] == fast_move_name:
                self.add_fast_move(move['name'],move['moveId'], move['type'], move['energyGain'], move['cooldown'])

    def load_charge_move(self, move_details,charge_move_name):
        for move in move_details:
            if move['moveId'] == charge_move_name:
                self.add_charge_move(move['name'],move['moveId'], move['type'], move['energy'], {})
                self.calculate_charge_move_counts(self.charge_moves[move['moveId']])
    
    def update_energy(self):
        current_time = time.time()
        time_difference = current_time - self.last_update_time
        self.energy += self.calculate_energy_gain(time_difference)
        self.last_update_time = current_time

    def calculate_energy_gain(self, time_difference):
        # logic for energy gain over time
        pass

    def calculate_charge_move_counts(self,charge_move):
        for fast_move_id, fast_move in self.fast_moves.items():
            self.charge_moves[charge_move.move_id].counts[fast_move_id] = self.charge_move_counts_helper(fast_move, charge_move)

    def charge_move_counts_helper(self,fast_move, charge_move):
        counts = []
        counts.append(math.ceil((charge_move.energy * 1) / fast_move.energy))
        counts.append(math.ceil((charge_move.energy * 2) / fast_move.energy) - counts[0])
        counts.append(math.ceil((charge_move.energy * 3) / fast_move.energy) - counts[0] - counts[1])

        return counts

def load_pk_data(info_name, pokemon_names, pokemon_details,moves_data,league_pok):
    temp_corrected_name = utils.closest_pokemon_name(info_name, pokemon_names)
    move_data = [item for item in pokemon_details if temp_corrected_name and item['speciesName'].startswith(temp_corrected_name)]
    
    pokemon_list = []
    
    for data in move_data:
        pokemon = Pokemon(data['speciesName'],data['speciesId'])
        
        for fast_move_name in data['fastMoves']:
            pokemon.load_fast_move(moves_data,fast_move_name)
        
        for charge_move_name in data['chargedMoves']:
            pokemon.load_charge_move(moves_data,charge_move_name)
        
        pokemon.set_recommended_moveset(league_pok)
        
        pokemon_list.append(pokemon)
    
    return pokemon_list


class Player:
    def __init__(self, name):
        self.name = name
        self.pokemons = [None, None, None]
        self.recommended_pk_ind = [None, None, None]
        self.current_pokemon_index = 0  # Index of the current Pokemon on the field
        self.shield_count = 2  
        self.pokemon_count = 0
        self.switch_lock = False
        self.switch_lock_timer = 0 
        self.switch_out_time = None
        self.oldest_pokemon_index = 0  # Added to track the index of the oldest Pokemon

    def add_pokemon(self, pokemon_list):
        if not pokemon_list:
            return False  # Skip this list if it's empty
        pokemon_list = [pokemon for pokemon in pokemon_list if pokemon.recommended_moveset]  # Filter out Pokemon with empty recommended_moveset
        # Check if any Pokemon in the list is already in the pokemons list
        for i, slot in enumerate(self.pokemons):
            if slot and any(p.species_id == p2.species_id for p2 in slot for p in pokemon_list):
                self.update_current_pokemon_index(i)  # Update index to the existing Pokemon
                return False  # Skip if Pokemon already in list

        if self.pokemon_count < 3:
            self.pokemons[self.pokemon_count] = pokemon_list
            self.update_current_pokemon_index(self.pokemon_count)
            self.update_recommended_pk()
            self.pokemon_count += 1
        else:
            self.pokemons[self.oldest_pokemon_index] = pokemon_list  # Replace the oldest Pokemon with the new one
            self.update_current_pokemon_index(self.oldest_pokemon_index)  
            self.update_recommended_pk()
            self.oldest_pokemon_index = (self.oldest_pokemon_index + 1) % 3  # Update the index of the oldest Pokemon
        return True


    def update_recommended_pk(self):
        max_rating_index = None
        max_rating = -1  # Initialize to a low value

        for i, pk in enumerate(self.pokemons[self.current_pokemon_index]):
            if pk.rating is not None and pk.rating > max_rating:
                max_rating = pk.rating
                max_rating_index = i

        self.recommended_pk_ind[self.current_pokemon_index] = max_rating_index


    def update_current_pokemon_index(self, new_index):
        if new_index != self.current_pokemon_index:
            self.current_pokemon_index = new_index
            self.switch_out_time = time.time()
            self.switch_lock = True
            self.switch_lock_timer = 60
            self.countdown_switch_lock()
 
    def countdown_switch_lock(self):
        self.switch_lock_timer = 60 - int(time.time() - self.switch_out_time)
        if self.switch_lock_timer <= 0:
            self.switch_out_time = None
            self.switch_lock_timer = 0
            self.switch_lock = False

    def use_shield(self):
        if self.shield_count > 0:
            self.shield_count -= 1

    def ui_helper_specific(self,pokemon_ind,chosen_pk_ind,fast_mv):
        pk_chosen = self.pokemons[pokemon_ind][chosen_pk_ind]
        
    def ui_helper(self,pokemon_ind=None,chosen_pk_ind=None):
        pk_name = []
        pk_fast_moves = []
        pk_charge_moves = []

        if pokemon_ind is None:
            pokemon_ind = self.current_pokemon_index
            for pk in self.pokemons[pokemon_ind]:
                pk_name.append(pk.species_name)

        if chosen_pk_ind is None:
            pk_recom = self.pokemons[pokemon_ind][self.recommended_pk_ind[self.current_pokemon_index]]
        else:
            pk_recom = self.pokemons[pokemon_ind][chosen_pk_ind]
            
        for fast_mv in pk_recom.fast_moves.values():
            pk_fast_moves.append(fast_mv.move_count_str())
        for charge_mv in pk_recom.charge_moves.values():
            pk_charge_moves.append(charge_mv.move_count_str(pk_recom.recommended_moveset[0]))
            
        return pk_name, pk_fast_moves, pk_charge_moves

    def __repr__(self):
        return f"Player(name={self.name}, " \
               f"pokemons={[str(pokemon) for pokemon in self.pokemons]}, " \
               f"current_pokemon_index={self.current_pokemon_index}, " \
               f"pokemon_count={self.pokemon_count}, " \
               f"shield_count={self.shield_count}, " \
               f"switch_lock={self.switch_lock}, " \
               f"switch_lock_timer={self.switch_lock_timer})"

# league_pok = f"json_files/rankings/Ultra League.json"
# with open(league_pok, 'r') as file:
#     league_pok = json.load(file)

# pk = load_pk_data('tapu bulu',pokemon_names,pokemon_details,moves,league_pok)
# my_player = Player('me')
# my_player.add_pokemon(pk)


# In[ ]:


class PokemonBattleAssistant(ctk.CTk):
    def __init__(self,feed_res,cup_names):
        super().__init__()
        self.title("Pokemon Battle Assistant")
        self.feed_res = feed_res

        mainframe = ctk.CTkFrame(self)
        mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=30, pady=20)

        # Add the drop down menu
        self.league_combobox = ctk.CTkComboBox(mainframe, values=cup_names,command=self.league_callback)
        self.league_combobox.grid(column=0, row=0,sticky="W", padx=0, pady=10)
        self.league_combobox.set('choose league')

        opponent_frame = ctk.CTkLabel(mainframe, text="Opponent's Pokemon", text_color= 'gray', anchor="nw")
        opponent_frame.grid(column=0, row=1, sticky=(tk.W, tk.E), padx=0, pady=0)
        opponent_frame.grid_columnconfigure((0, 1, 2), weight=1) 

        self.opp_pokemon_frames = [self.create_pokemon_frame(opponent_frame, i , 'opp') for i in range(3)]
        for i, frame in enumerate(self.opp_pokemon_frames):
            frame.grid(column=i, row=1, padx=10, pady=(5,0))

        info_frame = ctk.CTkFrame(mainframe)
        info_frame.grid(column=0, row=2, sticky=(tk.W, tk.E), padx=20, pady=20)

        info_frame.grid_columnconfigure(0, weight=1)
        info_frame.grid_columnconfigure(1, weight=1)

        self.switch_timer_label = ctk.CTkLabel(info_frame, text="Switch Timer: ")
        self.switch_timer_label.grid(column=0, row=2, sticky="W", padx=(10,70), pady=10)

        self.correct_alignment_label = ctk.CTkLabel(info_frame, text="Correct Alignment: ")
        self.correct_alignment_label.grid(column=1, row=2, sticky="W", padx=(0,70), pady=10)

        ctk.CTkButton(info_frame, text="Reset UI", command=self.reset_ui).grid(column=2, row=2, padx=(30,0), pady=10)
        self.start_button = ctk.CTkButton(info_frame, text="Start Recording", command=self.recoding)
        self.start_button.grid(column=5,sticky='E', row=2, padx=10, pady=10)

        my_frame = ctk.CTkLabel(mainframe, text="Your Pokemon",text_color= 'gray',anchor="nw")
        my_frame.grid(column=0, row=3, sticky=(tk.W, tk.E), padx=0, pady=0)
        my_frame.grid_columnconfigure((0, 1, 2), weight=1) 

        self.my_pokemon_frames = [self.create_pokemon_frame(my_frame,i , "me") for i in range(3)]
        for i, frame in enumerate(self.my_pokemon_frames):
            frame.grid(column=i, row=3, padx=10, pady=(5,10))

        pil_image = Image.new("RGB", feed_res)

        self.elapsed_time_label = ctk.CTkLabel(mainframe, text="0")
        self.elapsed_time_label.grid(column=0, row=5, sticky='E', padx=0, pady=0) 

        if display_img:
            self.my_image = ctk.CTkImage(light_image=pil_image,dark_image=pil_image, size=feed_res)
            self.image_label = ctk.CTkLabel(mainframe, text='',image=self.my_image)
            self.image_label.grid(column=0, row=4, pady=0)

        self.prev_my_roi_img = np.array([])
        self.prev_opp_roi_img = np.array([])
        self.prev_corrected_my_name = None
        self.prev_corrected_opp_name = None
        self.threshold = 500
        self.league = None

        self.league_pok = None
        self.move_type = ['fast_move','charge_move1','charge_move2']
        self.move_type_disp = ['Fast Move','Charge Move 1','Charge Move 2']
        self.current_opp_pokemon_index = None
        self.current_my_pokemon_index = None
        self.my_energy_start = time.time()
        self.opp_energy_start = time.time()



        self.vid_res = (int(client.resolution[0]/2), int(client.resolution[1]/2))
        self.record_vid = False
        self.out = None

        self.my_player = Player('me')
        self.opp_player = Player('opp')

        self.frames_map = {'me': self.my_pokemon_frames, 'opp': self.opp_pokemon_frames}
        self.player_map = {'me': self.my_player, 'opp': self.opp_player}


    def create_pokemon_frame(self, master, num, side):
        frame = ctk.CTkFrame(master)
        box_width = 250

        frame.pokemon_name_label = ctk.CTkComboBox(frame, values=[f"Pokemon {num+1}"], font=("Arial", 20), command=lambda choice, box_type='pokemon_name_label', side=side, num=num: self.pk_callback(choice, box_type, side, num), width=200)
        frame.pokemon_name_label.grid(column=0, row=0, sticky="W", padx=10, pady=10)

        frame.fast_move = ctk.CTkComboBox(frame, values=["Fast Move"], command=lambda choice, box_type='fast_move', side=side, num=num: self.pk_callback(choice, box_type, side, num), width=box_width)
        frame.fast_move.grid(column=0, row=1, sticky="W", padx=10, pady=10)

        frame.charge_move1 = ctk.CTkComboBox(frame, values=["Charge Move 1"], width=box_width)
        frame.charge_move1.grid(column=0, row=2, sticky="W", padx=10, pady=10)

        frame.charge_move1_progress = ctk.CTkProgressBar(frame, orientation="horizontal", height=20, width=50,corner_radius=0)
        frame.charge_move1_progress.set(0) 
        frame.charge_move1_progress.grid(column=1, row=2, sticky="W", padx=10, pady=10)

        frame.charge_move2 = ctk.CTkComboBox(frame, values=["Charge Move 2"], width=box_width)
        frame.charge_move2.grid(column=0, row=3, sticky="W", padx=10, pady=10)

        frame.charge_move2_progress = ctk.CTkProgressBar(frame, orientation="horizontal", height=20, width=50,corner_radius=0,fg_color='#3281a8',progress_color='#103c52')
        frame.charge_move2_progress.set(0.5) 
        frame.charge_move2_progress.grid(column=1, row=3, sticky="W", padx=10, pady=10)

        return frame

    def pk_callback(self, choice, box_type, side, num):
 
        label = self.find_label(side,num,'pokemon_name_label')
        chosen_pk_ind = self.get_index(label)

        mv_recom = self.player_map[side].pokemons[num][chosen_pk_ind].recommended_moveset
        _, fast_moves, charge_moves = self.player_map[side].ui_helper(num,chosen_pk_ind)
        if box_type == 'pokemon_name_label':
            self.update_label(side, num, 'fast_move', fast_moves)
            fast_mv_label = self.find_label(side,num,'fast_move')
            fast_mv = self.player_map[side].pokemons[num][chosen_pk_ind].fast_moves[mv_recom[0]]
            fast_mv_label.set(fast_mv.move_count_str())

        else:
            fast_mv = choice.split(' - ')[0].upper().replace(" ", "_")
            fast_mv = self.player_map[side].pokemons[num][chosen_pk_ind].fast_moves[fast_mv]

        self.update_label(side,num, 'charge_move1', charge_moves)
        charge_label = self.find_label(side,num,'charge_move1')
        charge_label.set(self.player_map[side].pokemons[num][chosen_pk_ind].charge_moves[mv_recom[1]].move_count_str(fast_mv.move_id))

        self.update_label(side,num, 'charge_move2', charge_moves)
        charge_label = self.find_label(side,num,'charge_move2')
        charge_label.set(self.player_map[side].pokemons[num][chosen_pk_ind].charge_moves[mv_recom[2]].move_count_str(fast_mv.move_id))   

    def highlight_on(self, frame):
        if frame.cget('fg_color') != "green":
            frame.configure(fg_color="green")
        
    def highlight_off(self, frame):
        if frame.cget('fg_color') != "transparent":
            frame.configure(fg_color="transparent")

    def league_callback(self,choice):
        self.league = choice
        self.league_pok = f"json_files/rankings/{self.league}.json"
        try:
            with open(self.league_pok, 'r') as file:
                self.league_pok = json.load(file)
                print(f"Loaded {self.league} JSON data")
        except FileNotFoundError:
            print(f"Failed to load {self.league} JSON data")

    def recoding(self):
        if self.start_button.cget("text") == "Start Recording":
            self.start_button.configure(text="Stop Recording")
            filename = utils.get_next_filename('videos')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            self.out = cv2.VideoWriter(filename, fourcc, 60.0, self.vid_res)
            self.record_vid = True
            self.stream_vid = threading.Thread(target=self.vid_stream, daemon=False)
            self.stream_vid.start()
        else:
            self.start_button.configure(text="Start Recording")
            self.record_vid = False  # Ensure recording stops
            self.stream_vid.join()  # Wait for the thread to actually stop
            print('releasing video')
            self.out.release()

            
    def reset_ui(self):

        self.prev_my_roi_img = np.array([])
        self.prev_opp_roi_img = np.array([])
        self.prev_corrected_my_name = None
        self.prev_corrected_opp_name = None
        self.league = None
        self.league_pok = None

        self.league_combobox.set('choose league')

        self.current_opp_pokemon_index = None
        self.current_my_pokemon_index = None
        for number in range(len(self.my_pokemon_frames)):
            self.update_label('me',number, 'pokemon_name_label', f'Pokemon {number+1}')
            for move,move_disp in zip(self.move_type,self.move_type_disp):
                self.update_label('me',number, move, move_disp)
        for number in range(len(self.opp_pokemon_frames)):
            self.update_label('opp',number, 'pokemon_name_label', f'Pokemon {number+1}')
            for move,move_disp in zip(self.move_type,self.move_type_disp):
                self.update_label('opp',number, move, move_disp)
        self.switch_timer_label.configure(text=f"Switch Timer: ")
        self.correct_alignment_label.configure(text="Correct Alignemnt: ")
        for my_frame,opp_frame in zip(self.my_pokemon_frames,self.opp_pokemon_frames):
            self.highlight_off(my_frame)
            self.highlight_off(opp_frame)

    def vid_stream(self):
        while self.record_vid:
            screen = client.last_frame
            resized_frame = cv2.resize(screen, self.vid_res)
            self.out.write(resized_frame)
            time.sleep(0.01) 


    def moveset_update(self,side):
        current_ind = self.player_map[side].current_pokemon_index
        pk_recom = self.player_map[side].recommended_pk_ind[current_ind]
        if pk_recom is None:
            print(f"No recommended Pokemon index set for player {side}, Pokemon {current_ind}")
            return
        mv_recom = self.player_map[side].pokemons[current_ind][pk_recom].recommended_moveset
        pk_names, fast_moves, charge_moves = self.player_map[side].ui_helper()

        self.update_label(side,self.player_map[side].current_pokemon_index,'pokemon_name_label',pk_names)

        self.update_label(side,current_ind, 'fast_move', fast_moves)
        fast_mv_label = self.find_label(side,current_ind,'fast_move')
        fast_mv = self.player_map[side].pokemons[current_ind][pk_recom].fast_moves[mv_recom[0]]
        fast_mv_label.set(fast_mv.move_count_str())

        self.update_label(side,current_ind, 'charge_move1', charge_moves)
        charge_label = self.find_label(side,current_ind,'charge_move1')
        charge_label.set(self.player_map[side].pokemons[current_ind][pk_recom].charge_moves[mv_recom[1]].move_count_str(fast_mv.move_id))

        self.update_label(side,current_ind, 'charge_move2', charge_moves)
        charge_label = self.find_label(side,current_ind,'charge_move2')
        charge_label.set(self.player_map[side].pokemons[current_ind][pk_recom].charge_moves[mv_recom[2]].move_count_str(fast_mv.move_id))        

    def update_highlight(self,side):
        current_ind = self.player_map[side].current_pokemon_index
        for i, frame in enumerate(self.frames_map[side]):
            if i == current_ind:
                self.highlight_on(frame)
            else:
                self.highlight_off(frame)
        
    def update_ui(self,client):
        time_start = time.time()
        screen = client.last_frame
        # screen = cv2.imread('templates/screenshot.png')
        if screen is not None:
            my_roi_img = screen[my_roi[1]:my_roi[1] + my_roi[3], my_roi[0]:my_roi[0] + my_roi[2]]
            opp_roi_img = screen[opp_roi[1]:opp_roi[1] + opp_roi[3], opp_roi[0]:opp_roi[0] + opp_roi[2]]
            msgs_roi_img = screen[msgs_roi[1]:msgs_roi[1] + msgs_roi[3], msgs_roi[0]:msgs_roi[0] + msgs_roi[2]]

            if utils.mse(my_roi_img, self.prev_my_roi_img) > self.threshold or utils.mse(opp_roi_img, self.prev_opp_roi_img) > self.threshold:
                self.prev_my_roi_img  = my_roi_img.copy()
                self.prev_opp_roi_img = opp_roi_img.copy()
                self.prev_msg_roi_img = msgs_roi_img.copy()
                gray_my_roi  = cv2.cvtColor(my_roi_img, cv2.COLOR_BGR2GRAY)
                gray_opp_roi = cv2.cvtColor(opp_roi_img, cv2.COLOR_BGR2GRAY)
                gray_msg_roi = cv2.cvtColor(msgs_roi_img, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur to the grayscale images
                blur_my_roi  = cv2.GaussianBlur(gray_my_roi, (5, 5), 0)
                blur_opp_roi = cv2.GaussianBlur(gray_opp_roi, (5, 5), 0)
                blur_msg_roi = cv2.GaussianBlur(gray_msg_roi, (5, 5), 0)

                # Apply binary thresholding
                _, thresh_my_roi  = cv2.threshold(blur_my_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, thresh_opp_roi = cv2.threshold(blur_opp_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, thresh_msg_roi = cv2.threshold(blur_msg_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Apply morphological operations to remove noise
                kernel = np.ones((2, 2), np.uint8)
                thresh_my_roi = cv2.morphologyEx(thresh_my_roi, cv2.MORPH_OPEN, kernel)
                thresh_opp_roi = cv2.morphologyEx(thresh_opp_roi, cv2.MORPH_OPEN, kernel)
                thresh_msg_roi = cv2.morphologyEx(thresh_msg_roi, cv2.MORPH_OPEN, kernel)

                thresh_my_roi = Image.fromarray(thresh_my_roi)
                thresh_opp_roi = Image.fromarray(thresh_opp_roi)
                thresh_msg_roi = Image.fromarray(thresh_msg_roi)

                with PyTessBaseAPI(psm=PSM.AUTO_OSD) as api:
                    api.SetImage(thresh_my_roi)
                    api.Recognize()
                    my_info = api.GetUTF8Text()

                    api.SetImage(thresh_opp_roi)
                    api.Recognize()
                    opp_info = api.GetUTF8Text()

                # Display the extracted Pokémon name and CP value
                if print_out:
                    print("My Info:", my_info)
                    print("Opponent Info:", opp_info)

                # for auto-detection if league not chosen
                if self.league_combobox.get() == 'choose league':
                    # Extract CP values using regex
                    my_cp = re.search(r'\bCP\s+(\d+)\b', my_info)
                    opp_cp = re.search(r'\bCP\s+(\d+)\b', opp_info)
                    if my_cp:
                        my_cp = int(my_cp.group(1))
                        print(f"My Pokémon CP: {my_cp}")
                    if opp_cp:
                        opp_cp = int(opp_cp.group(1))
                        print(f"Opponent Pokémon CP: {opp_cp}")
                    if my_cp and opp_cp:
                        higher_cp = max(my_cp, opp_cp)
                        if higher_cp <= 500:
                            self.league = "Little Cup"
                        elif higher_cp <= 1500 > 500:
                            self.league = "Great League"
                        elif 1500 < higher_cp <= 2500:
                            self.league = "Ultra League"
                        else:
                            self.league = "Master League"
                        self.league_combobox.set(self.league)
                        self.league_pok = f"json_files/rankings/{self.league}.json"
                        try:
                            with open(self.league_pok, 'r') as file:
                                self.league_pok = json.load(file)
                                print(f"Loaded {self.league} JSON data")
                        except FileNotFoundError:
                            print(f"Failed to load {self.league} JSON data")
                    else:
                        if print_out:
                            print("Could not determine league")    
                    
                # Extract Pokémon names
                my_info_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', my_info)
                opp_info_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', opp_info)

                if self.league:
                    if my_info_match and opp_info_match:
                        my_info_name = my_info_match.group(0)
                        opp_info_name = opp_info_match.group(0)

                        if not self.prev_corrected_opp_name or not self.prev_corrected_my_name :
                            self.my_energy_start = time.time()
                            self.opp_energy_start = time.time()

                        my_pk = load_pk_data(my_info_name,pokemon_names,pokemon_details,moves,self.league_pok)
                        opp_pk = load_pk_data(opp_info_name,pokemon_names,pokemon_details,moves,self.league_pok)

                        update_me = self.my_player.add_pokemon(my_pk)
                        update_opp = self.opp_player.add_pokemon(opp_pk)
                        if update_me:
                            self.moveset_update('me')
                        if update_opp:
                            self.moveset_update('opp')
                        self.update_highlight('me')
                        self.update_highlight('opp')
 
                    else:
                        self.my_enery_start = time.time()
                        self.opp_enery_start = time.time()
                        with PyTessBaseAPI(psm=PSM.AUTO_OSD) as api:
                            api.SetImage(thresh_msg_roi)
                            api.Recognize()
                            msg_info = api.GetUTF8Text()
                            print('heeeeeerererere',msg_info)

            # opponent switch lock timer
            if self.opp_player.switch_lock:
                self.opp_player.countdown_switch_lock()
                self.switch_timer_label.configure(text=f"Switch Timer: {self.opp_player.switch_lock_timer}")

            # Draw rectangles around the ROI
            roi_color = (0, 0, 0)  
            roi_thick = 7
            screen_with_rois = cv2.rectangle(screen.copy(), (my_roi[0], my_roi[1]), (my_roi[0] + my_roi[2], my_roi[1] + my_roi[3]), roi_color, roi_thick)
            screen_with_rois = cv2.rectangle(screen_with_rois, (opp_roi[0], opp_roi[1]), (opp_roi[0] + opp_roi[2], opp_roi[1] + opp_roi[3]), roi_color, roi_thick)
            screen_with_rois = cv2.rectangle(screen_with_rois, (msgs_roi[0], msgs_roi[1]), (msgs_roi[0] + msgs_roi[2], msgs_roi[1] + msgs_roi[3]), roi_color, roi_thick)

            if self.current_my_pokemon_index is not None and self.current_opp_pokemon_index is not None:
                my_count = self.find_label('me',self.current_my_pokemon_index,'fast_move').get()[-1]
                opp_count = self.find_label('opp',self.current_opp_pokemon_index,'fast_move').get()[-1]
                try:
                    correct_count = alignment_df.loc[int(my_count), opp_count]
                except KeyError:
                    correct_count = "Unknown"
                self.correct_alignment_label.configure(text=f"Correct Alignemnt: {correct_count}")
            
            if display_img:
                resized_image = cv2.resize(screen_with_rois, self.feed_res, interpolation=cv2.INTER_AREA)
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                pil_img  = Image.fromarray(resized_image)
                self.my_image.configure(light_image=pil_img,dark_image=pil_img) 

        time_elapsed = time.time() - time_start
        self.elapsed_time_label.configure(text=f'{time_elapsed:0.3f}')
        self.after(update_timer, lambda: app.update_ui(client)) 

    def update_label(self,side, index, name_of_variable, new_value):
        label = self.find_label(side,index, name_of_variable)
        if isinstance(new_value[0], list):
            new_value = [' - '.join(str(inner_item) for inner_item in sublist) for sublist in new_value]
        elif isinstance(new_value, str):
            new_value = [new_value]
        label.configure(values=new_value)
        label.set(new_value[0])  

    def find_label(self, side, index, name_of_variable):
        frame = self.frames_map[side][index]
        return getattr(frame, name_of_variable)


    def get_index(self,combo_box):
        current_value = combo_box.get()
        values = combo_box.cget("values")
        return values.index(current_value)

if __name__ == "__main__":
    app = PokemonBattleAssistant(feed_res,cup_names_combo_box)
    app.after(update_timer, lambda: app.update_ui(client)) 
    app.mainloop()

