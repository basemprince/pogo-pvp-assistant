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
import customtkinter as ctk
import threading
import pickle


# In[ ]:


# parameters
record_video = False
print_out = False
display_img = True
img_scale = 0.1
update_timer = 1
update_json_files = False
update_pokemon = False
phones = ['Pixel 3 XL', 'Pixel 7 Pro','Nexus 6P']


# In[ ]:


# Load the JSON files
pokemon_names = utils.load_pokemon_names()
pokemon_details = utils.load_pokemon_details()
moves = utils.load_moves_info()

# load alignment info
alignment_df = utils.load_alignment_df()
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
        opponent_frame.grid(column=0, row=1, sticky=(tk.W, tk.E), padx=0, pady=(0,20))
        opponent_frame.grid_columnconfigure((0, 1, 2), weight=1) 

        self.opp_pokemon_frames = [self.create_pokemon_frame(opponent_frame, i , 'opp') for i in range(3)]
        for i, frame in enumerate(self.opp_pokemon_frames):
            frame.grid(column=i, row=1, padx=10, pady=(20,0))

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

        self.my_pokemon_frames = [self.create_pokemon_frame(my_frame,i , "my") for i in range(3)]
        for i, frame in enumerate(self.my_pokemon_frames):
            frame.grid(column=i, row=3, padx=10, pady=(20,0))

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
        self.opp_pokemon_memory = []
        self.my_pokemon_memory = []
        self.switch_out_time = None
        self.switch_out_countdown = None
        self.opp_switch_timer_label = None
        self.switch_memory = []
        self.league_pok = None
        self.move_type = ['fast_move','charge_move1','charge_move2']
        self.move_type_disp = ['Fast Move','Charge Move 1','Charge Move 2']
        self.current_opp_pokemon_index = None
        self.current_my_pokemon_index = None

        self.vid_res = (int(client.resolution[0]/2), int(client.resolution[1]/2))
        self.record_vid = False
        self.out = None


    def create_pokemon_frame(self, master, num, side):
        frame = ctk.CTkFrame(master)
        box_width = 250

        frame.pokemon_name_label = ctk.CTkComboBox(frame, values=[f"Pokemon {num+1}"], font=("Arial", 20), command=lambda choice, box_type='pokemon_name_label', side=side, num=num: self.pk_callback(choice, box_type, side, num), width=200)
        frame.pokemon_name_label.grid(column=0, row=0, sticky="W", padx=10, pady=10)

        frame.fast_move = ctk.CTkComboBox(frame, values=["Fast Move"], command=lambda choice, box_type='fast_move', side=side, num=num: self.pk_callback(choice, box_type, side, num), width=box_width)
        frame.fast_move.grid(column=0, row=1, sticky="W", padx=10, pady=10)

        frame.charge_move1 = ctk.CTkComboBox(frame, values=["Charge Move 1"], width=box_width)
        frame.charge_move1.grid(column=0, row=2, sticky="W", padx=10, pady=10)

        frame.charge_move2 = ctk.CTkComboBox(frame, values=["Charge Move 2"], width=box_width)
        frame.charge_move2.grid(column=0, row=3, sticky="W", padx=10, pady=10)

        return frame

    def pk_callback(self, choice, box_type, side, num):
        # print(f"{box_type} ComboBox with choice {choice} from {side} frame: {num} was selected")
        mem_map = {'my': self.my_pokemon_memory, 'opp': self.opp_pokemon_memory}
        label = self.find_label(side,num,'pokemon_name_label')
        chosen_pk_ind = self.get_index(label)
        move_counts = mem_map[side][num][1][chosen_pk_ind]
        chosen_moveset = mem_map[side][num][2]
        if box_type == 'pokemon_name_label':
            self.update_label(side,num, 'fast_move', [pk_fast['fast_move'] for pk_fast in move_counts])
            fast_mv_label = self.find_label(side,num,'fast_move')
            values = fast_mv_label.cget("values")
            for val in values:
                if val.startswith(chosen_moveset[chosen_pk_ind][0]):
                    fast_mv_label.set(val)
                    break        
            choice = fast_mv_label.get()

        for move_data in move_counts:
            if ' - '.join(str(item) for item in move_data['fast_move']) == choice:
                charged_moves = move_data['charged_moves']
                self.update_label(side,num, 'charge_move1', charged_moves)
                self.update_label(side,num, 'charge_move2', charged_moves)
                charge_label = self.find_label(side,num,'charge_move1')
                values = charge_label.cget("values")
                for val in values:
                    if val.startswith(chosen_moveset[chosen_pk_ind][1]):
                        charge_label.set(val)
                        break
                charge_label = self.find_label(side,num,'charge_move2')
                values = charge_label.cget("values")
                for val in values:
                    if val.startswith(chosen_moveset[chosen_pk_ind][2]):
                        charge_label.set(val)
                        break
                break

    def highlight_on(self, frame):
        frame.configure(fg_color="green")
        
    def highlight_off(self, frame):
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
        self.league_combobox.set('choose league')
        self.prev_corrected_my_name = None
        self.prev_corrected_opp_name = None
        self.league = None
        self.opp_pokemon_memory = []
        self.my_pokemon_memory = []
        self.switch_out_time = None
        self.switch_out_countdown = None
        self.switch_memory = []
        self.opp_switch_timer_label = None
        self.league_pok = None
        self.current_opp_pokemon_index = None
        self.current_my_pokemon_index = None
        # self.league_combobox.set('choose league')
        for number in range(len(self.my_pokemon_frames)):
            self.update_label('my',number, 'pokemon_name_label', f'Pokemon {number+1}')
            for move,move_disp in zip(self.move_type,self.move_type_disp):
                self.update_label('my',number, move, move_disp)
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

    def update_pokemon_data(self, info_name, prev_corrected_name):
        temp_corrected_name = utils.closest_pokemon_name(info_name, pokemon_names)

        move_data = [item for item in pokemon_details if temp_corrected_name and item['speciesName'].startswith(temp_corrected_name)]
        if temp_corrected_name == prev_corrected_name:
            return prev_corrected_name, prev_corrected_name,move_data
        
        if print_out:
            for entry in move_data:
                print(f"Species Name: {entry['speciesName']}")
                print(f"Fast Moves: {', '.join(entry['fastMoves'])}")
                print(f"Charged Moves: {', '.join(entry['chargedMoves'])}")
                print("\n")

        if len(move_data) != 0:
            temp_corrected_name = move_data[0]['speciesName']
        else:
            temp_corrected_name = None

        if temp_corrected_name:
            corrected_name = temp_corrected_name
            prev_corrected_name = corrected_name
        else:
            corrected_name = prev_corrected_name
            if print_out:
                print(f"Using previous Pokémon name for Pokémon: {corrected_name}")

        return corrected_name, prev_corrected_name,move_data


    def update_ui(self,client):
        time_start = time.time()
        screen = client.last_frame
        # screen = cv2.imread('templates/screenshot.png')
        corrected_my_name = None
        corrected_opp_name = None
        if screen is not None:
            my_roi_img = screen[my_roi[1]:my_roi[1] + my_roi[3], my_roi[0]:my_roi[0] + my_roi[2]]
            opp_roi_img = screen[opp_roi[1]:opp_roi[1] + opp_roi[3], opp_roi[0]:opp_roi[0] + opp_roi[2]]

            if utils.mse(my_roi_img, self.prev_my_roi_img) > self.threshold or utils.mse(opp_roi_img, self.prev_opp_roi_img) > self.threshold:
                self.prev_my_roi_img = my_roi_img.copy()
                self.prev_opp_roi_img = opp_roi_img.copy()
                gray_my_roi = cv2.cvtColor(my_roi_img, cv2.COLOR_BGR2GRAY)
                gray_opp_roi = cv2.cvtColor(opp_roi_img, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur to the grayscale images
                blur_my_roi = cv2.GaussianBlur(gray_my_roi, (5, 5), 0)
                blur_opp_roi = cv2.GaussianBlur(gray_opp_roi, (5, 5), 0)

                # Apply binary thresholding
                _, thresh_my_roi = cv2.threshold(blur_my_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, thresh_opp_roi = cv2.threshold(blur_opp_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Apply morphological operations to remove noise
                kernel = np.ones((2, 2), np.uint8)
                thresh_my_roi = cv2.morphologyEx(thresh_my_roi, cv2.MORPH_OPEN, kernel)
                thresh_opp_roi = cv2.morphologyEx(thresh_opp_roi, cv2.MORPH_OPEN, kernel)

                thresh_my_roi = Image.fromarray(thresh_my_roi)
                thresh_opp_roi = Image.fromarray(thresh_opp_roi)
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

                if my_info_match and opp_info_match and self.league is not None:
                    my_info_name = my_info_match.group(0)
                    opp_info_name = opp_info_match.group(0)

                    corrected_opp_name, self.prev_corrected_opp_name, opp_move_data = self.update_pokemon_data(opp_info_name, self.prev_corrected_opp_name)
                    corrected_my_name,  self.prev_corrected_my_name, my_move_data  = self.update_pokemon_data(my_info_name,  self.prev_corrected_my_name)
                    
                    if corrected_opp_name:
                        if not self.switch_memory or corrected_opp_name != self.switch_memory[-1]:
                            if len(self.switch_memory) >= 3:
                                self.switch_memory.pop(0)
                            self.switch_memory.append(corrected_opp_name)
                            self.switch_out_time = time.time()

                        move_dic = utils.get_moveset_and_counts_2(opp_move_data, moves)

                        chosen_moveset = []
                        if not any(pokemon[0][0] == corrected_opp_name for pokemon in self.opp_pokemon_memory):
                            for pokemon in self.league_pok:
                                    for opp_pok in opp_move_data:
                                        if opp_pok['speciesName'].lower() == pokemon['speciesName'].lower():
                                            chosen_moveset.append(pokemon['moveset'])
                                            break
                            if len(self.opp_pokemon_memory) >= 3:
                                self.opp_pokemon_memory.pop(0)
                            self.opp_pokemon_memory.append([[pok['speciesName'] for pok in opp_move_data], move_dic,chosen_moveset])

                        for pokemon_number, pokemon_data in enumerate(self.opp_pokemon_memory):
                            pokemon_name, move_counts,chosen_moveset = pokemon_data
                            if print_out:
                                print(f"pokemon_name: {pokemon_name}, move_counts: {move_counts}")
                            self.update_label('opp',pokemon_number, 'pokemon_name_label', pokemon_name)
                            pk_label = self.find_label('opp',pokemon_number,'pokemon_name_label')
                            chosen_pk_ind = self.get_index(pk_label)
                            self.update_label('opp',pokemon_number, 'fast_move', [pk_fast['fast_move'] for pk_fast in move_counts[chosen_pk_ind]])
                            fast_mv_label = self.find_label('opp',pokemon_number,'fast_move')
                            values = fast_mv_label.cget("values")
                            for val in values:
                                if val.startswith(chosen_moveset[chosen_pk_ind][0]):
                                    fast_mv_label.set(val)
                                    break

                            if move_counts is not None:
                                for move_data in move_counts[chosen_pk_ind]:
                                    if ' - '.join(str(item) for item in move_data['fast_move']) == fast_mv_label.get():
                                        charged_moves = move_data['charged_moves']
                                        self.update_label('opp',pokemon_number, 'charge_move1', charged_moves)
                                        self.update_label('opp',pokemon_number, 'charge_move2', charged_moves)
                                        charge_label = self.find_label('opp',pokemon_number,'charge_move1')
                                        values = charge_label.cget("values")
                                        for val in values:
                                            if val.startswith(chosen_moveset[chosen_pk_ind][1]):
                                                charge_label.set(val)
                                                break
                                        charge_label = self.find_label('opp',pokemon_number,'charge_move2')
                                        values = charge_label.cget("values")
                                        for val in values:
                                            if val.startswith(chosen_moveset[chosen_pk_ind][2]):
                                                charge_label.set(val)
                                                break
                                        break

                        self.current_opp_pokemon_index = next((index for index, pokemon_data in enumerate(self.opp_pokemon_memory) if pokemon_data[0][0] == corrected_opp_name), None)
                        if self.current_opp_pokemon_index is not None:
                            for i, frame in enumerate(self.opp_pokemon_frames):
                                if i == self.current_opp_pokemon_index:

                                    self.highlight_on(frame)
                                else:
                                    self.highlight_off(frame)


                    if corrected_my_name:
                        move_dic = utils.get_moveset_and_counts_2(my_move_data, moves)
                        chosen_moveset = []
                        if not any(pokemon[0][0] == corrected_my_name for pokemon in self.my_pokemon_memory):
                            for pokemon in self.league_pok:
                                for my_pok in my_move_data:
                                    if my_pok['speciesName'].lower() == pokemon['speciesName'].lower():
                                        chosen_moveset.append(pokemon['moveset'])
                                        break
                            if len(self.my_pokemon_memory) >= 3:
                                self.my_pokemon_memory.pop(0)
                            self.my_pokemon_memory.append([[pok['speciesName'] for pok in my_move_data], move_dic,chosen_moveset])

                        for pokemon_number, pokemon_data in enumerate(self.my_pokemon_memory):
                            pokemon_name, move_counts,chosen_moveset = pokemon_data
                            self.update_label('my',pokemon_number, 'pokemon_name_label', pokemon_name)
                            pk_label = self.find_label('my',pokemon_number,'pokemon_name_label')
                            chosen_pk_ind = self.get_index(pk_label)
                            self.update_label('my',pokemon_number, 'fast_move', [pk_fast['fast_move'] for pk_fast in move_counts[chosen_pk_ind]])
                            fast_mv_label = self.find_label('my',pokemon_number,'fast_move')
                            values = fast_mv_label.cget("values")
                            for val in values:
                                if val.startswith(chosen_moveset[chosen_pk_ind][0]):
                                    fast_mv_label.set(val)
                                    break

                            if move_counts is not None:
                                for move_data in move_counts[chosen_pk_ind]:
                                    if ' - '.join(str(item) for item in move_data['fast_move']) == fast_mv_label.get():
                                        charged_moves = move_data['charged_moves']
                                        self.update_label('my',pokemon_number, 'charge_move1', charged_moves)
                                        self.update_label('my',pokemon_number, 'charge_move2', charged_moves)
                                        charge_label = self.find_label('my',pokemon_number,'charge_move1')
                                        values = charge_label.cget("values")
                                        for val in values:
                                            if val.startswith(chosen_moveset[chosen_pk_ind][1]):
                                                charge_label.set(val)
                                                break
                                        charge_label = self.find_label('my',pokemon_number,'charge_move2')
                                        values = charge_label.cget("values")
                                        for val in values:
                                            if val.startswith(chosen_moveset[chosen_pk_ind][2]):
                                                charge_label.set(val)
                                                break
                                        break

                        self.current_my_pokemon_index = next((index for index, pokemon_data in enumerate(self.my_pokemon_memory) if pokemon_data[0][0] == corrected_my_name), None)
                        if self.current_my_pokemon_index is not None:
                            for i, frame in enumerate(self.my_pokemon_frames):
                                if i == self.current_my_pokemon_index:
                                    self.highlight_on(frame)
                                else:
                                    self.highlight_off(frame)

            if self.switch_out_time is not None:
                switch_out_countdown = 60 - int(time.time() - self.switch_out_time)
                if switch_out_countdown <= 0:
                    self.switch_out_time = None
                    switch_out_countdown = 0
                self.switch_timer_label.configure(text=f"Switch Timer: {switch_out_countdown}")

            # Draw rectangles around the ROI
            roi_color = (0, 255, 0)  
            screen_with_rois = cv2.rectangle(screen.copy(), (my_roi[0], my_roi[1]), (my_roi[0] + my_roi[2], my_roi[1] + my_roi[3]), roi_color, 2)
            screen_with_rois = cv2.rectangle(screen_with_rois, (opp_roi[0], opp_roi[1]), (opp_roi[0] + opp_roi[2], opp_roi[1] + opp_roi[3]), roi_color, 2)

            if self.current_my_pokemon_index is not None and self.current_opp_pokemon_index is not None:
                my_count = self.find_label('my',self.current_my_pokemon_index,'fast_move').get()[-1]
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
        frames_map = {'my': self.my_pokemon_frames, 'opp': self.opp_pokemon_frames}
        frame = frames_map[side][index]
        return getattr(frame, name_of_variable)


    def get_index(self,combo_box):
        current_value = combo_box.get()
        values = combo_box.cget("values")
        return values.index(current_value)

if __name__ == "__main__":
    app = PokemonBattleAssistant(feed_res,cup_names_combo_box)
    app.after(update_timer, lambda: app.update_ui(client)) 
    app.mainloop()

