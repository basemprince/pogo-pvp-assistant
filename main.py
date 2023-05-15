#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import time
# import matplotlib.pyplot as plt
# %matplotlib inline
import json
import re
import math
from IPython.display import clear_output
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import pandas as pd
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
import os
import utils
import tkinter as tk
import customtkinter as ctk
import threading


# In[5]:


# parameters
record_video = False
print_out = False
display_img = True
img_scale = 0.1
update_timer = 1
update_json_files = False
phones = ['Pixel 3 XL', 'Pixel 7 Pro']


# In[6]:


# Load the JSON files
pokemon_names = utils.load_pokemon_names()
moves = utils.load_moves_info()
# load alignment info
alignment_df = utils.load_alignment_df()
# update json files if prompted:
if update_json_files:
    utils.update_json_files()
# connect to phone
client = utils.connect_to_device("127.0.0.1:5037")
phone_t = phones.index(client.device_name)


# In[7]:


roi_adjust =[[50,370,860],[50,350,860]]
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


# In[8]:


class PokemonFrame(ctk.CTkFrame):
    def __init__(self, master, name):
        super().__init__(master)

        self.pokemon_name_label = ctk.CTkLabel(self, text=name, font=("Arial", 20))
        self.pokemon_name_label.grid(column=0, row=0, sticky="W", padx=10, pady=10)

        self.fast_move = ctk.CTkLabel(self, text="Fast Move")
        self.fast_move.grid(column=0, row=1, sticky="W", padx=10, pady=10)

        self.charge_move1 = ctk.CTkLabel(self, text="Charge Move 1")
        self.charge_move1.grid(column=0, row=2, sticky="W", padx=10, pady=10)

        self.charge_move2 = ctk.CTkLabel(self, text="Charge Move 2")
        self.charge_move2.grid(column=0, row=3, sticky="W", padx=10, pady=10)

    def highlight_on(self):
        self.configure(fg_color="green")
    def highlight_off(self):
        self.configure(fg_color="transparent")

class PokemonBattleAssistant(ctk.CTk):
    def __init__(self,feed_res):
        super().__init__()
        self.title("Pokemon Battle Assistant")
        self.feed_res = feed_res

        mainframe = ctk.CTkFrame(self)
        mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=30, pady=20)

        # Add the drop down menu
        self.league_combobox = ctk.CTkComboBox(mainframe, values=['choose league','Great League', 'Ultra League', 'Master League'])
        self.league_combobox.grid(column=0, row=0,sticky="W", padx=0, pady=10)

        opponent_frame = ctk.CTkLabel(mainframe, text="Opponent's Pokemon", text_color= 'gray', anchor="nw")
        opponent_frame.grid(column=0, row=1, sticky=(tk.W, tk.E), padx=0, pady=(0,20))
        opponent_frame.grid_columnconfigure((0, 1, 2), weight=1) 

        self.opp_pokemon_frames = [PokemonFrame(opponent_frame, f"Pokemon {i + 1}") for i in range(3)]
        for i, frame in enumerate(self.opp_pokemon_frames):
            frame.grid(column=i, row=1, padx=10, pady=(20,0))

        info_frame = ctk.CTkFrame(mainframe)
        info_frame.grid(column=0, row=2, sticky=(tk.W, tk.E), padx=20, pady=20)

        self.switch_timer_label = ctk.CTkLabel(info_frame, text="Switch Timer: ")
        self.switch_timer_label.grid(column=0, row=2, sticky="W", padx=(10,70), pady=10)

        self.correct_alignment_label = ctk.CTkLabel(info_frame, text="Correct Alignment: ")
        self.correct_alignment_label.grid(column=1, row=2, sticky="W", padx=(0,70), pady=10)

        ctk.CTkButton(info_frame, text="Reset UI", command=self.reset_ui).grid(column=2, row=2, padx=10, pady=10)
        self.start_button = ctk.CTkButton(info_frame, text="Start Recording", command=self.recoding)
        self.start_button.grid(column=3, row=2, padx=10, pady=10)

        my_frame = ctk.CTkLabel(mainframe, text="Your Pokemon",text_color= 'gray',anchor="nw")
        my_frame.grid(column=0, row=3, sticky=(tk.W, tk.E), padx=0, pady=0)
        my_frame.grid_columnconfigure((0, 1, 2), weight=1) 

        self.my_pokemon_frames = [PokemonFrame(my_frame, f"Pokemon {i + 1}") for i in range(3)]
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

        self.vid_res = (int(client.resolution[0]/2), int(client.resolution[1]/2))
        self.record_vid = False
        self.out = None

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
        self.opp_pokemon_memory = []
        self.my_pokemon_memory = []
        self.switch_out_time = None
        self.switch_out_countdown = None
        self.switch_memory = []
        self.opp_switch_timer_label = None
        self.league_pok = None
        for number in range(len(self.my_pokemon_frames)):
            self.update_my_label(number, 'pokemon_name_label', f'Pokemon {number+1}')
            for move,move_disp in zip(self.move_type,self.move_type_disp):
                self.update_my_label(number, move, move_disp)
        for number in range(len(self.opp_pokemon_frames)):
            self.update_opp_label(number, 'pokemon_name_label', f'Pokemon {number+1}')
            for move,move_disp in zip(self.move_type,self.move_type_disp):
                self.update_opp_label(number, move, move_disp)
        self.switch_timer_label.configure(text=f"Switch Timer: ")
        self.correct_alignment_label.configure(text="Correct Alignemnt: ")
        for my_frame,opp_frame in zip(self.my_pokemon_frames,self.opp_pokemon_frames):
            my_frame.highlight_off()
            opp_frame.highlight_off()

    def vid_stream(self):
        while self.record_vid:
            screen = client.last_frame
            resized_frame = cv2.resize(screen, self.vid_res)
            self.out.write(resized_frame)
            time.sleep(0.01) 

    def update_ui(self,clinet):
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

                if self.league is None:
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
                        if higher_cp <= 1500:
                            self.league = "great-league"
                        elif 1500 < higher_cp <= 2500:
                            self.league = "ultra-league"
                        else:
                            self.league = "master-league"

                        print(f"League: {self.league}")
                        self.league_pok = f"json_files/{self.league.lower()}.json"
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

                    # Calculate the closest Pokémon names
                    temp_corrected_my_name = utils.closest_pokemon_name(my_info_name, pokemon_names)
                    temp_corrected_opp_name = utils.closest_pokemon_name(opp_info_name, pokemon_names)

                    # Check if either Pokémon is Giratina and change form based on the league
                    if temp_corrected_my_name == "Giratina":
                        if self.league in ["great-league", "ultra-league"]:
                            temp_corrected_my_name = "Giratina (Altered)"
                        else:
                            temp_corrected_my_name = "Giratina (Origin)"

                    if temp_corrected_opp_name == "Giratina":
                        if self.league in ["great-league", "ultra-league"]:
                            temp_corrected_opp_name = "Giratina (Altered)"
                        else:
                            temp_corrected_opp_name = "Giratina (Origin)"

                    if temp_corrected_my_name:
                        corrected_my_name = temp_corrected_my_name
                        self.prev_corrected_my_name = corrected_my_name
                    else:
                        corrected_my_name = self.prev_corrected_my_name
                        if print_out:
                            print(f"Using previous Pokémon name for My Pokémon: {corrected_my_name}")

                    if temp_corrected_opp_name:
                        corrected_opp_name = temp_corrected_opp_name
                        self.prev_corrected_opp_name = corrected_opp_name
                        if not self.switch_memory or corrected_opp_name != self.switch_memory[-1]:
                            if len(self.switch_memory) >= 3:
                                self.switch_memory.pop(0)
                            self.switch_memory.append(corrected_opp_name)
                            self.switch_out_time = time.time()
                    else:
                        corrected_opp_name = self.prev_corrected_opp_name
                        if print_out:
                            print(f"Using previous Pokémon name for Opp Pokémon: {corrected_opp_name}")


                    if corrected_opp_name:
                        opp_move_counts, opp_fast_move_turns = utils.get_moveset_and_counts_udpated(corrected_opp_name, self.league_pok, moves)
                        if opp_move_counts is not None:
                            # Check if the Pokémon is already in the memory before adding it
                            if not any(pokemon[0] == corrected_opp_name for pokemon in self.opp_pokemon_memory):
                                # If memory is already full, remove the oldest Pokémon
                                if len(self.opp_pokemon_memory) >= 3:
                                    self.opp_pokemon_memory.pop(0)
                                # Save the Pokémon and their moveset into a memory
                                self.opp_pokemon_memory.append([corrected_opp_name, opp_move_counts])

                            # Updating opponent Pokémon
                            for pokemon_number, pokemon_data in enumerate(self.opp_pokemon_memory):
                                pokemon_name, move_counts = pokemon_data
                                self.update_opp_label(pokemon_number, 'pokemon_name_label', pokemon_name)
                                if move_counts is not None:
                                    for move in self.move_type:
                                        self.update_opp_label(pokemon_number, move, move_counts[move])
  
                            current_pokemon_index = next((index for index, pokemon_data in enumerate(self.opp_pokemon_memory) if pokemon_data[0] == corrected_opp_name), None)
                            # Toggle highlight for the current Pokémon
                            if current_pokemon_index is not None:
                                for i, frame in enumerate(self.opp_pokemon_frames):
                                    if i == current_pokemon_index:
                                        frame.highlight_on()
                                    else:
                                        frame.highlight_off()

                            if print_out:
                                print(f"Opponent Pokémon: {corrected_opp_name}")
                                print(opp_move_counts)
                        else:
                            if print_out:
                                print("Error getting opponent moveset.")

                    if corrected_my_name:
                        my_move_counts, my_fast_move_turns = utils.get_moveset_and_counts_udpated(corrected_my_name, self.league_pok, moves)
                        if my_move_counts is not None:
                            # Check if the Pokémon is already in the memory before adding it
                            if not any(pokemon[0] == corrected_my_name for pokemon in self.my_pokemon_memory):
                                # If memory is already full, remove the oldest Pokémon
                                if len(self.my_pokemon_memory) >= 3:
                                    self.my_pokemon_memory.pop(0)
                                # Save the Pokémon and their moveset into a memory
                                self.my_pokemon_memory.append([corrected_my_name, my_move_counts])

                            # Updating my Pokémon
                            for pokemon_number, pokemon_data in enumerate(self.my_pokemon_memory):
                                pokemon_name, move_counts = pokemon_data
                                self.update_my_label(pokemon_number, 'pokemon_name_label', pokemon_name)
                                if move_counts is not None:
                                    for move in self.move_type:
                                        self.update_my_label(pokemon_number, move, move_counts[move])

                            current_pokemon_index = next((index for index, pokemon_data in enumerate(self.my_pokemon_memory) if pokemon_data[0] == corrected_my_name), None)
                            # Toggle highlight for the current Pokémon
                            if current_pokemon_index is not None:
                                for i, frame in enumerate(self.my_pokemon_frames):
                                    if i == current_pokemon_index:
                                        frame.highlight_on()
                                    else:
                                        frame.highlight_off()

                            if print_out:
                                print(f"My Pokémon: {corrected_my_name}")
                                print(my_move_counts)
                        else:
                            if print_out:
                                print("Error getting my moveset.")


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

            if corrected_my_name is not None and corrected_opp_name is not None:
                my_count = my_fast_move_turns
                opp_count = opp_fast_move_turns
                try:
                    correct_count = alignment_df.loc[my_count, str(opp_count)]
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

    def update_opp_label(self, index, name_of_variable, new_value):
        frame = self.opp_pokemon_frames[index]
        label = getattr(frame, name_of_variable)
        label.configure(text=new_value)


    def update_my_label(self, index, name_of_variable, new_value):
        frame = self.my_pokemon_frames[index]
        label = getattr(frame, name_of_variable)
        label.configure(text=new_value)

if __name__ == "__main__":
    app = PokemonBattleAssistant(feed_res)
    app.after(update_timer, lambda: app.update_ui(client)) 
    app.mainloop()

