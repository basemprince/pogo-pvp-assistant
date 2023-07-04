#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import battle_tracker
import sys


# In[2]:


# parameters
print_out = False
display_img = True
img_scale = 0.1
update_timer = 50
alignment_count_display = 5
roi_color = (0, 0, 0)  
roi_thick = 12
update_json_files = False
update_pokemon = False
ui_printout = True


# In[3]:


# Load the JSON files
pokemon_names = utils.load_pokemon_names()
pokemon_details = utils.load_pokemon_details()
moves = utils.load_moves_info()

# load alignment info
alignment_df = utils.load_alignment_df(alignment_count_display)
    
if update_pokemon:
    utils.update_pk_info()
    utils.update_move_info()
cup_names_combo_box = utils.update_leagues_and_cups(update_json_files)


# In[4]:


# connect to phone
client = utils.connect_to_device("127.0.0.1:5037")
roi_dict = utils.get_phone_data(client)
feed_res = (int(client.resolution[0]*img_scale), int(client.resolution[1]*img_scale))


# In[5]:


class PokemonBattleAssistant(ctk.CTk):
    def __init__(self,update_timer,feed_res,cup_names):
        super().__init__()
        self.title("Pokemon Battle Assistant")
        self.feed_res = feed_res
        self.update_timer = update_timer
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

        # frame to hold command line output and image
        output_image_frame = ctk.CTkFrame(mainframe)
        output_image_frame.grid(column=0, row=4, sticky=(tk.W, tk.E), padx=0, pady=0)

        # Add command line output 
        self.command_line_output = tk.Text(output_image_frame, bg='black', fg='white', height=pil_image.height/17, width=110)

        if display_img:
            output_image_frame.grid_columnconfigure(0, weight=1)
            output_image_frame.grid_columnconfigure(1, weight=1)
            self.command_line_output.grid(column=0, row=0, sticky=(tk.W, tk.E), padx=0, pady=0)

            # Add image
            self.my_image = ctk.CTkImage(light_image=pil_image,dark_image=pil_image, size=feed_res)
            self.image_label = ctk.CTkLabel(output_image_frame, text='',image=self.my_image)
            self.image_label.grid(column=1, row=0, pady=0)
        else:
            output_image_frame.grid_columnconfigure(0, weight=2)  # Set the weight to a larger value
            self.command_line_output.grid(column=0, row=0, sticky=(tk.W, tk.E), padx=0, pady=0)


        self.vid_res = (int(client.resolution[0]/2), int(client.resolution[1]/2))
        self.threshold = 500
        self.ui_reset_counter = 10
        self.get_ready_keywords = ['get', 'ready']
        self.attack_incoming_keywords = ['attack', 'incoming']


        # to push output to UI
        if ui_printout:
            sys.stdout = utils.TextRedirector(self.command_line_output)
            sys.stderr = utils.TextRedirector(self.command_line_output)
        self.initialize_variables()

    def create_pokemon_frame(self, master, num, side):
        frame = ctk.CTkFrame(master)
        box_width = 250

        frame.pokemon_name_label = ctk.CTkComboBox(frame, values=[f"Pokemon {num+1}"], font=("Arial", 20), command=lambda choice,
                                box_type='pokemon_name_label', side=side, num=num: self.pk_callback(choice, box_type, side, num), width=200)
        frame.pokemon_name_label.grid(column=0, row=0, sticky="W", padx=10, pady=10)

        frame.fast_move = ctk.CTkComboBox(frame, values=["Fast Move"], command=lambda choice, box_type='fast_move', 
                                          side=side, num=num: self.pk_callback(choice, box_type, side, num), width=box_width)
        frame.fast_move.grid(column=0, row=1, sticky="W", padx=10, pady=10)

        frame.charge_move1 = ctk.CTkComboBox(frame, values=["Charge Move 1"], command=lambda choice, box_type='charge_move1', 
                                          side=side, num=num: self.pk_charge_callback(choice, box_type, side, num), width=box_width)
        frame.charge_move1.grid(column=0, row=2, sticky="W", padx=10, pady=10)

        frame.charge_move1_progress = ctk.CTkProgressBar(frame, orientation="horizontal", height=20, width=50,corner_radius=0,
                                                         fg_color='gray',border_color='black')
        frame.charge_move1_progress.set(0) 
        frame.charge_move1_progress.grid(column=1, row=2, sticky="W", padx=10, pady=10)

        frame.charge_move2 = ctk.CTkComboBox(frame, values=["Charge Move 2"], command=lambda choice, box_type='charge_move2', 
                                          side=side, num=num: self.pk_charge_callback(choice, box_type, side, num), width=box_width)
        frame.charge_move2.grid(column=0, row=3, sticky="W", padx=10, pady=10)

        frame.charge_move2_progress = ctk.CTkProgressBar(frame, orientation="horizontal", height=20, width=50,corner_radius=0,
                                                        fg_color='gray',border_color='black')
        frame.charge_move2_progress.set(0) 
        frame.charge_move2_progress.grid(column=1, row=3, sticky="W", padx=10, pady=10)

        return frame

    def initialize_variables(self):

        self.prev_my_roi_img = np.array([])
        self.prev_opp_roi_img = np.array([])
        self.prev_my_pokeballs_img = np.array([])
        self.prev_opp_pokeballs_img = np.array([])
        self.prev_my_typing_img = np.array([])
        self.prev_opp_typing_img = np.array([])

        self.my_info_match = None
        self.opp_info_match = None
        self.league = None
        self.league_pok = None
        self.extract_throw_time_helper = 0
        self.Player_throw_move = None

        self.move_type = ['fast_move','charge_move1','charge_move2']
        self.move_type_disp = ['Fast Move','Charge Move 1','Charge Move 2']

        self.league_detector = utils.LeagueDetector()

        self.record_vid = False
        self.out = None

        self.my_player = battle_tracker.Player('me')
        self.opp_player = battle_tracker.Player('opp')
        self.match = battle_tracker.Match(alignment_count_display)

        self.frames_map = {'me': self.my_pokemon_frames,'my': self.my_pokemon_frames, 'opp': self.opp_pokemon_frames}
        self.player_map = {'me': self.my_player, 'my': self.my_player, 'opp': self.opp_player}

        self.my_typing_is_correct = False
        self.opp_typing_is_correct = False

        self.my_emblems = []
        self.opp_emblems = []

    def reset_ui(self,reset_league = True):
        self.initialize_variables()
        if reset_league:
            self.league_combobox.set('choose league')
        else:
            current_choice = self.league_combobox.get()
            self.league_callback(current_choice)
            print('UI has been reset and chosen league reloaded')
        for side in ['opp','me']:
            for number in range(len(self.frames_map[side])):
                self.update_label(side,number, 'pokemon_name_label', f'Pokemon {number+1}')
                for move,move_disp in zip(self.move_type,self.move_type_disp):
                    self.update_label(side,number, move, move_disp)
                for charge_mv_num in [1,2]:
                    progress = self.find_label(side,number,f'charge_move{charge_mv_num}_progress')
                    progress.set(0)
                    colors = self.progress_bar_color(0)
                    progress.configure(fg_color=colors[1],progress_color=colors[2])
                self.highlight_off(self.frames_map[side][number])

        self.switch_timer_label.configure(text=f"Switch Timer: ")
        self.correct_alignment_label.configure(text="Correct Alignemnt: ")


    def pk_callback(self, choice, box_type, side, num):
 
        label = self.find_label(side,num,'pokemon_name_label')
        chosen_pk_ind = self.get_index(label)
        self.player_map[side].ui_chosen_pk_ind[num] = chosen_pk_ind

        mv_recom = self.player_map[side].pokemons[num][chosen_pk_ind].recommended_moveset
        _, fast_moves, charge_moves = self.player_map[side].ui_helper(num,chosen_pk_ind)
        if box_type == 'pokemon_name_label':
            self.update_label(side, num, 'fast_move', fast_moves)
            fast_mv_label = self.find_label(side,num,'fast_move')
            fast_mv = self.player_map[side].pokemons[num][chosen_pk_ind].fast_moves[mv_recom[0]]
            fast_mv_label.set(fast_mv.move_count_str())
        else:
            fast_mv = choice.split(' - ')[0].upper().replace(" ", "_")
            self.player_map[side].pokemons[num][chosen_pk_ind].ui_chosen_moveset[0] = fast_mv
            fast_mv = self.player_map[side].pokemons[num][chosen_pk_ind].fast_moves[fast_mv]

        for i in [1,2]:
            self.update_label(side,num, f'charge_move{i}', charge_moves)
            charge_label = self.find_label(side,num,f'charge_move{i}')
            charge_label.set(self.player_map[side].pokemons[num][chosen_pk_ind].charge_moves[mv_recom[i]].move_count_str(fast_mv.move_id))

    def pk_charge_callback(self,choice,box_type,side,num):
        charge_num= int(box_type[-1])
        move = choice.split(' - ')[0].upper().replace(" ", "_")
        self.player_map[side].pokemons[num][self.player_map[side].ui_chosen_pk_ind[num]].ui_chosen_moveset[charge_num] = move


    def charge_move_progress(self):
        try:
            for side in ['opp','me']:
                if self.player_map[side].initialized:
                    num = self.player_map[side].current_pokemon_index
                    chosen_pk_ind = self.player_map[side].ui_chosen_pk_ind[num]

                    fast_mv = self.player_map[side].pokemons[num][chosen_pk_ind].ui_chosen_moveset[0]
                    for i in [1,2]:
                        charge_mv = self.player_map[side].pokemons[num][chosen_pk_ind].ui_chosen_moveset[i]
                        accum_energy = self.player_map[side].pokemons[num][chosen_pk_ind].charge_moves[charge_mv].accum_energy[fast_mv]
                        progress = self.find_label(side,num,f'charge_move{i}_progress')
                        colors = self.progress_bar_color(accum_energy)
                        progress.set(colors[0])
                        progress.configure(fg_color=colors[1],progress_color=colors[2])
        except Exception as e:
            print(f"Error in charge_move_progress: {str(e)}")

    def progress_bar_color(self,energy):
        if 0 <= energy < 1:
            return (energy,"gray", "#378df0")
        elif 1 <= energy < 2:
            return (energy-1,"#378df0", "#1a2182")
        elif 2 <= energy <= 3:
            return (energy-2,"#1a2182", "#080936")

    def highlight_on(self, frame):
        if frame.cget('fg_color') != "green":
            frame.configure(fg_color="green")

    def highlight_red(self, frame):
        if frame.cget('fg_color') != "red":
            frame.configure(fg_color="red")

    def highlight_off(self, frame):
        if frame.cget('fg_color') != "transparent":
            frame.configure(fg_color="transparent")

    def update_highlight(self, side):
        player = self.player_map[side]
        current_ind = player.current_pokemon_index
        frames = self.frames_map[side]
        for i, frame in enumerate(frames):
            if player.pk_fainted[i]:
                self.highlight_red(frame)
            elif i == current_ind:
                self.highlight_on(frame)
            else:
                self.highlight_off(frame)

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


    def extract_thrown_move(self, ocr_output,time_threshold = 2):
        current_time = time.time()

        if current_time - self.extract_throw_time_helper < time_threshold:
            return None, None

        ocr_output_strip = ocr_output.lower().strip()
        # Check for who threw the move
        if any(word in ocr_output_strip for word in self.get_ready_keywords):
            self.Player_throw_move = self.my_player
            self.extract_throw_time_helper = current_time
            print('Detected my player throwing a move')
        elif any(word in ocr_output_strip for word in self.attack_incoming_keywords):
            self.Player_throw_move = self.opp_player
            self.extract_throw_time_helper = current_time
            print('Detected opp player throwing a move')

        match = re.search(r'(.+?) used (.+)', ocr_output)
        if match:
            pokemon, move = match.groups()

            if self.Player_throw_move:
                # Get the list of the player's pokemons
                player_pokemons = [pokemon for pokemon in self.Player_throw_move.pk_battle_name if pokemon is not None]
                closest = utils.closest_name(pokemon, player_pokemons)

                if closest in player_pokemons:
                    closest_index = player_pokemons.index(closest)
                    self.Player_throw_move.pokemon_energy_consumer(move, closest_index)
                    self.extract_throw_time_helper = current_time
                    self.Player_throw_move = None 
                else:
                    print(f"Message info did not match {pokemon, move}")
            else:
                my_pk = self.my_player.pk_battle_name[self.my_player.current_pokemon_index]
                opp_pk = self.opp_player.pk_battle_name[self.opp_player.current_pokemon_index]
                closest = utils.closest_name(pokemon, [my_pk, opp_pk])
                self.Player_throw_move = None 
                if closest == my_pk:
                    self.my_player.pokemon_energy_consumer(move)
                    self.extract_throw_time_helper = current_time
                elif closest == opp_pk:
                    self.opp_player.pokemon_energy_consumer(move)
                    self.extract_throw_time_helper = current_time
                else:
                    print(f"Message info did not match {pokemon, move}")

            return pokemon, move
        else:
            return None, None
      
    def vid_stream(self):
        while self.record_vid:
            frame = client.last_frame
            resized_frame = cv2.resize(frame, self.vid_res)
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
        pk_label = self.find_label(side,current_ind,'pokemon_name_label')
        pk_name = self.player_map[side].pokemons[current_ind][self.player_map[side].recommended_pk_ind[current_ind]].species_name
        pk_label.set(pk_name)

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


    def ball_counter(self, player_id, pokeballs_roi):
        player = self.player_map[player_id]
        pokeballs_count = utils.count_pokeballs(pokeballs_roi)
        if pokeballs_count < player.pokeball_count:
            print(f"{player_id.capitalize()}'s Pokemon fainted!")
            player.pokemon_energy_updater(False)
            player.pokeball_count = pokeballs_count
            player.pk_fainted[player.current_pokemon_index] = True
        else:
            player.pokemon_energy_updater(True)
        return pokeballs_count

    def update_pokeballs_counts(self, roi_images):
        fainted = False

        for side in ['my', 'opp']:
            pokeballs_roi = roi_images[f'{side}_pokeballs_roi']
            player = self.player_map[side]

            pokeballs_count = utils.count_pokeballs(pokeballs_roi)

            if pokeballs_count < player.pokeball_count:
                print(f"{side.capitalize()}'s Pokemon fainted!")
                player.pokeball_count = pokeballs_count
                player.pk_fainted[player.current_pokemon_index] = True
                fainted = True

        for player in self.player_map.values():
            player.pokemon_energy_updater(not fainted)
        

    def ocr_detect(self,img):
        with PyTessBaseAPI(psm=PSM.AUTO_OSD) as api:
            api.SetImage(img)
            api.Recognize()
            return api.GetUTF8Text()


    def handle_emblem_update(self, update_me, update_opp, roi_images):
        try:
            if update_me:
                self.moveset_update('me')
                if 'my_typing_roi' in roi_images:
                    my_emblems = utils.detect_emblems(roi_images['my_typing_roi'])
                    self.my_typing_is_correct = True if set(my_emblems) == set(self.my_player.on_field_typing) else False
                else:
                    print("'my_typing_roi' not found in 'roi_images'.")

            if update_opp:
                self.moveset_update('opp')
                if 'opp_typing_roi' in roi_images:
                    opp_emblems = utils.detect_emblems(roi_images['opp_typing_roi'])
                    self.opp_typing_is_correct = True if set(opp_emblems) == set(self.opp_player.on_field_typing) else False
                else:
                    print("'opp_typing_roi' not found in 'roi_images'.")
        except Exception as e:
            print(f"Error in handle_emblem_update: {str(e)}")

    def update_ui(self,client):
        loop_start_time = time.time()
        frame = client.last_frame
        # frame = cv2.imread('templates/screenshot.png')
        if frame is not None:

            roi_images = utils.get_roi_images(frame,roi_dict)

            # if self.match.match_started():
            #     if not self.match.charge_mv_event:
            #         self.my_emblems = utils.detect_emblems(roi_images['my_typing_roi'],30)
            #         self.opp_emblems = utils.detect_emblems(roi_images['opp_typing_roi'],30)
            #         # print(f'my typings: ({len(self.my_emblems)}) {self.my_emblems}, opp typings: ({len(self.opp_emblems)}){self.opp_emblems}')
            #         # print(f'my_player_typing: {self.my_player.on_field_typing}, opp_player_typing: {self.opp_player.on_field_typing}')

            #         if self.my_typing_is_correct and set(self.my_emblems) != set(self.my_player.on_field_typing):
            #             # print("early my pokemon switch detected")
            #             self.my_typing_is_correct = False
            #             # self.my_player.pokemon_energy_updater(False)

            #         if self.opp_typing_is_correct and set(self.opp_emblems) != set(self.opp_player.on_field_typing):
            #             # print("early opp pokemon switch detected")
            #             self.opp_typing_is_correct = False
            #             # self.opp_player.pokemon_energy_updater(False)

                                        
            if utils.mse(roi_images['my_roi'], self.prev_my_roi_img) > self.threshold \
            or utils.mse(roi_images['opp_roi'], self.prev_opp_roi_img) > self.threshold:
                
                self.prev_my_roi_img, thresh_my_roi = utils.process_image(roi_images['my_roi'])
                self.prev_opp_roi_img, thresh_opp_roi = utils.process_image(roi_images['opp_roi'])
                self.prev_msg_roi_img, thresh_msg_roi = utils.process_image(roi_images['msgs_roi'])

                my_info = self.ocr_detect(thresh_my_roi)
                opp_info = self.ocr_detect(thresh_opp_roi)

                if print_out:
                    print("My Info:", my_info)
                    print("Opponent Info:", opp_info)

                # for auto-detection if league not chosen
                if self.league_combobox.get() == 'choose league':
                    self.league, self.league_pok = self.league_detector.detect_league(my_info, opp_info)
                    if self.league:
                        self.league_combobox.set(self.league)
                    
                # Extract Pokémon names
                self.my_info_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', my_info)
                self.opp_info_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', opp_info)

                if self.league:
                    if self.my_info_match and self.opp_info_match:
                        
                        my_info_name = self.my_info_match.group(0)
                        opp_info_name = self.opp_info_match.group(0)

                        my_pk, my_pk_name = battle_tracker.load_pk_data(my_info_name,pokemon_names,pokemon_details,moves,self.league_pok)
                        opp_pk, opp_pk_name = battle_tracker.load_pk_data(opp_info_name,pokemon_names,pokemon_details,moves,self.league_pok)


                        update_me = self.my_player.add_pokemon(my_pk,my_pk_name)
                        update_opp = self.opp_player.add_pokemon(opp_pk,opp_pk_name)

                        if not self.match.match_started() and self.my_player.pokemon_count !=0:
                            print('Battle Started')
                            self.match.start_match()

                        self.my_player.start_update()
                        self.opp_player.start_update()

                        self.handle_emblem_update(update_me, update_opp, roi_images)
                    
                        self.update_highlight('me')
                        self.update_highlight('opp')
                        self.match.charge_mv_event = False
                        self.update_timer = update_timer
                    else:
                        if self.match.match_started():
                            self.match.charge_mv_event = True
                            self.update_timer = 20
                            self.opp_player.pokemon_energy_updater(False)
                            self.my_player.pokemon_energy_updater(False)
                            
                            msg_info = self.ocr_detect(thresh_msg_roi)
                            pk, chr_mv = self.extract_thrown_move(msg_info)               

        
            if self.match.match_started():
                if not self.match.charge_mv_event:
                    self.update_pokeballs_counts(roi_images)

                    if self.my_player.pokeball_count == 0 and self.opp_player.pokeball_count == 0 and not self.match.end_time is not None:
                        print(f'End of match detected. UI resets in {self.ui_reset_counter} seconds')
                        self.match.end_match()

                self.charge_move_progress()
                correct_count = self.match.calculate_correct_alignment(self.my_player,self.opp_player)
                self.correct_alignment_label.configure(text=f"Correct Alignemnt: {correct_count}")
                # opponent switch lock timer
                if self.opp_player.switch_lock and self.opp_player.pokemon_count>1:
                    self.opp_player.countdown_switch_lock()
                    self.switch_timer_label.configure(text=f"Switch Timer: {self.opp_player.switch_lock_timer}")

            # Draw ROIs and display frames
            if display_img:
                pil_img = utils.draw_display_frames(frame, roi_dict, self.feed_res)
                self.my_image.configure(light_image=pil_img,dark_image=pil_img)

        if self.match.end_time is not None:
            if loop_start_time - self.match.end_time >= self.ui_reset_counter:
                self.reset_ui(False)
        time_elapsed = time.time() - loop_start_time
        self.elapsed_time_label.configure(text=f'{time_elapsed:0.3f}')
        self.after(self.update_timer, lambda: app.update_ui(client)) 

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
    app = PokemonBattleAssistant(update_timer,feed_res,cup_names_combo_box)
    app.after(update_timer, lambda: app.update_ui(client)) 
    app.mainloop()

